# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import pdb

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm
import torch.distributed as dist

from base import BaseTrainer, Multi_BaseTrainer_dist
from model.model import sim_matrix
from utils import inf_loop

class AllGather_multi(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, n_gpu, args):
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = args.rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None, None,
        )

class Multi_Trainer_dist_HT100M_Classification(Multi_BaseTrainer_dist):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, args, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, writer=None,
                 visualizer=None, tokenizer=None, max_samples_per_epoch=50000, start_epoch=1):
        super().__init__(args, model, loss, metrics, optimizer, config, writer, start_epoch=start_epoch)
        self.config = config
        self.args = args
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            # take the min
            self.len_epoch = min([len(x) for x in data_loader])
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.visualizer = visualizer
        self.val_chunking = True
        self.batch_size = self.data_loader[0].batch_size
        self.val_batch_size = self.valid_data_loader[0].batch_size
        self.log_step = int(np.sqrt(self.batch_size))
        self.total_batch_sum = sum([x.batch_size for x in self.data_loader])
        self.tokenizer = tokenizer
        self.max_samples_per_epoch = max_samples_per_epoch
        self.n_gpu = self.args.world_size
        self.allgather = AllGather_multi.apply
        # self.writer = writer
        # Freeze weights for whole network except the head
        for name, param in self.model.named_parameters():
            if 'ht100m' not in name:
                param.requires_grad = False


    def _eval_metrics(self, output):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output)
            # if self.writer is not None:
            #     self.writer.log_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _adjust_learning_rate(self, optimizer, epoch, args):
        lr = args.learning_rate1
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
        print('[INFO] Learning rate for next epoch is: {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        print('Starting for epoch: {}, Trainable parameters: {}'.format(epoch, sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        self.model.train()
        total_loss = [0] * len(self.data_loader)
        total_metrics = np.zeros(len(self.metrics))
        for loader in self.data_loader:
            loader.train_sampler.set_epoch(epoch)
        for batch_idx, data_li in enumerate(zip(*self.data_loader)):
            if (batch_idx + 1) * self.total_batch_sum > self.max_samples_per_epoch:
                break
            for dl_idx, data in enumerate(data_li):
                data['video'] = data['video'].to(self.device)
                data['label'] = data['label'].to(self.device)
                # w_embeds = data['relation'].to(self.device)

                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    video_embeds = self.model.module.compute_video(data['video'])
                    video_embeds = self.model.module.compute_video_aggregation(video_embeds, self.batch_size)
                    
                    video_predictions = self.model.module.head_ht100m_linear_probe(video_embeds)
                    
                    video_predictions = self.allgather(video_predictions.contiguous(), self.n_gpu, self.args)
                    video_labels = self.allgather(data['label'], self.n_gpu, self.args)
                    # w_embeds = self.allgather(w_embeds, self.n_gpu, self.args)

                    loss = self.loss(video_predictions, video_labels)
                    # loss = self.loss(output, w_embeds)
                loss.backward()

                self.optimizer.step()

                if self.writer is not None and self.args.rank == 0:
                    # self.writer.log_scalar(f'loss_train_{dl_idx}', loss.detach().item())
                    total = int(self.data_loader[dl_idx].n_samples/self.n_gpu)
                    current = batch_idx * self.data_loader[dl_idx].batch_size
                    final_total = (epoch-1) * total + current
                    self.writer.add_scalar(f'Loss_training/loss_{dl_idx}', loss.detach().item(), final_total)

                total_loss[dl_idx] += loss.detach().item()

                # if batch_idx % self.log_step == 0 and self.args.local_rank == 0:
                if batch_idx % self.log_step == 0 and self.args.rank == 0:
                    self.logger.info('Train Epoch: {} dl{} {} Loss: {:.6f}'.format(
                        epoch,
                        dl_idx,
                        self._progress(batch_idx, dl_idx),
                        loss.detach().item()))
                    print('Train Epoch: {} dl{} {} Loss: {:.6f}'.format(
                        epoch,
                        dl_idx,
                        self._progress(batch_idx, dl_idx),
                        loss.detach().item()), flush=True)

                self.optimizer.zero_grad()
            if batch_idx == self.len_epoch:
                break

        log = {
            f'loss_{dl_idx}': total_loss[dl_idx] / self.len_epoch for dl_idx in range(len(self.data_loader))
        }

        if self.writer is not None and self.args.rank == 0:
            for dl_idx in range(len(self.data_loader)):
                tl = total_loss[dl_idx] / self.len_epoch
                self.writer.add_scalar(f'Loss_training/loss_total_{dl_idx}', tl, epoch-1)

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            if self.args.rank == 0:
                log.update(val_log)

        #self._adjust_learning_rate(self.optimizer, epoch, self.args)

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = [0] * len(self.valid_data_loader)
        meta_arr = {x: [] for x in range(len(self.valid_data_loader))}
        vid_target_arr = {x: [] for x in range(len(self.valid_data_loader))}
        vid_pred_arr = {x: [] for x in range(len(self.valid_data_loader))}

        with torch.no_grad():
            # for validation we switch the nested loop order, because alternate batches not needed...
            # ... and dataloaders can be of different length
            for dl_idx, dl in enumerate(self.valid_data_loader):
                for data in tqdm(dl, desc=f"Validating dl{dl_idx}"):
                    meta_arr[dl_idx].append(data['meta'])
                    data['video'] = data['video'].to(self.device)
                    data['label'] = data['label'].to(self.device)

                    # Note that if the batch is not scattered among all the GPUs, `DataParallel` will fail because
                    # the model's mandatory argument `data` will not be passed to some of them.
                    # It can happen with the last batch of the dataset, depending on its size.
                    # It could be safely ignored during training but on validation/test we want accurate metrics.
                    # This avoids using `DataParallel` in this case, and supposes this batch fits in one GPU.
                    # TODO: Correct last batch issue: current_batch_size = data['video'].shape[0]
                    current_batch_size = data['label'].shape[0]
                    if current_batch_size != self.val_batch_size:
                        print('Skipping last batch...')
                        continue
                    #if isinstance(self.model, nn.DataParallel) and current_batch_size < (dl.batch_size or 1):
                    #    scattered_len = len(self.model.scatter([torch.empty(current_batch_size)], {},
                    #                                           self.model.device_ids)[0])
                    #    avoid_data_parallel = scattered_len < len(self.model.device_ids)
                    #else:
                    #    avoid_data_parallel = False

                    #if avoid_data_parallel:
                    #    vid_embed = self.model.module.compute_video(data['video'], return_embeds=True)
                    #    vid_embed = self.model.module.compute_video_aggregation(vid_embed, )
                    #else:
                    #    text_embed, vid_embed = self.model(data, return_embeds=True)
                    vid_embed = self.model.module.compute_video(data['video'])
                    vid_embed = self.model.module.compute_video_aggregation(vid_embed, self.val_batch_size)
                    vid_predictions = self.model.module.head_ht100m_linear_probe(vid_embed)

                    vid_predictions = self.allgather(vid_predictions.contiguous(), self.n_gpu, self.args)
                    video_labels = self.allgather(data['label'], self.n_gpu, self.args)

                    vid_pred_arr[dl_idx].append(vid_predictions.cpu())
                    vid_target_arr[dl_idx].append(video_labels.cpu())
                    loss = self.loss(vid_predictions, video_labels)
                    total_val_loss[dl_idx] += loss.item()

        for dl_idx in range(len(self.valid_data_loader)):
            # TODO: this needs a clean
            #if self.writer is not None:
            #    self.writer.log_scalar(f'loss_val_{dl_idx}',
            #                           total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx]))
            nested_metrics = {x: {} for x in range(len(self.valid_data_loader))}

            vid_labels = torch.cat(vid_target_arr[dl_idx])
            vid_preds = torch.cat(vid_pred_arr[dl_idx])

            for metric in self.metrics:
                metric_name = metric.__name__
                res = metric(vid_preds, vid_labels)
                if self.args.rank == 0:
                    print(res)
                    print(type(res))
                verbose(epoch=epoch, metrics=res, name=self.valid_data_loader[dl_idx].dataset_name)
                nested_metrics[dl_idx][metric_name] = res

                if self.writer is not None and self.args.rank == 0:
                    to_write = format_nested_metrics_for_writer(res, mode=metric_name,
                                                                name=self.valid_data_loader[dl_idx].dataset_name)
                    #for key, val in to_write.items():
                    #    self.writer.log_scalar(key, val)

                #if self.visualizer is not None and self.args.rank == 0:
                #    meta_arr_cat = {key: [] for key in meta_arr[0]}
                #    for meta in meta_arr:
                #        for key, val in meta.items():
                #            meta_arr_cat[key] += val
                #    self.visualizer.visualize_ranking(sims, epoch, meta_arr_cat, nested_metrics)

        res_dict = {}

        if self.args.rank == 0:
            res_dict = {f'val_loss_{dl_idx}': total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx])
                        for dl_idx in range(len(self.valid_data_loader))}
            res_dict['nested_val_metrics'] = nested_metrics

        return res_dict

    def _progress(self, batch_idx, dl_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader[dl_idx], 'n_samples'):
            current = batch_idx * self.data_loader[dl_idx].batch_size
            total = int(self.data_loader[dl_idx].n_samples / self.n_gpu)
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

def verbose(epoch, metrics, name="TEST"):
    acc = metrics['accuracy']
    msg = f"{name:s} epoch {epoch}, Acc: {acc:.1f}"
    print(msg)
    return msg

def format_nested_metrics_for_writer(metrics, mode, name="TEST"):
    res = {}
    for key, val in metrics.items():
        log_name = f"[{mode}]{name}_{key}"
        res[log_name] = val
    return res
