# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm
import torch.distributed as dist
from datetime import datetime
import random

from base import Multi_BaseTrainer_dist
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

class Multi_Trainer_dist_EgoAgg(Multi_BaseTrainer_dist):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, args, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, agg_data_loader=None, agg_valid_data_loader=None, lr_scheduler=None, len_epoch=None, writer=None,
                 visualizer=None, tokenizer=None, max_samples_per_epoch=50000, additional_losses=None, start_epoch=1):
        super().__init__(args, model, loss, metrics, optimizer, config, writer, start_epoch=start_epoch)
        self.config = config
        self.args = args
        self.data_loader = data_loader
        self.agg_data_loader = agg_data_loader
        self.agg_valid_data_loader = agg_valid_data_loader
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
        self.agg_batch_size = self.agg_data_loader[0].batch_size
        self.log_step = int(np.sqrt(self.batch_size))
        self.total_batch_sum = sum([x.batch_size for x in self.data_loader])
        self.tokenizer = tokenizer
        self.max_samples_per_epoch = max_samples_per_epoch
        self.n_gpu = self.args.world_size
        self.allgather = AllGather_multi.apply
        self.agg_count = 0
        self.additional_losses = additional_losses #Format is [intra-video, intra-text, inter-parent-video, inter-parent-text]
        self.do_hierarchical = self.config['training_methods']['hierarchical']['intra-modal'] or self.config['training_methods']['hierarchical']['inter-modal']
        # self.writer = writer

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
    
    def _train_step(self, data, epoch, batch_idx, dl_idx, hierarchy='child'):
        if hierarchy == 'child':
            batch_size = self.batch_size
        elif hierarchy == 'parent':
            batch_size = self.agg_batch_size


        if 'video_neg' in data.keys():  # w/ negative sampling
            data['text'] = data['text'] + data['text_neg']
            data['video'] = torch.cat( (data['video'], data['video_neg']), axis = 0)
            data['noun_vec'] = torch.cat((data['noun_vec'], data['noun_vec_neg']), axis=0)
            data['verb_vec'] = torch.cat((data['verb_vec'], data['verb_vec_neg']), axis=0)

        if self.tokenizer is not None:
            data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True,
                                            truncation=True)
            if 'aggregated_text' in data.keys():
                data['aggregated_text'] = self.tokenizer(data['aggregated_text'], return_tensors='pt', padding=True,
                                            truncation=True)
        data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
        data['video'] = data['video'].to(self.device)
        n_embeds = data['noun_vec'].to(self.device)
        v_embeds = data['verb_vec'].to(self.device)

        if 'aggregated_text' in data.keys():
            data['aggregated_text'] = {key: val.to(self.device) for key, val in data['aggregated_text'].items()}
            agg_n_embeds = data['aggregated_noun_vec'].to(self.device)
            agg_v_embeds = data['aggregated_verb_vec'].to(self.device)

        self.optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            '''
            Clarification
            -------------
            1. When hierarchy != 'parent': The data dict will not contain 'aggregated_text' parameter
            2. When hierarchy == 'parent': 'text' will always contain summary text and 'aggregated_text' will contain narrations
            '''
            if hierarchy == 'parent':
                # text_embeds and video_embeds are the aggregated parent embeddings passed through self.aggregation
                summary_embeds, video_stacked_embeds, text_embeds, video_embeds, text_stacked_embeds = self.model(data, do_aggregation=True, batch_size=batch_size)
            else:
                text_embeds, video_embeds = self.model(data)

            # Special treatment when we want to aggregate features
            # This part of the code gets the child features based on the selected parent features
            # This has nothing to do with the aggregation strategy
            # However, latest discussion (July end) suggests parent-child matching is not good, only parent-parent and child-child makes sense.
            if hierarchy == 'parent' and self.do_hierarchical:
                # For handling video aggregation
                video_stacked_embeds = video_stacked_embeds.view(batch_size, -1, video_stacked_embeds.shape[1])
                # Do video child feature sampling
                num_positives = self.config['training_methods']['hierarchical']['num_positives']
                assert video_stacked_embeds.shape[1] > num_positives
                pos_indices = random.sample(range(video_stacked_embeds.shape[1]), num_positives)
                video_clip_embeds = video_stacked_embeds[:, pos_indices, :]

                # video_embeds = torch.mean(video_embeds, dim=1) # Now handled in the forward function of the model. Can be safely removed
                # For handling text aggregation
                text_stacked_embeds = text_stacked_embeds.view(batch_size, -1, text_stacked_embeds.shape[1])
                #print(n_embeds.shape)
                #print(agg_n_embeds.shape)
                #exit()
                agg_n_embeds = agg_n_embeds.view(batch_size, -1, agg_n_embeds.shape[1])
                agg_v_embeds = agg_v_embeds.view(batch_size, -1, agg_v_embeds.shape[1])
                # Do text child if text aggregation method is used. For summary, we need to invoke a call to model again
                #if self.config['training_methods']['text aggregation']:
                if True:
                    num_positives = self.config['training_methods']['hierarchical']['num_positives']
                    assert text_stacked_embeds.shape[1] > num_positives
                    assert agg_n_embeds.shape[1] > num_positives
                    assert agg_v_embeds.shape[1] > num_positives
                    pos_indices = random.sample(range(text_stacked_embeds.shape[1]), num_positives)
                    text_clip_embeds = text_stacked_embeds[:, pos_indices, :]
                    n_clip_embeds = agg_n_embeds[:, pos_indices, :]
                    v_clip_embeds = agg_v_embeds[:, pos_indices, :]
                else:
                    # We need to evaluate on the model again with text aggregation because the current text_embeds only has summary embeddings
                    text_aggregated_embeds = self.model.module.compute_text(data['aggregated_text'])
                    text_aggregated_embeds = text_aggregated_embeds.view(batch_size, -1, text_aggregated_embeds.shape[1])
                    num_positives = self.config['training_methods']['hierarchical']['num_positives']
                    assert text_aggregated_embeds.shape[1] > num_positives
                    assert agg_n_embeds.shape[1] > num_positives
                    assert agg_v_embeds.shape[1] > num_positives
                    pos_indices = random.sample(range(text_aggregated_embeds.shape[1]), num_positives)
                    text_clip_embeds = text_aggregated_embeds[:, pos_indices, :]
                    agg_n_embeds = agg_n_embeds.view(batch_size, -1, agg_n_embeds.shape[1])
                    agg_v_embeds = agg_v_embeds.view(batch_size, -1, agg_v_embeds.shape[1])
                    n_clip_embeds = agg_n_embeds[:, pos_indices, :]
                    v_clip_embeds = agg_v_embeds[:, pos_indices, :]
                # Now that we have samples from n_embeds and v_embeds, we can safely average them
                #n_embeds = (torch.mean(n_embeds, dim=1) > 0.5).float()
                #v_embeds = (torch.mean(v_embeds, dim=1) > 0.5).float()
            #text_embeds = torch.mean(text_embeds, dim=1) # Now handled in the forward function of the model. Can be safely removed
            video_embeds = self.allgather(video_embeds, self.n_gpu, self.args)
            text_embeds = self.allgather(text_embeds, self.n_gpu, self.args)
            if hierarchy == 'parent':
                summary_embeds = self.allgather(summary_embeds, self.n_gpu, self.args)
            n_embeds = self.allgather(n_embeds, self.n_gpu, self.args)
            v_embeds = self.allgather(v_embeds, self.n_gpu, self.args)

            if hierarchy == 'parent' and self.do_hierarchical:
                video_clip_embeds = self.allgather(video_clip_embeds, self.n_gpu, self.args)
                video_clip_embeds = video_clip_embeds.view(-1, video_clip_embeds.shape[-1])
                text_clip_embeds = self.allgather(text_clip_embeds, self.n_gpu, self.args)
                text_clip_embeds = text_clip_embeds.view(-1, text_clip_embeds.shape[-1])
                n_clip_embeds = self.allgather(n_clip_embeds, self.n_gpu, self.args)
                n_clip_embeds = n_clip_embeds.view(-1, n_clip_embeds.shape[-1])
                v_clip_embeds = self.allgather(v_clip_embeds, self.n_gpu, self.args)
                v_clip_embeds = v_clip_embeds.view(-1, v_clip_embeds.shape[-1])

                assert video_embeds.shape[0] == text_embeds.shape[0]
                num_positives_MILNCE = video_embeds.shape[0]

                if False:#self.config['training_methods']['hierarchical']['intra-modal']:
                    intra_video_loss = self.additional_losses[0](sim_matrix(video_embeds, video_clip_embeds), num_samples=num_positives_MILNCE)
                    intra_text_loss = self.additional_losses[1](sim_matrix(text_embeds, text_clip_embeds), num_samples=num_positives_MILNCE)
                    total_intra_loss = intra_video_loss + intra_text_loss
                else:
                    total_intra_loss = None

                if False:#self.config['training_methods']['hierarchical']['inter-modal']:
                    inter_parent_video_loss = self.additional_losses[2](sim_matrix(video_embeds, text_clip_embeds), num_samples=num_positives_MILNCE)
                    inter_parent_text_loss = self.additional_losses[3](sim_matrix(video_clip_embeds, text_embeds), num_samples=num_positives_MILNCE)
                    total_inter_loss = inter_parent_video_loss + inter_parent_text_loss
                else:
                    total_inter_loss = None

            if hierarchy == 'parent':
                output1 = sim_matrix(text_embeds, summary_embeds)
                output2 = sim_matrix(video_embeds, summary_embeds)
            else:
                output = sim_matrix(text_embeds, video_embeds)

            only_video_with_summary_baseline = False
            only_sa_no_summary_baseline = False #Baseline where we don't use the summary. Only self-attention between aggregated video and text features
            if hierarchy == 'parent' and only_sa_no_summary_baseline:
                output = sim_matrix(text_embeds, video_embeds)

            if self.config['loss']['type'] == 'EgoNCE':
                sim_v = sim_matrix(v_embeds, v_embeds)
                sim_n = sim_matrix(n_embeds, n_embeds)
                #print('sim_n shape: {}'.format(sim_n.shape))
                if hierarchy == 'parent' and not only_sa_no_summary_baseline:
                    if not only_video_with_summary_baseline:
                        clip_loss = self.loss(output1, sim_v, sim_n) + self.loss(output2, sim_v, sim_n)
                    else:
                        clip_loss = self.loss(output2, sim_v, sim_n) #output1 is text and summary
                else:
                    clip_loss = self.loss(output, sim_v, sim_n)
            else:
                if hierarchy == 'parent' and not only_sa_no_summary_baseline:
                    if not only_video_with_summary_baseline:
                        clip_loss = self.loss(output1) + self.loss(output2)
                    else:
                        clip_loss = self.loss(output2) #output1 is text and summary
                else:
                    clip_loss = self.loss(output)

            intra_loss_exists = (hierarchy == 'parent' and self.do_hierarchical and total_intra_loss is not None)
            inter_loss_exists = (hierarchy == 'parent' and self.do_hierarchical and total_inter_loss is not None)
            if not intra_loss_exists and not inter_loss_exists:
                loss = clip_loss
            elif intra_loss_exists and not inter_loss_exists:
                loss = clip_loss + total_intra_loss
            elif not intra_loss_exists and inter_loss_exists:
                loss = clip_loss + total_inter_loss
            elif intra_loss_exists and inter_loss_exists:
                loss = clip_loss + total_intra_loss + total_inter_loss
            else:
                raise ValueError

        loss.backward()
        self.optimizer.step()

        if self.writer is not None and self.args.rank == 0 and hierarchy == 'child':
            # self.writer.log_scalar(f'loss_train_{dl_idx}', loss.detach().item())
            total = int(self.data_loader[dl_idx].n_samples/self.n_gpu)
            current = batch_idx * self.data_loader[dl_idx].batch_size
            final_total = (epoch-1) * total + current
            self.writer.add_scalar(f'Loss_training/loss_{dl_idx}', loss.detach().item(), final_total)

        # if batch_idx % self.log_step == 0 and self.args.local_rank == 0:
        if batch_idx % self.log_step == 0 and self.args.rank == 0 and hierarchy == 'child':
            print('[{}] Train Epoch: {} dl{} {} Loss: {:.6f}'.format(
                datetime.now().strftime(r'%m%d_%H:%M:%S'),
                epoch,
                dl_idx,
                self._progress(batch_idx, dl_idx),
                loss.detach().item()))

        if hierarchy == 'parent' and self.args.rank == 0:
            print('[{}] Parent Train Epoch: {} dl{} {} Loss: {:.6f}'.format(
                datetime.now().strftime(r'%m%d_%H:%M:%S'),
                epoch,
                dl_idx,
                'NA',
                loss.detach().item()))
            self.writer.add_scalar(f'Agg_Loss_training/loss_{dl_idx}', loss.detach().item(), self.agg_count)
            self.agg_count+=1

        if self.args.rank == 0:
            for param_group in self.optimizer.param_groups:
                curr_lr = param_group['lr']
            #print('Current learning rate is : {}'.format(curr_lr))

        self.optimizer.zero_grad()
        return loss.detach().item()

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

        self.model.train()
        total_loss = [0] * len(self.data_loader)
        total_agg_loss = [0] * len(self.agg_data_loader)
        total_metrics = np.zeros(len(self.metrics))
        for loader in self.data_loader:
            loader.train_sampler.set_epoch(epoch)
        for agg_loader in self.agg_data_loader:
            agg_loader.train_sampler.set_epoch(epoch)
        if len(self.agg_data_loader) != 1:
            print('Unexpected')
            raise ValueError
        self.agg_data_iter = iter(self.agg_data_loader[0])
        for batch_idx, data_li in enumerate(zip(*self.data_loader)):
            if (batch_idx + 1) * self.total_batch_sum > self.max_samples_per_epoch:
                break
            for dl_idx, data in enumerate(data_li):
                catastrophic_forgetting_baseline = False
                if not catastrophic_forgetting_baseline: #In this baseline we only FT for summary
                    # then assume we must tokenize the input, e.g. its a string
                    loss = self._train_step(data, epoch, batch_idx, dl_idx)
                    total_loss[dl_idx] += loss
                    del data

                #Aggregation training step
                if (batch_idx+1) % self.agg_train_freq == 0:
                    try:
                        agg_batch = next(self.agg_data_iter)
                    except StopIteration:
                        self.agg_data_iter = iter(self.agg_data_loader[0])
                        agg_batch = next(self.agg_data_iter)
                    agg_loss = self._train_step(agg_batch, epoch, 0, dl_idx, hierarchy='parent')
                    total_agg_loss[dl_idx] += agg_loss
                    del agg_batch

            if batch_idx == self.len_epoch:
                break

        log = {
            f'loss_{dl_idx}': total_loss[dl_idx] / self.len_epoch for dl_idx in range(len(self.data_loader))
        }

        if self.writer is not None and self.args.rank == 0:
            for dl_idx in range(len(self.data_loader)):
                tl = total_loss[dl_idx] / self.len_epoch
                self.writer.add_scalar(f'Loss_training/loss_total_{dl_idx}', tl, epoch-1)

        if self.writer is not None and self.args.rank == 0:
            for dl_idx in range(len(self.agg_data_loader)):
                tl = total_agg_loss[dl_idx] / self.len_epoch
                self.writer.add_scalar(f'Agg_Loss_training/loss_total_{dl_idx}', tl, epoch-1)

        if self.do_validation:
            val_logs = self._valid_epoch(epoch)
            if self.args.rank == 0:
                log.update(val_logs[0])
                log.update(val_logs[1])

        #self._adjust_learning_rate(self.optimizer, epoch, self.args)

        return log

    def _valid_epoch_per_dataloader(self, epoch, data_loader, group_list=None, task=None):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = [0] * len(data_loader)
        total_val_metrics = [np.zeros(len(self.metrics))] * len(data_loader)

        gt_arr = {x: [] for x in range(len(data_loader))}
        pred_arr = {x: [] for x in range(len(data_loader))}
        type_arr = {x: [] for x in range(len(data_loader))}

        with torch.no_grad():
            # for validation we switch the nested loop order, because alternate batches not needed...
            # ... and dataloaders can be of different length
            for dl_idx, dl in enumerate(data_loader):
                for batch_idx, data in enumerate(tqdm(dl)):
                    data['video'] = data['video'][0]  # remove batch
                    data['text'] = data['text']
                    #print(data['video'].shape) #torch.Size([5, 16, 4, 3, 224, 224]) for agg type

                    if self.tokenizer is not None:
                        data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                    data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
                    data['video'] = data['video'].to(self.device)

                    # Do the process per answer option since 5x16=80 will give OOM
                    if len(data['video'].shape) > 5: #5 allowed is B, T, C, H, W
                        vid_embed = []
                        for option_idx in range(data['video'].shape[0]):
                            # Inside this loop the batch size is 1
                            data_option = {}
                            data_option['text'] = data['text']
                            if task == 'SummaryMCQ':
                                data_option['video'] = data['video'][option_idx]
                            elif task == 'ShuffleMCQ':
                                if option_idx == data['correct'].item():
                                    data_option['video'] = data['video'][option_idx]
                                else:
                                    correct_ordered_video = data['video'][data['correct'].item()]
                                    data_option['video'] = correct_ordered_video[torch.randperm(correct_ordered_video.shape[0]), :, :, :, :] #After removing B and #options, the shape is [num_clips, num_frames, C, H, W]
                            else:
                                raise NotImplementedError
                            text_embed_option, _, _, vid_embed_option, _ = self.model(data_option, do_aggregation=True, batch_size=1)
                            vid_embed.append(vid_embed_option)
                            #text_embed_option, vid_embed_option = self.model(data_option, return_embeds=True)
                            #vid_embed.append(torch.mean(vid_embed_option, dim=0))
                        text_embed = text_embed_option
                        vid_embed = torch.cat(vid_embed, dim=0)
                    else:
                        text_embed, vid_embed = self.model(data, return_embeds=True)

                    data_gt = data['correct'][0].to(self.device).unsqueeze(0)
                    data_pred = sim_matrix(text_embed, vid_embed)
                    data_type = data['type'][0].to(self.device).unsqueeze(0)

                    # if isinstance(self.model, nn.DataParallel) and data["video"].shape[0] < len(self.model.device_ids):
                    # Note that if some batch has size smaller than the GPU size, `DataParallel` will fail.
                    # It can happen with the last batch of the dataset, depending on its size.
                    # This avoids using `DataParallel` in this case, and supposes the entire batch fits in one GPU.
                    #    text_embed, vid_embed = self.model.module(data, return_embeds=True)
                    # else:
                    #    text_embed, vid_embed = self.model(data, return_embeds=True)
                    data_gt_all = [torch.zeros_like(data_gt) for _ in range(self.n_gpu)]
                    torch.distributed.all_gather(data_gt_all, data_gt)
                    data_gt_all = torch.cat(data_gt_all, dim=0)

                    data_pred_all = [torch.zeros_like(data_pred) for _ in range(self.n_gpu)]
                    torch.distributed.all_gather(data_pred_all, data_pred)
                    data_pred_all = torch.cat(data_pred_all, dim=0)

                    data_type_all = [torch.zeros_like(data_type) for _ in range(self.n_gpu)]
                    torch.distributed.all_gather(data_type_all, data_type)
                    data_type_all = torch.cat(data_type_all, dim=0)

                    gt_arr[dl_idx].append(data_gt_all.cpu())
                    pred_arr[dl_idx].append(data_pred_all.cpu())
                    type_arr[dl_idx].append(data_type_all.cpu())

            if self.writer is not None and self.args.rank == 0:
                for dl_idx in range(len(self.valid_data_loader)):
                    tl = total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx])
                    self.writer.add_scalar(f'Loss_val/loss_total_{dl_idx}', tl, epoch-1)

        for dl_idx in range(len(self.valid_data_loader)):
            nested_metrics = {x: {} for x in range(len(self.valid_data_loader))}

            gt_arr_cat = torch.cat(gt_arr[dl_idx])
            pred_arr_cat = torch.cat(pred_arr[dl_idx])
            type_cat = torch.cat(type_arr[dl_idx])

            for metric in self.metrics:
                metric_name = metric.__name__
                res = metric(pred_arr_cat, gt_arr_cat, type_cat, group_list)
                if self.args.rank == 0:
                    print(
                        verbose(epoch=epoch, metrics=res, name=self.valid_data_loader[dl_idx].dataset_name))
                nested_metrics[dl_idx][metric_name] = res

                if self.writer is not None and self.args.rank == 0:
                    to_write = format_nested_metrics_for_writer(res, mode=metric_name,
                                                                name=self.valid_data_loader[dl_idx].dataset_name)
                    # for key, val in to_write.items():
                    #     self.writer.log_scalar(key, val)
                    for key, val in to_write.items():
                        key = key.replace('[', '_').replace(']', '_')
                        self.writer.add_scalar(f'Val_metrics_{dl_idx}/{key}', val, epoch - 1)

        res_dict = {}
        if self.args.rank == 0:
            res_dict = {f'val_loss_{dl_idx}': total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx])
                        for dl_idx in range(len(self.valid_data_loader))}
            res_dict['nested_val_metrics'] = nested_metrics

        return res_dict

    def _valid_epoch(self, epoch):
        """
        Validate two data_loaders

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        summarymcq_dict = self._valid_epoch_per_dataloader(epoch, self.agg_valid_data_loader, group_list=["SummaryMCQ"], task='SummaryMCQ')
        shufflemcq_dict = self._valid_epoch_per_dataloader(epoch, self.agg_valid_data_loader, group_list=["ShuffleMCQ"], task='ShuffleMCQ')
        egomcq_dict = self._valid_epoch_per_dataloader(epoch, self.valid_data_loader)
        if self.args.rank == 0:
            print('EGOMCQ Result: {}'.format(egomcq_dict))
            print('EGOSUMMARY Result: {}'.format(summarymcq_dict))
            print('SHUFFLE MCQ: {}'.format(shufflemcq_dict))
        return (egomcq_dict, summarymcq_dict)

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
    msg = ""
    for key in metrics.keys():
        acc = metrics[key]
        msg += f"{name:s} epoch {epoch}, {key:s}, Acc: {acc:.1f};    "
    print(msg)
    return msg

def format_nested_metrics_for_writer(metrics, mode, name="TEST"):
    res = {}
    for key, val in metrics.items():
        log_name = f"[{mode}]{name}_{key}"
        res[log_name] = val
    return res
