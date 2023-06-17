# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models

import transformers
from tensorboardX import SummaryWriter
import data_loader.data_loader as module_data
from trainer import Multi_Trainer_dist_MIR
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import utils.visualizer as module_vis
from sacred import Experiment
ex = Experiment('train')
import collections
from parse_config import ConfigParser
import utils.visualizer as module_vis
from utils.util import replace_nested_dict_item, load_checkpoint_after_preemption

from run.distributed_epic import main_worker as epic_main_worker
from run.distributed_egoclip import main_worker as egoclip_main_worker
from run.distributed_charades import main_worker as charades_main_worker
from run.distributed_egoaggregation import main_worker as egoaggregation_main_worker
from run.distributed_howto100m import main_worker as howto100m_main_worker

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def find_free_port():
    import socket
    s = socket.socket()
    s.bind(('', 0))            # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.


def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    #parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
    #                    metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-file', default=None, type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
    parser.add_argument('--experiment', default='egoaggregation', type=str,
                        help='Experiment name.')
    ###################################################################
    parser.add_argument('-c', '--config', default='configs/pt/egoclip.json', type=str,
                      help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    parser.add_argument('-o', '--observe', action='store_true',
                      help='Whether to observe (neptune)')
    parser.add_argument('-l', '--launcher', choices=['none', 'pytorch'], default='none',help='job launcher')
    parser.add_argument('-lr1', '--learning_rate1', type=float, default=2e-4)
    parser.add_argument('-sc', '--schedule', default=[10, 40, 70])
    parser.add_argument('-ek_margin', '--epic_loss_margin', default=0.2, type=float)

    #######################
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size')),
    ]
    config = ConfigParser(parser, options)
    args = parser.parse_args()
    try:
        recovered_checkpoint, recovered_epoch = load_checkpoint_after_preemption(config)
    except:
        recovered_checkpoint, recovered_epoch = None, None
    if recovered_checkpoint is not None:
        config["arch"]["args"]["load_checkpoint"] = recovered_checkpoint
        config["trainer"]["start_epoch"] = recovered_epoch + 1
    if args.experiment == "epic_mir":
        # Only for EPIC-MIR
        if config["loss"]["args"]["margin"] != args.epic_loss_margin:
            print('Different margin in config and command line args. Setting command line margin value: {}...'.format(args.epic_loss_margin))
            config["loss"]["args"]["margin"] = args.epic_loss_margin
    ex.add_config(config._config)
    ##########################


    #args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    # slurm available
    import os
    if args.world_size == -1 and "SLURM_NPROCS" in os.environ:
        args.world_size = int(os.environ["SLURM_NPROCS"])
        args.rank = int(os.environ["SLURM_PROCID"])
        jobid = os.environ["SLURM_JOBID"]
        try:
            restart_count = os.environ["SLURM_RESTART_COUNT"]
        except:
            restart_count = '0'
        hostfile = "dist_url." + jobid  + '.' + restart_count + ".txt"
        if args.dist_file is not None:
            args.dist_url = "file://{}.{}".format(os.path.realpath(args.dist_file), jobid)
        elif args.rank == 0:
            import socket
            ip = socket.gethostbyname(socket.gethostname())
            port = find_free_port()
            args.dist_url = "tcp://{}:{}".format(ip, port)
            with open(hostfile, "w") as f:
                f.write(args.dist_url)
        else:
            import os
            import time
            while not os.path.exists(hostfile):
                time.sleep(1)
            with open(hostfile, "r") as f:
                args.dist_url = f.read()
        print("dist-url:{} at PROCID {} / {}".format(args.dist_url, args.rank, args.world_size))

    # if args.dist_url == "env://" and args.world_size == -1:
    #     args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        if args.experiment == "epic_mir":
            mp.spawn(epic_main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, config))
        elif args.experiment == "egoclip":
            mp.spawn(egoclip_main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, config))
        elif args.experiment == "egoaggregation":
            mp.spawn(egoaggregation_main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, config))
        elif args.experiment == "charades":
            mp.spawn(charades_main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, config))
        elif args.experiment == "howto100m":
            mp.spawn(howto100m_main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, config))
    else:
        raise NotImplementedError
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, config)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    main()
