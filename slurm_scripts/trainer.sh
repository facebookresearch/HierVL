#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --partition=enter_your_partitions
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=80           # number of cores per tasks
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 12:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=/path/to/slurm/log/%x-%j.out           # output file name
#SBATCH --error=/path/to/slurm/log/%x-%j.err
#SBATCH --constraint='volta32gb'
#SBATCH --mem=480G
#SBATCH --open-mode=append

#SBATCH --mail-user=email@email.com
#SBATCH --mail-type=begin,end,fail # mail once the job finishes

echo "SLURM_JOBID: " $SLURM_JOBID

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
GPUS_PER_NODE=8
NNODES=$SLURM_NNODES
export JOB_NAME=$SLURM_JOB_NAME

# Load Anaconda distribution of Python
module load anaconda3
source ~/.bashrc
conda activate hiervl

srun -N8 --gres gpu:8 python distributed_main.py --multiprocessing-distributed --config ./configs/pt/egoaggregation.json --experiment egoaggregation