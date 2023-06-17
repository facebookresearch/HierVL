#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

if [ "$1" = "" ]; then
    echo "Job name cannot be empty"
    exit 1
fi

export JOB_NAME=$(date '+%Y-%m-%d_%H:%M:%S_')$1
export DESTINATION_DIR='/path/to/experiments/'

mkdir $DESTINATION_DIR/$JOB_NAME/
cp -R base/ $DESTINATION_DIR/$JOB_NAME/
cp -R configs/ $DESTINATION_DIR/$JOB_NAME/
cp -R data_loader/ $DESTINATION_DIR/$JOB_NAME/
cp -R logger/ $DESTINATION_DIR/$JOB_NAME/
cp -R model/ $DESTINATION_DIR/$JOB_NAME/
cp -R run/ $DESTINATION_DIR/$JOB_NAME/
cp -R trainer/ $DESTINATION_DIR/$JOB_NAME/
cp -R utils/ $DESTINATION_DIR/$JOB_NAME/
cp parse_config.py $DESTINATION_DIR/$JOB_NAME/
cp distributed_main.py $DESTINATION_DIR/$JOB_NAME/
cp trainer.sh $DESTINATION_DIR/$JOB_NAME/trainer_local.sh

cd $DESTINATION_DIR/$JOB_NAME/

sbatch -J $JOB_NAME trainer_local.sh

