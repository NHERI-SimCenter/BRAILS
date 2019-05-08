#!/bin/bash
!<<!
/*-------------------------------------------------------------------------*
| This script evaluates an InceptionV3 model on the roof validation set.   |
|                                                                          |
| Authors: Charles Wang, Qian Yu                                           |
|          UC Berkeley c_w@berkeley.edu                                    |
|                                                                          |
| Date:    05/02/2019                                                      |
*-------------------------------------------------------------------------*/
!

# Usage:
# sh finetune_inception_v3_on_roof_eval.sh
set -e

cd ..
baseDir=$(pwd)
workDir=${baseDir}/2_train
tmpDir=${baseDir}/tmp
SLIM_DIR=${tmpDir}/models/research/slim

cd ${workDir}
cp eval_roof_classifier_prob.py ${SLIM_DIR}/.
#cp roof.py ${SLIM_DIR}/datasets/.
#cp dataset_factory.py ${SLIM_DIR}/datasets/.
#unzip dataset.zip

# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=${tmpDir}/checkpoints

# Where the checkpoint to be evaluated.
TRAIN_DIR=${tmpDir}/roof-traindir/all

# Where the dataset is saved to.
DATASET_DIR=${tmpDir}/dataset/roof/
DATASET_NAME=roof

# set model
MODEL_NAME=inception_v3

# set subset
SET=validation

# Run evaluation.

echo "Extracting ${SET} Features..."
python eval_roof_classifier_prob.py \
  --slim_dir=${SLIM_DIR} \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=${SET} \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME}

