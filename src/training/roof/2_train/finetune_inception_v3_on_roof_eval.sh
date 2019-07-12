#!/bin/bash
!<<!
/*-------------------------------------------------------------------------*
| This script evaluates an InceptionV3 model on the roof validation set.   |
|                                                                          |
| Authors: Chaofeng Wang, Qian Yu                                           |
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
CLASSIFIER=eval_roof_classifier.py

cd ${workDir}
cp ${CLASSIFIER} ${SLIM_DIR}/.
#cp roof.py ${SLIM_DIR}/datasets/.
#cp dataset_factory.py ${SLIM_DIR}/datasets/.
#unzip dataset.zip

# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=${tmpDir}/checkpoints

# Where the checkpoint to be evaluated.
TRAIN_DIR=${tmpDir}/roof-traindir/all
checkpoint_file=${TRAIN_DIR}/model.ckpt-119999 # you should know the file name here

# Where the dataset is saved to.
DATASET_DIR=${tmpDir}/dataset/roof/
DATASET_NAME=roof
TEST_DIR=/Users/simcenter/Codes/SimCenter/BIM.AI/data/images/raw/roof/roof_photos_shape/test/gabled/

# set model
MODEL_NAME=inception_v3

# set subset
SET=validation

# Run evaluation.

echo "Extracting ${SET} Features..."
python ${CLASSIFIER} \
  --slim_dir=${SLIM_DIR} \
  --checkpoint_path=${checkpoint_file} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=${SET} \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --infile=${TEST_DIR}

