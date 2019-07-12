#!/bin/bash
!<<!
/*-------------------------------------------------------------------------*
| This script fine-tunes an InceptionV3 model on the roof training set.    |
| Modified from finetune_inception_v3_on_flowers.sh in slim repo.          |
|                                                                          |
| Authors: Chaofeng Wang, Qian Yu                                           |
|          UC Berkeley c_w@berkeley.edu                                    |
|                                                                          |
| Date:    05/02/2019                                                      |
*-------------------------------------------------------------------------*/
!
# Usage:
# sh finetune_inception_v3_on_roof_train.sh
set -e

# set up directories, better not change these
cd ..
baseDir=$(pwd)
tmpDirName=tmp
workDir=${baseDir}/2_train
tmpDir=${baseDir}/${tmpDirName}

# clean the tmp dir
if [ -d "$tmpDir" ]; then
  rm -rf ${tmpDir}
fi
mkdir ${tmpDirName}
mkdir ${tmpDirName}/roof-traindir

# download slim
cd ${tmpDir}
git clone https://github.com/tensorflow/models.git
SLIM_DIR=${tmpDir}/models/research/slim

# you need to modify roof.py and dataset_factory.py
cd ${workDir}
cp roof.py ${SLIM_DIR}/datasets/.
cp dataset_factory.py ${SLIM_DIR}/datasets/.

# get dataset. this dataset contains 3 categories, each has 2000 pics.
cd ${tmpDir}
wget -v -O dataset.zip https://berkeley.box.com/shared/static/g4tcoamv7lgzzq71y9c7x4vxpmcnv30s.zip
unzip dataset.zip
cd ${workDir}
cp ${tmpDir}/dataset/roof/dataset.json . 


#################
# Dataset Flags #
#################
# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=${tmpDir}/roof-traindir

# Where the dataset is saved to.
DATASET_DIR=${tmpDir}/dataset/roof/
DATASET_NAME=roof

# set subset
SET=train

#################################
# Model and training parameters #
#################################

# set model
MODEL_NAME=inception_v3

#max_number_of_steps: The maximum number of training steps. Default: 100000
MAX_TRAIN_STEPS=20000 #150000 120000

#summary_snapshot_steps: Summary save steps. Default: 2000
SUMMARY_STEPS=500

#model_snapshot_steps: Model save steps. Default: 10000
MODEL_SNAPSHOT_STEPS=1000

#log_every_n_steps: The frequency with which logs are print. Default: 10
LOG_EVERY_N_STEPS=10

#batch_size: The number of samples in each batch. Default: 32
BATCH_SIZE=64

######################
# Fine-tuning Flags #
######################
# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=${tmpDir}/checkpoints

CKPT_EXCLUDE_SCOPES=InceptionV3/Logits,InceptionV3/AuxLogits
TRAINABLE_SCOPES=${CKPT_EXCLUDE_SCOPES}

######################
# Learning rate flag #
######################
#learning_rate_decay_type: "fixed", "exponential", or "polynomial"' Default:exponential
######

#learning_rate: Initial learning rate. Default:0.01
INI_LR1=0.01
INI_LR2=0.001
######

#end_learning_rate: The minimal end learning rate used by a 'polynomial' decay learning rate. Default:0.0001
######

#label_smoothing: The amount of label smoothing. Default: 0.0
# Does it means label smoothing weight ???
# LABEL_SMOOTH=0.1 #0.1
######

#learning_rate_decay_factor: Learning rate decay factor. Default:0.94
# LR_DECAY_FACTOR=0.95
######

######################
# Optimization Flags #
######################
# Search the corresponding optimizer name, and chenge the default parameters

#weight_decay: The weight decay on the model weights. Default: 0.00004
WEIGHT_DECAY=0.00004

# final fc W regularisation weight
#FFC_W_reg_weight=0.0001
######

#optimizer: "adadelta", "adagrad", "adam", "ftrl", "momentum", "sgd" or "rmsprop".' Default: rmsprop
OPTIMIZER=rmsprop
######

#adadelta_rho: The decay rate for adadelta. Default: 0.95

#adagrad_initial_accumulator_value: Starting value for the AdaGrad accumulators. Default: 0.1

#adam_beta1: The exponential decay rate for the 1st moment estimates. Default: 0.9
ADAM_BETA1=0.9

#adam_beta2: The exponential decay rate for the 2nd moment estimates. Default: 0.999
ADAM_BETA2=0.999

#opt_epsilon: Epsilon term for the optimizer. Default: 1.0
#OPT_EPS=1e-5

#ftrl_learning_rate_power: The learning rate power. Default: -0.5

#ftrl_initial_accumulator_value: Starting value for the FTRL accumulators. Default: 0.1

#ftrl_l1: The FTRL l1 regularization strength. Default:0.0

#ftrl_l2: The FTRL l2 regularization strength. Default:0.0

#momentum: The momentum for the MomentumOptimizer and RMSPropOptimizer. Default: 0.9
######

#rmsprop_momentum: Momentum. Default: 0.9
######

#rmsprop_decay: Decay term for RMSProp. Default: 0.9
######

########################################
## Download the pre-trained checkpoint #
########################################
if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir ${PRETRAINED_CHECKPOINT_DIR}
fi
if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/${MODEL_NAME}.ckpt ]; then
  wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
  tar -xvf inception_v3_2016_08_28.tar.gz
  mv inception_v3.ckpt ${PRETRAINED_CHECKPOINT_DIR}/${MODEL_NAME}.ckpt
  rm inception_v3_2016_08_28.tar.gz
fi

## Download the dataset
#python download_and_convert_data.py \
#  --dataset_name=flowers \
#  --dataset_dir=${DATASET_DIR}

##################
# Start training #
##################
# Fine-tune only the new layers for 500 steps.
python train_roof_ft.py \
  --slim_dir=${SLIM_DIR} \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=${SET} \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/${MODEL_NAME}.ckpt \
  --checkpoint_exclude_scopes=${CKPT_EXCLUDE_SCOPES} \
  --trainable_scopes=${TRAINABLE_SCOPES} \
  --max_number_of_steps=3000 \
  --summary_snapshot_steps=300 \
  --model_snapshot_steps=300 \
  --log_every_n_steps=${LOG_EVERY_N_STEPS} \
  --batch_size=${BATCH_SIZE} \
  --weight_decay=${WEIGHT_DECAY} \
  --learning_rate=${INI_LR1} \
  --optimizer=${OPTIMIZER} \
#  --adam_beta1=${ADAM_BETA1} \
#  --adam_beta2=${ADAM_BETA2}



# Fine-tune all the new layers for MAX_TRAIN_STEPS steps.
python train_roof_ft.py \
  --slim_dir=${SLIM_DIR} \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=${SET} \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --checkpoint_path=${TRAIN_DIR} \
  --max_number_of_steps=${MAX_TRAIN_STEPS} \
  --batch_size=${BATCH_SIZE} \
  --summary_snapshot_steps=${SUMMARY_STEPS} \
  --model_snapshot_steps=${MODEL_SNAPSHOT_STEPS} \
  --log_every_n_steps=${LOG_EVERY_N_STEPS} \
  --weight_decay=${WEIGHT_DECAY} \
  --learning_rate=${INI_LR2} \
  --optimizer=${OPTIMIZER} \
#  --adam_beta1=${ADAM_BETA1} \
#  --adam_beta2=${ADAM_BETA2}


