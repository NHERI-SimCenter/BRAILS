#!/bin/bash
wget -O ./checkpoints/best_masked.pkl https://zenodo.org/record/4044228/files/best_masked.pkl

MODEL_NAME=ade20k-resnet50dilated-ppm_deepsup
MODEL_PATH=csail_segmentation_tool/csail_seg/$MODEL_NAME
RESULT_PATH=./

ENCODER=$MODEL_NAME/encoder_epoch_20.pth
DECODER=$MODEL_NAME/decoder_epoch_20.pth

# Download model weights and image
if [ ! -e $MODEL_PATH ]; then
  mkdir -p $MODEL_PATH
fi
if [ ! -e $ENCODER ]; then
  wget -O $MODEL_PATH/encoder_epoch_20.pth http://sceneparsing.csail.mit.edu/model/pytorch/$ENCODER
fi
if [ ! -e $DECODER ]; then
  wget -O $MODEL_PATH/decoder_epoch_20.pth http://sceneparsing.csail.mit.edu/model/pytorch/$DECODER
fi
