# -*- coding: utf-8 -*-
"""
/*------------------------------------------------------*
|                         BRAILS                        |
|                                                       |
| Author: Charles Wang,  UC Berkeley, c_w@berkeley.edu  |
|                                                       |
| Date:    10/15/2020                                   |
*------------------------------------------------------*/
"""

import argparse
import os
import tensorflow as tf
import numpy as np
from glob import glob
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser(description='Train a classifier')

parser.add_argument('--imageDir',help='The path of a folder containing images',type=str)
parser.add_argument('--imageList',help='A list of image paths', nargs='+')

parser.add_argument('--modelDir',help='The path of a folder where the model (h5) resides',type=str,required=True)
parser.add_argument('--modelFile',default='classifier.h5',help='The name of the model file (*.h5)',type=str)
parser.add_argument('--resultFile',default='preds.csv',help='The name of the result file (*.csv)',type=str)

parser.add_argument('--classNames',help='List of class names', nargs='+')

args = parser.parse_args()

imageDir = args.imageDir
imgList = args.imageList
modelDir = args.modelDir
modelFile = args.modelFile
resultFile = args.resultFile
classNames = args.classNames

if imgList is None:
    if imageDir is None:
        print("You need to provide --imageDir or --imageList")
        exit()
    else:
        imgList = []
        imgList += glob(os.path.join(imageDir, "*.png"))
        imgList += glob(os.path.join(imageDir, "*.jpg"))
        imgList += glob(os.path.join(imageDir, "*.jpeg"))

print(os.path.join(modelDir, modelFile))
model = load_model(os.path.join(modelDir, modelFile))


predictions = []
for img_path in imgList:
  img = image.load_img(img_path, target_size=(256, 256))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  prediction = model.predict(x)
  prediction = np.argmax(prediction[0])
  predictions.append(prediction)
  #classNames[prediction]


if classNames:
    for i,p in enumerate(predictions):
        predictions[i] = classNames[p]

df = pd.DataFrame(list(zip(imgList, predictions)), columns =['image', 'prediction']) 
df.to_csv(resultFile, index=False)

for img, pred in zip(imgList, predictions): 
    print ("Image :  %s     Class : %s" %(img, pred)) 
