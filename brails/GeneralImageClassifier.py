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


class ImageClassifier:
    """ A General Image Classifier. """


    def __init__(self, modelFile, classNames=None, resultFile='preds.csv'):
        '''
        modelFile: path to the model
        classNames: a list of classnames
        '''
        self.modelFile = modelFile
        self.classNames = classNames
        self.resultFile = resultFile
        if os.path.exists(modelFile):
            self.model = load_model(modelFile)
        else:
            print('Model file {} doesn\'t exist.'.format(modelFile))

    def predictOne(self,imagePath):

        img = image.load_img(imagePath, target_size=(256, 256))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        prediction = self.model.predict(x)
        prob = max(prediction[0])
        prediction = np.argmax(prediction[0])
        if self.classNames: prediction = self.classNames[prediction]
        
        print("Image :  {}     Class : {} ({}%)".format(imagePath, prediction, prob)) 

        return prediction

    def predictMulti(self,imagePathList):
        predictions = []
        probs = []
        for imagePath in imagePathList:
            img = image.load_img(imagePath, target_size=(256, 256))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            prediction = self.model.predict(x)
            probs.append(max(prediction[0]))
            prediction = np.argmax(prediction[0])
            if self.classNames: prediction = self.classNames[prediction]
            predictions.append(prediction)

        for img, pred, prob in zip(imagePathList, predictions, probs): 
            print("Image :  {}     Class : {} ({}%)".format(img, pred, prob)) 

        df = pd.DataFrame(list(zip(imagePathList, predictions, probs)), columns =['image', 'prediction', 'probability']) 
        df.to_csv(self.resultFile, index=False)
        print('Results written in file {}'.format(self.resultFile))

        return predictions
    
    def predict(self,image):
        if type(image) is list: self.predictMulti(image)
        elif type(image) is str: self.predictOne(image)
        else: print("The parameter of this function should be string or list.")



def main():
    pass

if __name__ == '__main__':
    main()




