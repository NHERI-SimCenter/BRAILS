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
import random
import tensorflow as tf
import numpy as np
from glob import glob
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.inception_v3 import preprocess_input


class ImageClassifier:
    """ A General Image Classifier. """

    def __init__(self, modelName=None, classNames=None, resultFile='preds.csv', workDir='tmp'):
        '''
        modelFile: path to the model
        classNames: a list of classnames
        '''

        if not os.path.exists(workDir): os.makedirs(workDir)

        if not modelName:

            modelName = 'myCoolModelv0.1'
            print('You didn\'t specify modelName, a default one is assigned {}.'.format(modelName))

        modelFile = os.path.join(workDir,'{}.h5'.format(modelName))

        self.workDir = workDir
        self.modelFile = modelFile
        self.classNames = classNames
        self.resultFile = resultFile

        if os.path.exists(modelFile):
            self.model = load_model(modelFile)
            print('Model found locally: {} '.format(modelFile))
        else:
            print('Model file {} doesn\'t exist locally. You are going to train your own model.'.format(modelFile))

    def predictOne(self,imagePath):

        img = image.load_img(imagePath, target_size=(256, 256))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        prediction = self.model.predict(x)
        prob = max(prediction[0])
        prediction = np.argmax(prediction[0])
        if self.classNames: prediction = self.classNames[prediction]
        
        print("Image :  {}     Class : {} ({}%)".format(imagePath, prediction, str(round(prob*100,2)))) 

        return [imagePath,prediction,prob]

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
            print("Image :  {}     Class : {} ({}%)".format(img, pred, str(round(prob*100,2)))) 

        df = pd.DataFrame(list(zip(imagePathList, predictions, probs)), columns =['image', 'prediction', 'probability']) 
        df.to_csv(self.resultFile, index=False)
        print('Results written in file {}'.format(self.resultFile))

        return df
    
    def predict(self,image):
        if type(image) is list: pred = self.predictMulti(image)
        elif type(image) is str: pred = self.predictOne(image)
        else: 
            print("The parameter of this function should be string or list.")
            pred = []
        return pred

    def loadData(self, imgDir, randomseed=1993, image_size=(256, 256), batch_size = 32, split=[0.8,0.2]):

        #print('* First split the data with 8:2.')
        self.train_ds = image_dataset_from_directory(imgDir,
        validation_split=split[0],
        subset="training",
        seed=randomseed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical')

        self.val_ds = image_dataset_from_directory(
            imgDir,
            validation_split=split[1],
            subset="validation",
            seed=randomseed,
            image_size=image_size,
            batch_size=batch_size,
            label_mode='categorical'
        )

        newClassNames = self.train_ds.class_names

        # Configure the dataset for performance 
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.train_ds = self.train_ds.prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.prefetch(buffer_size=AUTOTUNE)

        if self.classNames == None:
            self.classNames = newClassNames
        print('The names of the classes are: ', newClassNames)

        if newClassNames != self.classNames:
            print('Error. Folder names {} mismatch predefined classNames: {}'.format(newClassNames, self.classNames))
            return


    def retrain(self,lr1=0.0001,initial_epochs=10):
        '''
        if self.train_ds.class_names != self.classNames:
            print('Can not retrain. Folder names mismatch predefined classNames: {}'.format(self.classNames))
            return
        '''

        ### Compile the model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr1),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        self.model.summary()

        ### Train the model
        history = self.model.fit(self.train_ds, epochs=initial_epochs, validation_data=self.val_ds)

    def save(self, newModelName=None):
        if newModelName != None: 
            newFileName = os.path.join(self.workDir,'{}.h5'.format(newModelName))
        else:
            newFileName = self.modelFile
        self.model.save(newFileName) 
        print('Model saved at ', newFileName)



def main():
    pass

if __name__ == '__main__':
    main()




