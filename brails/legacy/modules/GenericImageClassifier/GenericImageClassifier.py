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

import os
import json
import types
import random
import pathlib
import argparse
import warnings
from glob import glob

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.inception_v3 import preprocess_input
from PIL import UnidentifiedImageError
from brails.modules.ModelZoo import zoo
import matplotlib.pyplot as plt

from brails.utils.plotUtils import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score

class ImageClassifier:
    """ A Generic Image Classifier. """

    def __init__(self, modelName=None, classNames=None, resultFile='preds.csv', workDir='tmp', printRes=True):
        '''
        modelFile: path to the model
        classNames: a list of classnames
        '''

        if not os.path.exists(workDir): os.makedirs(workDir)

        if not modelName:

            modelName = 'myCoolModelv0.1'
            print('You didn\'t specify modelName, a default one is assigned {}.'.format(modelName))

        modelFile = os.path.join(workDir,'{}.h5'.format(modelName))
        modelDetailFile = os.path.join(workDir,'{}.json'.format(modelName))

        self.workDir = workDir
        self.modelFile = modelFile
        self.classNames = classNames
        self.resultFile = os.path.join(workDir,resultFile)
        self.modelName = modelName
        self.modelDetailFile = modelDetailFile
        self.printRes = printRes

        if os.path.exists(modelFile):
            self.model = load_model(modelFile)
            print('Model found locally: {} '.format(modelFile))
            
            # check if a local definition of the model exists.
            if os.path.exists(self.modelDetailFile):
                with open(self.modelDetailFile) as f:
                    self.classNames = json.load(f)['classNames']
                    print('Class names found in the detail file: {} '.format(self.classNames))

        else:
            print('Model file {} doesn\'t exist locally. You are going to train your own model.'.format(modelFile))


    def predictOne(self,imagePath,color_mode='rgb'):

        img = image.load_img(imagePath, color_mode=color_mode, target_size=(256, 256))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        prediction = self.model.predict(x)
        prob = max(prediction[0])
        prediction = np.argmax(prediction[0])
        if self.classNames: prediction = self.classNames[prediction]

        if os.path.getsize(imagePath)/1024 < 9: # small image, likely to be empty
            #print("{imagePath} is blank. No predictions.")
            #return [imagePath,None, None]
            print("Image :  {}     Class : {} ({}%)".format(imagePath, prediction, str(round(0*100,2)))) 
            return [imagePath,prediction,0]
        else:
            print("Image :  {}     Class : {} ({}%)".format(imagePath, prediction, str(round(prob*100,2)))) 
            return [imagePath,prediction,prob]

    def predictMulti(self,imagePathList,color_mode='rgb'):
        predictions = []
        probs = []
        for imagePath in imagePathList:
            '''
            if os.path.getsize(imagePath)/1024 < 9: # small image, likely to be empty
                probs.append(0)
                predictions.append(None)
            else:
                img = image.load_img(imagePath, target_size=(256, 256))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                prediction = self.model.predict(x)
                probs.append(max(prediction[0]))
                prediction = np.argmax(prediction[0])
                if self.classNames: prediction = self.classNames[prediction]
                predictions.append(prediction)
            '''
            try:
                img = image.load_img(imagePath, color_mode=color_mode, target_size=(256, 256))
            except UnidentifiedImageError:
                warnings.warn(f"Image format error: skipping image '{imagePath}'")
                probs.append(None)
                predictions.append(None)
                continue
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            prediction = self.model.predict(x)
            if os.path.getsize(imagePath)/1024 < 9: # small image, likely to be empty
                probs.append(0)
            else:
                probs.append(max(prediction[0]))
            prediction = np.argmax(prediction[0])
            if self.classNames: prediction = self.classNames[prediction]
            predictions.append(prediction)

        if self.printRes:
            for img, pred, prob in zip(imagePathList, predictions, probs): 
                print("Image :  {}     Class : {} ({}%)".format(img, pred, str(round(prob*100,2)))) 
        df = pd.DataFrame(list(zip(imagePathList, predictions, probs)), columns =['image', 'prediction', 'probability'])
        print(df)
        df.to_csv(self.resultFile, index=False)
        print('Results written in file {}'.format(self.resultFile))

        return df
    
    def predict(self,image,color_mode='rgb'):
        if isinstance(image, types.GeneratorType):
            image = list(image)
        if isinstance(image, list): 
            pred = self.predictMulti(image,color_mode=color_mode)
        elif isinstance(image, (str, pathlib.Path)):
            pred = self.predictOne(image,color_mode=color_mode)
        else: 
            raise TypeError("")
        return pred

    def loadData(self, imgDir, valimgDir='', randomseed=1993, color_mode='rgb', image_size=(256, 256), batch_size = 32, split=[0.8,0.2]):

        if valimgDir == '':
            #print('* First split the data with 8:2.')
            self.train_ds = image_dataset_from_directory(imgDir,
            validation_split=split[1],
            subset="training",
            seed=randomseed,
            color_mode=color_mode,
            image_size=image_size,
            batch_size=batch_size,
            label_mode='categorical')


            self.val_ds = image_dataset_from_directory(
                imgDir,
                validation_split=split[1],
                subset="validation",
                seed=randomseed,
                color_mode=color_mode,
                image_size=image_size,
                batch_size=batch_size,
                label_mode='categorical'
            )
        else:
            self.train_ds = image_dataset_from_directory(imgDir,
            color_mode=color_mode,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True,
            label_mode='categorical')


            self.val_ds = image_dataset_from_directory(
                valimgDir,
                color_mode=color_mode,
                image_size=image_size,
                batch_size=batch_size,
                shuffle=True,
                label_mode='categorical'
            )        

        self.image_size = image_size
        self.batch_size = batch_size

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

    def train(self,baseModel='InceptionV3',lr1=0.0001,initial_epochs=10,
                    fine_tune_at=300,lr2=0.00001,fine_tune_epochs=50,color_mode='rgb',
                    horizontalFlip=False,verticalFlip=False,dropout=0.6,randomRotation=0.0,callbacks=[],plot=True):
        
        ## 1. Model zoo
         
        modelDict = {
            'Xception': tf.keras.applications.Xception	,
            'VGG16': tf.keras.applications.VGG16	    ,
            'VGG19': tf.keras.applications.VGG19	    ,
            'ResNet50': tf.keras.applications.ResNet50	, # 175
            'ResNet101': tf.keras.applications.ResNet101	, #345
            'ResNet152': tf.keras.applications.ResNet152	,
            'ResNet50V2': tf.keras.applications.ResNet50V2	,
            'ResNet101V2': tf.keras.applications.ResNet101V2,	
            'ResNet152V2': tf.keras.applications.ResNet152V2,	
            'InceptionV3': tf.keras.applications.InceptionV3,	#311
            'InceptionResNetV2': tf.keras.applications.InceptionResNetV2,
            'MobileNet': tf.keras.applications.MobileNet	,
            'MobileNetV2': tf.keras.applications.MobileNetV2,	
            'DenseNet121': tf.keras.applications.DenseNet121,	
            'DenseNet169': tf.keras.applications.DenseNet169,	
            'DenseNet201': tf.keras.applications.DenseNet201,	
            'NASNetMobile': tf.keras.applications.NASNetMobile	,
            'NASNetLarge': tf.keras.applications.NASNetLarge,	
            'EfficientNetB0': tf.keras.applications.EfficientNetB0,	
            'EfficientNetB1': tf.keras.applications.EfficientNetB1,	
            'EfficientNetB2': tf.keras.applications.EfficientNetB2,	
            'EfficientNetB3': tf.keras.applications.EfficientNetB3,	
            'EfficientNetB4': tf.keras.applications.EfficientNetB4,	
            'EfficientNetB5': tf.keras.applications.EfficientNetB5,	
            'EfficientNetB6': tf.keras.applications.EfficientNetB6,	
            'EfficientNetB7': tf.keras.applications.EfficientNetB7 #813
            }

        ## 2. Create model

        ### 2.1 Load the pre-trained base model

        if baseModel not in modelDict.keys():
            print('{} is not found or not supported. \n Choose from {}'.format(baseModel, modelDict.keys()))
            return

        # Load InceptionV3 model pre-trained on imagenet
        if color_mode=='rgb':
            imgDim = 3
        elif color_mode=='rgba':
            imgDim = 4
        else:
            imgDim = 1
        base_model = modelDict[baseModel](input_shape=self.image_size + (imgDim,), # self.image_size is defined in loadData
                                                       include_top=False,
                                                       weights='imagenet')
        # Freeze the base model
        base_model.trainable = False
          

        ### 2.2 Add preprocessing layers and a classification head to build the model

        # Augmentation
        aug_list = []
        if horizontalFlip: 
            aug_list.append(tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'))
            print('Horizontal flip applied')
        if verticalFlip: 
            aug_list.append(tf.keras.layers.experimental.preprocessing.RandomFlip('vertical'))
            print('Vertical flip applied')
        if randomRotation>0.0: 
            aug_list.append(tf.keras.layers.experimental.preprocessing.RandomRotation(randomRotation))
            print('Random rotation = {} applied'.format(randomRotation))
        if len(aug_list)>0: data_augmentation = tf.keras.Sequential(aug_list)

        # Pre-processing layer
        inputs = tf.keras.Input(shape=self.image_size + (imgDim,))
        if len(aug_list)>0: 
            x = data_augmentation(inputs) # augment
            x = preprocess_input(x) 
        else: x = preprocess_input(inputs) 


        # Then go into the backbone model
        x = base_model(x)

        # Then go into the classification header
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(dropout)(x) # You can change the dropout rate 
        prediction_layer = tf.keras.layers.Dense(len(self.classNames), activation='softmax')
        outputs = prediction_layer(x)

        # Put them together
        self.model = tf.keras.Model(inputs, outputs)

        ### Compile the model
        self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr1, momentum=0.9),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        self.model.summary()

        ### 3. Train the model
        #callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
        history = self.model.fit(self.train_ds, epochs=initial_epochs, validation_data=self.val_ds, callbacks=callbacks)

        # Plot learning curves

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        if plot:
            plt.figure(figsize=(8, 8))
            plt.subplot(2, 1, 1)
            plt.plot(acc, label='Training Accuracy')
            plt.plot(val_acc, label='Validation Accuracy')
            plt.legend(loc='lower right')
            plt.ylabel('Accuracy')
            plt.ylim([min(plt.ylim()),1])
            plt.title('Training and Validation Accuracy')

            plt.subplot(2, 1, 2)
            plt.plot(loss, label='Training Loss')
            plt.plot(val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.ylabel('Cross Entropy')
            plt.ylim([0,1.0])
            plt.title('Training and Validation Loss')
            plt.xlabel('epoch')
            plt.show()

        ## 4. Fine tuning

        ### 4.1 Un-freeze the top layers of the model

        # Un-freeze the whole base model
        base_model.trainable = True

        # Fine-tune from this layer onwards
        #fine_tune_at = 300 # There are a total of 311 layer

        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
          layer.trainable =  False

        ### 4.2 Compile the model

        self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr2, momentum=0.9),
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        #model.summary()

        ### 4.3 Continue training the model

        total_epochs =  initial_epochs + fine_tune_epochs
        history_fine = self.model.fit(self.train_ds, epochs=total_epochs, initial_epoch=history.epoch[-1], validation_data=self.val_ds, callbacks=callbacks)

        # Plot learning curves

        acc += history_fine.history['accuracy']
        val_acc += history_fine.history['val_accuracy']

        loss += history_fine.history['loss']
        val_loss += history_fine.history['val_loss']

        if plot:
            plt.figure(figsize=(8, 8))
            plt.subplot(2, 1, 1)
            plt.plot(acc, label='Training Accuracy')
            plt.plot(val_acc, label='Validation Accuracy')
            plt.ylim([0.8, 1])
            plt.plot([initial_epochs-1,initial_epochs-1],
                      plt.ylim(), label='Start Fine Tuning')
            plt.legend(loc='lower right')
            plt.title('Training and Validation Accuracy')

            plt.subplot(2, 1, 2)
            plt.plot(loss, label='Training Loss')
            plt.plot(val_loss, label='Validation Loss')
            plt.ylim([0, 1.0])
            plt.plot([initial_epochs-1,initial_epochs-1],
                     plt.ylim(), label='Start Fine Tuning')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Loss')
            plt.xlabel('epoch')
            plt.show()

        ## Evaluate the performance of the model
        # Evaluate the overall performance on the val_ds set
        #loss, accuracy = self.model.evaluate(self.val_ds)
        #print('Val accuracy :', accuracy)

        ## 5. Save model

        # Save the model in the current folder 
        self.model.save(self.modelFile) 
        print('Model saved at ', self.modelFile)

        modelDetails = {
                'modelName' : self.modelName,
                'classNames' : self.classNames
            }
        with open(self.modelDetailFile, 'w') as outfile:
            json.dump(modelDetails, outfile)
        print('Model details saved in ', self.modelDetailFile)

    def evaluate(self):
        prediction_tensor = self.model.predict(self.val_ds)
        prediction_int = list(np.argmax(prediction_tensor,axis=1))
        labels_int = []
        for i,gt in self.val_ds:
          for j in gt:
            labels_int.append(np.argmax(j))

        class_names = materialClassifier.classNames
        prediction = [class_names[i] for i in prediction_int]
        label = [class_names[i] for i in labels_int]

        print(f'Accuracy is   : {accuracy_score(prediction,label)}')
        print(f'F1 score is   : {accuracy_score(f1_score,label)}')
        cnf_matrix = confusion_matrix(prediction,label)
        plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix',normalize=True,xlabel='Labels',ylabel='Predictions')



    def retrain(self,lr1=0.0001,initial_epochs=10,callbacks=[]):
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
        history = self.model.fit(self.train_ds, epochs=initial_epochs, validation_data=self.val_ds,callbacks=callbacks)

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




