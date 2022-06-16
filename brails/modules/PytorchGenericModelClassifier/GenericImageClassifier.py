# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 The Regents of the University of California
#
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
#


"""
This module has classes and methods for training and evaluating models.

.. rubric:: Contents

.. autosummary::

    CustomDataset
    PytorchImageClassifier

"""


import os
import json
import types
import random
import pathlib
import argparse
import warnings
from tqdm import tqdm
from glob import glob
from sys import exit

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR

import timm

import numpy as np
import pandas as pd


from PIL import Image
from PIL import UnidentifiedImageError
import matplotlib.pyplot as plt


from brails.utils.plotUtils import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


class CustomDataset(Dataset):
   
    """
    A customized dataset for constructing validation set

    
    Parameters
    ----------
    dataset: a Pytorch dataset object
    transform: the transformations that can be applied on the input

    """

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            x = self.transform(self.dataset[index][0])
        else:
            x = self.dataset[index][0]
        
        y = self.dataset[index][1]
        return x, y
    
    def __len__(self):

        return len(self.dataset)


class PytorchImageClassifier:

    """
    A Generic Image Classifier. Can be used for training and evaluating the model. 

    
    Parameters
    ----------
    modelName: architecture of the model. Please refer to https://github.com/rwightman/pytorch-image-models for supported models.
    imgDir: directories for training data
    valimgDir: directories for validation data
    random_split: ratio to split the data into a training set and validation set if validation data is not provided.
    resultFile: name of the result file for predicting multple images.
    workDir: the working directory
    download: False,
    printRes: show the probability and prediction
    printConfusionMatrix: whether to print the confusion matrix or not

    """

    def __init__(self, modelName=None, imgDir='', valimgDir='', download=False, random_split=[0.8, 0.2], resultFile='preds.csv', workDir='./tmp', printRes=True, printConfusionMatrix=False):


        #######################################################
        if not download:

            # the name of the trained model
            modelFile = os.path.join(workDir,'{}.pkl'.format(modelName))

            # meta data contains model name, 
            modelDetailFile = os.path.join(workDir,'{}.json'.format(modelName))

        else:

            modelFile = os.path.join(workDir + "/models/", '{}.pkl'.format(modelName))

            # meta data contains model name, 
            modelDetailFile = os.path.join(workDir + "/models/", '{}.json'.format(modelName))
        #######################################################

        self.download = download
        self.workDir = workDir
        self.modelFile = modelFile
        self.resultFile = os.path.join(workDir,resultFile)
        self.modelName = modelName
        self.modelDetailFile = modelDetailFile
        self.printRes = printRes
        self.printConfusionMatrix = printConfusionMatrix
        self.random_split = random_split
        #######################################################
        # create model

        if self.arch == 'transformer':

            self.model = timm.create_model('vit_base_patch16_224', pretrained=True)

        else:
            self.model = timm.create_model(arch, pretrained=True)

        #######################################################

        if imgDir:

            print ("Loading data")
            self.loadData(imgDir=imgDir, valimgDir=valimgDir)

            
        self.criterion =  nn.CrossEntropyLoss()

        # load local model
        if os.path.exists(modelFile):

            print('\nModel found locally: {} '.format(modelFile))

            if imgDir:

                print ("You are going to fine-tune the local model.")


            # check if a local definition of the model exists.
            if os.path.exists(self.modelDetailFile):
                
                with open(self.modelDetailFile) as f:
                    
                    self.classNames = json.load(f)['classNames']
                    
                    print('Class names found in the detail file: {} '.format(self.classNames))


            # change the number of output class
            self.model.reset_classifier(len(self.classNames))
            self.model = nn.DataParallel(self.model)

            self.load_model(modelFile)

        else:
                
            self.model.reset_classifier(len(self.classNames))

            if imgDir:
                
                print('Model file {} doesn\'t exist locally. You are going to train your own model.'.format(modelFile))


        #######################################################
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model.to(self.device)

        #######################################################

    def load_model(self, modelFile):

        """
        Load a pre-trained model given the path of the model.
        
        Parameters
        ----------
        modelFile: string
            The path to the pre-trained model

        """
        print ("Loading ", modelFile)

        if torch.cuda.is_available():

            state_dict = torch.load(modelFile)

        else:

            state_dict = torch.load(modelFile, map_location='cpu')

        self.model.load_state_dict(state_dict)


    def predictOneImage(self, imagePath):

        """
        Predict the labels of one image given the image path
        
        Parameters
        ----------
        imagePath: string
            The path to the image


        Return
        -------
        imagePath: string 
            The path to the image

        prediction: int or string
            The predicton of the model on the image

        prob: float
            The probabilty of the prediction
        """


        image = Image.open(imagePath).convert("RGB")

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        loader = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

        image = loader(image).float()

        self.model.eval()

        output = self.model(image.unsqueeze(0).to(self.device))
        prob, prediction = torch.max(torch.nn.functional.softmax(output, dim=-1), 1)


        if self.classNames: 

            prediction = self.classNames[prediction]

        else:
            prediction = prediction.cpu().item()

        print("Image :  {}     Class : {} ({})".format(imagePath, prediction, prob.cpu().item()))
        
        return [imagePath, prediction, prob]


    def predictMultipleImages(self, imagePathList, resultFile=None):
        
        """
        Predict the labels of mutiple images given the image paths
        
        Parameters
        ----------
         imagePathList: list
            A list of image paths
         
         resultFile: string
            The name of the result file. If not given, use the default name
        
        Return
        -------
        df: pandas dataframe

            The pandas dataframe which contains all the predictions

        """

        if resultFile:

            self.resultFile = os.path.join(self.workDir, resultFile)


        predictions = []
        probs = []

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.model.eval()

        loader = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

        for imagePath in imagePathList:

            try:

                image = Image.open(imagePath).convert("RGB")

            except UnidentifiedImageError:
                
                warnings.warn(f"Image format error: skipping image '{imagePath}'")
                probs.append(None)
                predictions.append(None)
                continue
            
            image = loader(image).float()
            output = self.model(image.unsqueeze(0).to(self.device))
            prob, prediction = torch.max(torch.nn.functional.softmax(output, dim=-1), 1)


            if os.path.getsize(imagePath)/1024 < 9: # small image, likely to be empty
                probs.append(0)
            else:
                probs.append(prob.cpu().item())

            if self.classNames: 

                prediction = self.classNames[prediction]
            else:
                prediction = prediction.cpu().item()

            predictions.append(prediction)

        
        if self.printRes:

            for img, pred, prob in zip(imagePathList, predictions, probs): 
            
                print("Image :  {}     Class : {} ({}%)".format(img, pred, str(round(prob*100,2)))) 
        

        df = pd.DataFrame(list(zip(imagePathList, predictions, probs)), columns =['image', 'prediction', 'probability'])
        print(df)
        df.to_csv(self.resultFile, index=False)
        print('Results written in file {}'.format(self.resultFile))

        return df

    def get_transform(self):

        """
        The transformations that are used for training and testing
        
        
        Return
        -------
        train_transforms: pytorch transformations
        
            The transformations used for training

        val_transforms: pytorch transformations
        
            The transformations used for validation

        """

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_transforms = [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.)),
            transforms.RandomGrayscale(p=0.5),
            transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize]


        train_transforms = transforms.Compose(train_transforms)

        val_transforms = [transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize]
 
        val_transforms = transforms.Compose(val_transforms)

        return train_transforms, val_transforms 


    def predictOneDirectory(self, directory_name=None, resultFile=None):

        """
        Predict the labels of all the images in one directory
        
        Parameters
        ----------
          directory_name: string
            The directory which has all the images
         
         resultFile: string
            The name of the result file. If not given, use the default name
        
        Return
        -------

        df: pandas dataframe

            The pandas dataframe which contains all the predictions

        """

        if not directory_name:
            print ("Please provide the directory for saving the images.")
            return

        if resultFile:

            self.resultFile = os.path.join(self.workDir,resultFile)


        self.printRes = False
        train_transforms, val_transforms = self.get_transform()

        image_path  = []

        valid_images = [".jpg",".gif",".png",".tga"]
        for f in os.listdir(directory_name):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue

            image_path .append(os.path.join(directory_name, f))

        if len(image_path ) == 0:

            print ("Not image files.")
            exit()
        else:
            print ("Found %d images." % (len(image_path)))

        self.predictMultipleImages(image_path)


    def Get_Images(self, root_dir):
 
        """
        Load the training images
        
        Parameters
        ----------
          root_dir: string
            The directory which has all the training images
        
        Return
        -------
        
        data: Pytorch ImageFolder class

            A generic data loader where the images are arranged in this way by default:

                root/dog/xxx.png
                root/dog/xxy.png
                root/dog/[...]/xxz.png

                root/cat/123.png
                root/cat/nsdf3.png
                root/cat/[...]/asd932_.png

        """

        data = torchvision.datasets.ImageFolder(root=root_dir)

        return data


    def loadData(self, imgDir="", valimgDir=''):

        """
        Load the training images. If valimgDir is not given, split the images into a training set and validation set
        
        Parameters
        ----------
          imgDir: string
            The directory which has all the images
        
          valimgDir: string
            The directory which has validation images

        Return
        -------

        """

        train_transforms, val_transforms = self.get_transform()

        if not valimgDir:

            print('No validation dataset. Split the data with 8:2.')

            dataset =  self.Get_Images(imgDir)

            print ('The class name to index: ', dataset.class_to_idx)

            self.classNames = list(dataset.class_to_idx.keys())

            train_size = int(len(dataset)*self.random_split[0])
            val_size   = int(len(dataset)*self.random_split[1])

            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])


        else:

            train_dataset = self.Get_Images(imgDir)
            val_dataset   = self.Get_Images(valimgDir)

            print ('The class name to index: ', train_dataset.class_to_idx)
            
            self.classNames = list(train_dataset.class_to_idx.keys())


        self.train_dataset = CustomDataset(train_dataset, train_transforms)
        self.val_dataset   = CustomDataset(val_dataset,   val_transforms)


    def fine_tuning(self, lr=0.001, batch_size=32, epochs=10, plot=False):
        """
        Fine-tune the model using the provided images
    
        Parameters
        ----------
          lr: float
            The learning rate for training the model
    
          batch_size: int
            The batch size for training the model. 
        
          epoch: int
            The number of epochs to training the model

          plot: bool

            Whether or not to plot the training accuracy and validation accuracy

        Return
        -------

        """    

        self.train(lr=lr, batch_size=batch_size, epochs=epochs, plot=False)
        

    def train(self, lr=0.01, batch_size=64, epochs=10, plot=False):

        """
        Training the model with the training set and evaluate on the validation set. The model and metadata will be saved. 
        
        Parameters
        ----------
          lr: float
            The learning rate for training the model
    
          batch_size: int
            The batch size for training the model. 
        
          epoch: int
            The number of epochs to training the model

          plot: bool

            Whether or not to plot the training accuracy and validation accuracy

        Return
        -------
        
        """

        if not hasattr(self, 'train_dataset'):
            print ("No training data. Please provide the directory to training dataset when you initialize the ImageClassifier.")
            exit()

        ############################################################
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size = batch_size, shuffle=True, num_workers=4)
        self.val_loader   = torch.utils.data.DataLoader(self.val_dataset,   batch_size = batch_size, shuffle=False, num_workers=4)


        ############################################################
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, momentum=0.9)

        scheduler = StepLR(optimizer, step_size=10, gamma=0.95)


        all_train_acc = []
        all_train_loss = []

        all_val_acc = []
        all_val_loss = []

        for epoch in range(epochs):

            print ("Epoch: ", epoch)

            self.model.train()

            predictions = []
            ground_truths = []
            correct = 0

            avg_loss = 0

            for i, (images, labels) in enumerate(tqdm(self.train_loader)):

                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                output = self.model(images.float())

                loss = self.criterion(output, labels)
    
                loss.backward()
                optimizer.step()

                avg_loss += loss.detach().cpu().numpy()


                _, pred = torch.max(output, 1)

                predictions.extend(pred.detach().cpu().tolist())  # Should do the detach internally
                ground_truths.extend(labels.detach().cpu().tolist())
            
                correct += pred.eq(labels.view_as(pred)).sum().item()


            print ('Train Accuracy: %s'% (100.0 * correct / len(self.train_loader.dataset)))

            print ('Train Loss: %s'% (avg_loss / len(self.train_loader)))

            scheduler.step()

            all_train_acc.append(100.0 * correct / len(self.train_loader.dataset))
            all_train_loss.append(avg_loss / len(self.train_loader))

            val_acc, val_loss = self.evaluate()

            all_val_acc.append(val_acc)
            all_val_loss.append(val_loss)

            print ("\n")

        # Plot learning curves

        if plot:

            plt.figure(figsize=(8, 8))
            plt.subplot(2, 1, 1)
            plt.plot(all_train_acc, label='Training Accuracy')
            plt.plot(all_val_acc, label='Validation Accuracy')
            plt.legend(loc='lower right')
            plt.ylabel('Accuracy')
            plt.ylim([min(plt.ylim()),1])
            plt.title('Training and Validation Accuracy')

            plt.subplot(2, 1, 2)
            plt.plot(all_train_loss, label='Training Loss')
            plt.plot(all_val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.ylabel('Cross Entropy')
            plt.ylim([0,1.0])
            plt.title('Training and Validation Loss')
            plt.xlabel('epoch')
            plt.show()


        # Save the model in the working folder 
        self.save() 

        modelDetails = {
                'modelName' : self.modelName,
                'classNames' : self.classNames,
                'final train accuracy': all_train_acc[-1],
                'final val accuracy': all_val_acc[-1]
            }

        with open(self.modelDetailFile, 'w') as outfile:
            json.dump(modelDetails, outfile)

        print('Model details saved in ', self.modelDetailFile)


    def evaluate(self):

        """
        Evaluate the performance of the model on the validation set        

        Parameters
        ----------


        Return
        -------
            
        validation accuracy: float
            The accuracy on the validation set


        validation loss: float
            The avarage loss on the validation set
        """


        self.model.eval()

        predictions = []
        ground_truths = []
        correct = 0
        avg_loss = 0.0
        for i, (images, labels ) in enumerate(self.val_loader):

            images = images.to(self.device)
            labels = labels.to(self.device)

            output = self.model(images.float())

            loss = self.criterion(output, labels)

            _, pred = torch.max(output, 1)

            predictions.extend(pred.detach().cpu().tolist())  # Should do the detach internally
            ground_truths.extend(labels.detach().cpu().tolist())
        
            correct += pred.eq(labels.view_as(pred)).sum().item()

            avg_loss += loss.detach().cpu().numpy()


        print ('Val Accuracy: %s'% (100.0 * correct / len(self.val_loader.dataset)))

        cnf_matrix = confusion_matrix(predictions, ground_truths)

        print ("validation Confusion Matrix")
        print (cnf_matrix)

        if self.printConfusionMatrix:
            plot_confusion_matrix(cnf_matrix, classes=self.classNames, title='Confusion matrix',normalize=True, xlabel='Labels', ylabel='Predictions')

        return 100.0 * correct / len(self.val_loader.dataset), avg_loss / len(self.val_loader)

    def save(self):


        """
        Save the model with the name of self.modelFile

        Parameters
        ----------


        Return
        -------
            
        """

        torch.save(self.model.state_dict(), self.modelFile)

        print('Model saved at ', self.modelFile)


def main():
        
    # Please refer to https://github.com/rwightman/pytorch-image-models for supported models
    # imgDir/valimgDir: directories for training/validation images. Arranged in this way:

    #    root/dog/xxx.png
    #    root/dog/xxy.png
    #    root/dog/[...]/xxz.png

    #    root/cat/123.png
    #    root/cat/nsdf3.png
    #    root/cat/[...]/asd932_.png

    work = PytorchImageClassifier(modelName='transformer_rooftype_v1', imgDir='./roofType/')

    work.fine_tuning(lr=0.001, batch_size=64, epochs=5)

    work.predictOneDirectory("./roofType/flat")

    #work.predictOneImage("./roofType/flat/TopViewx-76.84779286x38.81642318.png")

    #work.predictMultipleImages(["./roofType/flat/TopViewx-76.84779286x38.81642318.png", "./roofType/flat/TopViewx-76.96240924000001x38.94450328.png"])


if __name__ == '__main__':
    main()
