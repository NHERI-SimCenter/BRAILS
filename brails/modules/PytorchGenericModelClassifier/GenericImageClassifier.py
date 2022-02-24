# -*- coding: utf-8 -*-
"""
/*------------------------------------------------------*
|                         BRAILS                        |
|                                                       |
| Author: Yunhui Guo,  UC Berkeley, yunhui@berkeley.edu  |
|                                                       |
| Date:    2/17/2022                                   |
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
    """ A Generic Image Classifier. Can be used for training and evaluating the model. """

    def __init__(self, modelName=None, imgDir='', valimgDir='', random_split=[0.8, 0.2], resultFile='preds.csv', workDir='./tmp', printRes=True, printConfusionMatrix=False):
        '''
        modelName: architecture of the model. Please refer to https://github.com/rwightman/pytorch-image-models for supported models.
        imgDir: directories for training data
        valimgDir: directories for validation data
        random_split: ratio to split the data into a training set and validation set if validation data is not provided.
        resultFile: name of the result file for predicting multple images.
        workDir: the working directory
        printRes: show the probability and prediction
        printConfusionMatrix: whether to print the confusion matrix or not
        '''

        if not os.path.exists(workDir): 
            os.makedirs(workDir)

        if not modelName:

            modelName = 'resnet18_v1'
            arch = 'resnet18'
            print('You didn\'t specify modelName, a default one is assigned {}.'.format(modelName))
        else:

            task, arch, version = modelName.split("_")


        # the name of the trained model
        modelFile = os.path.join(workDir,'{}.pickle'.format(modelName))
        
        # meta data contains model name, 
        modelDetailFile = os.path.join(workDir,'{}.json'.format(modelName))


        self.workDir = workDir
        self.modelFile = modelFile
        self.resultFile = os.path.join(workDir,resultFile)
        self.modelName = modelName
        self.modelDetailFile = modelDetailFile
        self.printRes = printRes
        self.printConfusionMatrix = printConfusionMatrix
        self.random_split = random_split
        self.classNames = None
        #######################################################
        # create model
        self.model = timm.create_model(arch, pretrained=True)

        #######################################################

        if imgDir:
            print ("Loading data")
            self.loadData(imgDir=imgDir, valimgDir=valimgDir)

            self.model.reset_classifier(len(self.classNames))

            
        self.criterion =  nn.CrossEntropyLoss()

        # load local model

        if os.path.exists(modelFile):
            print('Model found locally: {} '.format(modelFile))

            if imgDir:
                print ("You are going to fine-tune the local model.")


            # check if a local definition of the model exists.
            if os.path.exists(self.modelDetailFile):
                with open(self.modelDetailFile) as f:
                    self.classNames = json.load(f)['classNames']
                    print('Class names found in the detail file: {} '.format(self.classNames))

            # change the number of output class
            self.model.reset_classifier(len(self.classNames))

            self.load_model(modelFile)

        else:

            if imgDir:
                
                print('Model file {} doesn\'t exist locally. You are going to train your own model.'.format(modelFile))
            else:
                print ("Pre-trained model does not exist. Need to provide training data.")
                exit()

        #######################################################
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        #######################################################

    def load_model(self, modelFile):

        state_dict = torch.load(modelFile)

        self.model.load_state_dict(state_dict)


    def predictOneImage(self, imagePath):

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


    def predictMultipleImages(self,imagePathList, resultFile=None):
        
        if resultFile:

            self.resultFile = os.path.join(self.workDir, resultFile)


        predictions = []
        probs = []

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.model.eval()

        loader = transforms.Compose([transforms.Scale(224), transforms.ToTensor(), normalize])

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

        if not directory_name:
            print ("Please provide the directory for saving the images.")
            return

        if resultFile:

            self.resultFile = os.path.join(self.workDir,resultFile)


        self.printRes = False
        train_transforms, val_transforms = self.get_transform()

        image_path = list( pathlib.Path(directory_name).rglob('*.[pP][nN][gG]') )

        self.predictMultipleImages(image_path)


    def Get_Images(self, root_dir):
        
        data = torchvision.datasets.ImageFolder(root=root_dir)

        return data


    def loadData(self, imgDir="", valimgDir=''):

        train_transforms, val_transforms = self.get_transform()

        if not valimgDir:

            print('No validation dataset. Split the data with 8:2.')

            dataset =  self.Get_Images(imgDir)

            newClassNames = list(dataset.class_to_idx.keys())

            train_size = int(len(dataset)*self.random_split[0])
            val_size   = int(len(dataset)*self.random_split[1])

            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])


            self.train_dataset = CustomDataset(train_dataset, train_transforms)
            self.val_dataset   = CustomDataset(val_dataset,   val_transforms)

        else:

            self.train_dataset = Get_Images(imgDir, train_transforms)
            self.val_dataset   = Get_Images(valimgDir, val_transforms)
        

            newClassNames = list(train_dataset.class_to_idx.keys())
        
        self.classNames = newClassNames

        print('The names of the classes are: ', self.classNames)
        

    def train(self, lr=0.01, batch_size=64, epochs=10, plot=False):

        if not hasattr(self, 'train_dataset'):
            print ("No training data. Please provide the directory to training dataset when you initialize the ImageClassifier.")
            return

        ############################################################
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size= batch_size, shuffle=True, num_workers=4)
        self.val_loader   = torch.utils.data.DataLoader(self.val_dataset,     batch_size = batch_size, shuffle=False, num_workers=4)


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

            for i, (images, labels ) in enumerate(self.train_loader):

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
            plot_confusion_matrix(cnf_matrix, classes=self.class_names, title='Confusion matrix',normalize=True,xlabel='Labels',ylabel='Predictions')

        return 100.0 * correct / len(self.val_loader.dataset), avg_loss / len(self.val_loader)

    def save(self):

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

    work = PytorchImageClassifier(modelName='rooftype_resnet18_v1', imgDir='/home/yunhui/SimCenter/train_BRAILS_models/datasets/roofType/')

    work.train(lr=0.01, batch_size=64, epochs=5)

    work.predictOneDirectory("/home/yunhui/SimCenter/train_BRAILS_models/datasets/roofType/flat")

    #work.predictOneImage("/home/yunhui/SimCenter/train_BRAILS_models/datasets/roofType/flat/TopViewx-76.84779286x38.81642318.png")

    #work.predictMultipleImages(["/home/yunhui/SimCenter/train_BRAILS_models/datasets/roofType/flat/TopViewx-76.84779286x38.81642318.png", "/home/yunhui/SimCenter/train_BRAILS_models/datasets/roofType/flat/TopViewx-76.96240924000001x38.94450328.png"])


if __name__ == '__main__':
    main()
