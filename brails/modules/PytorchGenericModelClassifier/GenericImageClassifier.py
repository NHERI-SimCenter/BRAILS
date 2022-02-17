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


class MyLazyDataset(Dataset):
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


class ImageClassifier:
    """ A Generic Image Classifier. Can be used for  """

    def __init__(self, modelName=None, imgDir="/home/yunhui/SimCenter/train_BRAILS_models/datasets/roofType", valimgDir='', batch_size=256, resultFile='preds.csv', workDir='./tmp', printRes=True, printConfusionMatrix=False):
        '''
        modelFile: path to the model
        classNames: a list of classnames
        '''

        if not os.path.exists(workDir): 
            os.makedirs(workDir)

        if not modelName:

            modelName = 'resnet18'
            print('You didn\'t specify modelName, a default one is assigned {}.'.format(modelName))


        # the name of the trained model
        modelFile = os.path.join(workDir,'{}.pickle'.format(modelName))
        
        # meta data
        modelDetailFile = os.path.join(workDir,'{}.json'.format(modelName))

        self.workDir = workDir
        self.modelFile = modelFile
        self.resultFile = os.path.join(workDir,resultFile)
        self.modelName = modelName
        self.modelDetailFile = modelDetailFile
        self.printRes = printRes
        self.printConfusionMatrix = printConfusionMatrix
        self.batch_size = batch_size
        #######################################################
        # create model
        self.model = timm.create_model(modelName, pretrained=True)

        if imgDir != '':
            
            self.loadData(imgDir=imgDir, valimgDir=valimgDir)

        self.criterion =  nn.CrossEntropyLoss()

        #######################################################
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        if os.path.exists(modelFile):

            self.load_model(modelFile)

            print('Model found locally: {} '.format(modelFile))
            
            # check if a local definition of the model exists.
            if os.path.exists(self.modelDetailFile):
                with open(self.modelDetailFile) as f:
                    self.classNames = json.load(f)['classNames']
                    print('Class names found in the detail file: {} '.format(self.classNames))

        else:
            print('Model file {} doesn\'t exist locally. You are going to train your own model.'.format(modelFile))


    def load_model(self, modelFile):

        state_dict = torch.load(modelFile)

        self.model.load_state_dict(state_dict)


    def predictOne(self, imagePath):

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


    def predictMulti(self,imagePathList):
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
    
    def predict(self, image):
        if isinstance(image, types.GeneratorType):
            image = list(image)

        if isinstance(image, list): 
            pred = self.predictMulti(image)

        elif isinstance(image, (str, pathlib.Path)):
            pred = self.predictOne(image)
        else: 
            raise TypeError("")
        
        return pred


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

    def Get_Images(self, root_dir):
        
        data = torchvision.datasets.ImageFolder(root=root_dir)

        return data


    def loadData(self, imgDir="/home/yunhui/SimCenter/train_BRAILS_models/datasets/roofType", valimgDir='', randomseed=1993, split=[0.8,0.2]):

        train_transforms, val_transforms = self.get_transform()

        if valimgDir == '':

            #print('* First split the data with 8:2.')

            dataset =  self.Get_Images(imgDir)

            newClassNames = list(dataset.class_to_idx.keys())

            train_size = int(len(dataset)*split[0])
            val_size = int(len(dataset)*split[1])

            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])


            train_dataset = MyLazyDataset(train_dataset, train_transforms)
            val_dataset   = MyLazyDataset(val_dataset, val_transforms)

        else:

            train_dataset = Get_Images(imgDir, train_transforms)
            val_dataset   = Get_Images(valimgDir, val_transforms)
        

            newClassNames = list(train_dataset.class_to_idx.keys())
        

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= self.batch_size, shuffle=True, num_workers=4)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = self.batch_size, shuffle=False, num_workers=4)


        self.classNames = newClassNames

        print('The names of the classes are: ', self.classNames)
        
        # change the number of output class
        self.model.reset_classifier(len(self.classNames))


    def train(self, lr=0.01, epochs=10, plot=False):

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


        ## 5. Save model

        # Save the model in the current folder 
        self.save() 

        modelDetails = {
                'modelName' : self.modelName,
                'classNames' : self.classNames
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

        print ("Confusion Matrix")
        print (cnf_matrix)

        if self.printConfusionMatrix:
            plot_confusion_matrix(cnf_matrix, classes=self.class_names, title='Confusion matrix',normalize=True,xlabel='Labels',ylabel='Predictions')

        return 100.0 * correct / len(self.val_loader.dataset), avg_loss / len(self.val_loader)

    def save(self):

        torch.save(self.model.state_dict(), self.modelFile)

        print('Model saved at ', self.modelFile)


def main():
        
    # Please refer to https://fastai.github.io/timmdocs/ for supported models
    # imgDir/valimgDir: directories for training/validation images. Arranged in this way:

    #    root/dog/xxx.png
    #    root/dog/xxy.png
    #    root/dog/[...]/xxz.png

    #    root/cat/123.png
    #    root/cat/nsdf3.png
    #    root/cat/[...]/asd932_.png

    work = ImageClassifier(modelName='resnet18', imgDir="/home/yunhui/SimCenter/train_BRAILS_models/datasets/roofType", valimgDir='', batch_size=64)


    #work.train(lr=0.01, epochs=10)

    #work.predictOne("/home/yunhui/SimCenter/train_BRAILS_models/datasets/roofType/flat/TopViewx-76.84779286x38.81642318.png")

    work.predictMulti(["/home/yunhui/SimCenter/train_BRAILS_models/datasets/roofType/flat/TopViewx-76.84779286x38.81642318.png", "/home/yunhui/SimCenter/train_BRAILS_models/datasets/roofType/flat/TopViewx-76.96240924000001x38.94450328.png"])



if __name__ == '__main__':
    main()




