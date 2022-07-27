# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 The Regents of the University of California
#
# This file is part of BRAILS.
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
# You should have received a copy of the BSD 3-Clause License along with
# BRAILS. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Barbaros Cetiner


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
import sys
import requests
import zipfile

class ImageClassifier:

    def __init__(self, modelArch="efficientnetv2s"): 
    
        self.modelArch = modelArch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        self.batchSize = None
        self.nepochs = None
        self.trainDataDir = None
        self.testDataDir = None
        self.classes = None
        self.lossHistory = None
        self.preds = None
        
        if "resnet" in modelArch:
            input_size = 224
    
        elif "efficientnetv2" in modelArch:
            if 's' in modelArch:
                input_size = 384
            elif 'm' in modelArch:
                input_size = 480
            elif 'l' in modelArch:
                input_size = 480
            else:
                sys.exit("Model name or architecture not defined!")
            
        elif "convnext" in modelArch:
            if 's' in modelArch:
                input_size = 384
            elif 'b' in modelArch:
                input_size = 480
            elif 'l' in modelArch:
                input_size = 480
            else:
                sys.exit("Model name or architecture not defined!")
            
        elif "regnet" in modelArch:
            if '16' in modelArch:
                input_size = 384
            elif '32' in modelArch:
                input_size = 384
            else:
                sys.exit("Model name or architecture not defined!")

    
        elif "vit" in modelArch:
            if '14' in modelArch:
                input_size = 518
            elif '16' in modelArch:
                input_size = 512
            else:
                sys.exit("Model name or architecture not defined!")                

        else:
            sys.exit("Model name or architecture not defined!")
          
        self.modelInputSize = input_size          

    def train(self, trainDataDir='tmp/hymenoptera_data', batchSize=8, nepochs=100, plotLoss=True):
        
        if trainDataDir=='tmp/hymenoptera_data':
            print('Downloading default dataset...')
            url = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'
            req = requests.get(url)
            zipdir = os.path.join('tmp',url.split('/')[-1])
            os.makedirs('tmp',exist_ok=True)
            with open(zipdir,'wb') as output_file:
                output_file.write(req.content)
            print('Download complete.')
            with zipfile.ZipFile(zipdir, 'r') as zip_ref:
                zip_ref.extractall('tmp')
        
        def train_model(model, dataloaders, criterion, optimizer, num_epochs=100, es_tolerance=10):
            since = time.time()
        
            val_acc_history = []
            
            best_model_wts = copy.deepcopy(model.state_dict())
            best_acc = 0.0
            es_counter = 0
        
            for epoch in range(num_epochs):
                print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                print('-' * 10)
        
                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:
                    if phase == 'train':
                        model.train()  # Set model to training mode
                    else:
                        model.eval()   # Set model to evaluate mode
        
                    running_loss = 0.0
                    running_corrects = 0
        
                    # Iterate over data.
                    for inputs, labels in dataloaders[phase]:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
        
                        # zero the parameter gradients
                        optimizer.zero_grad()
        
                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            # Get model outputs and calculate loss
                            # Special case for inception because in training it has an auxiliary output. In train
                            #   mode we calculate the loss by summing the final output and the auxiliary output
                            #   but in testing we only consider the final output.
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
        
                            _, preds = torch.max(outputs, 1)
        
                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()
        
                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
        
                    epoch_loss = running_loss / len(dataloaders[phase].dataset)
                    epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
        
                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase.capitalize(), epoch_loss, epoch_acc))
        
                    # deep copy the model
                    if phase == 'val' and epoch_acc > best_acc:
                        es_counter = 0
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                    if phase == 'val':
                        es_counter += 1
                        val_acc_history.append(epoch_acc)
                if es_counter>=es_tolerance:
                  print('Early termination criterion satisfied.')
                  break
                print()
        
            time_elapsed = time.time() - since
            print('Best val Acc: {:4f}'.format(best_acc))
            print('Elapsed time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            # load best model weights
            model.load_state_dict(best_model_wts)
            return model, val_acc_history
        
        def set_parameter_requires_grad(model, feature_extracting):
            if feature_extracting:
                for param in model.parameters():
                    param.requires_grad = False
            else:
                for param in model.parameters():
                    param.requires_grad = True

        def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
            # Initialize these variables which will be set in this if statement. Each of these
            #   variables is model specific.
            model_ft = None
        
            modelname = ''.join(filter(str.isalnum, model_name.lower()))
        
            if "resnet" in modelname:
                model_ft = models.resnet18(pretrained=use_pretrained)
                set_parameter_requires_grad(model_ft, feature_extract)
                num_ftrs = model_ft.fc.in_features
                model_ft.fc = nn.Linear(num_ftrs, num_classes)
        
            elif "efficientnetv2" in modelname:
                if 's' in modelname:
                    if use_pretrained:
                        model_ft = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
                    else:
                        model_ft = models.efficientnet_v2_s()

                elif 'm' in modelname:
                    if use_pretrained:
                        model_ft = models.efficientnet_v2_m(weights='IMAGENET1K_V1')
                    else:
                        model_ft = models.efficientnet_v2_m()

                elif 'l' in modelname:
                    if use_pretrained:
                        model_ft = models.efficientnet_v2_l(weights='IMAGENET1K_V1')
                    else:
                        model_ft = models.efficientnet_v2_l()
        
                set_parameter_requires_grad(model_ft, feature_extract)
                num_ftrs = model_ft.classifier[-1].in_features
                model_ft.classifier[-1] = nn.Linear(num_ftrs,num_classes)
        
            elif "convnext" in modelname:
                if 's' in modelname:
                    if use_pretrained:
                        model_ft = models.convnext_small(weights='IMAGENET1K_V1')
                    else:
                        model_ft = models.convnext_small()

                elif 'b' in modelname:
                    if use_pretrained:
                        model_ft = models.convnext_base(weights='IMAGENET1K_V1')
                    else:
                        model_ft = models.convnext_base()

                elif 'l' in modelname:
                    if use_pretrained:
                        model_ft = models.convnext_large(weights='IMAGENET1K_V1')
                    else:
                        model_ft = models.convnext_large()

        
                set_parameter_requires_grad(model_ft, feature_extract)
                num_ftrs = model_ft.classifier[-1].in_features
                model_ft.classifier[-1] = nn.Linear(num_ftrs,num_classes)        
        
            elif "regnet" in modelname:
                if '16' in modelname:
                    if use_pretrained:
                        model_ft = models.regnet_y_16gf(weights='IMAGENET1K_SWAG_E2E_V1')
                    else:
                        model_ft = models.regnet_y_16gf()

                elif '32' in modelname:
                    if use_pretrained:
                        model_ft = models.regnet_y_32gf(weights='IMAGENET1K_SWAG_E2E_V1')
                    else:
                        model_ft = models.regnet_y_32gf()

        
            elif "vit" in modelname:
                if '14' in modelname:
                    if use_pretrained:
                        model_ft = models.vit_h_14(weights='IMAGENET1K_SWAG_E2E_V1')
                    else:
                        model_ft = models.vit_h_14()

                elif '16' in modelname:
                    if use_pretrained:
                        model_ft = models.vit_l_16(weights='IMAGENET1K_SWAG_E2E_V1')
                    else:
                        model_ft = models.vit_l_16()
                    
        
            else:
                sys.exit("Model name or architecture not defined!")
            
            return model_ft              
        
        self.batchSize = batchSize
        self.trainDataDir = trainDataDir

        classes = os.listdir(os.path.join(self.trainDataDir,'train'))        
        self.classes = sorted(classes)
        num_classes = len(self.classes)
        
        if isinstance(nepochs, int):
            nepochs_it = round(nepochs/2)
            nepochs_ft = nepochs - nepochs_it
        elif isinstance(nepochs, list) and len(nepochs)>=2:
            nepochs_it = nepochs[0]
            nepochs_ft = nepochs[1]
        else:
            sys.exit('Incorrect nepochs entry. Number of epochs should be defined as an integer or a list of two integers!')
            
        self.nepochs = [nepochs_it,nepochs_ft]
        
        # Initialize the model for this run
        model_ft = initialize_model(self.modelArch, num_classes, feature_extract=False, use_pretrained=True)
        
        # Data augmentation and normalization for training
        # Just normalization for validation
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(self.modelInputSize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(self.modelInputSize),
                transforms.CenterCrop(self.modelInputSize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        # Create training and validation datasets
        image_datasets = {x: datasets.ImageFolder(os.path.join(self.trainDataDir, x), data_transforms[x]) for x in ['train', 'val']}
        # Create training and validation dataloaders
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchSize, shuffle=True, num_workers=4) for x in ['train', 'val']}
        
        # Send the model to GPU
        model_ft = model_ft.to(self.device)
        
        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are 
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        params_to_update = model_ft.parameters()
        
        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)  
        
        # Setup the loss fxn
        criterion = nn.CrossEntropyLoss()
        
        # Train and evaluate
        model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=nepochs_it)
        print('New classifier head trained using transfer learning.')  
        
        # Initialize the non-pretrained version of the model used for this run
        print('\nFine-tuning the model...')
        set_parameter_requires_grad(model_ft,feature_extracting=False)
        final_model = model_ft.to(self.device)
        final_optimizer = optim.SGD(final_model.parameters(), lr=0.0001, momentum=0.9)
        final_criterion = nn.CrossEntropyLoss()
        _,final_hist = train_model(final_model, dataloaders_dict, final_criterion, final_optimizer, num_epochs=nepochs_ft)
        print('Training complete.')
        os.makedirs('tmp/models', exist_ok=True)
        torch.save(final_model, 'tmp/models/trained_model.pth')
        self.modelPath = 'tmp/models/trained_model.pth'
        
        
        # Plot the training curves of validation accuracy vs. number 
        #  of training epochs for the transfer learning method and
        #  the model trained from scratch
      
        plothist = [h.cpu().numpy() for h in hist] + [h.cpu().numpy() for h in final_hist]
        self.lossHistory = plothist
        if plotLoss:      
            plt.title("Validation Accuracy vs. Number of Training Epochs")
            plt.xlabel("Training Epochs")
            plt.ylabel("Validation Accuracy")
            plt.plot(range(1,len(plothist)+1),plothist)
            plt.ylim((0.4,1.))
            plt.xticks(np.arange(1, len(plothist)+1, 1.0))
            plt.show()

    def retrain(self, modelPath='tmp/models/trained_model.pth', 
                trainDataDir='tmp/hymenoptera_data', batchSize=8, 
                nepochs=100, plotLoss=True):
        
        if trainDataDir=='tmp/hymenoptera_data':
            print('Downloading default dataset...')
            url = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'
            req = requests.get(url)
            zipdir = os.path.join('tmp',url.split('/')[-1])
            os.makedirs('tmp',exist_ok=True)
            with open(zipdir,'wb') as output_file:
                output_file.write(req.content)
            print('Download complete.')
            with zipfile.ZipFile(zipdir, 'r') as zip_ref:
                zip_ref.extractall('tmp')
        
        def train_model(model, dataloaders, criterion, optimizer, num_epochs=100, es_tolerance=10):
            since = time.time()
        
            val_acc_history = []
            
            best_model_wts = copy.deepcopy(model.state_dict())
            best_acc = 0.0
            es_counter = 0
        
            for epoch in range(num_epochs):
                print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                print('-' * 10)
        
                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:
                    if phase == 'train':
                        model.train()  # Set model to training mode
                    else:
                        model.eval()   # Set model to evaluate mode
        
                    running_loss = 0.0
                    running_corrects = 0
        
                    # Iterate over data.
                    for inputs, labels in dataloaders[phase]:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
        
                        # zero the parameter gradients
                        optimizer.zero_grad()
        
                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            # Get model outputs and calculate loss
                            # Special case for inception because in training it has an auxiliary output. In train
                            #   mode we calculate the loss by summing the final output and the auxiliary output
                            #   but in testing we only consider the final output.
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
        
                            _, preds = torch.max(outputs, 1)
        
                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()
        
                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
        
                    epoch_loss = running_loss / len(dataloaders[phase].dataset)
                    epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
        
                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase.capitalize(), epoch_loss, epoch_acc))
        
                    # deep copy the model
                    if phase == 'val' and epoch_acc > best_acc:
                        es_counter = 0
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                    if phase == 'val':
                        es_counter += 1
                        val_acc_history.append(epoch_acc)
                if es_counter>=es_tolerance:
                  print('Early termination criterion satisfied.')
                  break
                print()
        
            time_elapsed = time.time() - since
            print('Best val Acc: {:4f}'.format(best_acc))
            print('Elapsed time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            # load best model weights
            model.load_state_dict(best_model_wts)
            return model, val_acc_history
        
        def set_parameter_requires_grad(model, feature_extracting):
            if feature_extracting:
                for param in model.parameters():
                    param.requires_grad = False
            else:
                for param in model.parameters():
                    param.requires_grad = True
        
        # Data augmentation and normalization for training
        # Just normalization for validation
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(self.modelInputSize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(self.modelInputSize),
                transforms.CenterCrop(self.modelInputSize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        self.batchSize = batchSize
        self.trainDataDir = trainDataDir

        classes = os.listdir(os.path.join(self.trainDataDir,'train'))        
        self.classes = sorted(classes)
        
        if isinstance(nepochs, int):
            self.nepochs = [0, nepochs]
        else:
            sys.exit('Incorrect nepochs entry. For retraining, number of epochs should be defined as an integer')

        
        # Create training and validation datasets
        image_datasets = {x: datasets.ImageFolder(os.path.join(self.trainDataDir, x), data_transforms[x]) for x in ['train', 'val']}
        # Create training and validation dataloaders
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchSize, shuffle=True, num_workers=4) for x in ['train', 'val']}
        
        # Send the model to GPU
        model = torch.load(modelPath)    
        set_parameter_requires_grad(model,feature_extracting=False)
        final_model = model.to(self.device)
        final_optimizer = optim.SGD(final_model.parameters(), lr=0.0001, momentum=0.9)
        final_criterion = nn.CrossEntropyLoss()
        _,final_hist = train_model(final_model, dataloaders_dict, final_criterion, final_optimizer, num_epochs=nepochs)
        print('Training complete.')
        os.makedirs('tmp/models', exist_ok=True)
        torch.save(final_model, 'tmp/models/retrained_model.pth')
        self.modelPath = 'tmp/models/retrained_model.pth'
        
        
        # Plot the training curves of validation accuracy vs. number 
        #  of training epochs for the transfer learning method and
        #  the model trained from scratch
        plothist = [h.cpu().numpy() for h in final_hist]
        self.lossHistory = plothist
        
        if plotLoss:        
            plt.plot(range(1,len(plothist)+1),plothist)
            plt.title("Validation Accuracy vs. Number of Training Epochs")
            plt.xlabel("Training Epochs")
            plt.ylabel("Validation Accuracy")
            #plt.ylim((0.4,1.))
            plt.xticks(np.arange(1, len(plothist)+1, 1.0))
            plt.show()    

    def predict(self, modelPath='tmp/models/trained_model.pth', 
                testDataDir='tmp/hymenoptera_data/val/ants',
                classes=['Ants','Bees']):
        
        self.modelPath = modelPath
        self.testDataDir = testDataDir
        self.classes = sorted(classes)
        
        loader = transforms.Compose([
                transforms.Resize(self.modelInputSize),
                transforms.CenterCrop(self.modelInputSize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        def image_loader(image_name):
            image = Image.open(image_name).convert("RGB")
            image = loader(image).float()
            image = image.unsqueeze(0)  
            return image.to(self.device) 
        
        model = torch.load(modelPath)
        model.eval()
        
        preds = []
        if os.path.isdir(testDataDir):
            for im in os.listdir(testDataDir):
                if ('jpg' in im) or ('jpeg' in im) or ('png' in im):
                    image = image_loader(os.path.join(testDataDir,im))
                    _, pred = torch.max(model(image),1)
                    preds.append((im, classes[pred]))   
            self.preds = preds
        elif os.path.isfile(testDataDir):
            img = plt.imread(testDataDir)[:,:,:3]
            image = image_loader(testDataDir)
            _, pred = torch.max(model(image),1)
            pred = classes[pred]
            if pred.islower():
                pred = pred.capitalize()
            plt.imshow(img)
            plt.title((f"Predicted class: {pred}"))
            plt.show()
            print((f"Predicted class: {pred}"))
            self.preds = pred