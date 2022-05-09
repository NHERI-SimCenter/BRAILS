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
# Sascha Hornauer
# Barbaros Cetiner
#
# Last updated:
# 05-08-2022   

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

from torch.autograd.variable import Variable
import torch.nn as nn
from tqdm import tqdm

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import seaborn as sns
from .lib.datasets import YearBuiltFolder


sm = nn.Softmax()

class YearBuiltClassifier():

    def __init__(self, checkpoint='', onlycpu=False, workDir='tmp', printRes=False):
        '''
        checkpoint (str): Path to checkpoint. Defaults to best pretrained version.
        onlycpu (bool): Use CPU only, use GPU by default.
        '''

        if onlycpu:
            self.device='cpu'
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
        self.checkpoint = checkpoint
        self.onlycpu = onlycpu
        self.workDir = workDir
        self.printRes = printRes

        self.checkpointsDir = os.path.join(workDir,'models')
        os.makedirs(self.checkpointsDir,exist_ok=True)
        weight_file_path = os.path.join(self.checkpointsDir,'yearBuiltv0.1.pth')

        print('\nDetermining the era of construction for each building...')
        if self.checkpoint != '':
            self.modelFile = self.checkpoint
            
        else:
            if not os.path.isfile(weight_file_path):
                print(f'Loading default era of construction classifier model file to {self.checkpointsDir} folder...')
                torch.hub.download_url_to_file('https://zenodo.org/record/4310463/files/model_best.pth',
                                               weight_file_path, progress=False)
                print('\nDefault era of construction classifier model loaded')
            else:
                print(f"Default era of construction classifier model at {self.checkpointsDir} loaded")
            self.modelFile = weight_file_path
            
        if not torch.cuda.is_available():
            self.model = torch.load(self.modelFile, map_location=torch.device('cpu'))
        else:
            self.model = torch.load(self.modelFile) 


        # test_transforms
        self.test_transforms = transforms.Compose([
            transforms.Resize((550, 550)),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])   

        try:
            self.num_classes = self.model.classifier1[4].out_features # Take amount of classes from model file
        except AttributeError:
            raise RuntimeError('The model, either provided or downloaded, does not fit the format in this implementation. Please check if the checkpoint is correct and fits to the model.')

        
        self.model.eval()
        self.model = self.model.to(self.device)

    def predict(self,image=''):
        '''
        image (str): Path to one image or a folder containing images.
        '''

        dataset = YearBuiltFolder(image,  transforms=self.test_transforms, classes=range(self.num_classes),calc_perf = False)
        # Do not change the batchsize.
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0) 

        predictions = []
        probs = []
        imagePathList = []

        predictions_data = self.evaluate_to_stats(test_loader)
        
        print("Performing construction era classifications...")
        for prediction in tqdm(predictions_data):
            imagePathList.append(str(prediction['filename']).replace('\\','/'))
            predictions.append(prediction['prediction'][0])
            p = prediction['probability']
            probs.append(p)
            if self.printRes: print(f"Image :  {str(prediction['filename'])}     Class : {prediction['prediction'][0]} ({str(round(p*100,2))}%)")
        print('\n')
        pred_clear = []
        for prediction in predictions:
            if prediction==0:
                pred_str = 'Pre-1970'
            elif prediction==1:
                pred_str = '1970-1979'
            elif prediction==2:
                pred_str = '1980-1989'
            elif prediction==3:
                pred_str = '1990-1999'
            elif prediction==4:
                pred_str = '2000-2009'
            else:
                pred_str = 'Post-2010'
            pred_clear.append(pred_str)
            
        df = pd.DataFrame(list(zip(imagePathList, pred_clear, probs)), columns =['image', 'prediction', 'probability']) 
        self.results_df = df.copy()
        
    def evaluate_to_stats(self, testloader):
        self.model.eval()

        self.calc_perf = False

        predictions = []

        

        with torch.no_grad():
            for batch_idx, (inputs, label, filename) in enumerate(testloader):
            
                if inputs.shape[0] > 1:
                    raise NotImplementedError('Please choose a batch size of 1. Saving the results is not compatible with larger batch sizes in this version.')        
                inputs = inputs.to(self.device)
                inputs = Variable(inputs)
                output_1, output_2, output_3, output_concat= self.model(inputs)
                outputs_com = output_1 + output_2 + output_3 + output_concat

                
                output = sm(outputs_com.data)
                _, predicted_com = torch.max(output, 1)

                y_pred = predicted_com[0].flatten().cpu().numpy()

                p = output[0][y_pred][0].item()

                y_fn = filename[0]

                if self.calc_perf:
                    predictions.append({'filename':y_fn,'prediction':y_pred,'probability':p, 'ground truth':label.cpu().numpy()})
                else:
                    predictions.append({'filename':y_fn,'prediction':y_pred,'probability':p})

        return predictions

    def construct_confusion_matrix_image(classes, con_mat):
        con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

        con_mat_df = pd.DataFrame(con_mat_norm,
                                  index=classes,
                                  columns=classes)

        figure = plt.figure(figsize=(8, 8))
        sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        figure.canvas.draw()

        return figure