import argparse

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import csv

from PIL import Image
from torch.utils.data import Dataset
from torch.autograd.variable import Variable
import torch.nn as nn
import sklearn
#from sklearn.metrics.classification import precision_recall_fscore_support

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import seaborn as sns
from .lib.datasets import YearBuiltFolder


sm = nn.Softmax()

class YearBuiltClassifier():

    def __init__(self, checkpoint='', onlycpu=False, workDir='tmp', resultFile='YearBuilt.csv', printRes=True):
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
        self.outFilePath = os.path.join(workDir, resultFile)
        self.printRes = printRes

        self.checkpointsDir = os.path.join(workDir,'checkpoints')
        os.makedirs(self.checkpointsDir,exist_ok=True)
        weight_file_path = os.path.join(self.checkpointsDir,'yearBuiltv0.1.pth')

        if self.checkpoint != '':
            self.modelFile = self.checkpoint
            
        else:
            if not os.path.isfile(weight_file_path):
                print('Loading remote model file to the weights folder..')
                torch.hub.download_url_to_file('https://zenodo.org/record/4310463/files/model_best.pth', weight_file_path)
            self.modelFile = weight_file_path

        if not torch.cuda.is_available():
            self.model = torch.load(self.modelFile, map_location=torch.device('cpu'))
        else:
            self.model = torch.load(self.modelFile) 


        # test_transforms
        self.test_transforms = transforms.Compose([
            transforms.Scale((550, 550)),
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

        for prediction in predictions_data:
            imagePathList.append(str(prediction['filename']))
            predictions.append(prediction['prediction'][0])
            p = prediction['probability']
            probs.append(p)
            if self.printRes: print(f"Image :  {str(prediction['filename'])}     Class : {prediction['prediction'][0]} ({str(round(p*100,2))}%)")

        df = pd.DataFrame(list(zip(imagePathList, predictions, probs)), columns =['image', 'prediction', 'probability']) 
        df.to_csv(self.outFilePath, index=False)
        print(f'Results written in file {self.outFilePath}')

        return df
        
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


                if batch_idx % 50 == 0:
                    print('Testing image {} from {}'.format(batch_idx,len(testloader)))

            '''
            if self.calc_perf:
                y_gt = []
                y_pred = []
                for prediction in predictions:
                    y_gt.append(prediction['ground truth'])
                    y_pred.append(prediction['prediction'])

                y_gt = np.array(y_gt,dtype=np.uint8)
                y_pred = np.array(y_pred,dtype=np.uint8)

                precision, recall, f1, support = precision_recall_fscore_support(y_gt,
                                                                             y_pred, average='macro')

                confusion_matrix = sklearn.metrics.confusion_matrix(y_gt, y_pred, labels=range(len(testloader.dataset.classes)))

                cm_fig = construct_confusion_matrix_image(testloader.dataset.classes, confusion_matrix)

                cm_fig.savefig('result_confusion_matrix.png',dpi=300)

                print('F1 {}, precision, {}, recall, {}'.format(f1,precision,recall))   
            '''

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



if __name__ == '__main__':
    main()
