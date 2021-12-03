import argparse

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import csv

from PIL import Image
from torch.utils.data import Dataset
from torch.autograd.variable import Variable
import sklearn
from sklearn.metrics.classification import precision_recall_fscore_support

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import seaborn as sns
from lib.datasets import YearBuiltFolder

parser = argparse.ArgumentParser(description='Classify Year Built')

parser.add_argument('--image-path',help='Path to one image or a folder containing images.',required=True)
parser.add_argument('--checkpoint', default=None,type=str,
                    help='Path to checkpoint. Defaults to best pretrained version.')
parser.add_argument('--only-cpu', action='store_true', help='Use CPU only, disregard GPU.')

parser.add_argument('--model',help='Pretrained model, options ["foundation_v0.1"]', type=str)
parser.add_argument('--calc-perf', help='Calculate the performance and save a confusion matrix.',action='store_true')


args = parser.parse_args()

if args.only_cpu:
    device='cpu'
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    
    test_transforms = transforms.Compose([
        transforms.Scale((550, 550)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])    
      
    if args.checkpoint is not None:
        model = torch.load(args.checkpoint)
    else:
        weight_file_path = './weights/model_weights.pth'
        if not os.path.isfile(weight_file_path):
            print('Loading remote model file to the weights folder..')
            torch.hub.download_url_to_file('https://zenodo.org/record/4310463/files/model_best.pth', weight_file_path)
            
        model = torch.load(weight_file_path)

    try:
        num_classes = model.classifier1[4].out_features # Take amount of classes from model file
    except AttributeError:
        raise RuntimeError('The model, either provided or downloaded, does not fit the format in this implementation. Please check if the checkpoint is correct and fits to the model.')
    
    dataset = YearBuiltFolder(args.image_path,  transforms=test_transforms, classes=range(num_classes),calc_perf = args.calc_perf)
    # Do not change the batchsize.
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0) 

    model.eval()
    model = model.to(device)
    
    print('Checking {} images'.format(len(dataset)))

    predictions = evaluate_to_stats(model, test_loader)

    with open('{}_prediction_results.csv'.format(os.path.basename(os.path.normpath(args.image_path))), 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        if args.calc_perf:
            wr.writerow(['filename','prediction','ground truth'])
        else:
            wr.writerow(['filename','prediction'])
            
        for prediction in predictions:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            if args.calc_perf:
                wr.writerow([str(prediction['filename']),str(prediction['prediction'][0]),str(prediction['ground truth'][0])])
            else:
                wr.writerow([str(prediction['filename']),str(prediction['prediction'][0])])

    print ('Classification finished. Results written to {}'.format('{}_prediction_results.csv'.format(os.path.basename(os.path.normpath(args.image_path)))))
    
def evaluate_to_stats(net, testloader):
    net.eval()
    
    predictions = []
    
    with torch.no_grad():
        for batch_idx, (inputs, label, filename) in enumerate(testloader):
    
            if inputs.shape[0] > 1:
                raise NotImplementedError('Please choose a batch size of 1. Saving the results is not compatible with larger batch sizes in this version.')        
            inputs = inputs.to(device)
            inputs = Variable(inputs)
            output_1, output_2, output_3, output_concat= net(inputs)
            outputs_com = output_1 + output_2 + output_3 + output_concat
    
            _, predicted_com = torch.max(outputs_com.data, 1)
            
            y_pred = predicted_com[0].flatten().cpu().numpy()
            
            y_fn = filename[0]
            
            if args.calc_perf:
                predictions.append({'filename':y_fn,'prediction':y_pred, 'ground truth':label.cpu().numpy()})
            else:
                predictions.append({'filename':y_fn,'prediction':y_pred})
                
               
            if batch_idx % 50 == 0:
                print('Testing image {} from {}'.format(batch_idx,len(testloader)))
                
        if args.calc_perf:
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