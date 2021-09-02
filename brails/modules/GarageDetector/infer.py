import os
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
# Author: Barbaros Cetiner

from lib.infer_detector import Infer
import torch
import time
from tqdm import tqdm
import warnings
import argparse
import csv

# Ignore Divide by Zero Warnings for dyOverdx
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser('EfficientDet-based garage detection model')
    parser.add_argument('--im_path', type=str, default="datasets/test",
                        help='Path for the building images')
    parser.add_argument('--model_path', type=str, default="models/efficientdet-d4_trained.pth",
                        help='Path for the pretrained inference model.' 
                             'Do NOT define this argument if the pretrained model bundled with the module will be used')
    parser.add_argument('--gpu_enabled', type=boolean_string, default=True,
                        help='Enable GPU processing (Enter False for CPU-based inference)')    
    parser.add_argument('--csv_out', type=str, default="nFloorPredict.csv",
                        help='Name of the CSV output file where the inference results will be written')

    args = parser.parse_args()
    return args

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean entry')
    return s == 'True'

def install_default_model(model_path):
    if model_path == "models/efficientdet-d4_trained.pth":
        os.makedirs('models',exist_ok=True)
        model_path = os.path.join('models','efficientdet-d4_trained.pth')

        if not os.path.isfile(model_path):
            print('Loading default model file to the models folder...')
            torch.hub.download_url_to_file('https://zenodo.org/record/4421613/files/efficientdet-d4_trained.pth',
                                           model_path, progress=False)
            print('Default model loaded.')
    else:
        print(f'Inferences will be performed using the custom model at {model_path}.')

def infer(opt):
    install_default_model(opt.model_path)
    
    # Start Program Timer
    startTime = time.time()
    
    # Get the Image List
    imgList = os.listdir(opt.im_path)
        
    # Create and Define the Inference Model
    classes = ["garage"]
    
    print("Performing inferences on images...")
    gtfInfer = Infer()
    gtfInfer.load_model(opt.model_path, classes, use_gpu=opt.gpu_enabled)

    rows = []
    for imFile in tqdm(imgList):
        img = cv2.imread(os.path.join(imdir,imFile))
        bldgID = int(imFile.split('.')[0])
        cv2.imwrite("input.jpg", img)
        scores, labels, boxes = gtf.predict("input.jpg", threshold=0.35)
        if len(boxes)>=1:
          rows.append([bldgID,1])
        else:
          rows.append([bldgID,0])
           
        with open('garageOut.csv', 'w') as csvfile: 
            # creating a csv writer object 
            csvwriter = csv.writer(csvfile) 
    
            # writing the data rows 
            csvwriter.writerows(rows)
    
    # End Program Timer and Display Execution Time
    endTime = time.time()
    hours, rem = divmod(endTime-startTime, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nTotal execution time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    
    # Cleanup the Root Folder
    if os.path.isfile("input.jpg"):
        os.remove("input.jpg")
    if os.path.isfile("output.jpg"):
        os.remove("output.jpg")
        
if __name__ == '__main__':
    opt = get_args()
    infer(opt)