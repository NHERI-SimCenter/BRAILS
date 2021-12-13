# Author: Barbaros Cetiner

import os
import cv2
from lib.infer_detector import Infer
import torch
import time
from tqdm import tqdm
import argparse
import csv
import warnings
import shutil

# Ignore warning messages:
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser('EfficientDet-based chimney detection model')
    parser.add_argument('--im_path', type=str, default="datasets/test",
                        help='Path for the building images')
    parser.add_argument('--model_path', type=str, default="models/efficientdet-d4_chimneyDetector.pth",
                        help='Path for the pretrained inference model.' 
                             'Do NOT define this argument if the pretrained model bundled with the module will be used')
    parser.add_argument('--gpu_enabled', type=boolean_string, default=True,
                        help='Enable GPU processing (Enter False for CPU-based inference)')    
    parser.add_argument('--csv_out', type=str, default="chimneyOut.csv",
                        help='Name of the CSV output file where the inference results will be written')

    args = parser.parse_args()
    return args

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean entry')
    return s == 'True'

def install_default_model(model_path):
    if model_path == "models/efficientdet-d4_chimneyDetector.pth":
        os.makedirs('models',exist_ok=True)
        model_path = os.path.join('models','efficientdet-d4_chimneyDetector.pth')

        if not os.path.isfile(model_path):
            print('Loading default model file to the models folder...')
            torch.hub.download_url_to_file('https://zenodo.org/record/5775292/files/efficientdet-d4_chimneyDetector.pth',
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
    classes = ["chimney"]
    
    print("Performing inferences on images...")
    gtfInfer = Infer()
    gtfInfer.load_model(opt.model_path, classes, use_gpu=opt.gpu_enabled)

    rows = []
    count = 1
    for imFile in tqdm(imgList):
        img = cv2.imread(os.path.join(opt.im_path,imFile))
        bldgID = imFile.split('.')[0]
        cv2.imwrite("input.jpg", img)
        scores, labels, boxes = gtfInfer.predict("input.jpg", threshold=0.2)
        shutil.copyfile("output.jpg",f"output{count}.jpg")
        count += 1
        if len(boxes)>=1:
          rows.append([bldgID,1])
        else:
          rows.append([bldgID,0])
           
    with open(opt.csv_out, 'w', newline='', encoding='utf-8') as csvfile: 
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
    #if os.path.isfile("input.jpg"):
    #    os.remove("input.jpg")
    #if os.path.isfile("output.jpg"):
    #    os.remove("output.jpg")
        
if __name__ == '__main__':
    opt = get_args()
    infer(opt)