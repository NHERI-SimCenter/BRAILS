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

# Ignore Divide by Zero Warnings for dyOverdx
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser('EfficientDet-based number of floor detection model')
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

def create_polygon(bb):
    polygon = Polygon([(bb[0], bb[1]), (bb[0], bb[3]),
                       (bb[2], bb[3]), (bb[2], bb[1])])
    return polygon

def intersect_polygons(poly1, poly2):
    if poly1.intersects(poly2):
        polyArea = poly1.intersection(poly2).area
        if poly1.area!=0 and poly2.area!=0:
            overlapRatio = polyArea/poly1.area*100
        else: 
            overlapRatio = 0
    else:
        overlapRatio = 0
    return overlapRatio

def check_threshold_level(boxesPoly):
    thresholdChange = False
    thresholdIncrease = False
    if not boxesPoly:
        thresholdChange = True
        thresholdIncrease = False
    else:
        falseDetect = np.zeros(len(boxesPoly))
        for k in range(len(boxesPoly)):
            #for m in range(len(boxesPoly)):
            #    overlapRatio = np.array([intersect_polygons(boxesPoly[m], boxesPoly[k])],dtype=float)
            #    print(f"{m}, {k}")
            overlapRatio = np.array([intersect_polygons(p, boxesPoly[k]) for p in boxesPoly],dtype=float)
            falseDetect[k] = len([idx for idx, val in enumerate(overlapRatio) if val > 75][1:])
        thresholdChange = any(falseDetect>2)
        if thresholdChange: thresholdIncrease = True
    return thresholdChange, thresholdIncrease

def compute_derivative(centBoxes):
    nBoxes = centBoxes.shape[0] 
    dyOverdx = np.zeros((nBoxes,nBoxes)) + 10
    for k in range(nBoxes):
        for m in range(nBoxes):
            dx = abs(centBoxes[k,0]-centBoxes[m,0])
            dy = abs(centBoxes[k,1]-centBoxes[m,1])
            if k!=m:
                dyOverdx[k,m] = dy/dx
    return dyOverdx

def infer(opt):
    install_default_model(opt.model_path)
    
    # Start Program Timer
    startTime = time.time()
    
    # Get the Image List
    imgList = os.listdir(opt.im_path)
    nImages = len(imgList)
    
    # Initiate a CSV File to Write Program Output If More Than One Image in imageDir
    if nImages!=1:
        csvFile = open(opt.csv_out,'w+')
        csvFile.write("Image, nFloors\n")
        
    # Create and Define the Inference Model
    classes = ["floor"]
    
    print("Performing inferences on images...")
    gtfInfer = Infer()
    gtfInfer.load_model(opt.model_path, classes, use_gpu=opt.gpu_enabled)
    for imgno in tqdm(range(nImages)):
        # Perform Iterative Inference
        imgPath = os.path.join(opt.im_path, imgList[imgno])
        img = cv2.imread(imgPath)
        img = cv2.resize(img,(640,640))
        cv2.imwrite("input.jpg",img)
        _, _, boxes = gtfInfer.predict("input.jpg",threshold=0.2)
        boxesPoly = [create_polygon(bb) for bb in boxes]
        
        multiplier = 1
        while check_threshold_level(boxesPoly)[0]:
            if check_threshold_level(boxesPoly)[1]:
                confThreshold = 0.2 + multiplier*0.1
                if confThreshold>1: break
            else:
                confThreshold = 0.2 - multiplier*0.02
                if confThreshold==0: break
            _, _, boxes = gtfInfer.predict("input.jpg",threshold=confThreshold)                    
            multiplier +=1        
            boxesPoly = [create_polygon(bb) for bb in boxes]
        
        # Postprocessing    
        boxesPoly = [create_polygon(bb) for bb in boxes]
        
        nestedBoxes = np.zeros((10*len(boxes)),dtype=int)
        counter = 0
        for bbPoly in boxesPoly:    
            overlapRatio = np.array([intersect_polygons(p, bbPoly) for p in boxesPoly],dtype=float)
            ind = [idx for idx, val in enumerate(overlapRatio) if val > 75][1:]
            nestedBoxes[counter:counter+len(ind)] = ind
            counter += len(ind)
        nestedBoxes  = np.unique(nestedBoxes[:counter])
        
        counter = 0
        for x in nestedBoxes:
            del boxes[x-counter] 
            counter += 1
        
        nBoxes = len(boxes)
        
        boxesPoly = []
        boxesExtendedPoly = []
        centBoxes = np.zeros((nBoxes,2))
        for k in range(nBoxes):    
            bb = boxes[k]  
            tempPoly = create_polygon(bb)
            boxesPoly.append(tempPoly)
            x,y = tempPoly.centroid.xy
            centBoxes[k,:] = np.array([x[0],y[0]])
            boxesExtendedPoly.append(create_polygon([0.9*bb[0],0,1.1*bb[2],len(img)-1]))
        
        stackedInd = []
        for bb in boxesExtendedPoly:    
            overlapRatio = np.array([intersect_polygons(p, bb) for p in boxesExtendedPoly],dtype=float)
            stackedInd.append([idx for idx, val in enumerate(overlapRatio) if val > 10])
            
        uniqueStacks0 = [list(x) for x in set(tuple(x) for x in stackedInd)]
        
        dyOverdx = compute_derivative(centBoxes)
        stacks = np.where(dyOverdx>1.3)
        
        counter = 0
        uniqueStacks = [[] for i in range(nBoxes)]
        for k in range(nBoxes):
            while counter<len(stacks[0]) and k==stacks[0][counter]:
                uniqueStacks[k].append(stacks[1][counter])
                counter +=1
        
        uniqueStacks = [list(x) for x in set(tuple(x) for x in uniqueStacks)]
        
        if len(uniqueStacks0)==1 or len(uniqueStacks)==1:
            nFloors = len(uniqueStacks0[0])
        else:
            lBound = len(img)/5
            uBound = 4*len(img)/5
            middlePoly = Polygon([(lBound,0),(lBound,len(img)),(uBound,len(img)),(uBound,0)])
            overlapRatio = np.empty(len(uniqueStacks))
            for k in range(len(uniqueStacks)):
                poly = unary_union([boxesPoly[x] for x in uniqueStacks[k]])
                overlapRatio[k] = (intersect_polygons(poly, middlePoly))
        
            
            indKeep = np.argsort(-overlapRatio)[0:2]
            stack4Address = []
            for k in range(2):
                if overlapRatio[indKeep[k]]>10: stack4Address.append(indKeep[k])
            if len(stack4Address)!=0:
                nFloors = max([len(uniqueStacks[x]) for x in stack4Address])
            else:
                nFloors = len(uniqueStacks[0])
                
           
        if nImages!=1: 
                csvFile.write(os.path.splitext(imgList[imgno])[0] + f", {nFloors}\n")
        else:
            print(f"Number of floors:{nFloors}\n")        
    
    csvFile.close()
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