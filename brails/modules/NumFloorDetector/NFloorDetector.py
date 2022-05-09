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
#
# Last updated:
# 05-08-2022

from .lib.train_detector import Detector
import os
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from lib.infer_detector import Infer
import torch
import time
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
if torch.cuda.is_available():
    useGPU=True
else:    
    useGPU=False
    
class NFloorDetector():
    def __init__(self):
        self.system_dict = {}
        self.system_dict["train"] = {}
        self.system_dict["train"]["data"] = {}
        self.system_dict["train"]["model"] = {}        
        self.system_dict["infer"] = {}
        self.system_dict["infer"]["params"] = {}

        self.set_fixed_params()
    
    def set_fixed_params(self):        
        self.system_dict["train"]["data"]["trainSet"] = "train"
        self.system_dict["train"]["data"]["validSet"] = "valid"
        self.system_dict["train"]["data"]["classes"] = ["floor"]
        self.system_dict["train"]["model"]["valInterval"] = 1
        self.system_dict["train"]["model"]["saveInterval"] = 5
        self.system_dict["train"]["model"]["esMinDelta"] = 0.0
        self.system_dict["train"]["model"]["esPatience"] = 0
        
    def load_train_data(self, rootDir="datasets/", nWorkers=0, batchSize=2):
        self.system_dict["train"]["data"]["rootDir"] = rootDir
        self.system_dict["train"]["data"]["nWorkers"] = nWorkers
        self.system_dict["train"]["data"]["batchSize"] = batchSize        
        

    def train(self, compCoeff=3, topOnly=False, optim="adamw", lr=1e-4, numEpochs=25, nGPU=1):
        self.system_dict["train"]["model"]["compCoeff"] = compCoeff
        self.system_dict["train"]["model"]["topOnly"] = topOnly
        self.system_dict["train"]["model"]["optim"] = optim          
        self.system_dict["train"]["model"]["lr"] = lr
        self.system_dict["train"]["model"]["numEpochs"] = numEpochs
        self.system_dict["train"]["model"]["nGPU"] = nGPU        
        
        # Create the Object Detector Object
        gtf = Detector()

        gtf.set_train_dataset(self.system_dict["train"]["data"]["rootDir"],
                              "",
                              "",
                              self.system_dict["train"]["data"]["trainSet"],
                              classes_list=self.system_dict["train"]["data"]["classes"],
                              batch_size=self.system_dict["train"]["data"]["batchSize"],
                              num_workers=self.system_dict["train"]["data"]["nWorkers"])        

        gtf.set_val_dataset(self.system_dict["train"]["data"]["rootDir"],
                            "",
                            "",
                            self.system_dict["train"]["data"]["validSet"])
        
        # Define the Model Architecture
        coeff = self.system_dict["train"]["model"]["compCoeff"]
        modelArchitecture = f"efficientdet-d{coeff}.pth"
        
        gtf.set_model(model_name=modelArchitecture,
                      num_gpus=self.system_dict["train"]["model"]["nGPU"],
                      freeze_head=self.system_dict["train"]["model"]["topOnly"])
        
        # Set Model Hyperparameters    
        gtf.set_hyperparams(optimizer=self.system_dict["train"]["model"]["optim"], 
                            lr=self.system_dict["train"]["model"]["lr"],
                            es_min_delta=self.system_dict["train"]["model"]["esMinDelta"], 
                            es_patience=self.system_dict["train"]["model"]["esPatience"])
        
        # Train    
        gtf.train(num_epochs=self.system_dict["train"]["model"]["numEpochs"],
                  val_interval=self.system_dict["train"]["model"]["valInterval"],
                  save_interval=self.system_dict["train"]["model"]["saveInterval"])
        
    def retrain(self, optim="adamw", lr=1e-4, numEpochs=25, nGPU=1):
        self.system_dict["train"]["model"]["compCoeff"] = 4
        self.system_dict["train"]["model"]["topOnly"] = False
        self.system_dict["train"]["model"]["optim"] = optim          
        self.system_dict["train"]["model"]["lr"] = lr
        self.system_dict["train"]["model"]["numEpochs"] = numEpochs
        self.system_dict["train"]["model"]["nGPU"] = nGPU        
        
        # Create the Object Detector Object
        gtf = Detector()

        gtf.set_train_dataset(self.system_dict["train"]["data"]["rootDir"],
                              "",
                              "",
                              self.system_dict["train"]["data"]["trainSet"],
                              classes_list=self.system_dict["train"]["data"]["classes"],
                              batch_size=self.system_dict["train"]["data"]["batchSize"],
                              num_workers=self.system_dict["train"]["data"]["nWorkers"])        

        gtf.set_val_dataset(self.system_dict["train"]["data"]["rootDir"],
                            "",
                            "",
                            self.system_dict["train"]["data"]["validSet"])
        
        # Define the Model Architecture
        coeff = self.system_dict["train"]["model"]["compCoeff"]
        
        model_path = os.path.join('pretrained_weights',f"efficientdet-d{coeff}.pth")
        
        os.makedirs('pretrained_weights',exist_ok=True)
        if not os.path.isfile(model_path):
            print('Loading default floor detector model file to the pretrained folder...')
            torch.hub.download_url_to_file('https://zenodo.org/record/4421613/files/efficientdet-d4_trained.pth',
                                           model_path, progress=False)
            
        gtf.set_model(model_name=f"efficientdet-d{coeff}.pth",
                      num_gpus=self.system_dict["train"]["model"]["nGPU"],
                      freeze_head=self.system_dict["train"]["model"]["topOnly"])
        
        # Set Model Hyperparameters    
        gtf.set_hyperparams(optimizer=self.system_dict["train"]["model"]["optim"], 
                            lr=self.system_dict["train"]["model"]["lr"],
                            es_min_delta=self.system_dict["train"]["model"]["esMinDelta"], 
                            es_patience=self.system_dict["train"]["model"]["esPatience"])
        
        # Train    
        gtf.train(num_epochs=self.system_dict["train"]["model"]["numEpochs"],
                  val_interval=self.system_dict["train"]["model"]["valInterval"],
                  save_interval=self.system_dict["train"]["model"]["saveInterval"])        

    def predict(self, images, 
                modelPath='tmp/models/efficientdet-d4_nfloorDetector.pth',
                gpuEnabled=useGPU):
        self.system_dict["infer"]["images"] = images
        self.system_dict["infer"]["modelPath"] = modelPath
        self.system_dict["infer"]["gpuEnabled"] = gpuEnabled
        self.system_dict["infer"]['predictions'] = []
        
        print('\nDetermining the number of floors for each building...')
        
        def install_default_model(model_path):
            if model_path == 'tmp/models/efficientdet-d4_nfloorDetector.pth':
                os.makedirs('tmp/models',exist_ok=True)
        
                if not os.path.isfile(model_path):
                    print('Loading default floor detector model file to tmp/models folder...')
                    torch.hub.download_url_to_file('https://zenodo.org/record/4421613/files/efficientdet-d4_trained.pth',
                                                   model_path, progress=False)
                    print('Default floor detector model loaded')
                else: 
                    print(f"Default floor detector model at {model_path} loaded")
            else:
                print(f'Inferences will be performed using the custom model at {model_path}')
        
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
        
        install_default_model(self.system_dict["infer"]["modelPath"])
        
        # Start Program Timer
        startTime = time.time()
        
        # Get the Image List
        try: 
            imgList = os.listdir(self.system_dict["infer"]["images"])
            for imgno in range(len(imgList)):
                imgList[imgno] = os.path.join(self.system_dict["infer"]["images"],imgList[imgno])
        except:
            imgList = self.system_dict["infer"]["images"]
        
        nImages = len(imgList)
            
        # Create and Define the Inference Model
        classes = ["floor"]
        
        print("Performing floor detections...")
        gtfInfer = Infer()
        gtfInfer.load_model(self.system_dict["infer"]["modelPath"], classes, use_gpu=self.system_dict["infer"]["gpuEnabled"])
        predictions = []
        for imgno in tqdm(range(nImages)):
            # Perform Iterative Inference
            imgPath = str(imgList[imgno])
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
                         
            predictions.append(nFloors)
        
        self.system_dict["infer"]['predictions'] = predictions
        self.system_dict["infer"]["images"] = imgList
        
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