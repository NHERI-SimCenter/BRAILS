# Author: Barbaros Cetiner

from .lib.train_detector import Detector
import os
import csv
import cv2
from lib.infer_detector import Infer
import torch
import time
from tqdm import tqdm
import pandas as pd
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
        self.system_dict["train"]["data"]["classes"] = ["garage"]
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

    def predict(self, images, modelPath="models/efficientdet-d4_trained.pth", gpuEnabled=useGPU, outFile="garageOut.csv"):
        self.system_dict["infer"]["images"] = images
        self.system_dict["infer"]["modelPath"] = modelPath
        self.system_dict["infer"]["gpuEnabled"] = gpuEnabled
        self.system_dict["infer"]["outFile"] = outFile
        
        def install_default_model(model_path):
            if model_path == "models/efficientdet-d4_trained.pth":
                os.makedirs('models',exist_ok=True)
                model_path = os.path.join('models','efficientdet-d4_trained.pth')
        
                if not os.path.isfile(model_path):
                    print('Loading default garage detector model file to the models folder...')
                    torch.hub.download_url_to_file('https://zenodo.org/record/5384012/files/efficientdet-d4_trained.pth',
                                                   model_path, progress=False)
                    print('Default garage detector model loaded.')                    
            else:
                print(f'Inferences will be performed using the custom model at {model_path}.')
        
        
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
            
        # Create and Define the Inference Model
        classes = ["garage"]
        
        print("Performing inferences on images...")
        gtfInfer = Infer()
        gtfInfer.load_model(self.system_dict["infer"]["modelPath"], classes, use_gpu=self.system_dict["infer"]["gpuEnabled"])
        
        rows = []
        predictions = []
        for img in tqdm(imgList):
            img = cv2.imread(img)
            bldgID = os.path.basename(img).split('.')[0]
            cv2.imwrite("input.jpg", img)
            scores, labels, boxes = gtfInfer.predict("input.jpg", threshold=0.35)
            if len(boxes)>=1:
              rows.append([bldgID,1])
              predictions.append(1)
            else:
              rows.append([bldgID,0])
              predictions.append(0)

        nImages = len(imgList)               
        if nImages!=1: 
            with open(self.system_dict["infer"]["outFile"], 'w', newline='', encoding='utf-8') as csvfile: 
                # creating a csv writer object 
                csvwriter = csv.writer(csvfile) 
        
                # writing the data rows 
                rows.insert(0, "Image, garageExist")
                csvwriter.writerows(rows)  
        else:
            print(f"Garage exists: {bool(predictions)}\n")        
        
        df = pd.DataFrame(list(zip(imgList, predictions)), columns =['image', 'prediction',])
        
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
       
        return df    