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
# 05-12-2022

import os
import numpy as np
import time
import pandas as pd

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torchvision.transforms as transforms
from .utils.datasets import RoofImages
import torchvision.models as models

class RoofCoverClassifier():
    def __init__(self):
        self.system_dict = {}
        self.system_dict["infer"] = {}
        self.system_dict["infer"]["params"] = {}
        self.system_dict["infer"]["params"]['batch_size'] = 64
        self.system_dict["infer"]["params"]['nworkers'] = 0
                      
    def predict(self, images='tmp/images/satellite', modelPath="tmp/models/weights_resnet_34.ckp"):
        self.system_dict["infer"]["images"] = images
        self.system_dict["infer"]["modelPath"] = modelPath
        self.system_dict["infer"]['predictions'] = []
        
        
        print('\nClassifying roof cover material for each building...')
        
        os.makedirs('tmp/models',exist_ok=True)
        if not os.path.isfile(self.system_dict["infer"]["modelPath"]):
            print('Loading default roof cover classifer model file to tmp/models folder...')
            torch.hub.download_url_to_file('https://zenodo.org/record/4394542/files/weights_resnet_34.ckp',
                                           self.system_dict["infer"]["modelPath"],
                                           progress=False)
            print('Default roof cover classifier loaded')
        else: 
            print(f"Default roof cover classifier at {modelPath} loaded")  
    
        # Create the classifier model from the downloaded checkpoint:
        model = models.__dict__['resnet34'](num_classes=4)
    
        checkpoint = torch.load(modelPath, map_location=torch.device('cpu'))
        if not torch.cuda.is_available():
            print('Could not find a GPU accelerator, using CPU for inferences')
            from collections import OrderedDict
            cpu_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:] # remove module.
                cpu_state_dict[name] = v
            model.load_state_dict(cpu_state_dict)
        else:
            model = torch.nn.DataParallel(model).cuda()
            model.load_state_dict(torch.load(modelPath),strict=False)
        
        cudnn.benchmark = True
        
        # Get the Image List
        try: 
            imgList = os.listdir(self.system_dict["infer"]["images"])
            for imgno in range(len(imgList)):
                imgList[imgno] = os.path.join(self.system_dict["infer"]["images"],imgList[imgno])
        except:
            imgList = self.system_dict["infer"]["images"]
        

        df = pd.DataFrame()
        df['coarse_class'] = [os.path.dirname(im) for im in imgList]
        df['filenames'] = imgList
        valdir = 'tmp_val_set.csv'
        df.to_csv(valdir,sep=',')
    
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    
        classes = checkpoint['classes']
    
        val_loader = torch.utils.data.DataLoader(
            RoofImages(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]),test_mode=True, classes=classes),
            batch_size=self.system_dict["infer"]["params"]['batch_size'],
            shuffle=False,
            num_workers=self.system_dict["infer"]["params"]['nworkers'],
            pin_memory=True)
    
        # Perform inferences
        model.eval()
        pred_all = []
        all_indexes = []
    
        # Start Program Timer
        startTime = time.time()
        
        with torch.no_grad():
            for i, (self.system_dict["infer"]["images"], target, index) in enumerate(val_loader):
    
                if torch.cuda.is_available():
                    target = target.cuda(None, non_blocking=True)
    
                # compute output
                output = model(self.system_dict["infer"]["images"])     
                pred = output.max(1, keepdim=True)[1]  # .squeeze() # get the index of the max log-probability
                pred_all.append(pred.flatten().cpu().numpy())
                all_indexes.append(index.flatten().cpu().numpy()) # Making sure the index fits to the label
                
        y_pred = np.concatenate(pred_all)
            
        all_indexes = np.concatenate(all_indexes)
        filenames = val_loader.dataset.data_df.iloc[all_indexes]['filenames']
        
        results = pd.DataFrame()
        results['image'] = filenames        
        preds = [val_loader.dataset.classes[y_class] for y_class in y_pred]
        preds_final = []
        for pred in preds:
            if pred=='shingles asphalt wood':
                preds_final.append('Shingle')
            elif pred=='tiles':
                preds_final.append('Tiles')
            elif pred=='slate':
                preds_final.append('Slate')        
            elif pred=='standing seam (metal)':
                preds_final.append('Metal')
                
        results['prediction']  = preds_final
        self.system_dict["infer"]['predictions'] = results
        self.system_dict["infer"]["images"] = imgList
        
        
        # End Program Timer and Display Execution Time
        endTime = time.time()
        hours, rem = divmod(endTime-startTime, 3600)
        minutes, seconds = divmod(rem, 60)
        print("\nTotal execution time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
        
        # Cleanup the Root Folder
        if os.path.isfile('tmp_val_set.csv'):
            os.remove('tmp_val_set.csv')