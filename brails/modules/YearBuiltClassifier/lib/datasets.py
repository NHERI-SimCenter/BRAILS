import torch
from PIL import Image
from torch.utils.data import Dataset

import numpy as np
import os
from collections import defaultdict
from scipy import signal

class YearBuiltFolder(Dataset):
    
    def __init__(self, image_folder, soft_labels=False, gaussian_std=1.5,  transforms=None, classes=None, calc_perf=False):

        self.transforms = transforms

        self.img_paths = []
        self.filenames = []
        self.labels = []
        self.calc_perf = calc_perf
        self.soft_labels = soft_labels
        '''
        if not os.path.isdir(image_folder):
            if os.path.isfile(image_folder):
                # The following format is to be consistent with os.walk output
                file_list = [[os.path.split(image_folder)[0],'None',[os.path.split(image_folder)[1]]]]
            else:
                print('Error: Image folder or file {} not found.'.format(image_folder))
                exit()
        else:
            file_list = os.walk(image_folder, followlinks=True)
        '''

        if isinstance(image_folder, list): #a list of images
            file_list = [[os.path.split(i)[0],'None',[os.path.split(i)[1]]] for i in image_folder]
        elif isinstance(image_folder, str): 
            if not os.path.isdir(image_folder):
                if os.path.isfile(image_folder):
                    # The following format is to be consistent with os.walk output
                    file_list = [[os.path.split(image_folder)[0],'None',[os.path.split(image_folder)[1]]]]
                else:
                    print('Error: Image folder or file {} not found.'.format(image_folder))
                    exit()

            else:# dir
                file_list = os.walk(image_folder, followlinks=True)

        class_counts = defaultdict(lambda:0) 

        for root, _, fnames in sorted(file_list):
            for fname in sorted(fnames):
                
                if 'jpg' in fname or 'png' in fname:
                    img_path = os.path.join(root, fname)

                    self.filenames.append(fname)
                    self.img_paths.append(img_path)
                    
                    if calc_perf:
                        label = root.split(os.path.sep)[-1]
                        self.labels.append(label)
                        class_counts[label] = class_counts[label] + 1 # count labels
        
        if classes is None:
            self.classes = np.unique(self.labels)
        else:
            self.classes=classes
        ########### Calculate train weights for weighted sampling        
        self.class_weights = {}
        self.train_weights = []

        for _class in self.classes:
            self.class_weights[_class] = sum(np.array([label == _class for label in self.labels]))
        
        for entry in self.labels:
            self.train_weights.append(1 / self.class_weights[entry])       

        ################################
        # Optionally, create soft labels by pre-calculating unimodal gaussians and assigning
        # them according to the class label
            
        if soft_labels:
            res = 100
            class_n = len(self.classes)
            window = signal.gaussian(res*class_n, std=(res/10)*(class_n-1)*float(gaussian_std))
            
            center_id = int((res*class_n)/2)
            
            samples = []
            for i in range(-class_n//2,class_n//2+1):
                pos_of_int = window[min(max(center_id+res*i,0),window.shape[0]-1)]
                samples.append(pos_of_int)
                
            samples = np.array(samples)
        
            label_lookup = defaultdict()
            for class_id in self.classes:
                class_soft_labels = []
                
                max_class_id = class_n
                for i in range(max_class_id):
                    sample_id = (i+3)-np.flatnonzero(self.classes == class_id)
            
                    if sample_id < 0 or sample_id >= max_class_id:
                        class_soft_labels.append(0.0)
                    else:
                        class_soft_labels.append(samples[sample_id].squeeze())
                        
                class_soft_labels = np.array(class_soft_labels)
                
                label_lookup[class_id] = class_soft_labels
        
            year_built_softlabels = [label_lookup[label] for label in self.labels]
            
            self.labels = year_built_softlabels
            
            

    def __len__(self):
        return len(self.filenames)

    def loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
        
    def __getitem__(self, index):

        
        path = self.img_paths[index]
        img = self.loader(path)

        if self.transforms is not None:
            img = self.transforms(img)
        
        if self.calc_perf:
            label = self.labels[index]
            # The label is either a probability distribution or the class number
            if self.soft_labels:
                target = label
                target = torch.FloatTensor(target) 
            else:
                target = [np.flatnonzero(self.classes == label)]
                target = torch.LongTensor(target).squeeze()
                
            
        else:
            target = []
    

        return img, target, path

  
    
