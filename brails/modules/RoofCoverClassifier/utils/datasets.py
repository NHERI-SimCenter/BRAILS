import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class RoofImages(Dataset):

    def __init__(self, csv_file, transforms, npid=False,test_mode=False, classes = None):
        
        self.transforms = transforms
        self.data_df = pd.read_csv(csv_file)
        self.npid = npid
        
        classes_file = self.data_df['coarse_class'].unique()
        classes_file = np.delete(classes_file,np.argwhere(classes_file == 'disregard'))
        classes_file = np.sort(classes_file)
        
        # It is checked if the classes from the checkpoint should be loaded or if this is 
        # actually a new training process
        self.classes = classes_file if classes is None else classes
        self.classes = np.sort(self.classes)
        
        self.data_df = self.data_df[self.data_df.coarse_class != 'disregard']

        self.class_weights = {}
        self.train_weights = []
        for _class in self.classes:
            self.class_weights[_class] = len(self.data_df[_class == self.data_df['coarse_class']])
        if not test_mode:
            for entry in self.data_df['coarse_class']:
                self.train_weights.append(1/self.class_weights[entry])
            self.train_weights = np.array(self.train_weights)

        self.test_mode = test_mode
        
        
    @staticmethod
    def to_csv_datasource(image_folder, csv_filename, calc_perf=False):
        
        img_paths = []
        filenames = []
        labels = []
        
        if not os.path.isdir(image_folder):
            if os.path.isfile(image_folder):
                # The following format is to be consistent with os.walk output
                file_list = [[os.path.split(image_folder)[0],'None',[os.path.split(image_folder)[1]]]]
            else:
                print('Error: Image folder or file {} not found.'.format(image_folder))
                exit()
        else:
            file_list = os.walk(image_folder, followlinks=True)

        for root, _, fnames in sorted(file_list):
            for fname in sorted(fnames):
                
                if 'jpg' in fname or 'png' in fname:
                    img_path = os.path.join(root, fname)

                    img_paths.append(img_path)
                    
                    if calc_perf:
                        label = root.split(os.path.sep)[-1]
                        labels.append(label)
                 
           
           
        df = pd.DataFrame()
        if calc_perf:            
            df['coarse_class'] = labels
        df['filenames'] = img_paths
        df.to_csv(csv_filename,sep=',')
            

    def __len__(self):
        return len(self.data_df)

    def loader(self,path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, index):

        target = self.data_df.iloc[index]['coarse_class']
        path= self.data_df.iloc[index]['filenames']
        img = self.loader(path)

        if self.transforms is not None:
            img = self.transforms(img)
        
        if self.test_mode:
            target = 0 # In test mode, no labels are available
        
        target = np.flatnonzero(self.classes == target).flatten()
        target = torch.LongTensor(target).squeeze()
        
        return img, target, index
