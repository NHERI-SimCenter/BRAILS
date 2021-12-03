from __future__ import print_function, division

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

eps = np.finfo(float).eps #small number to avoid zeros

class Foundation_Type_Testset(Dataset):
    def __init__(self, image_folder, transform=None,mask_buildings=False, load_masks=False):

        self.transform = transform

        self.img_paths = []
        self.mask_paths = []
        self.filenames = []

        self.mask_buildings = mask_buildings
        self.load_masks = load_masks
        '''
        if not os.path.isdir(image_folder):
            if os.path.isfile(image_folder):
                # The following format is to be consistent with os.walk output
                file_list = [[os.path.split(image_folder)[0],'None',[os.path.split(image_folder)[1]]]]
            elif isinstance(image_folder, list): #a list of images
                file_list = image_folder
            else:
                print('Error: Image folder or file {} not found.'.format(image_folder))
                exit()
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

        for root, _, fnames in sorted(file_list):
            for fname in sorted(fnames):
                if 'jpg' in fname or 'png' in fname:
                    if 'mask' in fname:
                        continue
                    img_path = os.path.join(root, fname)

                    if self.load_masks:
                        _, file_extension = os.path.splitext(img_path)
                        mask_filename = fname.replace(file_extension, '-mask.png')
                        mask_path = os.path.join(root, mask_filename)
                        if not os.path.isfile(mask_path):
                            print('No mask for {}. Skipping'.format(fname))
                            continue
                        self.mask_paths.append(mask_path)
                    self.filenames.append(fname)
                    self.img_paths.append(img_path)



    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.img_paths[idx]
        #image = Image.open(img_name)
        image = Image.open(img_name).convert('RGB')

        if self.mask_buildings and self.load_masks:
            image = np.array(image)
            mask_filename = self.mask_paths[idx]
            mask = Image.open(mask_filename)
            mask = np.array(mask)
            # Filter building labels
            mask[np.where((mask != 25) & (mask != 1))] = 0
            image[mask == 0, :] = 0
            image = Image.fromarray(np.uint8(image))

        if (self.transform):
            image = self.transform(image)

        fname = self.filenames[idx]
        return (image, fname)

class Foundation_Type_Binary(Dataset):
    def __init__(self, image_folder, transform=None,mask_buildings=False, load_masks=False):

        self.transform = transform
        self.classes = ['Raised','Not Raised']
        self.img_paths = []
        self.mask_paths = []
        labels = []
        self.mask_buildings = mask_buildings
        self.load_masks = load_masks

        assert os.path.isdir(image_folder),'Image folder {} not found or not a path'.format(image_folder)

        for root, _, fnames in sorted(os.walk(image_folder, followlinks=True)):
            for fname in sorted(fnames):
                if 'jpg' in fname or 'png' in fname:
                    if 'mask' in fname:
                        continue
                    img_path = os.path.join(root, fname)

                    _, file_extension = os.path.splitext(img_path)
                    mask_filename = fname.replace(file_extension, '-mask.png')
                    mask_path = os.path.join(root, mask_filename)
                    if not os.path.isfile(mask_path):
                        print('No mask for {}. Skipping'.format(fname))
                        continue

                    labels.append(os.path.dirname(img_path).split(os.path.sep)[-1])
                    self.img_paths.append(img_path)
                    self.mask_paths.append(mask_path)

        self.train_labels = np.zeros(len(labels))

        for class_id in ['5001', '5005', '5002', '5003']:
            idx = np.where(np.array(labels) == class_id)[0]
            self.train_labels[idx] = 0
        for class_id in ['5004', '5006']: # Piles Piers and Posts
            idx = np.where(np.array(labels) == class_id)[0]
            self.train_labels[idx] = 1
        
        # Train weights for optional weighted sampling    
        self.train_weights = np.ones(len(self.train_labels))
        self.train_weights[self.train_labels == 0] = np.sum(self.train_labels == 0) / len(self.train_labels)
        self.train_weights[self.train_labels == 1] = np.sum(self.train_labels == 1) / len(self.train_labels)
        self.train_weights = 1-self.train_weights
        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.img_paths[idx]
        image = Image.open(img_name).convert('RGB')

        if self.mask_buildings and self.load_masks:
            image = np.array(image)
            mask_filename = self.mask_paths[idx]
            mask = Image.open(mask_filename)
            mask = np.array(mask)
            # Filter building labels
            mask[np.where((mask != 25) & (mask != 1))] = 0
            image[mask == 0, :] = 0
            image = Image.fromarray(np.uint8(image))

        class_id = torch.FloatTensor([self.train_labels[idx]])

        if (self.transform):
            image = self.transform(image)
        return (image, class_id, idx)