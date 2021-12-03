import torchvision.datasets as datasets
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import torch

class ImageFolderInstance(datasets.ImageFolder):
    """: Folder datasets which returns the index of the image as well::
    """
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

class MaskFolderInstance(datasets.ImageFolder):
    """: Slightly hackish way to train on masks quick
    """
    def __init__(self, *args, **kwargs):
        if 'mask_images' in kwargs.keys():
            self.mask_images = kwargs['mask_images']
            kwargs.pop('mask_images')
        else:
            self.mask_images = False
        super().__init__(*args, **kwargs)
        # Another hackish thing to get rid of a problem with missin masks.
        # This class has to be redesigned from scratch

        # The following lines can be used to remove images without mask. This is only relevant
        # when doing the split
        # no_mask_ids = []
        # for i,entry in enumerate(self.imgs):
        #    if not os.path.isfile('/home/saschaho/Simcenter/Floor_Elevation_Data/Streetview_Irma/Streetview_Irma/images/' + os.path.basename(entry[0]).replace('.jpg','.png')):
        #        no_mask_ids.append(i)
        # # reverse order will preserver order during deletion
        # for i in sorted(no_mask_ids, reverse=True):
        #     del self.imgs[i]


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]

        img = self.loader(path)

        if self.mask_images:
            # replace jpg image with mask which is saved somewhere else.
            # This implementation needs to be more generalized
            #mask_filename = '/home/saschaho/Simcenter/Floor_Elevation_Data/Streetview_Irma/Streetview_Irma/images/' + os.path.basename(path).replace('.jpg','.png')
            mask_filename = os.path.basename(path).replace('.jpg','.png')
            mask = Image.open(mask_filename)
            mask = np.array(mask)
            # Filter building labels
            mask[np.where((mask != 25) & (mask != 1))] = 0
            img = np.array(img)
            img[mask == 0, :] = 0
            img = Image.fromarray(np.uint8(img))

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class CombinedMaskDataset(Dataset):
    """: Slightly hackish way to train on masks quick
    """
    def __init__(self, other_data_path=None, csv_root_folder=None, data_csv=None, transform=None, mask_images=True, attribute=None):
        self.mask_images = mask_images
        self.transform = transform
        self.imgs = []
        self.targets = []
        self.target_transform = None

        # Add only training images from the training data
        train_data_pd = pd.read_csv(data_csv)
        for entry in train_data_pd.iterrows():
            jpg_file = os.path.join(csv_root_folder, entry[1]['filename'])
            # If image and mask file exists. Add to list
            if os.path.isfile(jpg_file) and os.path.isfile(os.path.join(csv_root_folder, entry[1]['filename'].replace('.jpg', '.png'))):
                self.imgs.append(jpg_file)
                self.targets.append(entry[1][attribute])

        if other_data_path is not None:
            # If this is training data add all images from the other data
            for subdirs,_,filenames in os.walk(other_data_path,followlinks=True):
                for filename in filenames:
                    if 'jpg' in filename:
                        self.imgs.append(os.path.join(subdirs,filename))
                        self.targets.append(-1)

        self.train_labels = self.targets


    def __len__(self):
        return len(self.imgs)

    def loader(self,path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path= self.imgs[index]
        target = torch.FloatTensor([float(self.targets[index])])

        img = self.loader(path)

        if self.mask_images:
            # replace jpg image with mask which is saved somewhere else.
            # This implementation needs to be more generalized
            #mask_filename = '/home/saschaho/Simcenter/Floor_Elevation_Data/Streetview_Irma/Streetview_Irma/images/' + os.path.basename(path).replace('.jpg','.png')
            mask_filename = path.replace('.jpg','.png')
            mask = Image.open(mask_filename)
            mask = np.array(mask)
            # Filter building labels
            mask[np.where((mask != 25) & (mask != 1))] = 0
            img = np.array(img)
            img[mask == 0, :] = 0
            img = Image.fromarray(np.uint8(img))

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index