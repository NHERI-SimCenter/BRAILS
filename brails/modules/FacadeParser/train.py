import os
import csv
import copy
import time
import sys
from pathlib import Path
from typing import Any, Callable, Optional
from PIL import Image
import torch
import torchvision.models.segmentation as models
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score#, roc_auc_score
import numpy as np
from tqdm import tqdm
import argparse

def get_args():
    parser = argparse.ArgumentParser('Pytorch-based segmentation model for facade parsing')
    parser.add_argument('--data_path', type=str, default='dataset',
                        help='Path for the root folder of dataset')
    parser.add_argument('--architecture', type=str, default="deeplabv3_resnet101",
                        help='Available options: fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, deeplabv3_resnet101')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers of Dataloader')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='The number of images per batch')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--optim', type=str, default='adam',
                        help='Select optimizer for training, '
                             'Use \'adam\' until the last stage'
                             'then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of training epochs')

    args = parser.parse_args()
    return args

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean entry')
    return s == 'True'

class Dataset(VisionDataset):
    def __init__(self,
                 root: str,
                 imageFolder: str,
                 maskFolder: str,
                 transforms: Optional[Callable] = None) -> None:

        super().__init__(root, transforms)
        imageFolderPath = Path(self.root) / imageFolder
        maskFolderPath = Path(self.root) / maskFolder
        if not imageFolderPath.exists():
            raise OSError(f"{imageFolderPath} does not exist!")
        if not maskFolderPath.exists():
            raise OSError(f"{maskFolderPath} does not exist!")

        self.image_names = sorted(imageFolderPath.glob("*"))
        self.mask_names = sorted(maskFolderPath.glob("*"))

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> Any:
        imagePath = self.image_names[index]
        maskPath = self.mask_names[index]
        with open(imagePath, "rb") as imFile, open(maskPath,
                                                        "rb") as maskFile:
            image = Image.open(imFile); 
            mask = Image.open(maskFile); 
            sample = {"image": image, "mask": mask}
            if self.transforms:
                sample["image"] = self.transforms(sample["image"])
                sample["mask"] = torch.tensor(np.array(sample["mask"],dtype=np.uint8), dtype=torch.long)
            return sample

def train(opt):
    # Specify the Model Architecture
    if opt.architecture.lower()=="deeplabv3_resnet50":
        model = models.deeplabv3_resnet50(pretrained=True,progress=True)
        model.classifier = models.deeplabv3.DeepLabHead(2048,5)
    elif  opt.architecture.lower()=="deeplabv3_resnet101":    
        model = models.deeplabv3_resnet101(pretrained=True,progress=True)
        model.classifier = models.deeplabv3.DeepLabHead(2048,5)
    elif  opt.architecture.lower()=="fcn_resnet50":    
        model = models.fcn_resnet50(pretrained=True,progress=True)
        model.classifier = models.fcn.FCNHead(2048,5)
    elif  opt.architecture.lower()=="fcn_resnet101":    
        model = models.fcn_resnet101(pretrained=True,progress=True)
        model.classifier = models.fcn.FCNHead(2048,5)
    
    # Define Optimizer    
    if opt.optim.lower()=="adam":
        modelOptim = torch.optim.Adam(model.parameters(), lr = opt.lr)
    elif opt.optim.lower()=="sgd":
        modelOptim = torch.optim.SGD(model.parameters(), lr = opt.lr)
        
    # Define Loss Function 
    lossFnc = torch.nn.CrossEntropyLoss()
    
    # Set Training and Validation Datasets
    dataTransforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    segdata = {
        x: Dataset(root=Path(opt.data_path) / x,
        imageFolder="images",
        maskFolder="masks",
        transforms=dataTransforms)
        for x in ["train", "valid"]
        }
    
    dataLoaders = {
        x: DataLoader(segdata[x],
                       batch_size=opt.batch_size,
                       shuffle=True,
                       num_workers=opt.num_workers)
        for x in ["train", "valid"]    
        }
    
    # Set Training Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create and Initialize Training Log File
    #perfMetrics = {"f1-score": f1_score, "auroc": roc_auc_score}
    perfMetrics = {"f1-score": f1_score}
    fieldnames = ['epoch', 'train_loss', 'valid_loss'] + \
        [f'train_{m}' for m in perfMetrics.keys()] + \
        [f'valid_{m}' for m in perfMetrics.keys()]
    with open(os.path.join('log.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    #model = torch.load('trainedModel25.pth')
    # Train
    startTimer = time.time()
    for epoch in range(1,opt.num_epochs+1):
        print('-' * 60)
        print("Epoch: {}/{}".format(epoch,opt.num_epochs))
        batchsummary = {a: [0] for a in fieldnames}
    
        for phase in ["train", "valid"]:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            # Iterate over data.
            for sample in tqdm(iter(dataLoaders[phase]), file=sys.stdout):
                inputs = sample['image'].to(device)
                masks = sample['mask'].to(device)
                # zero the parameter gradients
                modelOptim.zero_grad()
    
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = lossFnc(outputs['out'], masks)
                    y_pred = outputs['out'].data.cpu().numpy().ravel()
                    y_true = masks.data.cpu().numpy().ravel()
    
                    for name, metric in perfMetrics.items():
                        if name == 'f1-score':
                            # Use a classification threshold of 0.1
                            f1Classes = np.zeros(4)
                            nPixels = np.zeros(4)
                            for classID in range(4):
                                f1Classes[classID]  = metric(y_true==classID,y_pred[classID*len(y_true):(classID+1)*len(y_true)]>0.1)
                                nPixels[classID] = np.count_nonzero(y_true==classID)
                            f1weights = nPixels/(np.sum(nPixels))
                            f1 = np.matmul(f1Classes,f1weights)
                            batchsummary[f'{phase}_{name}'].append(f1)
                              
                        else:
                            batchsummary[f'{phase}_{name}'].append(
                                metric(y_true.astype('uint8'), y_pred))
    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        modelOptim.step()
            batchsummary['epoch'] = epoch
            epoch_loss = loss
            batchsummary[f'{phase}_loss'] = epoch_loss.item()
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        print((f'train loss: {batchsummary["train_loss"]: .4f}, '
               f'valid loss: {batchsummary["valid_loss"]: .4f}, '
               f'train f1-score: {batchsummary["train_f1-score"]: .4f}, '
               f'valid f1-score: {batchsummary["valid_f1-score"]: .4f}, '))
               #f'train auroc: {batchsummary["train_auroc"]: .4f}, '
               #f'valid auroc: {batchsummary["valid_auroc"]: .4f}, '))
        with open(os.path.join('log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            # deep copy the model
            if phase == 'valid' and loss < 1e10:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())
    
    time_elapsed = time.time() - startTimer
    print('-' * 60)
    print('Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print(f'Lowest validation loss: {best_loss: .4f}')
        
    # Load best model weights:
    model.load_state_dict(best_model_wts)

    # Save the best model:
    torch.save(model, 'facadeParser.pth')
    
if __name__ == '__main__':
    opt = get_args()
    train(opt)