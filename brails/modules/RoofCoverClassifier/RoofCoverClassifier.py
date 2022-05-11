import argparse
import datetime
import os
import random
import shutil
import time
import warnings

import matplotlib
matplotlib.use('agg')

import numpy as np
import sklearn

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import torch.multiprocessing as mp

import torch.utils.data.distributed
import torchvision.transforms as transforms
from sklearn.metrics import precision_recall_fscore_support
from tensorboardX import SummaryWriter

from .utils.datasets import RoofImages
import torchvision.models as models

from .utils.eval_tools import construct_confusion_matrix_image
from .utils.radam import RAdam
import pandas as pd

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))



'''
args.gpu
args.evaluate
args.resume
args.arch
args.batch_size
args.workers
args.pretrained
args.val_data
'''

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--val-data',
                    help='path to val csv, either a csv file ending in .csv, a folder or a single image. For testing, provide the test data with this flag and  choose the -e eval command line parameter')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--resume', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--val-resize', type = int, default=None,
                    help='validation image resizing')

best_f1 = 0

os.makedirs('weights',exist_ok=True)
weight_file_path = os.path.join('weights','weights_resnet_34.ckp')


if not os.path.isfile(weight_file_path):
    print('Loading remote model file to the weights folder..')
    torch.hub.download_url_to_file('https://zenodo.org/record/4394542/files/weights_resnet_34.ckp', weight_file_path)
    
def main():
    args = parser.parse_args()
    
    if args.evaluate and args.resume == None:
        args.resume = weight_file_path

    
    
    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_f1
    args.gpu = gpu

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
        model.fc = nn.Linear(512 * 1, 4) # replace final classifier
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch == 'residual_attention_network':
            from model.residual_attention_network import ResidualAttentionModel_92
            model = ResidualAttentionModel_92(num_classes=4)
        else:
            model = models.__dict__[args.arch](num_classes=4)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss(reduction='none').cuda(args.gpu)
    
    # Better Adam optimizer: https://github.com/LiyuanLucasLiu/RAdam
    optimizer = RAdam(model.parameters(),lr=args.lr)

    # optionally resume from a checkpoint
    checkpoint = None
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
                from collections import OrderedDict
                cpu_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:] # remove module.
                    cpu_state_dict[name] = v
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_f1 = checkpoint['best_f1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_f1 = best_f1.to(args.gpu)
            
            if args.gpu is None:
                model.load_state_dict(cpu_state_dict)
            else:
                model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    val_data = args.val_data
    

    if not args.resume:
        print('Evaluation is chosen without resuming from a checkpoint. Please choose a checkpoint to load with the --resume parameter.')
        exit(1)
    
    if not (os.path.isfile(val_data) and os.path.splitext(val_data)[-1] == '.csv'):        
        RoofImages.to_csv_datasource(val_data,csv_filename='tmp_val_set.csv', calc_perf=True)
        valdir = 'tmp_val_set.csv'
    else:
        valdir = args.val_data
        
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])



    try:
        classes = checkpoint['classes']
    except:
        classes = None

    val_loader = torch.utils.data.DataLoader(
        RoofImages(valdir, transforms.Compose([
            transforms.Resize(256 if args.val_resize is None else args.val_resize),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]),test_mode=args.evaluate, classes=classes),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        results = validate(val_loader, model, criterion, args, 0, test_mode=True)
     
        return results







def validate(val_loader, model, criterion, args, epoch, test_mode=False):
    print('Start validation')
    # switch to evaluate mode
    model.eval()
    pred_all = []
    real_all = []
    loss_vec = []
    all_indexes = []

    with torch.no_grad():
        for i, (images, target, index) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            
            if not test_mode:
                
                loss = criterion(output, target)
                loss = loss.mean() # Because I need to reduce to none because of something else
                real_all.append(target.cpu().numpy().flatten())
                loss_vec.append(loss.item())
                
            pred = output.max(1, keepdim=True)[1]  # .squeeze() # get the index of the max log-probability
            pred_all.append(pred.flatten().cpu().numpy())
            
            all_indexes.append(index.flatten().cpu().numpy()) # Making sure the index fits to the label
            

    

    
    y_pred = np.concatenate(pred_all)
    
        
    all_indexes = np.concatenate(all_indexes)
    
    filenames = val_loader.dataset.data_df.iloc[all_indexes]['filenames']
    
    results = pd.DataFrame()
    results['file'] = filenames
#           
    results['prediction'] = [val_loader.dataset.classes[y_class] for y_class in y_pred]
    
    return  results


