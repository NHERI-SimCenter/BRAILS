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

from utils.datasets import RoofImages
import torchvision.models as models

from utils.eval_tools import construct_confusion_matrix_image
from utils.radam import RAdam
import pandas as pd

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--train-data',
                    help='path to training data, either a csv file ending in .csv, a folder or a single image')
parser.add_argument('--val-data',
                    help='path to val csv, either a csv file ending in .csv, a folder or a single image. For testing, provide the test data with this flag and  choose the -e eval command line parameter')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
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
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--loss-weighting', action='store_true',
                    help='loss weighting.')
parser.add_argument('--val-resize', type = int, default=None,
                    help='validation image resizing')
parser.add_argument('--weighted-sampling', action='store_true')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--checkpoint-dir', default='.')
parser.add_argument('--result-file', default='results.csv', help='Name of the csv file to save the predictions of all files.')

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

    currentTime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    args.log_dir = os.path.join('runs', 'run_{}'.format(args.checkpoint_dir) + currentTime)
    os.makedirs('runs', exist_ok=True)

    ## Tensorboard ##
    writer = SummaryWriter(args.log_dir)
    writer.add_text('args', str(args), 0)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, writer))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, writer)


def main_worker(gpu, ngpus_per_node, args, writer):
    global best_f1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
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
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
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
    train_data = args.train_data
    val_data = args.val_data
    
    if not args.evaluate:
        if not (os.path.isfile(train_data) and os.path.splitext(train_data)[-1] == '.csv'):
            RoofImages.to_csv_datasource(train_data,csv_filename='tmp_train_set.csv', calc_perf=True)
            traindir = 'tmp_train_set.csv'
        else:
            traindir = args.train_data
        
        if not (os.path.isfile(val_data) and os.path.splitext(val_data)[-1] == '.csv'):
            RoofImages.to_csv_datasource(val_data,csv_filename='tmp_val_set.csv', calc_perf=True)
            valdir = 'tmp_val_set.csv'
        else:
            valdir = args.val_data
    else:
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

    if not args.evaluate:
        train_dataset = RoofImages(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.ColorJitter(0.4,0.4,0.4,0.4),
                transforms.RandomGrayscale(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        if args.distributed:
            if args.weighted_sampling:
                print ('Warning: Weighted sampling not implemented for distributed training. So no weighted sampling will be performed')
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            if args.weighted_sampling:
                train_weights = np.array(train_dataset.train_weights)
                train_sampler = torch.utils.data.WeightedRandomSampler(train_weights, len(train_weights), replacement=True)
            else:
                train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

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
        f1, results = validate(val_loader, model, criterion, args, 0, writer, test_mode=True)
        results.to_csv(args.result_file)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        idx_vec, loss_vec = train(train_loader, model, criterion, optimizer, epoch, args, writer)

        if args.loss_weighting:
            print ('Weighting losses')
            train_set_order = np.argsort(idx_vec)
            loss_vec = (loss_vec - loss_vec.min())/(loss_vec.max() - loss_vec.min())
            loss_vec_in_order = loss_vec[train_set_order]
            train_sampler.weights = torch.as_tensor(loss_vec_in_order, dtype=torch.double)

        # evaluate on validation set
        f1, results = validate(val_loader, model, criterion, args, epoch, writer)
        results.to_csv(args.result_file)

        # remember best acc@1 and save checkpoint
        is_best = f1 > best_f1
        best_f1 = max(f1, best_f1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_f1': best_f1,
                'optimizer' : optimizer.state_dict(),
                'classes' : train_dataset.classes # Save the exact classnames this checkpoint was created with
            }, is_best=is_best, file_folder = args.log_dir)


def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    print('Start training')
    # switch to train mode
    model.train()
    loss_vec = []
    idx_vec = []

    for i, (images, target, idx) in enumerate(train_loader):
        # measure data loading time

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)
        with torch.no_grad():
            loss_vec.extend(loss.flatten().cpu().numpy())
            idx_vec.extend(idx.flatten().cpu().numpy())

        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss = loss.mean()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print('Instance {} Loss {}'.format(i,loss.mean()))

    writer.add_scalar('Train/Loss', np.mean(loss_vec), epoch)
    return np.array(idx_vec), np.array(loss_vec)


def validate(val_loader, model, criterion, args, epoch, writer, test_mode=False):
    print('Start validation')
    # switch to evaluate mode
    model.eval()
    pred_all = []
    real_all = []
    loss_vec = []
    all_indexes = []

    with torch.no_grad():
        end = time.time()
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
            

    
    if not test_mode:
        y_gt = np.concatenate(real_all)
    
    y_pred = np.concatenate(pred_all)
    
    if not test_mode:
        writer.add_scalar('Val/Loss', np.mean(loss_vec), epoch)
        precision, recall, f1, support = precision_recall_fscore_support(y_gt,
                                                                         y_pred, average='macro')
        writer.add_scalar('Val/F1', f1, epoch)
        writer.add_scalar('Val/Precision', precision, epoch)
        writer.add_scalar('Val/Recall', recall, epoch)
        classes = val_loader.dataset.classes
        
        
        confusion_matrix = sklearn.metrics.confusion_matrix(y_gt, y_pred, labels=range(len(classes)))
        cm, cm_fig = construct_confusion_matrix_image(classes, confusion_matrix)
        writer.add_image('Val/confusion_matrix', cm, epoch)
    else:
        f1 = None
        
    all_indexes = np.concatenate(all_indexes)
    
    filenames = val_loader.dataset.data_df.iloc[all_indexes]['filenames']
    
    results = pd.DataFrame()
    results['file'] = filenames
#           
    results['prediction'] = [val_loader.dataset.classes[y_class] for y_class in y_pred]
    
    return f1, results


def save_checkpoint(state, is_best, file_folder) :
    filename=os.path.join(file_folder,'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(file_folder,'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).cpu().numpy())
        return res


if __name__ == '__main__':
    main()