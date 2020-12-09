'''
@author: Sascha Hornauer
This code is a heavily modified but based on Progressive Multi-Granularity Training of
Du, Ruoyi and Chang, Dongliang and Bhunia, Ayan Kumar and Xie, Jiyang and Song, Yi-Zhe and Ma, Zhanyu and Guo, Jun

See the LICENSE file for further citation information
'''

from __future__ import print_function
import os

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from lib.datasets import YearBuiltFolder
from torchvision.transforms import transforms
from lib.utils import load_model, cosine_anneal_schedule, jigsaw_generator, \
    test_softlabels, test
from torch.autograd.variable import Variable
import argparse

parser = argparse.ArgumentParser(description='Classify Year Built')

parser.add_argument('--image-path', help='Path to one image or a folder containing images.', required=True)
parser.add_argument('--epochs', help='Epochs to train', default=200)
parser.add_argument('--batch-size', help='Batch size to use for training. Recommended is 128 for 8 GPUs or significantly less when no GPU is available. Default is 8', default=8)
parser.add_argument('--soft-labels', help='Activate soft labels', action='store_true')
parser.add_argument('--gaussian-std', help='Standard deviation of the unimodal gaussian used to model class membership', default=1.5)
parser.add_argument('--checkpoint', default='', type=str,
                    help='Path to checkpoint. Defaults to best pretrained version.')
parser.add_argument('--exp-name', default='unnamed_experiment', help='Name of this experiment for saved files.')

args = parser.parse_args()

def train(nb_epoch, batch_size, store_name, image_path, soft_labels=False, gaussian_std=1.5, resume=False, start_epoch=0, model_path=None):
    batch_size = int(batch_size)
    nb_epoch = int(nb_epoch)
    start_epoch = int(start_epoch)
    
    exp_dir = store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)

    use_cuda = torch.cuda.is_available()
    # GPU
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Scale((550, 550)),
        transforms.RandomCrop(448, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    transform_test = transforms.Compose([
        transforms.Scale((550, 550)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
        
    trainset = YearBuiltFolder(image_path, calc_perf=True,soft_labels=soft_labels, gaussian_std=gaussian_std, transforms=transform_train)
    
    train_weights = np.array(trainset.train_weights)
    train_sampler = torch.utils.data.WeightedRandomSampler(train_weights, len(train_weights),
                                                                   replacement=True)
        
        
    if use_cuda:
        device = 'cuda'
    else:
        device = 'cpu'
            
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=(train_sampler == None), num_workers=4, sampler=train_sampler)
    
    # Testing is on the same data but without the transforms. This whole class
    # only givey training performance and the user has to use the detect script
    # to check the performance.
    train_set_without_transforms = YearBuiltFolder(image_path,calc_perf=True, soft_labels=soft_labels, gaussian_std=gaussian_std, transforms=transform_test)
    
    train_test_loader = torch.utils.data.DataLoader(train_set_without_transforms, batch_size=3, shuffle=True, num_workers=4)
       
    # Model
    if resume:
        net = torch.load(model_path)
    else:
        net = load_model(model_name='resnet50_pmg', pretrain=True, require_grad=True, num_classes = len(trainset.classes))
    netp = torch.nn.DataParallel(net)

    net.to(device)
        
    if soft_labels:
        loss = nn.KLDivLoss(reduction='batchmean')
    else:
        loss = nn.CrossEntropyLoss()
    
    sm = nn.LogSoftmax(dim=1)
    
    optimizer = optim.SGD([
        {'params': net.classifier_concat.parameters(), 'lr': 0.002},
        {'params': net.conv_block1.parameters(), 'lr': 0.002},
        {'params': net.classifier1.parameters(), 'lr': 0.002},
        {'params': net.conv_block2.parameters(), 'lr': 0.002},
        {'params': net.classifier2.parameters(), 'lr': 0.002},
        {'params': net.conv_block3.parameters(), 'lr': 0.002},
        {'params': net.classifier3.parameters(), 'lr': 0.002},
        {'params': net.features.parameters(), 'lr': 0.0002}

    ],
        momentum=0.9, weight_decay=5e-4)

    max_train_acc = 0
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    for epoch in range(start_epoch, nb_epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0
        correct = 0
        total = 0
        idx = 0
        for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
            idx = batch_idx
            if inputs.shape[0] < batch_size:
                continue
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)

            # update learning rate
            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])

            # Step 1
            optimizer.zero_grad()
            inputs1 = jigsaw_generator(inputs, 8)            
            output_1, _, _, _ = netp(inputs1)

            loss1 = loss(sm(output_1), targets) * 1
            loss1.backward()
            optimizer.step()

            # Step 2
            optimizer.zero_grad()
            inputs2 = jigsaw_generator(inputs, 4)
            _, output_2, _, _ = netp(inputs2)
            loss2 = loss(sm(output_2), targets) * 1
            loss2.backward()
            optimizer.step()

            # Step 3
            optimizer.zero_grad()
            inputs3 = jigsaw_generator(inputs, 2)
            _, _, output_3, _ = netp(inputs3)
            loss3 = loss(sm(output_3), targets) * 1
            loss3.backward()
            optimizer.step()

            # Step 4
            optimizer.zero_grad()
            _, _, _, output_concat = netp(inputs)
            concat_loss = loss(sm(output_concat), targets) * 2
            concat_loss.backward()
            optimizer.step()

            #  training log
            _, predicted = torch.max(output_concat.data, 1)
            if soft_labels:
                _, targets = torch.max(targets.data, 1)
            
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            train_loss += (loss1.item() + loss2.item() + loss3.item() + concat_loss.item())
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss3 += loss3.item()
            train_loss4 += concat_loss.item()

            if batch_idx % 50 == 0:
                print(
                    'Step: %d | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    batch_idx, train_loss1 / (batch_idx + 1), train_loss2 / (batch_idx + 1),
                    train_loss3 / (batch_idx + 1), train_loss4 / (batch_idx + 1), train_loss / (batch_idx + 1),
                    100. * float(correct) / total, correct, total))

        train_acc = 100. * float(correct) / total
        train_loss = train_loss / (idx + 1)
        with open(os.path.join(exp_dir,'results_train.txt'), 'a') as file:
            file.write(
                'Iteration %d | train_acc = %.5f | train_loss = %.5f | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f |\n' % (
                epoch, train_acc, train_loss, train_loss1 / (idx + 1), train_loss2 / (idx + 1), train_loss3 / (idx + 1),
                train_loss4 / (idx + 1)))

        # if epoch < 5 or epoch >= 80:
        print ('Start testing')
        if soft_labels:
            train_acc, train_acc_com, train_loss, y_gt, y_pred = test_softlabels(net, loss, train_test_loader)
        else:
            train_acc, train_acc_com, train_loss, y_gt, y_pred = test(net, loss, train_test_loader)
        if train_acc_com > max_train_acc:
            max_train_acc = train_acc_com
            net.cpu()
            torch.save(net, os.path.join(store_name,'model_best.pth'))
            net.to(device)
        with open(os.path.join(exp_dir,'results_test.txt'), 'a') as file:
            file.write('Iteration %d, test_acc = %.5f, test_acc_combined = %.5f, test_loss = %.6f\n' % (
            epoch, train_acc, train_acc_com, train_loss))
        
        net.cpu()
        torch.save(net, os.path.join(store_name,'model_epoch_{}.pth'.format(epoch)))
        net.to(device)


try:
    train(nb_epoch=args.epochs,  # number of epoch
             batch_size=args.batch_size,  # batch size
             image_path=args.image_path,
             soft_labels=args.soft_labels,
             gaussian_std=args.gaussian_std,
             store_name=args.exp_name,  # folder for output
             resume=not (args.checkpoint == ''),  # resume training from checkpoint
             start_epoch=0,  # the start epoch number when you resume the training
             model_path=args.checkpoint
             )  # the saved model where you want to resume the training
except ValueError as ex:
    if 'Expected more than 1 value per channel when training' in str(ex):
        print('Error: The chosen batch size is not compatible with the size of the dataset due to the batch norm layers in the model. Please choose a different batch size, preferably larger.')
        exit(1)
