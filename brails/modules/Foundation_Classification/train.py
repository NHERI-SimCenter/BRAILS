import argparse
import datetime
import os
import matplotlib as mpl
from attention_utils.utils import evaluate
import numpy as np

mpl.use('Agg')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from attention_utils.radam import RAdam

from models.resnet_applied import resnet50

from torch.optim.lr_scheduler import StepLR

from utils.Datasets import Foundation_Type_Binary

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
np.random.seed(1337)

parser = argparse.ArgumentParser(description='Train Residual Attention')
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--epochs', default=500, type=int,
                    help='Train for n epochs')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='Start epoch')
parser.add_argument('--lr', default=0.001, type=float,
                    help='Learning rate')
parser.add_argument('--batch-size', default=4, type=int,
                    help='Size of mini batch')
parser.add_argument('--train-data', type=str,
                    help='Training csv file with image names and labels')
parser.add_argument('--val-data', type=str,
                    help='Validation  csv file with image names and labels')
parser.add_argument('--test-data', type=str,
                    help='Test csv file with image names and labels')
parser.add_argument('--eval', action='store_true',
                    help='Only do evaluation')
parser.add_argument('--exp-name', default='noname', type=str, help='Name of this experiment to add to path name of log')
parser.add_argument('--checkpoint', type=str,
                    help='Path to checkpoint for evaluation or pretraining')
parser.add_argument('--mask-buildings', action='store_true')
parser.add_argument('--freeze-layers', action='store_true')
parser.add_argument('--pretrained', action='store_true', help='Use pretrained model.')

args = parser.parse_args()
if args.eval:
    args.pretrained = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

currentTime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

log_dir = os.path.join('runs', 'run_{}'.format(args.exp_name) + currentTime)
summary_writer = SummaryWriter(log_dir)

lr = args.lr  # 0.1
summary_writer.add_text('Parameters', str(args))
if args.start_epoch > 0:
    args.epochs = args.start_epoch + args.epochs

def main():
    best_checkpoint = os.path.join(log_dir, 'checkpoints/best_self_trained.pkl')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transforms = [
        transforms.RandomResizedCrop(224, scale=(0.7, 1.)),
        transforms.RandomGrayscale(p=0.5),
        transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        normalize]

    train_transforms = transforms.Compose(train_transforms)

    val_transforms = [transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize]

    val_transforms = transforms.Compose(val_transforms)

    if not args.eval:
        
        train_dataset = Foundation_Type_Binary(args.train_data,
                                               transform=train_transforms,
                                               mask_buildings=args.mask_buildings, load_masks=True)

        val_dataset = Foundation_Type_Binary(args.val_data,
                                             transform=val_transforms,
                                             mask_buildings=args.mask_buildings, load_masks=True)
        
        train_weights = np.array(train_dataset.train_weights)
        train_sampler = torch.utils.data.WeightedRandomSampler(train_weights, len(train_weights),
                                                                   replacement=True)
    

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                  drop_last=False, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                drop_last=False)

    else:
        test_dataset = Foundation_Type_Binary(args.test_data,
                                             transform=val_transforms,
                                             mask_buildings=args.mask_buildings, load_masks=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                drop_last=False)

    model = resnet50(low_dim=1)

    # Freeze all layers apart from the final layer
    if args.freeze_layers:
        ct = 0
        for child in model.children():
            ct += 1
            if ct < 10:
                print('Freezing {}'.format(child))
                for param in child.parameters():
                    param.requires_grad = False

    summary_writer.add_text('Architecture', model.__class__.__name__)
    summary_writer.add_text('Train Transforms', str(train_transforms))
    summary_writer.add_text('Val Transforms', str(val_transforms))

    criterion = nn.BCEWithLogitsLoss()

    optimizer = RAdam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

    summary_writer.add_text('Criterion', str(criterion))
    summary_writer.add_text('Optimizer', str(optimizer))

    is_train = not args.eval

    best_perf = 1e6

    model = nn.DataParallel(model).to(device)
    if args.pretrained:
        try:
            state_dict = torch.load(args.checkpoint)['state_dict']
        except KeyError: # Format of NPID checkpoints, and checkpoints created with train.py differ
            state_dict = torch.load(args.checkpoint)

        missing, unexpected = model.load_state_dict(state_dict,
                                                    strict=False)
        if len(missing) or len(unexpected):
            print('Missing or unexpected keys: {},{}'.format(missing, unexpected))

    if is_train is True:
        for epoch in range(args.start_epoch, args.epochs):
            model.train()

            print('Training epoch: {}'.format(epoch))
            y_train_pred, y_train_gt, avg_train_loss = parse(model, train_loader, criterion, optimizer, 'train', epoch)
            scheduler.step()
            evaluate(summary_writer, 'Train', y_train_gt, y_train_pred, avg_train_loss,
                     train_loader.dataset.classes, epoch)

            print('Validation epoch: {}'.format(epoch))
            with torch.no_grad():
                y_val_pred, y_val_gt, avg_val_loss = parse(model, val_loader, criterion, None, 'val', epoch)

            current_perf = evaluate(summary_writer, 'Val',
                                    y_val_gt, y_val_pred, avg_val_loss,
                                    train_loader.dataset.classes, epoch)

            if current_perf > best_perf:
                best_perf = current_perf
                print('current best performance measure,', best_perf)
                torch.save(model.state_dict(), best_checkpoint)

            # Save regular checkpoint every epoch
            latest_model_path = os.path.join(log_dir, 'checkpoint_epoch_{}.pkl'.format(epoch))
            torch.save(model.state_dict(), latest_model_path)
        print('best performance measure mse' + str(best_perf))

    else:
        print('Only test mode:')
        with torch.no_grad():
            y_val_pred, y_val_gt, avg_val_loss = parse(model, test_loader, criterion, None, 'test', 0)

        current_perf = evaluate(summary_writer, 'Test',
                                y_val_gt, y_val_pred, avg_val_loss,
                                test_loader.dataset.classes, 0)

        print('F1: {}'.format(current_perf[2]))
        print('Precision: {}'.format(current_perf[0]))
        print('Recall: {}'.format(current_perf[1]))
        exit()


def parse(model, data_loader, criterion, optimizer, parse_type, epoch):
    train = False
    if parse_type == 'train':
        model.train()
        train = True
    elif parse_type == 'val' or parse_type == 'test':
        model.eval()
    else:
        raise ValueError('Wrong parse type. Options are train, val or test')

    predictions = []
    ground_truths = []

    avg_loss = 0

    for i, (images, labels, _) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad()

        prediction = model(images.float())
        loss = criterion(prediction.squeeze(), labels.squeeze())

        if train:
            loss.backward()
            optimizer.step()

        avg_loss += loss.detach().cpu().numpy()

        predictions.extend(torch.sigmoid(prediction).detach().cpu().tolist())  # Should do the detach internally
        ground_truths.extend(labels.detach().cpu().tolist())

        if (i + 1) % 100 == 0:
            print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" % (
                epoch + 1, args.epochs, i + 1, len(data_loader), loss.data.item()))

    return predictions, ground_truths, avg_loss / len(data_loader)


if __name__ == '__main__':
    main()
