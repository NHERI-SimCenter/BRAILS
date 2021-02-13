import argparse

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import csv

from models.resnet_applied import resnet50
from utils.Datasets import Foundation_Type_Testset

parser = argparse.ArgumentParser(description='Detect Foundation Type')

parser.add_argument('--image-path',help='Path to one image or a folder containing images.',required=True)
parser.add_argument('--checkpoint', default=None,type=str,
                    help='Path to checkpoint. Defaults to best pretrained version.')
parser.add_argument('--only-cpu', action='store_true', help='Use CPU only, disregard GPU.')
parser.add_argument('--mask-buildings', action='store_true')
parser.add_argument('--load-masks', action='store_true')
# This is no longer used 
#parser.add_argument('--model',help='Pretrained model, options ["foundation_v0.1"]', type=str)

os.makedirs('checkpoints',exist_ok=True)
weight_file_path = os.path.join('checkpoints','best_masked.pkl')

if not os.path.isfile(weight_file_path):
    print('Loading remote model file to the weights folder..')
    torch.hub.download_url_to_file('https://zenodo.org/record/4145934/files/best_masked.pkl', weight_file_path)
    
model_name='ade20k-resnet50dilated-ppm_deepsup'
model_path=os.path.join('csail_segmentation_tool','csail_seg',model_name)

encoder='{}/encoder_epoch_20.pth'.format(model_name)
decoder='{}/decoder_epoch_20.pth'.format(model_name)

os.makedirs(model_path,exist_ok=True)

if not os.path.isfile(os.path.join(model_path,'encoder_epoch_20.pth')):
    print('Loading remote model file to the weights folder..')
    torch.hub.download_url_to_file('http://sceneparsing.csail.mit.edu/model/pytorch/{}'.format(encoder),os.path.join(model_path,'encoder_epoch_20.pth'))

if not os.path.isfile(os.path.join(model_path,'decoder_epoch_20.pth')):
    print('Loading remote model file to the weights folder..')
    torch.hub.download_url_to_file('http://sceneparsing.csail.mit.edu/model/pytorch/{}'.format(decoder),os.path.join(model_path,'decoder_epoch_20.pth'))

args = parser.parse_args()

if args.only_cpu:
    device='cpu'
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    test_transforms = [transforms.Resize((224, 224)),
                      transforms.ToTensor(),
                      normalize]

    if args.mask_buildings and not args.load_masks:
        from csail_segmentation_tool.csail_segmentation import MaskBuilding
        test_transforms.insert(0,
                              MaskBuilding(device))

    test_transforms = transforms.Compose(test_transforms)
    dataset = Foundation_Type_Testset(args.image_path, transform=test_transforms, mask_buildings=args.mask_buildings,
                                      load_masks = args.load_masks)

    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = resnet50(low_dim=1)
    
    if args.checkpoint is not None:
        model_file = args.checkpoint
    else:
        weight_file_path = os.path.join('checkpoints','best_model_weights.pth')
        if not os.path.isfile(weight_file_path):
            print('Loading remote model file to the weights folder..')
            torch.hub.download_url_to_file('https://zenodo.org/record/4145934/files/best_masked.pkl', weight_file_path)
        model_file = weight_file_path

    if not torch.cuda.is_available():
        state_dict = torch.load(model_file, map_location=torch.device('cpu'))
    else:
        state_dict = torch.load(model_file)


    missing, unexpected = model.load_state_dict(state_dict,
                                                strict=False)
    if any(['module' in name for name in unexpected]):
        # Remapping to remove effects of DataParallel
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' of dataparallel
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict,
                              strict=False)
    else:
        if len(missing) or len(unexpected):
            print('Missing or unexpected keys: {},{}'.format(missing, unexpected))
            print('This should not happen. Check if checkpoint is correct')

    model.eval()
    model = model.to(device)

    predictions = []

    with torch.no_grad():
        for i, (images, filename) in enumerate(test_loader):
            print('Classifying {}'.format(filename))
            images = images.to(device)

            prediction = model(images.float())
            predictions.append({'filename':filename[0],'prediction':int(torch.sigmoid(prediction).cpu().numpy() > 0.5)})

    with open('{}_prediction_results.csv'.format(os.path.basename(os.path.normpath(args.image_path))), 'w', newline='') as myfile:
        for prediction in predictions:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow([str(prediction['filename']),str(prediction['prediction'])])

    print ('Classification finished. Results written to {}'.format('{}_prediction_results.csv'.format(os.path.basename(os.path.normpath(args.image_path)))))


if __name__ == '__main__':
    main()
