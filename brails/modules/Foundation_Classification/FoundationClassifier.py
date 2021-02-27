import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import csv

from .models.resnet_applied import resnet50
from .utils.Datasets import Foundation_Type_Testset

from .csail_segmentation_tool.csail_segmentation import MaskBuilding

class FoundationHeightClassifier():

    def __init__(self, checkpoint='', onlycpu=False, maskBuildings=False,loadMasks=False, workDir='tmp', resultFile='FoundationElevation.csv', printRes=True):
        '''
        checkpoint (str): Path to checkpoint. Defaults to best pretrained version.
        onlycpu (bool): Use CPU only, disregard GPU by default.
        maskBuildings (bool): Mask the parts of the image which are not a building. Slow.
        loadMasks (bool): Generate a mask on the fly if False and maskBuildings=True
        '''
    
        self.checkpoint = checkpoint
        self.onlycpu = onlycpu
        self.maskBuildings = maskBuildings
        self.loadMasks = loadMasks
        self.workDir = workDir
        self.outFilePath = os.path.join(workDir, resultFile)
        self.printRes = printRes

        self.checkpointsDir = os.path.join(workDir,'checkpoints')
        os.makedirs(self.checkpointsDir,exist_ok=True)
        weight_file_path = os.path.join(self.checkpointsDir,'best_masked.pkl')
        '''
        if not os.path.isfile(weight_file_path):
            print('Loading remote model file to the weights folder..')
            torch.hub.download_url_to_file('https://zenodo.org/record/4145934/files/best_masked.pkl', weight_file_path)
        '''

        if self.checkpoint != '':
            self.modelFile = self.checkpoint
        else:
            #weight_file_path = os.path.join(self.checkpointsDir,'best_model_weights.pth')
            if not os.path.isfile(weight_file_path):
                print('Loading remote model file to the weights folder..')
                torch.hub.download_url_to_file('https://zenodo.org/record/4145934/files/best_masked.pkl', weight_file_path)
            self.modelFile = weight_file_path

        # need to change this to tmp folder
        model_name='ade20k-resnet50dilated-ppm_deepsup'
        model_dir=os.path.join(workDir, 'csail_segmentation_tool','csail_seg',model_name)
        os.makedirs(model_dir, exist_ok=True)

        encoder=f'{model_name}/encoder_epoch_20.pth'
        decoder=f'{model_name}/decoder_epoch_20.pth'
        localEncoderFilePath = os.path.join(model_dir,'encoder_epoch_20.pth')
        localDecoderFilePath = os.path.join(model_dir,'decoder_epoch_20.pth')
        if not os.path.isfile(localEncoderFilePath):
            print('Loading remote model (encoder) file to the weights folder..')
            torch.hub.download_url_to_file(f'http://sceneparsing.csail.mit.edu/model/pytorch/{encoder}',localEncoderFilePath)

        if not os.path.isfile(localDecoderFilePath):
            print('Loading remote model (decoder) file to the weights folder..')
            torch.hub.download_url_to_file(f'http://sceneparsing.csail.mit.edu/model/pytorch/{decoder}',localDecoderFilePath)

        if self.onlycpu: 
            self.device='cpu'
        else: 
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


        # test_transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.test_transforms = [transforms.Resize((224, 224)),
                          transforms.ToTensor(),
                          normalize]

        if self.maskBuildings and not self.loadMasks:
            #from csail_segmentation_tool.csail_segmentation import MaskBuilding
            self.test_transforms.insert(0, MaskBuilding(self.device, model_dir=model_dir))

        self.test_transforms = transforms.Compose(self.test_transforms)

    def predict(self,image=''):
        '''
        image (str): Path to one image or a folder containing images.
        '''

        dataset = Foundation_Type_Testset(image, transform=self.test_transforms, mask_buildings=self.maskBuildings,
                                          load_masks = self.loadMasks)

        test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        model = resnet50(low_dim=1)


        model_file = self.modelFile

        if not torch.cuda.is_available():
            state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        else:
            state_dict = torch.load(model_file)


        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if any(['module' in name for name in unexpected]):
            # Remapping to remove effects of DataParallel
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove 'module.' of dataparallel
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict, strict=False)
        else:
            if len(missing) or len(unexpected):
                print('Missing or unexpected keys: {},{}'.format(missing, unexpected))
                print('This should not happen. Check if checkpoint is correct')

        model.eval()
        model = model.to(self.device)

        predictions = []
        probs = []
        imagePathList = []

        with torch.no_grad():
            for i, (images, filename) in enumerate(test_loader):
                images = images.to(self.device)

                prediction = model(images.float())
                score = torch.sigmoid(prediction).cpu().numpy()[0][0]

                p = int( score >= 0.5) # class: 0 or 1
                predictions.append(p)
                imagePathList.append(filename[0])
                prob = score if score >= 0.5 else 1.-score
                probs.append(prob)
                if self.printRes: print(f"Image :  {filename[0]}     Class : {p} ({str(round(prob*100,2))}%)") 

        
        df = pd.DataFrame(list(zip(imagePathList, predictions, probs)), columns =['image', 'prediction', 'probability']) 
        df.to_csv(self.outFilePath, index=False)
        print(f'Results written in file {self.outFilePath}')

        return df


if __name__ == '__main__':
    main()
