from time import sleep

import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import os

from .csail_seg.config import cfg
from .csail_seg.models import ModelBuilder
from scipy.io import loadmat
torch.multiprocessing.set_start_method('spawn', force=True)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

cur_dir_tmp = os.path.dirname(os.path.realpath(__file__))

class MaskBuilding(object):
    #colors = loadmat('./csail_segmentation_tool/csail_seg/data/color150.mat')['colors']
    #config_file = './csail_segmentation_tool/csail_seg/config/ade20k-resnet50dilated-ppm_deepsup.yaml'
    colors = loadmat(f'{cur_dir_tmp}/csail_seg/data/color150.mat')['colors']
    config_file = f'{cur_dir_tmp}/csail_seg/config/ade20k-resnet50dilated-ppm_deepsup.yaml'
    cfg.merge_from_file(config_file)
    cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
    cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()
    enc_weights = f'{cur_dir_tmp}/csail_seg/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth'
    dec_weights = f'{cur_dir_tmp}/csail_seg/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth'
    cfg.MODEL.weights_encoder = enc_weights
    cfg.MODEL.weights_decoder = dec_weights

    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p

    def __init__(self, device, model_dir=''):

        if model_dir!='':
            enc_weights = f'{model_dir}/encoder_epoch_20.pth'
            dec_weights = f'{model_dir}/decoder_epoch_20.pth'
            cfg.MODEL.weights_encoder = enc_weights
            cfg.MODEL.weights_decoder = dec_weights

        self.device = device
        self.net_encoder = ModelBuilder.build_encoder(
            arch=cfg.MODEL.arch_encoder,
            fc_dim=cfg.MODEL.fc_dim,
            weights=cfg.MODEL.weights_encoder)
        self.net_decoder = ModelBuilder.build_decoder(
            arch=cfg.MODEL.arch_decoder,
            fc_dim=cfg.MODEL.fc_dim,
            num_class=cfg.DATASET.num_class,
            weights=cfg.MODEL.weights_decoder,
            use_softmax=True)

        self.net_encoder.eval().to(self.device)
        self.net_decoder.eval().to(self.device)

    def get_mask(self,pic):

        ori_height = pic.size[0]
        ori_width = pic.size[1]
        print('Calculating mask for image')
        img_resized_list = []
        with torch.no_grad():
            for this_short_size in cfg.DATASET.imgSizes:
                # calculate target height and width
                scale = min(this_short_size / float(min(ori_height, ori_width)),
                            cfg.DATASET.imgMaxSize / float(max(ori_height, ori_width)))
                target_height, target_width = int(ori_height * scale), int(ori_width * scale)

                # to avoid rounding in network
                target_width = self.round2nearest_multiple(target_width, cfg.DATASET.padding_constant)
                target_height = self.round2nearest_multiple(target_height, cfg.DATASET.padding_constant)

                # resize and normalize images
                val_transform = transforms.Compose(
                    [transforms.Resize((target_width, target_height)), transforms.ToTensor(), normalize])

                # image transform, to torch float tensor 3xHxW
                img_resized = val_transform(pic)
                img_resized = torch.unsqueeze(img_resized.to(self.device), 0)
                img_resized_list.append(img_resized)

            scores = torch.zeros(img_resized_list[0].shape[0], cfg.DATASET.num_class, ori_width, ori_height).to(self.device)

            for img in img_resized_list:
                # forward pass
                pred_tmp = self.net_decoder(self.net_encoder(img, return_feature_maps=True),
                                            segSize=(ori_width, ori_height))
                scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)

        _, segm_pred = torch.max(scores, dim=1)

        return segm_pred.cpu()

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image): Image with building and surrounds
        Returns:
            Tensor: Cut out building with everything else black
        """
        segm_pred = self.get_mask(pic)

        image = np.array(pic)

        mask = np.array(segm_pred.cpu()).squeeze()
        # Filter building labels which are 25 and 1
        mask[np.where((mask != 25) & (mask != 1))] = 0
        image[mask == 0, :] = 0

        image = Image.fromarray(np.uint8(image))

        return image

    def __repr__(self):
        return self.__class__.__name__ + '()'
