import argparse
import os
import sys

import torch
from PIL import Image


from csail_segmentation_tool.csail_segmentation import MaskBuilding
import numpy as np

parser = argparse.ArgumentParser(description='Detect Foundataion Type')

parser.add_argument('--image-path',help='Path to one image or a folder containing images.',required=True)
parser.add_argument('--only-cpu', action='store_true')
args = parser.parse_args()

if args.only_cpu:
    device='cpu'
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(folder):
    if not args.only_cpu:
        print ('Will try to use available GPU.')
    mask_module = MaskBuilding(device)

    for root, _, fnames in sorted(os.walk(folder, followlinks=True)):
        for fname in sorted(fnames):

            if 'jpg' in fname or 'png' in fname:
                if 'mask' in fname:
                    print('Skipping {} because it has mask in filename. Presumably this is a mask already'.format(fname))
                    continue
                img_path = os.path.join(root, fname)

                _, file_extension = os.path.splitext(img_path)
                mask_filename = fname.replace(file_extension,'-mask.png')
                mask_path = os.path.join(root, mask_filename)
                if os.path.isfile(mask_path):
                    print('Skipping {} because mask for this file exists.'.format(fname))
                    continue

                try:
                    # Save the mask at the same name but add a '-mask suffix
                    with torch.no_grad():
                        mask = mask_module.get_mask(Image.open(img_path)).detach().cpu().squeeze().numpy()
                        mask_image = Image.fromarray(np.uint8(mask))
                        mask_image.save(mask_path)
                except RuntimeError as er:
                    if 'CUDA' in str(er):
                        print ('Error using GPU. If GPU exists maybe there is not enough GPU RAM. Try --only-cpu flag!')
                        sys.exit(1)
                    else:
                        print('Failed with unknown error: {}'.format(er))
                        sys.exit(1)


if __name__ =='__main__':
    main(args.image_path)
