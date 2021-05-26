# -*- coding: utf-8 -*-
import torch
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as T
from shapely import geometry
import argparse
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser('DeepLabv3-ResNet101-based facade parser model')
    parser.add_argument('--im_path', type=str, default="dataset/test/images",
                        help='Path for the building images')
    parser.add_argument('--model_path', type=str, default="models/facadeParser.pth",
                        help='Path for the pretrained segmentation model.' 
                             'Do NOT define this argument if the pretrained model bundled with the module will be used')
    parser.add_argument('--ref_file', type=str, default='refFile.csv',
                        help='CSV file containing the real world measurements of reference length and roof run.'
                             'This is a temporary argument that will be removed once the ray-casting method is debugged.')
    parser.add_argument('--gpu_enabled', type=boolean_string, default=True,
                        help='True if CUDA-compatible GPU available on the computing system.')
    parser.add_argument('--save_segimages', type=boolean_string, default=False,
                        help='True if segmentation masks need to be saved in PNG format.'
                             'False by default.')
    parser.add_argument('--segim_path', type=str, default="segmentedImages",
                        help='Path for the segmentation masks.'
                             'By default, masks are saved in the current directory under segmentedImages folder.')

    args = parser.parse_args()
    return args

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean entry')
    return s == 'True'


def install_default_model(model_path):
    if model_path == "models/facadeParser.pth":
        os.makedirs('models',exist_ok=True)
        model_path = os.path.join('models','facadeParser.pth')

        if not os.path.isfile(model_path):
            print('Loading default model file to the models folder...')
            torch.hub.download_url_to_file('https://zenodo.org/record/4784634/files/facadeParser.pth',
                                           model_path, progress=False)
            print('Default model loaded.')
    else:
        print(f'Inferences will be performed using the custom model at {model_path}.')
        
def gen_bbox(roofContour):   
    minx = min(roofContour[:,0])
    maxx = max(roofContour[:,0])
    miny = min(roofContour[:,1])
    maxy = max(roofContour[:,1])
    roofBBoxPoly = geometry.Polygon([(minx,miny),(minx,maxy),
                                     (maxx,maxy),(maxx,miny)])
    return roofBBoxPoly

def gen_bboxR0(roofBBoxPoly,facadeBBoxPoly):
    x,y = roofBBoxPoly.exterior.xy
    x = np.array(x).astype(int); y = np.array(y).astype(int)
    
    yBottom = max(y)
    ind = np.where(y==yBottom)
    xRoofBottom = np.unique(x[ind])
    yRoofBottom = np.tile(yBottom,len(xRoofBottom))
    
    x,y = facadeBBoxPoly.exterior.xy
    x = np.array(x).astype(int); y = np.array(y).astype(int)
    
    yBottom = max(y)
    ind = np.where(y==yBottom)
    xFacadeBottom = xRoofBottom
    yFacadeBottom = np.tile(yBottom,len(xFacadeBottom))
    
    R0BBoxPoly = geometry.Polygon([(xFacadeBottom[0],yFacadeBottom[0]),
                                   (xFacadeBottom[1],yFacadeBottom[1]),
                                   (xRoofBottom[1],yRoofBottom[1]),
                                   (xRoofBottom[0],yRoofBottom[0])])
    return R0BBoxPoly

def decode_segmap(image, nc=4):
    label_colors = np.array([(0, 0, 0),(255, 0, 0), (255, 165, 0), (0, 0, 255)])
                 # 0=background # 1=roof, 2=facade, 3=window

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
  
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    
    rgb = np.stack([r, g, b], axis=2)
    return rgb


def infer(opt):
    # Set the computing environment
    if opt.gpu_enabled:
        dev = 'cuda'
    else:
        dev = 'cpu'
    
    # Load the trained model and set the model to evaluate mode
    install_default_model(opt.model_path)
    model = torch.load(opt.model_path)
    model.eval()
    
    # Load the reference file
    df = pd.read_csv(opt.ref_file)
    
    # Create the output table
    dfOut = pd.DataFrame(columns=['img', 'R0', 'R1','pitch'])
    
    # Get the names of  images in im_path
    imgList = os.listdir(opt.im_path) 
    
    
    for ino in tqdm(range(len(imgList))):
        # Run image through the segmentation model
        imFile = imgList[ino]
        img = Image.open(os.path.join(opt.im_path,imFile))
        
        trf = T.Compose([T.Resize(640),
                         T.ToTensor(), 
                         T.Normalize(mean = [0.485, 0.456, 0.406], 
                                     std = [0.229, 0.224, 0.225])])
        
        inp = trf(img).unsqueeze(0).to(dev)
        scores = model.to(dev)(inp)['out']
        pred = torch.argmax(scores.squeeze(), dim=0).detach().cpu().numpy()
        
        # Extract component masks
        maskRoof = (pred==1).astype(np.uint8)
        maskFacade = (pred==2).astype(np.uint8)
        maskWin = (pred==3).astype(np.uint8)
        
        # Open and close masks
        kernel = np.ones((10,10), np.uint8)
        openedFacadeMask = cv2.morphologyEx(maskFacade, cv2.MORPH_OPEN, kernel)
        maskFacade = cv2.morphologyEx(openedFacadeMask, cv2.MORPH_CLOSE, kernel)
        openedWinMask = cv2.morphologyEx(maskWin, cv2.MORPH_OPEN, kernel)
        maskWin = cv2.morphologyEx(openedWinMask, cv2.MORPH_CLOSE, kernel)
        #plt.imshow(maskRoof)
        
        #plt.imshow(maskWin); plt.show()
        
        # Find roof contours
        contours, _ = cv2.findContours(maskRoof,cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        roofContour = max(contours, key = cv2.contourArea).squeeze()
        #plt.plot(roofContour[:,0],roofContour[:,1])
        
        # Find the mimumum area rectangle that fits around the primary roof contour
        roofMinRect = cv2.minAreaRect(roofContour)
        roofMinRect = cv2.boxPoints(roofMinRect)
        roofMinRect = np.int0(roofMinRect)
        roofMinRectPoly = geometry.Polygon([(roofMinRect[0,0],roofMinRect[0,1]),
                                            (roofMinRect[1,0],roofMinRect[1,1]),
                                            (roofMinRect[2,0],roofMinRect[2,1]),
                                            (roofMinRect[3,0],roofMinRect[3,1])])
        x,y = roofMinRectPoly.exterior.xy
        #plt.plot(x,y)
        
        roofBBoxPoly = gen_bbox(roofContour)
        x,y = roofBBoxPoly.exterior.xy
        roofPixHeight = max(y)-min(y)
        
        #plt.plot(x,y)
        #plt.show()
        
        # Find facade contours
        #plt.imshow(maskFacade)
        contours, _ = cv2.findContours(maskFacade,cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        facadeContour = max(contours, key = cv2.contourArea).squeeze()
        #plt.plot(facadeContour[:,0],facadeContour[:,1])
        
        facadeBBoxPoly = gen_bbox(facadeContour)
        x,y = facadeBBoxPoly.exterior.xy
        
        #plt.plot(x,y)
        
        R0BBoxPoly = gen_bboxR0(roofBBoxPoly,facadeBBoxPoly)
        x,y = R0BBoxPoly.exterior.xy
        R0PixHeight = max(y)-min(y)
        R1PixHeight = R0PixHeight + roofPixHeight
        refPixLength = max(x)-min(x)
        
        #plt.plot(x,y)
        #plt.show()
        
        # Extract reference measurements
        try:
            ref_length = df[df['image'].str.contains(imFile)].iloc[0]['refLength']
            roof_run = df[df['image'].str.contains(imFile)].iloc[0]['roofRun']
        except:
            print(f"Information for {imFile} could not be found in reference file.")
        
        # Calculate heigths of interest
        scale = ref_length/refPixLength
        R0 = R0PixHeight*scale
        R1 = R1PixHeight*scale
        roofPitch = (R1-R0)/roof_run
        dfOut.loc[ino] = [imFile, R0, R1, roofPitch]
        
        # Save segmented images
        if opt.save_segimages:
            rgb = decode_segmap(pred)
            #plt.imshow(rgb); plt.show()
            
            os.makedirs('segmentedImages',exist_ok=True)
            fname = os.path.splitext(imgList[ino])[0]
            fext = os.path.splitext(imgList[ino])[-1]
            
            rgb = Image.fromarray(rgb)
            rgb.save(os.path.join(opt.segim_path,fname + '_segmented' + fext))
    
    
    dfOut.to_csv('facadeParsingResults.csv')
    
    # Unload the model from GPU
    if opt.gpu_enabled:
        del model
        torch.cuda.empty_cache()
        
if __name__ == '__main__':
    opt = get_args()
    infer(opt)