# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 The Regents of the University of California
#
# This file is part of BRAILS.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You should have received a copy of the BSD 3-Clause License along with
# BRAILS. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Barbaros Cetiner
#
# Last updated:
# 05-19-2022

import math
import torch
import cv2
import os
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as T
from shapely.geometry import Polygon
from tqdm import tqdm
from random import random

class FacadeParser:
    
    def __init__(self):                
        self.footprints = []
        self.streetScales = []
        self.street_images = []        
        self.model_path = []

    def predict(self,imhandler,storymodel,model_path='tmp/models/facadeParser.pth',
                save_segimages=False):
        self.footprints = imhandler.footprints[:]
        self.streetScales = imhandler.streetScales[:]
        self.street_images = imhandler.street_images[:]        
        self.model_path = model_path
        
        def compute_roofrun(footprint):
            # Find the mimumum area rectangle that fits around the footprint
            # What failed: 1) PCA, 2) Rotated minimum area rectangle 
            # 3) Modified minimum area rectangle
            # Current implementation: Tight-fit bounding box
            
            # Convert lat/lon values to Cartesian coordinates for better coordinate resolution:
            
            xyFootprint = np.zeros((len(footprint),2))
            for k in range(1,len(footprint)):
                lon1 = footprint[0,0]
                lon2 = footprint[k,0]
                lat1 = footprint[0,1]
                lat2 = footprint[k,1]
                
                xyFootprint[k,0] = (lon1-lon2)*40075000*3.28084*math.cos((lat1+lat2)*math.pi/360)/360
                xyFootprint[k,1] = (lat1-lat2)*40075000*3.28084/360
                
            # Calculate the slope and length of each line segment comprising the footprint polygon:
            slopeSeg = np.diff(xyFootprint[:,1])/np.diff(xyFootprint[:,0])
            
            segments = np.arange(len(slopeSeg))
            lengthSeg = np.zeros(len(slopeSeg))
            for k in range(len(xyFootprint)-1):
                p1 = np.array([xyFootprint[k,0],xyFootprint[k,1]])
                p2 = np.array([xyFootprint[k+1,0],xyFootprint[k+1,1]])
                lengthSeg[k] = np.linalg.norm(p2-p1)
            
            # Cluster the line segments based on their slope values:
            slopeClusters = []
            totalLengthClusters = []
            segmentsClusters = []
            while slopeSeg.size>1:
                ind = np.argwhere(abs((slopeSeg-slopeSeg[0])/slopeSeg[0])<0.3).squeeze()
                slopeClusters.append(np.mean(slopeSeg[ind]))
                totalLengthClusters.append(np.sum(lengthSeg[ind]))
                segmentsClusters.append(segments[ind])
                
                indNot = np.argwhere(abs((slopeSeg-slopeSeg[0])/slopeSeg[0])>=0.3).squeeze()
                slopeSeg = slopeSeg[indNot]
                lengthSeg = lengthSeg[indNot]
                segments = segments[indNot]
            
            if slopeSeg.size==1:
                slopeClusters.append(slopeSeg)
                totalLengthClusters.append(lengthSeg)
                segmentsClusters.append(segments)
            
            # Mark the two clusters with the highest total segment lengths as the 
            # principal directions of the footprint
            principalDirSlopes = []
            principalDirSegments = []
            counter = 0
            for ind in np.flip(np.argsort(totalLengthClusters)):
                if type(segmentsClusters[ind]) is np.ndarray:
                    principalDirSlopes.append(slopeClusters[ind])
                    principalDirSegments.append(segmentsClusters[ind])
                    counter+=1
                if counter==2:
                    break

            xFootprint = xyFootprint[:,0]
            yFootprint = xyFootprint[:,1]
            slopeSeg = np.diff(xyFootprint[:,1])/np.diff(xyFootprint[:,0])
            
            bndLines = np.zeros((4,4))
            for cno,cluster in enumerate(principalDirSegments):
                xp = np.zeros((2*len(cluster)))
                yp = np.zeros((2*len(cluster)))
                for idx, segment in enumerate(cluster):
                    angle = math.pi/2 - math.atan(slopeSeg[segment])
                    x = xFootprint[segment:segment+2]
                    y = yFootprint[segment:segment+2]
                    xp[2*idx:2*idx+2] = x*math.cos(angle) - y*math.sin(angle)
                    yp[2*idx:2*idx+2] = x*math.sin(angle) + y*math.cos(angle)
                
                minLineIdx = int(np.argmin(xp)/2)
                maxLineIdx = int(np.argmax(xp)/2)
                
                
                minLineIdx = cluster[int(np.argmin(xp)/2)]
                maxLineIdx = cluster[int(np.argmax(xp)/2)]
                
                bndLines[2*cno:2*cno+2,:] = np.array([[xFootprint[minLineIdx],
                                                       yFootprint[minLineIdx],
                                                       xFootprint[minLineIdx+1],
                                                       yFootprint[minLineIdx+1]],
                                                      [xFootprint[maxLineIdx],
                                                       yFootprint[maxLineIdx],
                                                       xFootprint[maxLineIdx+1],
                                                       yFootprint[maxLineIdx+1]]])
            bbox = np.zeros((5,2))
            counter = 0
            for k in range(2):
                line1 = bndLines[k,:]
                x1 = line1[0]
                x2 = line1[2]
                y1 = line1[1]
                y2 = line1[3]
                for m in range(2,4):
                    line2 = bndLines[m,:]
                    x3 = line2[0]
                    x4 = line2[2]
                    y3 = line2[1]
                    y4 = line2[3]
                    d = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
                    bbox[counter,:] = [((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4))/d, 
                                       ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4))/d]
                    counter += 1
            bbox[4,:] = bbox[0,:]
            bbox[2:4,:] = np.flipud(bbox[2:4,:])
            
            sideLengths = np.linalg.norm(np.diff(bbox,axis=0),axis=1)
            roof_run = min(sideLengths)
            
            return roof_run
        
        def install_default_model(model_path='tmp/models/facadeParser.pth'):
            if model_path == "tmp/models/facadeParser.pth":
                os.makedirs('tmp/models',exist_ok=True)
                model_path = 'tmp/models/facadeParser.pth'
        
                if not os.path.isfile(model_path):
                    print('Loading default facade parser model file to tmp/models folder...')
                    torch.hub.download_url_to_file('https://zenodo.org/record/5809365/files/facadeParser.pth',
                                                   model_path, progress=False)
                    print('Default facade parser model loaded')
                else: 
                    print(f"Default facade parser model at {model_path} loaded")
            else:
                print(f'Inferences will be performed using the custom model at {model_path}')
                
        def gen_bbox(roofContour):   
            minx = min(roofContour[:,0])
            maxx = max(roofContour[:,0])
            miny = min(roofContour[:,1])
            maxy = max(roofContour[:,1])
            roofBBoxPoly = Polygon([(minx,miny),(minx,maxy),(maxx,maxy),(maxx,miny)])
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
            
            R0BBoxPoly = Polygon([(xFacadeBottom[0],yFacadeBottom[0]),
                                  (xFacadeBottom[1],yFacadeBottom[1]),
                                  (xRoofBottom[1],yRoofBottom[1]),
                                  (xRoofBottom[0],yRoofBottom[0])])
            return R0BBoxPoly
        
        def decode_segmap(image, nc=5):
            label_colors = np.array([(0, 0, 0),(255, 0, 0), (255, 165, 0), (0, 0, 255),
                                     (175,238,238)])
                         # 0=background # 1=roof, 2=facade, 3=window, 4=door
        
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
             
        # Set the computing environment
        if torch.cuda.is_available():
            dev = 'cuda'
        else:
            dev = 'cpu'
       
        # Load the trained model and set the model to evaluate mode
        print('\nDetermining the heights and roof pitch for each building...')
        install_default_model(self.model_path)
        model = torch.load(self.model_path,map_location=torch.device(dev))
        model.eval()
        
        # Create the output table
        self.predictions = pd.DataFrame(columns=['image',
                                                 'roofeaveheight',
                                                 'buildingheight',
                                                 'roofpitch'])
        
        for polyNo in tqdm(range(len(self.footprints))):
            if self.footprints[polyNo] is None:
                    continue
            
            # Extract building footprint information and download the image viewing
            # this footprint:
            footprint = np.fliplr(np.squeeze(np.array(self.footprints[polyNo])))                        

            if self.streetScales[polyNo] is None:
                    continue            


            # Run image through the segmentation model
            img = Image.open(self.street_images[polyNo])
            
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
            #maskDoor = (pred==4).astype(np.uint8)
            
            # Open and close masks
            kernel = np.ones((10,10), np.uint8)
            openedFacadeMask = cv2.morphologyEx(maskFacade, cv2.MORPH_OPEN, kernel)
            maskFacade = cv2.morphologyEx(openedFacadeMask, cv2.MORPH_CLOSE, kernel)
            openedWinMask = cv2.morphologyEx(maskWin, cv2.MORPH_OPEN, kernel)
            maskWin = cv2.morphologyEx(openedWinMask, cv2.MORPH_CLOSE, kernel)

            
            
            # Find roof contours
            contours, _ = cv2.findContours(maskRoof,cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours)!=0:
                roofContour = max(contours, key = cv2.contourArea).squeeze()
                
                if roofContour.ndim==2:
                    # Find the mimumum area rectangle that fits around the primary roof contour
                    roofMinRect = cv2.minAreaRect(roofContour)
                    roofMinRect = cv2.boxPoints(roofMinRect)
                    roofMinRect = np.int0(roofMinRect)
                    roofMinRectPoly = Polygon([(roofMinRect[0,0],roofMinRect[0,1]),
                                                        (roofMinRect[1,0],roofMinRect[1,1]),
                                                        (roofMinRect[2,0],roofMinRect[2,1]),
                                                        (roofMinRect[3,0],roofMinRect[3,1])])
                    x,y = roofMinRectPoly.exterior.xy
            
                    
                    roofBBoxPoly = gen_bbox(roofContour)
                    x,y = roofBBoxPoly.exterior.xy
                    roofPixHeight = max(y)-min(y)
                else:
                    roofPixHeight = 0
            else:
                roofPixHeight = 0
            
            # Find facade contours
            contours, _ = cv2.findContours(maskFacade,cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            try:
                facadeContour = max(contours, key = cv2.contourArea).squeeze()
            except:
                continue
            
            facadeBBoxPoly = gen_bbox(facadeContour)
            x,y = facadeBBoxPoly.exterior.xy

            
            R0BBoxPoly = gen_bboxR0(roofBBoxPoly,facadeBBoxPoly)
            x,y = R0BBoxPoly.exterior.xy
            R0PixHeight = max(y)-min(y)
            R1PixHeight = R0PixHeight + roofPixHeight
        
            
            # Calculate heigths of interest
            ind = storymodel.system_dict['infer']['images'].index(self.street_images[polyNo])
            normalizer = storymodel.system_dict['infer']['predictions'][ind]*14*(1.3-random()*0.4)
            R0 = R0PixHeight*self.streetScales[polyNo]
            if R0>normalizer:
                normfact = normalizer/R0
                R0 = normalizer
            elif R0<0.8*normalizer:
                normfact = normalizer/R0
                R0 = normalizer            
            else:
                normfact = 1
            R1 = R1PixHeight*self.streetScales[polyNo]*normfact
            if (R1-R0)>1.5*normalizer:
                R1 = R0 + (1.2-random()*0.4)*normalizer
                
            roof_run = compute_roofrun(footprint)
            roofPitch = (R1-R0)/roof_run
            self.predictions.loc[polyNo] = [self.street_images[polyNo], 
                                            R0, R1, roofPitch]
        
            # Save segmented images
            if save_segimages:
                rgb = decode_segmap(pred)
                
                rgb = Image.fromarray(rgb)
                rgb.save(self.street_images[polyNo].split('.')[0] + 
                         '_segmented.png')
            
        self.predictions = self.predictions.round({'roofeaveheight': 1, 
                                'buildingheight': 1,
                                'roofpitch': 2})
        
        # Unload the model from GPU
        if torch.cuda.is_available():
            del model
            torch.cuda.empty_cache()