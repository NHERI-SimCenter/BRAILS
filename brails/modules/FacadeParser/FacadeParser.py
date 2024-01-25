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
# 01-09-2024

import math
import torch
import cv2
import os
import pandas as pd
import numpy as np
import base64
import struct
import torchvision.transforms as T

from PIL import Image
from shapely.geometry import Polygon
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from copy import deepcopy

class FacadeParser:
    
    def __init__(self):                
        self.cam_elevs = []
        self.depthmaps = []
        self.footprints = []
        self.model_path = []        
        self.street_images = []        

    def predict(self,imhandler,model_path='tmp/models/facadeParser.pth',
                save_segimages=False):
        self.cam_elevs = imhandler.cam_elevs[:]
        self.depthmaps = imhandler.depthmaps[:]
        self.footprints = imhandler.footprints[:]
        self.model_path = model_path
        self.street_images = imhandler.street_images[:]        
        
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
        
        def get_bin(a):
            ba = bin(a)[2:]
            return "0"*(8 - len(ba)) + ba

        def getUInt16(arr, ind):
            a = arr[ind]
            b = arr[ind + 1]
            return int(get_bin(b) + get_bin(a), 2)

        def getFloat32(arr, ind):
            return bin_to_float("".join(get_bin(i) for i in arr[ind : ind + 4][::-1]))

        def bin_to_float(binary):
            return struct.unpack("!f", struct.pack("!I", int(binary, 2)))[0]

        def parse_dmap_str(b64_string):
            # Ensure correct padding (The length of string needs to be divisible by 4):
            b64_string += "="*((4 - len(b64_string)%4)%4)

            # Convert the URL safe format to regular format:
            data = b64_string.replace("-", "+").replace("_", "/")
            
            # Decode the string:
            data = base64.b64decode(data)  
            
            return np.array([d for d in data])

        def parse_dmap_header(depthMap):
            return {
                "headerSize": depthMap[0],
                "numberOfPlanes": getUInt16(depthMap, 1),
                "width": getUInt16(depthMap, 3),
                "height": getUInt16(depthMap, 5),
                "offset": getUInt16(depthMap, 7),
            }

        def parse_dmap_planes(header, depthMap):
            indices = []
            planes = []
            n = [0, 0, 0]

            for i in range(header["width"] * header["height"]):
                indices.append(depthMap[header["offset"] + i])

            for i in range(header["numberOfPlanes"]):
                byteOffset = header["offset"] + header["width"]*header["height"] + i*4*4
                n = [0, 0, 0]
                n[0] = getFloat32(depthMap, byteOffset)
                n[1] = getFloat32(depthMap, byteOffset + 4)
                n[2] = getFloat32(depthMap, byteOffset + 8)
                d = getFloat32(depthMap, byteOffset + 12)
                planes.append({"n": n, "d": d})

            return {"planes": planes, "indices": indices}

        def compute_dmap(header, indices, planes):
            v = [0, 0, 0]
            w = header["width"]
            h = header["height"]

            depthMap = np.empty(w * h)

            sin_theta = np.empty(h)
            cos_theta = np.empty(h)
            sin_phi = np.empty(w)
            cos_phi = np.empty(w)

            for y in range(h):
                theta = (h - y - 0.5)/h*np.pi
                sin_theta[y] = np.sin(theta)
                cos_theta[y] = np.cos(theta)

            for x in range(w):
                phi = (w - x - 0.5)/w*2*np.pi + np.pi/2
                sin_phi[x] = np.sin(phi)
                cos_phi[x] = np.cos(phi)

            for y in range(h):
                for x in range(w):
                    planeIdx = indices[y*w + x]

                    v[0] = sin_theta[y]*cos_phi[x]
                    v[1] = sin_theta[y]*sin_phi[x]
                    v[2] = cos_theta[y]

                    if planeIdx > 0:
                        plane = planes[planeIdx]
                        t = np.abs(
                            plane["d"]
                            / (
                                v[0]*plane["n"][0]
                                + v[1]*plane["n"][1]
                                + v[2]*plane["n"][2]
                            )
                        )
                        depthMap[y*w + (w - x - 1)] = t
                    else:
                        depthMap[y*w + (w - x - 1)] = 9999999999999999999.0
            return {"width": w, "height": h, "depthMap": depthMap}

        def get_depth_map(depthfile, imsize, bndangles):         
            # Decode depth map string:
            with open(depthfile,'r') as fout:
                depthMapStr = fout.read()  
            depthMapData = parse_dmap_str(depthMapStr)
            
            # Parse first bytes to get the data headers:
            header = parse_dmap_header(depthMapData)
            
            # Parse remaining bytes into planes of float values:
            data = parse_dmap_planes(header, depthMapData)
            
            # Compute position and depth values of pixels:
            depthMap = compute_dmap(header, data["indices"], data["planes"])
            
            # Process float 1D array into integer 2D array with pixel values ranging 
            # from 0 to 255:
            im = depthMap["depthMap"]
            im[np.where(im == max(im))[0]] = 255
            if min(im) < 0:
                im[np.where(im < 0)[0]] = 0
            im = im.reshape((depthMap["height"], depthMap["width"]))
            
            # Flip the 2D array to have it line up with pano image pixels:
            im = np.fliplr(im)
            
            # Read the 2D array into an image and resize this image to match the size 
            # of pano:
            imPanoDmap = Image.fromarray(im)
            imPanoDmap = imPanoDmap.resize(imsize)
            
            # Crop the depthmap such that it includes the building of interest only:
            imBldgDmap = imPanoDmap.crop((bndangles[0],0,bndangles[1],imsize[1]))     
            return imBldgDmap
             
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
            
            # Extract building footprint information :
            footprint = np.fliplr(np.squeeze(np.array(self.footprints[polyNo])))                        

            # Run building image through the segmentation model:
            try:
                img = Image.open(self.street_images[polyNo])
            except:
                self.predictions.loc[polyNo] = [self.street_images[polyNo], 
                                                None, None, None]
                continue
            
            imsize = img.size
            
            trf = T.Compose([T.Resize(round(1000/max(imsize)*min(imsize))),
                             T.ToTensor(), 
                             T.Normalize(mean = [0.485, 0.456, 0.406], 
                                         std = [0.229, 0.224, 0.225])]) #
            
            inp = trf(img).unsqueeze(0).to(dev)
            scores = model.to(dev)(inp)['out']
            predraw = torch.argmax(scores.squeeze(), dim=0).detach().cpu().numpy()
            pred = np.array(Image.fromarray(np.uint8(predraw)).resize(imsize))
            
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
        
            # Get the raw depthmap for the building and crop it so that it covers 
            # just the segmentation mask of the bbox for the building facade:
            (depthfile, imsize, bndangles) = self.depthmaps[polyNo]
            depthmap = get_depth_map(depthfile, imsize, bndangles)
            depthmapbbox = depthmap.crop((min(x),min(y),max(x),max(y)))
            
            # Convert the depthmap to a Numpy array for further processing and
            # sample the raw depthmap at its vertical centerline:
            depthmapbbox_arr = np.asarray(depthmapbbox)
            depthmap_cl = depthmapbbox_arr[:,round(depthmapbbox.size[0]/2)]
            
            # Calculate the vertical camera angles corresponding to the bottom
            # and top of the facade bounding box:
            imHeight = img.size[1]
            angleTop = ((imHeight/2 - min(y))/(imHeight/2))*math.pi/2
            angleBottom = ((imHeight/2 - max(y))/(imHeight/2))*math.pi/2
            
            # Take the first derivative of the depthmap with respect to vertical
            # pixel location and identify the depthmap discontinuity (break) 
            # locations:
            break_pts = [0]
            boolval_prev = True
            depthmap_cl_dx = np.append(abs(np.diff(depthmap_cl))<0.1,True)
            for (counter, boolval_curr) in enumerate(depthmap_cl_dx):
                if (boolval_prev==True and boolval_curr==False) or (boolval_prev==False and boolval_curr==True):
                    break_pts.append(counter)
                boolval_prev = boolval_curr
            break_pts.append(counter)
            
            # Identify the depthmap segments to keep for extrapolation, i.e., 
            # segments that are not discontinuities in the depthmap: 
            segments_keep = []
            for i in range(len(break_pts)-1):
                if all(depthmap_cl_dx[break_pts[i]:break_pts[i+1]]) and all(depthmap_cl[break_pts[i]:break_pts[i+1]]!=255):
                   segments_keep.append((break_pts[i],break_pts[i+1]))
            
            # Fit line models to individual (kept) segments of the depthmap and
            # determine the model that results in the smallest residual for 
            # all kept depthmap points:
            lm = LinearRegression(fit_intercept = True)
            x = np.arange(depthmapbbox.size[1])
            xKeep = np.hstack([x[segment[0]:segment[1]] for segment in segments_keep])
            yKeep = np.hstack([depthmap_cl[segment[0]:segment[1]] for segment in segments_keep])
            residualprev = 1e10
            model_lm = deepcopy(lm)
            for segment in segments_keep:
                xvect = x[segment[0]:segment[1]]
                yvect = depthmap_cl[segment[0]:segment[1]]
                
                # Fit model:
                lm.fit(xvect.reshape(-1, 1),yvect)
                preds = lm.predict(xKeep.reshape(-1,1))
                residual = np.sum(np.square(yKeep-preds))
                if residual<residualprev:
                    model_lm = deepcopy(lm)                
                residualprev = residual
            
            # Extrapolate depthmap using the best-fit model:
            depthmap_cl_depths = model_lm.predict(x.reshape(-1,1))
            
            # Calculate heigths of interest:
            R0 = (depthmap_cl_depths[0]*math.sin(angleTop) 
                      - depthmap_cl_depths[-1]*math.sin(angleBottom))*3.28084
            scale = R0/R0PixHeight
            R1 = R1PixHeight*scale                
        
            # Calculate roof pitch:            
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