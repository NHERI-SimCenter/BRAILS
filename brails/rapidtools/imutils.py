# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 The Regents of the University of California
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
# 07-23-2024

import rasterio
import rasterio.warp
from rasterio.crs import CRS
from rasterio.windows import Window
from brails.workflow.FootprintHandler import FootprintHandler
from shapely.geometry import box, Polygon
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

from PIL.ExifTags import TAGS
from brails.utils.geoTools import haversine_dist
import math


import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class imutils:
    def __init__(self): 
        self.footprints = []
        self.aerialImageList = []
        self.centroids = []
        
    def aerial_image_extractor(self,rasterMosaicFile,fpData='osm'):
        def orgcrs2wgs84_bbox(bbox_xy,orgcrs):
            # Project the feature to the desired CRS
            feature_proj = rasterio.warp.transform_geom(
                orgcrs,
                CRS.from_epsg(4326),
                bbox_xy
            ) 
            bbox_lonlat = feature_proj['coordinates'][0]
            return (bbox_lonlat[0][0],bbox_lonlat[0][1],bbox_lonlat[2][0],bbox_lonlat[2][1])
        
        def wgs842orgcrs(lonlat,orgcrs):
            # In GeoJSON format
            feature = {
                "type": "Point",
                "coordinates": lonlat
            }
            
            # Project the feature to the desired CRS
            feature_proj = rasterio.warp.transform_geom(
                CRS.from_epsg(4326),
                orgcrs,
                feature
            )    
            return feature_proj['coordinates']
        
        def get_image_for_fp(dataset,fp):
            fp_cent = Polygon(fp).centroid
            imname = f'images/aerial/imaerial_{fp_cent.y:.8f}{fp_cent.x:.8f}.jpg'
            fp_xy = []
            for vert in fp:
                fp_xy.append(wgs842orgcrs(vert,dataset.crs))
            poly_xy = Polygon(fp_xy).bounds
            bufferdist = max(abs(poly_xy[0]-poly_xy[2]),abs(poly_xy[1]-poly_xy[3]))*0.2
            poly_xy = Polygon(fp_xy).buffer(bufferdist).bounds
            row1, col1 = dataset.index(poly_xy[0],poly_xy[1])
            row2, col2 = dataset.index(poly_xy[2],poly_xy[3])
            ysize = max(row1,row2)-min(row1,row2)
            xsize = max(col1,col2)-min(col1,col2)
            
            # Create a Window and calculate the transform from the source dataset    
            window = Window(min(col1,col2),
                            min(row1,row2),
                            xsize,
                            ysize,
                            )
        
            imarray = dataset.read(window=window)
            unique, counts = np.unique(imarray, return_counts=True)
            try:
                zerocount = dict(zip(unique, counts))[0]
            except:
                zerocount = 0
            if (zerocount/imarray.size)<0.5:
                results = (fp,imname,[fp_cent.x,fp_cent.y])
                imout =  np.moveaxis(np.moveaxis(imarray, -1, 0),-1,1)
                imout = Image.fromarray(imout[:,:,:3])
                imout.save(imname)
            else:
                results = None
            return results
        
        # Read the aerial imagery GeoTIFF into a dataset object:
        dataset = rasterio.open(rasterMosaicFile, driver='GTiff', num_threads='all_cpus')
        
        # Compute the coordinates of the bounding box for the dataset
        bbox_wgs84 = orgcrs2wgs84_bbox(box(*dataset.bounds),dataset.crs)
        
        # Get the footprints contained in the determined bounding box:
        fpHandler = FootprintHandler()
        fpHandler.fetch_footprint_data(bbox_wgs84,fpSource=fpData)
        
        # Create the folder to extract the aerial imagery:
        os.makedirs('images',exist_ok=True)
        os.makedirs('images/aerial',exist_ok=True)
        
        # Extract building-wise aerial imagery:
        footprints = fpHandler.footprints
        print('\n')
        for fp in tqdm(footprints, desc='Extracting aerial imagery...'):
           res = get_image_for_fp(dataset,fp) 
           if res is not None:
               (fp,imout,fp_cent) = res
               self.footprints.append(fp)
               self.aerialImageList.append(imout)
               self.centroids.append(fp_cent)
        
        # Report the total number of images extracted and their directory:    
        print(f'\nExtracted aerial imagery for a total of {len(self.footprints)} buildings.')
        print(f'You can access the extracted images at {os.getcwd()}/images/aerial')
        
    def pulsar_image_extractor(self,imdir,fpData='usastr'):  
        def get_image_data(image_path):
            image = Image.open(image_path)
            exif_data = image._getexif()
            if exif_data is not None:
                for tag, value in exif_data.items():
                    tag_name = TAGS.get(tag, tag)
                    if tag_name=='GPSInfo':
                        lat = float(value[2][0] + value[2][1]/60 + value[2][2]/3600)
                        if value[1]=='S':
                            lat = -lat
                        
                        lon = float(value[4][0] + value[4][1]/60 + value[4][2]/3600)
                        if value[3]=='W':
                            lon = -lon
                        
                        heading = value[17]
         
            return (lat, lon, heading)

        def get_capture_limits(im_coords, buffer=50):
            lats = []
            lons = []
            for coord in im_coords:
                lats.append(coord[0])
                lons.append(coord[1])
            bbox = (min(lons),min(lats),max(lons),max(lats))
            
            # Conversion factor for 50 meters in degrees (approximation for latitude)
            meters_to_degrees = 1 / 111111
            
            # Calculate expanded bounding box
            expansion_distance_meters = buffer
            expansion_distance_degrees = expansion_distance_meters * meters_to_degrees
            
            expanded_bbox = (
                bbox[0] - expansion_distance_degrees,  # new xmin
                bbox[1] - expansion_distance_degrees,  # new ymin
                bbox[2] + expansion_distance_degrees,  # new xmax
                bbox[3] + expansion_distance_degrees   # new ymax
            )
            return expanded_bbox

        def get_3pt_angle(a, b, c):
            ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
            return ang + 360 if ang < 0 else ang

        def get_angle_from_heading(coord,heading):
            # Determine the cartesian coordinates of a point along the heading that
            # is 100 ft away from the origin:
            x0 = 100*math.sin(math.radians(heading))
            y0 = 100*math.cos(math.radians(heading))
            
            # Calculate the clockwise viewing angle for each coord with respect to the 
            # heading:
            ang = 360 - get_3pt_angle((x0,y0), (0,0), coord)
            
            # Return viewing angles such that anything to the left of the vertical 
            # camera axis is negative and counterclockwise angle measurement:
            return ang if ang<=180 else ang-360

        def get_bndangle(fp, cam_coord, cam_heading, imsize):
            (lat0,lon0) = cam_coord
            xy = []
            for vert in fp:
                lon1 = vert[0]
                lat1 = vert[1]
                x = (lon1 - lon0)*40075000*3.28084*math.cos((lat0 + lat1)*math.pi/360)/360
                y = (lat1 - lat0)*40075000*3.28084/360
                xy.append((x,y))
            
            # Calculate the theoretical viewing angle for each footprint vertex with 
            # respect to the camera heading angle
            camera_angles = []
            for coord in xy:
                camera_angles.append(get_angle_from_heading(coord,cam_heading))
            
            # Calculate the viewing angle values that encompass the building buffered 
            # 10 degrees in horizontal direction:
            bndAngles = np.rint((np.array([round(min(camera_angles),-1)-10,
                                           round(max(camera_angles),-1)+10]) + 180)/360*imsize[0])
            return bndAngles


        imfiles = os.listdir(imdir)
        imfiles = [imdir+im for im in imfiles if im.endswith('.jpg')]

        im_coords = []
        im_headings = []
        for im in imfiles:
            data = get_image_data(im)
            im_coords.append([data[0],data[1]])
            im_headings.append(data[2])

        bbox = get_capture_limits(im_coords)

        fpHandler = FootprintHandler()
        fpHandler.fetch_footprint_data(bbox, fpSource = 'usastr', lengthUnit = 'm', 
                                       outputFile = 'footprints.geojson')

        immatches = {}
        print('\nPairing footprint data to street-level imagery...')
        for fpind, fp in tqdm(enumerate(fpHandler.centroids)):
            cent = [fp.y, fp.x]
            keep = []
            for ind_im, coord in enumerate(im_coords):
                dist = haversine_dist(cent,coord)*0.3048 
                if dist < 50:
                    keep.append((ind_im,dist))
            
            dist = 1000
            best_im = None
            for ind_im, dist_im in keep:
                if dist_im<dist:
                    dist = dist_im
                    best_im = ind_im
            
            if best_im:
                try: 
                    immatches[best_im].append(fpind)
                except:
                    immatches[best_im] = [fpind]
        print('Pairing complete')


        print('\nGenerating building-specific imagery...')
        os.makedirs('parsed_imagery',exist_ok=True)
        imcounter = 0
        for im in tqdm(immatches.keys()):
            imfile = imfiles[int(im)]
            fpmatches = immatches[im]  
            cam_coord = im_coords[im]
            cam_heading = im_headings[im]
            image = Image.open(imfile)
            imsize = image.size
            split_point = imsize[0] // 4
            left_piece = image.crop((0, 0, split_point, imsize[1]*0.70))
            right_piece = image.crop((split_point, 0, imsize[0], imsize[1]*0.7))

            # Create a new blank image
            new_image = Image.new('RGB', imsize)
            
            # Paste images into the new image
            new_image.paste(right_piece, (0, 0))
            new_image.paste(left_piece, (imsize[0]-split_point, 0))

            for fpind in fpmatches:
                fp = fpHandler.footprints[fpind]
                bnd_angles = get_bndangle(fp, cam_coord, cam_heading, imsize)   
                
                bldg_im = new_image.crop((bnd_angles[0],0,bnd_angles[1],imsize[1]))
                bldg_im.save(f'parsed_imagery/{imcounter}.jpg')
                imcounter+=1
                
        print(f'Building-specific imagery processed and saved in {os.getcwd()}/parsed_imagery')

        model_id = "IDEA-Research/grounding-dino-base"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

        text = "a building. a car. a tree. a trash can"

        print('\nCleaning poor visibility images & improving image focus...')
        os.makedirs('filtered_imagery/',exist_ok=True)
        parsed_ims = os.listdir('parsed_imagery/')
        for im in tqdm(parsed_ims):
            image = Image.open('parsed_imagery/' + im)
            inputs = processor(images=image, text=text, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.2,
                text_threshold=0.25,
                target_sizes=[image.size[::-1]]
            )
            
            try:
                bldgind = [ind for ind, res in enumerate(results[0]['labels'])if 'building' in res]        
                indbest = np.argmax(results[0]['scores'].cpu().numpy()[bldgind])
                
                box = results[0]['boxes'].cpu().numpy()[bldgind[indbest],:].round()
                box = (box[0]-30,box[1]-30,box[2]+30,box[3]+30)
                cropped_image = image.crop(box)
                cropped_image.save(f'filtered_imagery/{im.replace(".jpg","_cropped.jpg")}')
            except:
                pass

        print(f'Cleaned imagery processed and saved in {os.getcwd()}/filtered_imagery')  
