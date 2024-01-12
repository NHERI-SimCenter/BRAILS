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
# 01-11-2024

import rasterio
import rasterio.warp
from rasterio.crs import CRS
from brails.workflow.FootprintHandler import FootprintHandler
from shapely.geometry import box, Polygon
from rasterio.windows import Window

class imutils:
    def __init__(self): 
        self.footprints = []
        self.aerialImageList = []
        
        def aerial_image_extractor(self,rasterMosaicFile):
            def orgcrs2wgs84_bbox(bbox_xy,orgcrs):
                # Project the feature to the desired CRS
                feature_proj = rasterio.warp.transform_geom(
                    orgcrs,
                    CRS.from_epsg(4326),
                    bbox_xy
                ) 
                bbox_lonlat = feature_proj['coordinates'][0]
                return (bbox_lonlat[0][0],bbox_lonlat[0][1],bbox_lonlat[2][0],bbox_lonlat[2][1])
            
            def wgs842orgcrs(latlon,orgcrs):
                # In GeoJSON format
                feature = {
                    "type": "Point",
                    "coordinates": latlon
                }
                
                # Project the feature to the desired CRS
                feature_proj = rasterio.warp.transform_geom(
                    CRS.from_epsg(4326),
                    orgcrs,
                    feature
                )    
                return feature_proj['coordinates']
            
            dataset = rasterio.open('Trinity5camera60acreTest_transparent_mosaic_group1-001.tif')
            bbox_wgs84 = orgcrs2wgs84_bbox(box(*dataset.bounds),dataset.crs)
            
            fpHandler = FootprintHandler()
            fpHandler.fetch_footprint_data(bbox_wgs84,fpSource='osm')
            
            self.footprints = fpHandler.footprints
            for fp in self.footprints:
                cent = Polygon(fp).centroid
                imname = f'imarial_{cent.y:.8f}{cent.x:.2f}.tif'
                self.aerialImageList.append(imname)
                fp_xy = []
                for vert in fp:
                    fp_xy.append(wgs842orgcrs(vert,dataset.crs))
                poly_xy = Polygon(fp_xy).buffer(0.2).bounds
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
                transform = dataset.window_transform(window)
            
                # Create a new cropped raster to write to
                profile = dataset.profile
                profile.update({
                    'height': xsize,
                    'width': ysize,
                    'transform': transform})
            
                with rasterio.open(imname, 'w', **profile) as dst:
                    # Read the data from the window and write it to the output raster
                    dst.write(dataset.read(window=window))