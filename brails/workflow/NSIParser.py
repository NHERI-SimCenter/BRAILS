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
# 02-01-2024  

import requests 
import numpy as np
import pandas as pd
import os
from requests.adapters import HTTPAdapter, Retry
from shapely.strtree import STRtree
from shapely import Polygon, Point

class NSIParser:
    def __init__(self): 
        self.inventory = []
        self.attributes = ['erabuilt','numstories','occupancy','constype']
    
    def GenerateBldgInventory(self,footprints,outFile=None):
        """
        Function that reads NSI buildings points and matches the data to a set
        of footprints. 
         
        Input:  List of footprint data defined as a list of lists of 
                coordinates in EPSG 4326, i.e., [[vert1],....[vertlast]].
                Vertices are defined in [longitude,latitude] fashion.
        Output: Building inventory for the footprints containing the attributes
                1)latitude, 2)longitude, 3)building plan area, 4)number of 
                floors, 5)year of construction, 6)replacement cost, 7)structure
                type, 8)occupancy class, 9)footprint polygon
        """  
        
        def get_bbox(footprints):
            """
            Function that determines the extent of the area covered by the 
            footprints as a tight-fit rectangular bounding box
            
            Input:  List of footprint data defined as a list of lists of 
                    coordinates in EPSG 4326, i.e., [[vert1],....[vertlast]].
                    Vertices are defined in [longitude,latitude] fashion
            Output: Tuple containing the minimum and maximum latitude and 
                    longitude values 
           """     
           # :
            minlat = footprints[0][0][1]
            minlon = footprints[0][0][0]
            maxlat = footprints[0][0][1]
            maxlon = footprints[0][0][0]
            for fp in footprints:
                for vert in fp:
                    if vert[1]>maxlat:
                        maxlat = vert[1]
                    if vert[0]>maxlon:
                        maxlon = vert[0]
                    if vert[1]<minlat:
                        minlat = vert[1]
                    if vert[0]<minlon:
                        minlon = vert[0]
            return (minlat,minlon,maxlat,maxlon)
        
        def get_nbi_data(box): 
            """
            Function that gets the NBI data for a bounding box entry
            Input:  Tuple containing the minimum and maximum latitude and 
                    longitude values 
            Output: Dictionary containing extracted NBI data keyed using the
                    NBI point coordinates
           """  
            
            # Unpack the bounding box coordinates:
            (minlat,minlon,maxlat,maxlon) = bbox
            
            # Construct the query URL for the bounding box input
            baseurl = "https://nsi.sec.usace.army.mil/nsiapi/structures?bbox="        
            bboxstr = (f'{minlon:.5f},{minlat:.5f},{minlon:.5f},{maxlat:.5f},'
                       f'{maxlon:.5f},{maxlat:.5f},{maxlon:.5f},{minlat:.5f},'
                       f'{minlon:.5f},{minlat:.5f}')
            url = baseurl + bboxstr                  
            
            # Define a retry stratey for common error codes to use in downloading
            # NBI data:
            s = requests.Session()
            retries = Retry(total=5, 
                            backoff_factor=0.1,
                            status_forcelist=[500, 502, 503, 504])
            s.mount('https://', HTTPAdapter(max_retries=retries))
            
            # Download NBI data using the defined retry strategy, read downloaded
            # GeoJSON data into a list:
            print('\nGetting National Structure Inventory (NSI) building data'
                  ' for the building footprints...')
            response = s.get(url)       
            datalist = response.json()['features']
            
            # Write the data in datalist into a dictionary for better data access,
            # and filtering the duplicate entries:
            datadict = {}
            for data in datalist:
                pt = Point(data['geometry']['coordinates'])
                datadict[pt] = data['properties']
            return datadict
        
        def get_inv_from_datadict(datadict,footprints):
            # Create an STR tree for the building points obtained from NBI: 
            points = list(datadict.keys())
            pttree = STRtree(points)
            
            # Find the data points that are enclosed in each footprint:
            ress = []
            for fp in footprints:
                res = pttree.query(Polygon(fp))
                if res.size!=0:
                    ress.append(res)
                else:
                    ress.append(np.empty(shape=(0, 0)))
            
            # Match NBI data to each footprint:
            footprints_out_json = []
            footprints_out = []
            lat = []
            lon = []
            planarea = []
            nstories = []
            yearbuilt = []
            repcost = []
            strtype = []
            occ = []
            for ind, fp in enumerate(footprints):
                if ress[ind].size!=0:
                    footprints_out.append(fp)
                    footprints_out_json.append(('{"type":"Feature","geometry":' + 
                    '{"type":"Polygon","coordinates":[' + 
                    f"""{fp}""" + 
                    ']},"properties":{}}'))
                    ptind = ress[ind][0]
                    ptres = datadict[points[ptind]]
                    lat.append(ptres['y'])
                    lon.append(ptres['x'])
                    planarea.append(ptres['sqft'])
                    nstories.append(ptres['num_story'])
                    yearbuilt.append(ptres['med_yr_blt'])
                    repcost.append(ptres['val_struct'])
                    bldgtype = ptres['bldgtype'] + '1'
                    if bldgtype=='M1':
                        strtype.append('RM1')
                    else:
                        strtype.append(bldgtype)
                    if '-' in ptres['occtype']:
                        occ.append(ptres['occtype'].split('-')[0])
                    else:
                        occ.append(ptres['occtype'])
                    
            
            # Display the number of footprints that can be matched to NSI points:
            print(f'Found a total of {len(footprints_out)} building points in'
                  ' NSI that match the footprint data.')
            
            # Write the extracted features into a Pandas dataframe:
            inventory = pd.DataFrame(pd.Series(lat,name='Latitude'))
            inventory['Longitude'] = lon
            inventory['PlanArea'] = planarea
            inventory['NumberOfStories'] = nstories
            inventory['YearBuilt'] = yearbuilt
            inventory['ReplacementCost'] = repcost
            inventory['StructureType'] = strtype
            inventory['OccupancyClass'] = occ
            inventory['Footprint'] = footprints_out
            return (inventory, footprints_out_json)

        # Determine the coordinates of the bounding box including the footprints:
        bbox = get_bbox(footprints)          

        # Get the NBI data for computed bounding box:
        datadict = get_nbi_data(bbox)  
        
        # Create a footprint-merged building inventory from extracted NBI data:
        (self.inventory, footprints_out_json) = get_inv_from_datadict(datadict,footprints)    
        
        # If requested, write the created inventory in R2D-compatible CSV format:
        if outFile:
            inventory = self.inventory.copy(deep=True)
            n = inventory.columns[1]
            inventory.drop(n, axis = 1, inplace = True)
            inventory[n] = footprints_out_json
            inventory.to_csv(outFile, index=True, index_label='id') 
            print(f'\nFinal inventory data available in {os.getcwd()}/{outFile}')