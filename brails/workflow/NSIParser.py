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
# 01-18-2024  

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
    
    def GenerateBldgInventory(self,footprints):
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
                    
        bboxstr = f'{minlon:.5f},{minlat:.5f},{minlon:.5f},{maxlat:.5f},{maxlon:.5f},{maxlat:.5f},{maxlon:.5f},{minlat:.5f},{minlon:.5f},{minlat:.5f}'
                    
        url = "https://nsi.sec.usace.army.mil/nsiapi/structures?bbox="
        
        # Getting the NSI building points for the 
        # Define a retry stratey fgor common error codes to use when
        # downloading tiles:
        s = requests.Session()
        retries = Retry(total=5, 
                        backoff_factor=0.1,
                        status_forcelist=[500, 502, 503, 504])
        s.mount('https://', HTTPAdapter(max_retries=retries))
        
        # Download tile using the defined retry strategy:
        print('\nGetting National Structure Inventory (NSI) building points for the building footprints...')
        response = s.get(url+bboxstr)       
        datalist = response.json()['features']
        
        datadict = {}
        count = 0
        for data in datalist:
            pt = Point(data['geometry']['coordinates'])
            if pt in datadict.keys():
                count+=1
            datadict[pt] = data['properties']
        points = list(datadict.keys())
        pttree = STRtree(points)
        
        ress = []
        for fp in footprints:
            res = pttree.query(Polygon(fp))
            if res.size!=0:
                ress.append(res)
            else:
                ress.append(np.empty(shape=(0, 0)))
        
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
                footprints_out.append(('{"type":"Feature","geometry":' + 
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
                strtype.append(ptres['bldgtype'])
                occ.append(ptres['occtype'])
        
        print(f'Found a total of {len(footprints_out)} building points in NSI that match the footprint data.')
        
        inventory = pd.DataFrame(pd.Series(lat,name='Latitude'))
        inventory['Longitude'] = lon
        inventory['PlanArea'] = planarea
        inventory['NumberOfStories'] = nstories
        inventory['YearBuilt'] = yearbuilt
        inventory['ReplacementCost'] = repcost
        inventory['StructureType'] = strtype
        inventory['OccupancyClass'] = occ
        inventory['Footprint'] = footprints_out
        
        self.inventory= inventory
        
        inventory.to_csv('inventory.csv', index=True, index_label='id') 
        print(f'\nFinal inventory data available in {os.getcwd()}/inventory.csv')