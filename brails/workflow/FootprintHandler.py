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
# 05-12-2022  

import requests
import sys
import json
from itertools import groupby

class FootprintHandler:
    def __init__(self): 
        self.footprints = []
        self.queryarea = []
        
    def fetch_footprint_data(self,queryarea):
        """
        Function that loads footprint data from OpenStreetMap
        
        Input: Location entry defined as a string or a list of strings 
               containing the area name(s) or a tuple containing the longitude
               and longitude pairs for the bounding box of the area of interest
        Output: Footprint information parsed as a list of lists with each
                coordinate described in longitude and latitude pairs   
        """
        def get_osm_footprints(queryarea):
            if isinstance(queryarea,str):
                # Search for the query area using Nominatim API:
                print(f"\nSearching for {queryarea}...")
                queryarea = queryarea.replace(" ", "+").replace(',','+')
                
                queryarea_formatted = ""
                for i, j in groupby(queryarea):
                    if i=='+':
                        queryarea_formatted += i
                    else:
                        queryarea_formatted += ''.join(list(j))
                
                nominatimquery = ('https://nominatim.openstreetmap.org/search?' +
                                  f"q={queryarea_formatted}&format=json")
                
                r = requests.get(nominatimquery)
                datalist = r.json()
                
                areafound = False
                for data in datalist:
                    queryarea_turboid = data['osm_id'] + 3600000000
                    queryarea_name = data['display_name']
                    if(data['osm_type']=='relation' and 
                       'university' in queryarea.lower() and
                       data['type']=='university'):
                        areafound = True
                        break
                    elif (data['osm_type']=='relation' and 
                         data['type']=='administrative'): 
                        areafound = True
                        break
                
                if areafound==True:
                    print(f"Found {queryarea_name}")
                else:
                    sys.exit(f"Could not locate an area named {queryarea}. " + 
                             'Please check your location query to make sure' +
                             'it was entered correctly.')
                    
                        
            elif isinstance(queryarea,tuple):
                pass
            else:
                sys.exit('Incorrect location entry. The location entry must be defined' + 
                         ' as a string or a list of strings containing the area name(s)' + 
                         ' or a tuple containing the longitude and latitude pairs for' +
                         ' the bounding box of the area of interest.')
                         
                         
            
            # Obtain and parse the footprint data for the determined area using Overpass API:
            if isinstance(queryarea,str):
                queryarea_printname = queryarea_name.split(",")[0]
            elif isinstance(queryarea,tuple):
                queryarea_printname = (f"the bounding box: [{queryarea[0]}," 
                                       f"{queryarea[1]}, {queryarea[2]}, "
                                       f"{queryarea[3]}]")
            
            print(f"\nFetching footprint data for {queryarea_printname}...")
            url = 'http://overpass-api.de/api/interpreter'
            
            if isinstance(queryarea,str):
                query = f"""
                [out:json][timeout:5000];
                area({queryarea_turboid})->.searchArea;
                way["building"](area.searchArea);
                out body;
                >;
                out skel qt;
                """
            elif isinstance(queryarea,tuple):
                bbox = [min(queryarea[1],queryarea[3]),
                        min(queryarea[0],queryarea[2]),
                        max(queryarea[1],queryarea[3]),
                        max(queryarea[0],queryarea[2])]
                query = f"""
                [out:json][timeout:5000];
                way["building"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
                out body;
                >;
                out skel qt;
                """
                
            r = requests.get(url, params={'data': query})
            
            datalist = r.json()['elements']
            nodedict = {}
            for data in datalist:
                if data['type']=='node':
                   nodedict[data['id']] = [data['lon'],data['lat']]
        
        
            footprints = []
            for data in datalist:
                if data['type']=='way':
                    nodes = data['nodes']
                    footprint = []
                    for node in nodes:
                        footprint.append(nodedict[node])
                    footprints.append(footprint)
            
            print(f"Found a total of {len(footprints)} building footprints in {queryarea_printname}")
            return footprints

        self.queryarea = queryarea
        if isinstance(queryarea,str):
            self.footprints = get_osm_footprints(queryarea)
        elif isinstance(queryarea,tuple):
            self.footprints = get_osm_footprints(queryarea)
        elif isinstance(queryarea,list):    
            self.footprints = []
            for query in queryarea: 
                self.footprints.extend(get_osm_footprints(query))
        else:
            sys.exit('Incorrect location entry. The location entry must be defined' + 
                     ' as a string or a list of strings containing the area name(s)' + 
                     ' or a tuple containing the latitude and longitude pairs for' +
                     ' the bounding box of the area of interest.')   

    def load_footprint_data(self,fpfile):
        """
        Function that loads footprint data from a GeoJSON file
        
        Input: A GeoJSON file containing footprint information
        Output: Footprint information parsed as a list of lists with each
                coordinate described in longitude and latitude pairs   
        """
        with open(fpfile) as f:
            data = json.load(f)['features']

        self.footprints = []
        for count, loc in enumerate(data):
            self.footprints.append(loc['geometry']['coordinates'][0][0])