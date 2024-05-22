# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 The Regents of the University of California
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
# 04-19-2024  

import requests
import copy
import json
from shapely.geometry import Polygon, LineString, Point, box
from shapely import wkt
from requests.adapters import HTTPAdapter, Retry
from brails.workflow.FootprintHandler import FootprintHandler 

class TransportationElementHandler:
    def __init__(self): 
        self.queryarea = ''
        self.output_files = {'roads':'Roads.geojson'}
        
    def fetch_transportation_elements(self, queryarea:str):

        def query_generator(bpoly: Polygon, eltype:str) -> str:
            # Get the bounds of the entered bounding polygon and lowercase the
            # entered element type:
            bbox = bpoly.bounds
            eltype = eltype.lower()
            
            # If element type is bridge, generate an NBI query for the bounds
            # of bpoly:
            if eltype=='bridge':
                query = ('https://geo.dot.gov/server/rest/services/Hosted/' + 
                          'National_Bridge_Inventory_DS/FeatureServer/0/query?'+
                          'where=1%3D1&outFields=*'+
                          f"&geometry={bbox[0]}%2C{bbox[1]}%2C{bbox[2]}%2C{bbox[3]}" +
                          '&geometryType=esriGeometryEnvelope&inSR=4326' + 
                          '&spatialRel=esriSpatialRelIntersects&outSR=4326&f=json')
            
            # If element type is tunnel, generate an NTI query for the bounds
            # of bpoly:
            elif eltype=='tunnel':
                query = ('https://geo.dot.gov/server/rest/services/Hosted/' +
                         'National_Tunnel_Inventory_DS/FeatureServer/0/query?' +
                         'where=1%3D1&outFields=*'
                         f"&geometry={bbox[0]}%2C{bbox[1]}%2C{bbox[2]}%2C{bbox[3]}" + 
                         '&geometryType=esriGeometryEnvelope&inSR=4326' + 
                         '&spatialRel=esriSpatialRelIntersects&outSR=4326&f=json')                    
        
            # If element type is railroad, generate an NARNL query for the 
            # bounds of bpoly:
            elif eltype=='railroad':
                query = ('https://geo.dot.gov/server/rest/services/Hosted/' + 
                         'North_American_Rail_Network_Lines_DS/FeatureServer/0/query?' + 
                          'where=1%3D1&outFields=*'+
                          f"&geometry={bbox[0]}%2C{bbox[1]}%2C{bbox[2]}%2C{bbox[3]}" +
                          '&geometryType=esriGeometryEnvelope&inSR=4326' + 
                          '&spatialRel=esriSpatialRelIntersects&outSR=4326&f=json')                         
        
            # If element type is primary_road, generate a TIGER query for the 
            # bounds of bpoly:
            elif eltype=='primary_road':
                query = ('https://tigerweb.geo.census.gov/arcgis/rest/services/' + 
                          'TIGERweb/Transportation/MapServer/2/query?where=&text='+
                          '&outFields=OID,NAME,MTFCC'
                          f"&geometry={bbox[0]}%2C{bbox[1]}%2C{bbox[2]}%2C{bbox[3]}" +
                          '&geometryType=esriGeometryEnvelope&inSR=4326' + 
                          '&spatialRel=esriSpatialRelIntersects&outSR=4326&f=json')                    
        
            # If element type is secondary_road, generate a TIGER query for the 
            # bounds of bpoly:
            elif eltype=='secondary_road':
                query = ('https://tigerweb.geo.census.gov/arcgis/rest/services/' + 
                          'TIGERweb/Transportation/MapServer/6/query?where=&text='+
                          '&outFields=OID,NAME,MTFCC'
                          f"&geometry={bbox[0]}%2C{bbox[1]}%2C{bbox[2]}%2C{bbox[3]}" +
                          '&geometryType=esriGeometryEnvelope&inSR=4326' + 
                          '&spatialRel=esriSpatialRelIntersects&outSR=4326&f=json')              
        
            # If element type is local_road, generate a TIGER query for the 
            # bounds of bpoly:
            elif eltype=='local_road':
                query = ('https://tigerweb.geo.census.gov/arcgis/rest/services/' + 
                          'TIGERweb/Transportation/MapServer/8/query?where=&text='+
                          '&outFields=OID,NAME,MTFCC'
                          f"&geometry={bbox[0]}%2C{bbox[1]}%2C{bbox[2]}%2C{bbox[3]}" +
                          '&geometryType=esriGeometryEnvelope&inSR=4326' + 
                          '&spatialRel=esriSpatialRelIntersects&outSR=4326&f=json')  
            
            # Otherwise, raise a NotImplementedError:
            else:
                raise NotImplementedError('Element type not implemented')
            return query
        
        def create_pooling_session():
            # Define a retry stratey for common error codes to use when
            # downloading data:
            s = requests.Session()
            retries = Retry(total=10, 
                            backoff_factor=0.1,
                            status_forcelist=[500, 502, 503, 504])
            s.mount('https://', HTTPAdapter(max_retries=retries))
            
            return s
        
        def get_el_counts(bpoly:Polygon, eltype: str) -> int:            
            # Create a persistent requests session:
            s = create_pooling_session()
            
            # Create the query required to get the element counts:
            query = query_generator(bpoly,eltype)
            query = query.replace('outSR=4326','returnCountOnly=true')
            
            # Download the element count for the bounding polygon using the 
            # defined retry strategy:
            r = s.get(query)
            elcount = r.json()['count']
            
            return elcount                 

        def get_max_el_count(eltype: str) -> int:            
            # Create a persistent requests session:
            s = create_pooling_session()
            
            # Create the query required to get the element counts:
            query = query_generator(box(-1,1,-1,1),eltype)
            query = query.split('/query?')
            query = query[0] + '?f=pjson'
            
            # Download the maximum element count for the bounding polygon using
            # the defined retry strategy:
            r = s.get(query)
            maxelcount = r.json()['maxRecordCount']
            
            return maxelcount
        
        def list2geojson(datalist:list, eltype:str, bpoly:Polygon) -> dict:
            # Lowercase the entered element type string:
            eltype = eltype.lower()
            
            # Populate the geojson header:
            geojson = {'type':'FeatureCollection', 
                       "crs": {"type": "name", "properties": 
                               {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                       'features':[]}
            
            # Scroll through each item in the datalist:
            for item in datalist:
                # If the element is a bridge or tunnel, parse its geometry as a
                # point and check if the extracted point is within the bounding
                # polygon:
                if eltype in ['bridge','tunnel']:
                    geometry = [item['geometry']['x'],item['geometry']['y']]
                    if bpoly.contains(Point(geometry)):
                        feature = {'type':'Feature',
                                   'properties':{},
                                   'geometry':{'type':'Point',
                                               'coordinates':[]}}
                    else:
                        continue
                else:
                # If the element is a road segment, parse it as a MultiLineString
                # and check if the extracted segment is within the bounding
                # polygon:
                    geometry = item['geometry']['paths']
                    if bpoly.intersects(LineString(geometry[0])):
                        feature = {'type':'Feature',
                                   'properties':{},
                                   'geometry':{'type':'MultiLineString',
                                               'coordinates':[]}}
                    else:
                        continue
                
                # Copy the obtained geometry in a feature dictionary:
                feature['geometry']['coordinates'] = geometry.copy()
                
                # Read item attributes 
                properties = item['attributes']
                
                # For each attribute:
                for prop in properties.keys():
                    # Clean up the property name from redundant numeric text:
                    if '_' in prop:
                        strsegs = prop.split('_')
                        removestr = ''
                        for seg in strsegs:
                            if any(char.isdigit() for char in seg):
                                removestr = '_' + seg
                        propname = prop.replace(removestr,'')
                    else:
                        propname = prop
                    
                    # Write the property in a feature dictionary:
                    feature['properties'][propname] = properties[prop]
                
                # Add the feature in the geojson dictionary:
                geojson['features'].append(feature)
            return geojson

        def print_el_counts(datalist: list, eltype:str):
            nel = len(datalist)
            eltype_print = eltype.replace('_',' ')
            
            if eltype in ['bridge','tunnel']:
                elntype = 'node'
            else:
                elntype = 'edge'
            
            if nel==1:
                suffix = ''
            else:
                suffix = 's'
            
            print(f'Found {nel} {eltype_print} {elntype}{suffix}')

        def write2geojson(bpoly:Polygon, eltype:str) -> dict:
            nel = get_el_counts(bpoly,eltype)
            
            # Create a persistent requests session:
            s = create_pooling_session()
            
            # Download element data using the defined retry strategy:
            query = query_generator(bpoly,eltype)
            r = s.get(query)
            
            # Check to see if the data was successfully downloaded:
            if 'error' in r.text:
                print(f"Data server for {eltype.replace('_',' ')}s is currently unresponsive." +
                         " Please try again later.")
                datalist = []
            else:
                datalist = r.json()['features']
            
            # If road data convert it into GeoJSON format:
            jsonout = list2geojson(datalist,eltype,bpoly)  
            
            # If not road data convert it into GeoJSON format and write it into
            # a file: 
            if '_road' not in eltype:      
                print_el_counts(jsonout['features'],eltype)
                if len(datalist)!=0:
                    output_filename = f'{eltype.title()}s.geojson'
                    with open(output_filename, 'w') as output_file:
                        json.dump(jsonout, output_file, indent=2)
                else:
                    jsonout = ''
            return jsonout

        def find(s, ch):
            return [i for i, ltr in enumerate(s) if ltr == ch]

        def lstring2xylist(lstring):
            coords = lstring.xy
            coordsout = []
            for i in range(len(coords[0])):
                coordsout.append([coords[0][i],coords[1][i]])    
            return coordsout

        def combine_write_roadjsons(roadjsons,bpoly):
            roadjsons_combined = {'type':'FeatureCollection', 
                       "crs": {"type": "name", "properties": 
                               {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                       'features':[]}
            rtype = {'primary_road':'P','secondary_road':'S','local_road':'L'}
            for key, roadjson in roadjsons.items(): 
                for item in roadjson['features']:  
                    # Get into the properties and add RTYPE into edge property  
                    item['properties']['RTYPE'] = rtype[key]  
                    # Get into the geometry and intersect it with the bounding polygon:
                    lstring = LineString(item['geometry']['coordinates'][0]) 
                    coords = lstring.intersection(bpoly)
                    # If output is LineString write the intersecting geometry in item and 
                    # append it to the combined the json file:
                    if coords.geom_type=='LineString':
                        coordsout = lstring2xylist(coords)
                        item['geometry']['coordinates'] = [coordsout]  
                        roadjsons_combined['features'].append(item)
                # If not, create multiple copies of the same json for each linestring:
                    else:
                        mlstring_wkt = coords.wkt
                        inds = find(mlstring_wkt,'(')[1:]
                        nedges = len(inds)
                        for i in range(nedges):
                            if i+1!=nedges:
                                edge = wkt.loads('LINESTRING ' + 
                                                 mlstring_wkt[inds[i]:inds[i+1]-2])
                            else:
                                edge = wkt.loads('LINESTRING ' + 
                                                 mlstring_wkt[inds[i]:-1])
                            coordsout = lstring2xylist(edge)  
                            newitem = copy.deepcopy(item)
                            newitem['geometry']['coordinates'] = [coordsout]
                            roadjsons_combined['features'].append(newitem)

            print_el_counts(roadjsons_combined['features'],'road')
            with open('Roads.geojson', 'w') as output_file:
                json.dump(roadjsons_combined, output_file, indent=2)
       
        self.queryarea = queryarea
        
        # Initialize FootprintHandler:
        fpHandler = FootprintHandler()
        
        # Run FootprintHandler to get the boundary for the entered location:
        if isinstance(queryarea,tuple):
            bpoly,_ = fpHandler._FootprintHandler__bbox2poly(queryarea)
        else:
            bpoly,_,_ = fpHandler._FootprintHandler__fetch_roi(queryarea)   

        # Define supported element types:
        eltypes = ['bridge', 'tunnel', 'railroad', 'primary_road',
                   'secondary_road', 'local_road']
        roadjsons = {'primary_road':[], 'secondary_road':[], 'local_road':[]}
        
        # Write the GeoJSON output for each element:
        for eltype in eltypes:
            if '_road' not in eltype:
                jsonout = write2geojson(bpoly,eltype)
                if jsonout!='':
                   self.output_files[eltype + 's'] = eltype.capitalize() + 's.geojson'
            else:
                print(f"Fetching {eltype.replace('_',' ')}s, may take some time...")
                roadjsons[eltype] = (write2geojson(bpoly,eltype))
        combine_write_roadjsons(roadjsons,bpoly)