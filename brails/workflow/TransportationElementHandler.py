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
# 06-23-2023  

import requests
import copy
import json
from shapely.geometry import Polygon, LineString, Point, box
from shapely import wkt
from itertools import groupby
import sys

class TransportationElementHandler:
    def __init__(self): 
        self.queryarea = []
        self.output_files = {'roads':'Roads.geojson',
                             'bridges':'Bridges.geojson',
                             'tunnels':'Tunnels.geojson',
                             'railroads':'Railroads.geojson'}
        
    def fetch_transportation_elements(self,queryarea):

        def fetch_roi(queryarea):
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
                    queryarea_osmid = data['osm_id']
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
                
                queryarea_printname = queryarea_name.split(",")[0]  
                
                print(f"\nFetching the transportation network for {queryarea_printname}...")
                url = 'http://overpass-api.de/api/interpreter'
                
                if isinstance(queryarea,str):
                    query = f"""
                    [out:json][timeout:5000];
                    rel({queryarea_osmid});
                    out geom;
                    """
                r = requests.get(url, params={'data': query})
                
                datastruct = r.json()['elements'][0]
                
                boundarypoly = []
                if datastruct['tags']['type']=='boundary':
                    for coorddict in datastruct['members']:
                        if coorddict['role']=='outer':
                            for coord in coorddict['geometry']:
                                boundarypoly.append([coord['lon'],coord['lat']])
                
                bpoly = Polygon(boundarypoly)                  
                                
            elif isinstance(queryarea,tuple):
                bpoly = box(queryarea[0],queryarea[1],queryarea[2],queryarea[3])
                print("\nFetching the transportation network for the bounding box: " +
                      f"[{queryarea[0]}, {queryarea[1]}, {queryarea[2]},{queryarea[3]}]...")
            else:
                sys.exit('Incorrect location entry. The location entry must be defined' + 
                         ' as a string or a list of strings containing the area name(s)' + 
                         ' or a tuple containing the longitude and latitude pairs for' +
                         ' the bounding box of the area of interest.')       
            return bpoly

        def query_generator(bpoly,eltype):
            bbox = bpoly.bounds
            eltype = eltype.lower()
            if eltype=='bridge':
                query = f'https://geo.dot.gov/mapping/rest/services/NTAD/National_Bridge_Inventory/MapServer/0/query?where=1%3D1&outFields=STRUCTURE_NUMBER_008,RECORD_TYPE_005A,ROUTE_PREFIX_005B,SERVICE_LEVEL_005C,ROUTE_NUMBER_005D,DIRECTION_005E,MIN_VERT_CLR_010,DETOUR_KILOS_019,YEAR_BUILT_027,TRAFFIC_LANES_ON_028A,TRAFFIC_LANES_UND_028B,ADT_029,YEAR_ADT_030,APPR_WIDTH_MT_032,MEDIAN_CODE_033,DEGREES_SKEW_034,STRUCTURE_FLARED_035,RAILINGS_036A,NAV_VERT_CLR_MT_039,NAV_HORR_CLR_MT_040,SERVICE_ON_042A,SERVICE_UND_042B,STRUCTURE_KIND_043A,STRUCTURE_TYPE_043B,APPR_KIND_044A,APPR_TYPE_044B,MAIN_UNIT_SPANS_045,APPR_SPANS_046,HORR_CLR_MT_047,MAX_SPAN_LEN_MT_048,STRUCTURE_LEN_MT_049,LEFT_CURB_MT_050A,RIGHT_CURB_MT_050B,ROADWAY_WIDTH_MT_051,DECK_WIDTH_MT_052,VERT_CLR_OVER_MT_053,VERT_CLR_UND_REF_054A,VERT_CLR_UND_054B,LAT_UND_REF_055A,LAT_UND_MT_055B,LEFT_LAT_UND_MT_056,YEAR_RECONSTRUCTED_106,PERCENT_ADT_TRUCK_109,SCOUR_CRITICAL_113,FUTURE_ADT_114,YEAR_OF_FUTURE_ADT_115,DECK_AREA&geometry={bbox[0]}%2C{bbox[1]}%2C{bbox[2]}%2C{bbox[3]}&geometryType=esriGeometryEnvelope&inSR=4326&spatialRel=esriSpatialRelIntersects&outSR=4326&f=json'
            elif eltype=='tunnel':
                query = f'https://geo.dot.gov/mapping/rest/services/NTAD/National_Tunnel_Inventory/MapServer/0/query?where=1%3D1&outFields=tunnel_number_i1,route_number_i7,route_direction_i8,route_type_i9,facility_carried_i10,year_built_a1,year_rehabilitated_a2,number_of_lanes_a3,annual_average_daily_traffic_a4,annual_average_daily_truck_traf,year_of_annual_average_daily_tr,detour_length_a7,service_in_tunnel_a8,direction_of_traffic_c3,tunnel_length_g1,minimum_vertical_clearance_over,roadway_width_curb_to_curb_g3,number_of_bores_s1,tunnel_shape_s2,portal_shape_s3,ground_conditions_s4,left_sidewalk_width_g4,right_sidewalk_width_g5&geometry={bbox[0]}%2C{bbox[1]}%2C{bbox[2]}%2C{bbox[3]}&geometryType=esriGeometryEnvelope&inSR=4326&spatialRel=esriSpatialRelIntersects&outSR=4326&f=json'    
            elif eltype=='railroad':
                query = f'https://geo.dot.gov/mapping/rest/services/NTAD/North_American_Rail_Network_Lines/MapServer/0/query?where=1%3D1&outFields=FRAARCID,TRACKS,MILES&geometry={bbox[0]}%2C{bbox[1]}%2C{bbox[2]}%2C{bbox[3]}&geometryType=esriGeometryEnvelope&inSR=4326&spatialRel=esriSpatialRelIntersects&outSR=4326&f=json'    
            elif eltype=='primary_road':
                query = f'https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Transportation/MapServer/2/query?where=&text=&outFields=OID,NAME&geometry={bbox[0]}%2C{bbox[1]}%2C{bbox[2]}%2C{bbox[3]}&geometryType=esriGeometryEnvelope&inSR=4326&spatialRel=esriSpatialRelIntersects&outSR=4326&f=json'
            elif eltype=='secondary_road':
                query = f'https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Transportation/MapServer/6/query?where=&text=&outFields=OID,NAME&geometry={bbox[0]}%2C{bbox[1]}%2C{bbox[2]}%2C{bbox[3]}&geometryType=esriGeometryEnvelope&inSR=4326&spatialRel=esriSpatialRelIntersects&outSR=4326&f=json'
            elif eltype=='local_road':
                query = f'https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Transportation/MapServer/8/query?where=&text=&outFields=OID,NAME&geometry={bbox[0]}%2C{bbox[1]}%2C{bbox[2]}%2C{bbox[3]}&geometryType=esriGeometryEnvelope&inSR=4326&spatialRel=esriSpatialRelIntersects&outSR=4326&f=json'
            else:
                raise NotImplementedError('Element type not implemented')
            return query

        def conv2geojson(datalist,eltype,bpoly):
            eltype = eltype.lower()
            geojson = {'type':'FeatureCollection', 
                       "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
                       'features':[]}
            for item in datalist:
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
                    geometry = item['geometry']['paths']
                    if bpoly.intersects(LineString(geometry[0])):
                        feature = {'type':'Feature',
                                   'properties':{},
                                   'geometry':{'type':'MultiLineString',
                                               'coordinates':[]}}
                    else:
                        continue
                feature['geometry']['coordinates'] = geometry.copy()
                properties = item['attributes']
                for prop in properties:
                    if prop.count('_')>1:
                        propname = prop[:prop.rindex('_')]
                    else:
                        propname = prop
                    feature['properties'][propname] = properties[prop]
                geojson['features'].append(feature)
            return geojson

        def print_el_counts(datalist,eltype):
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

        def write2geojson(bpoly,eltype):
            query = query_generator(bpoly,eltype)
            r = requests.get(query)
            if 'error' in r.text:
                sys.exit(f"Data server for {eltype.replace('_',' ')}s is currently unresponsive." +
                         " Please try again later.")
            
            datalist = r.json()['features']
            jsonout = conv2geojson(datalist,eltype,bpoly)  
            if '_road' not in eltype:      
                print_el_counts(jsonout['features'],eltype)
                if len(jsonout['features'])!=0:
                    output_filename = f'{eltype.title()}s.geojson'
                    with open(output_filename, 'w') as output_file:
                        json.dump(jsonout, output_file, indent=2)
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
                       "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
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
                                edge = wkt.loads('LINESTRING ' + mlstring_wkt[inds[i]:inds[i+1]-2])
                            else:
                                edge = wkt.loads('LINESTRING ' + mlstring_wkt[inds[i]:-1])
                            coordsout = lstring2xylist(edge)  
                            newitem = copy.deepcopy(item)
                            newitem['geometry']['coordinates'] = [coordsout]
                            roadjsons_combined['features'].append(newitem)

            print_el_counts(roadjsons_combined['features'],'road')
            with open('Roads.geojson', 'w') as output_file:
                json.dump(roadjsons_combined, output_file, indent=2)    
        
        self.queryarea = queryarea
        bpoly = fetch_roi(self.queryarea)

        eltypes = ['bridge','tunnel','railroad','primary_road','secondary_road','local_road']

        roadjsons = {'primary_road':[],'secondary_road':[],'local_road':[]}
        for eltype in eltypes:
            if '_road' not in eltype:
                write2geojson(bpoly,eltype)
            else:
                roadjsons[eltype] = (write2geojson(bpoly,eltype))
        combine_write_roadjsons(roadjsons,bpoly)