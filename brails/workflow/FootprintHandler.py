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
# 07-14-2023  

import math
import json
import requests
import sys
import pandas as pd
from tqdm import tqdm
from itertools import groupby
from shapely.geometry import Polygon, LineString, MultiPolygon, box
from shapely.ops import linemerge, unary_union, polygonize

class FootprintHandler:
    def __init__(self): 
        self.queryarea = []
        self.availableDataSources = ['osm','ms','usastr']
        self.fpSource = 'osm'
        self.footprints = []
        self.bldgheights = []
        
    def fetch_footprint_data(self,queryarea,fpSource='osm'):
        """
        Function that loads footprint data from OpenStreetMap, Microsoft or USA
        Structures data
        
        Input: Location entry defined as a string or a list of strings 
               containing the area name(s) or a tuple containing the longitude
               and longitude pairs for the bounding box of the area of interest
        Output: Footprint information parsed as a list of lists with each
                coordinate described in longitude and latitude pairs   
        """
        def get_osm_footprints(queryarea):
            def write_fp2geojson(footprints, output_filename='footprints.geojson'):
                geojson = {'type':'FeatureCollection', 
                           "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
                           'features':[]}
                for fp in footprints:
                    feature = {'type':'Feature',
                               'properties':{},
                               'geometry':{'type':'Polygon',
                                           'coordinates':[]}}
                    feature['geometry']['coordinates'] = [fp]
                    geojson['features'].append(feature)
                    
                with open(output_filename, 'w') as output_file:
                    json.dump(geojson, output_file, indent=2)
                
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
                                  f"q={queryarea_formatted}&format=jsonv2")
                
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
                             'Please check your location query to make sure ' +
                             'it was entered correctly.')
                 
                queryarea_printname = queryarea_name.split(",")[0]    
                        
            elif isinstance(queryarea,tuple):
                if len(queryarea)%2==0 and len(queryarea)!=0:                        
                    if len(queryarea)==4:
                        bpoly = [min(queryarea[1],queryarea[3]),
                                min(queryarea[0],queryarea[2]),
                                max(queryarea[1],queryarea[3]),
                                max(queryarea[0],queryarea[2])]
                        bpoly = f'{bpoly[0]},{bpoly[1]},{bpoly[2]},{bpoly[3]}'
                        queryarea_printname = (f"the bounding box: {list(queryarea)}")                        
                    elif len(queryarea)>4:
                        bpoly = 'poly:"'
                        queryarea_printname = 'the bounding box: ['
                        for i in range(int(len(queryarea)/2)):
                            bpoly+=f'{queryarea[2*i+1]} {queryarea[2*i]} '
                            queryarea_printname+= f'{queryarea[2*i]}, {queryarea[2*i+1]}, '
                        bpoly = bpoly[:-1]+'"'
                        queryarea_printname = queryarea_printname[:-2]+']'
                    else:
                        raise ValueError('Less than two latitude longitude pairs were entered to define the bounding box entry. ' + 
                                         'A bounding box can be defined by using at least two longitude/latitude pairs.') 
                else:
                        raise ValueError('Incorrect number of elements detected in the tuple for the bounding box. ' 
                                         'Please check to see if you are missing a longitude or latitude value.')                                       


                                     
            # Obtain and parse the footprint data for the determined area using Overpass API:
            
            print(f"\nFetching OSM footprint data for {queryarea_printname}...")
            url = 'http://overpass-api.de/api/interpreter'
            
            if isinstance(queryarea,str):
                query = f"""
                [out:json][timeout:5000][maxsize:2000000000];
                area({queryarea_turboid})->.searchArea;
                way["building"](area.searchArea);
                out body;
                >;
                out skel qt;
                """
            elif isinstance(queryarea,tuple):
                query = f"""
                [out:json][timeout:5000][maxsize:2000000000];
                way["building"]({bpoly});
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
            
            print(f"\nFound a total of {len(footprints)} building footprints in {queryarea_printname}")
            
            write_fp2geojson(footprints)
            
            return footprints
 
        def get_ms_footprints(self):
            def fetch_roi(queryarea):
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
                                  f"q={queryarea_formatted}&format=jsonv2")                
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
                             'Please check your location query to make sure ' +
                             'it was entered correctly.')
                
                queryarea_printname = queryarea_name.split(",")[0]  
                
                print(f"\nFetching Microsoft building footprints for {queryarea_printname}...")
                url = 'http://overpass-api.de/api/interpreter'
                
                if isinstance(queryarea,str):
                    query = f"""
                    [out:json][timeout:5000];
                    rel({queryarea_osmid});
                    out geom;
                    """
                r = requests.get(url, params={'data': query})
                
                datastruct = r.json()['elements'][0]
                
                if datastruct['tags']['type'] in ['boundary','multipolygon']:
                    lss = []
                    for coorddict in datastruct['members']:
                        if coorddict['role']=='outer':
                            ls = []
                            for coord in coorddict['geometry']:
                                ls.append([coord['lon'],coord['lat']])
                            lss.append(LineString(ls))
                
                    merged = linemerge([*lss])
                    borders = unary_union(merged) # linestrings to a MultiLineString
                    polygons = list(polygonize(borders)) 
                    
                    if len(polygons)==1:
                        bpoly = polygons[0]
                    else:
                        bpoly = MultiPolygon(polygons)
                
                else:
                    sys.exit(f"Could not retrieve the boundary for {queryarea}. " + 
                             'Please check your location query to make sure ' +
                             'it was entered correctly.')    
                return bpoly, queryarea_printname
            
            def deg2num(lat, lon, zoom):
                lat_rad = math.radians(lat)
                n = 2**zoom
                xtile = int((lon + 180)/360*n)
                ytile = int((1 - math.asinh(math.tan(lat_rad))/math.pi)/2*n)
                return (xtile,ytile)
            
            def determine_tile_coords(bbox):
                xlist = []; ylist = []
                for vert in bbox:
                    (lat, lon) = (vert[1],vert[0])
                    x,y = deg2num(lat, lon, 9)
                    xlist.append(x)
                    ylist.append(y)
            
                    xlist = list(range(min(xlist),max(xlist)+1))
                    ylist = list(range(min(ylist),max(ylist)+1))
                return (xlist,ylist)               
            
            def xy2quadkey(xtile,ytile):
                xtilebin = str(bin(xtile)); xtilebin = xtilebin[2:]
                ytilebin = str(bin(ytile)); ytilebin = ytilebin[2:]
                zpad = len(xtilebin)-len(ytilebin)
                if zpad<0:
                  xtilebin = xtilebin.zfill(len(xtilebin)-zpad)
                elif zpad>0:
                  ytilebin = ytilebin.zfill(len(ytilebin)+zpad)
                quadkeybin = "".join(i + j for i, j in zip(ytilebin, xtilebin))
                quadkey = ''
                for i in range(0, int(len(quadkeybin)/2)):
                    quadkey+=str(int(quadkeybin[2*i:2*(i+1)],2))
                return int(quadkey)
            
            def bbox2quadkeys(bpoly):
                bbox = bpoly.bounds
                bbox_coords = [[bbox[0],bbox[1]],
                               [bbox[2],bbox[1]],
                               [bbox[2],bbox[3]],
                               [bbox[0],bbox[3]]] 
            
            
                (xtiles,ytiles) = determine_tile_coords(bbox_coords)
                quadkeys = []
                for xtile in xtiles:
                    for ytile in ytiles:
                        quadkeys.append(xy2quadkey(xtile,ytile))
                quadkeys = list(set(quadkeys))
                return quadkeys
            
            def bbox2poly(queryarea):
                if len(queryarea)%2==0 and len(queryarea)!=0:                        
                    if len(queryarea)==4:
                        bpoly = box(*queryarea)
                        queryarea_printname = (f"the bounding box: {list(queryarea)}")                        
                    elif len(queryarea)>4:
                        queryarea_printname = 'the bounding box: ['
                        bpolycoords = []
                        for i in range(int(len(queryarea)/2)):
                            bpolycoords = bpolycoords.append([queryarea[2*i], queryarea[2*i+1]])
                            queryarea_printname+= f'{queryarea[2*i]}, {queryarea[2*i+1]}, '
                        bpoly = Polygon(bpolycoords)
                        queryarea_printname = queryarea_printname[:-2]+']'
                    else:
                        raise ValueError('Less than two latitude longitude pairs were entered to define the bounding box entry. ' + 
                                         'A bounding box can be defined by using at least two longitude/latitude pairs.') 
                else:
                        raise ValueError('Incorrect number of elements detected in the tuple for the bounding box. ' 
                                         'Please check to see if you are missing a longitude or latitude value.')  

                print(f"\nFetching Microsoft building footprints for {queryarea_printname}...")
                
                return bpoly, queryarea_printname
            
            def parse_file_size(strsize):
                strsize = strsize.lower()
                if 'gb' in strsize:
                    multiplier = 1e9
                    sizestr = 'gb' 
                elif 'mb' in strsize:
                    multiplier = 1e6
                    sizestr = 'mb' 
                elif 'kb' in strsize:
                    multiplier = 1e3
                    sizestr = 'kb'
                else:
                    multiplier = 1
                    sizestr = 'b'
                return float(strsize.replace(sizestr,''))*multiplier
                    
            def download_ms_tiles(quadkeys,bpoly):
                dftiles = pd.read_csv(
                    "https://minedbuildings.blob.core.windows.net/global-buildings/dataset-links.csv"
                )
                
                footprints = []
                bldgheights = []
                for quadkey in tqdm(quadkeys):
                    rows = dftiles[dftiles['QuadKey'] == quadkey]
                    if rows.shape[0] == 1:
                        url = rows.iloc[0]['Url']
                    elif rows.shape[0] > 1:
                        rows.loc[:,'Size'] = rows['Size'].apply(lambda x: parse_file_size(x))
                        url = rows[rows['Size']==rows['Size'].max()].iloc[0]['Url']
                    else:
                        continue
                    
                    df_fp = pd.read_json(url, lines=True)
                    for index, row in tqdm(df_fp.iterrows(), total=df_fp.shape[0]):
                        fp_poly = Polygon(row['geometry']['coordinates'][0])
                        if fp_poly.intersects(bpoly):
                            footprints.append(row['geometry']['coordinates'][0])
                            bldgheights.append(row['properties']['height']*3.28084)
            
                return (footprints, bldgheights)
            
            def write_fp2geojson(footprints, bldgheights, output_filename='footprints.geojson'):
                geojson = {'type':'FeatureCollection', 
                           "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
                           'features':[]}
                for fp,bldgheight in zip(footprints,bldgheights):
                    feature = {'type':'Feature',
                               'properties':{},
                               'geometry':{'type':'Polygon',
                                           'coordinates':[]}}
                    feature['geometry']['coordinates'] = [fp]
                    feature['properties']['bldgheight'] = bldgheight
                    geojson['features'].append(feature)
                with open(output_filename, 'w') as output_file:
                    json.dump(geojson, output_file, indent=2)
                        
            if isinstance(queryarea,tuple):
                bpoly, queryarea_printname = bbox2poly(queryarea)   
            elif isinstance(queryarea,str):
                bpoly, queryarea_printname = fetch_roi(queryarea)                                                
            
            quadkeys = bbox2quadkeys(bpoly)
            (footprints, bldgheights) = download_ms_tiles(quadkeys, bpoly)
            write_fp2geojson(footprints,bldgheights)
            
            print(f"\nFound a total of {len(footprints)} building footprints in {queryarea_printname}")
            return footprints        
     
        def get_usastruct_footprints(queryarea):
            if isinstance(queryarea,tuple):
                queryarea_printname = (f"the bounding box: [{queryarea[0]}," 
                                       f"{queryarea[1]}, {queryarea[2]}, "
                                       f"{queryarea[3]}]")

            elif isinstance(queryarea,str):
                sys.exit("This option is not yet available for FEMA USA Structures footprint data")

            print(f"\nFetching FEMA USA Structures footprint data for {queryarea_printname}...")

            query = f'https://services2.arcgis.com/FiaPA4ga0iQKduv3/ArcGIS/rest/services/USA_Structures_View/FeatureServer/0/query?geometry={queryarea[0]},{queryarea[1]},{queryarea[2]},{queryarea[3]}&geometryType=esriGeometryEnvelope&inSR=4326&spatialRel=esriSpatialRelIntersects&f=pjson'

            r = requests.get(query)
            if 'error' in r.text:
                sys.exit("Data server is currently unresponsive." +
                          " Please try again later.")

            datalist = r.json()['features']
            footprints = []
            for data in datalist:
                footprint = data['geometry']['rings'][0]
                footprints.append(footprint)

            print(f"\nFound a total of {len(footprints)} building footprints in {queryarea_printname}")
            return footprints    

        def polygon_area(lats, lons):
        
            radius = 20925721.784777 # Earth's radius in feet
            
            from numpy import arctan2, cos, sin, sqrt, pi, append, diff, deg2rad
            lats = deg2rad(lats)
            lons = deg2rad(lons)
        
            # Line integral based on Green's Theorem, assumes spherical Earth
        
            #close polygon
            if lats[0]!=lats[-1]:
                lats = append(lats, lats[0])
                lons = append(lons, lons[0])
        
            #colatitudes relative to (0,0)
            a = sin(lats/2)**2 + cos(lats)* sin(lons/2)**2
            colat = 2*arctan2( sqrt(a), sqrt(1-a) )
        
            #azimuths relative to (0,0)
            az = arctan2(cos(lats) * sin(lons), sin(lats)) % (2*pi)
        
            # Calculate diffs
            # daz = diff(az) % (2*pi)
            daz = diff(az)
            daz = (daz + pi) % (2 * pi) - pi
        
            deltas=diff(colat)/2
            colat=colat[0:-1]+deltas
        
            # Perform integral
            integrands = (1-cos(colat)) * daz
        
            # Integrate 
            area = abs(sum(integrands))/(4*pi)
        
            area = min(area,1-area)
            if radius is not None: #return in units of radius
                return area * 4*pi*radius**2
            else: #return in ratio of sphere total area
                return area
        
        def load_footprint_data(fpfile):
            """
            Function that loads footprint data from a GeoJSON file
            
            Input: A GeoJSON file containing footprint information
            Output: Footprint information parsed as a list of lists with each
                    coordinate described in longitude and latitude pairs   
            """
            
            def pluralsuffix(count):
                if count!=1:
                    suffix = 's'
                else:
                    suffix = ''
                return suffix
            
            with open(fpfile) as f:
                data = json.load(f)['features']

                footprints = []
                discardedfp_count = 0
                correctedfp_count = 0
                for count, loc in enumerate(data):
                    if loc['geometry']['type']=='Polygon':
                        temp_fp = loc['geometry']['coordinates']
                        if len(temp_fp)>1:
                            fp = temp_fp[:] 
                        elif len(temp_fp[0])>1:
                            fp = temp_fp[0][:] 
                        elif len(temp_fp[0][0])>1:
                            fp = temp_fp[0][0][:]
                        
                        if len(fp)==2:
                           list_len = [len(i) for i in fp]
                           fp = fp[list_len.index(max(list_len))]
                           correctedfp_count+=1
                        
                        footprints.append(fp)
                            
                    elif loc['geometry']['type']=='MultiPolygon':
                        discardedfp_count+=1   
                
                if discardedfp_count==0:   
                    print(f"Extracted a total of {len(footprints)} building footprints from {fpfile}")
                else: 
                    print(f"Corrected {correctedfp_count} building footprint{pluralsuffix(correctedfp_count)} with invalid geometry")
                    print(f"Discarded {discardedfp_count} building footprint{pluralsuffix(discardedfp_count)} with invalid geometry")
                    print(f"Extracted a total of {len(footprints)} building footprints from {fpfile}")
            return footprints

        def fp_source_selector(self):
            if self.fpSource=='osm':
                footprints = get_osm_footprints(self.queryarea)
            elif self.fpSource=='ms':
                 footprints = get_ms_footprints(self.queryarea)
            elif self.fpSource=='usastr':
                footprints = get_usastruct_footprints(self.queryarea)
            return footprints

        self.queryarea = queryarea
        self.fpSource = fpSource   
        
        if isinstance(self.queryarea,str):
            if 'geojson' in queryarea.lower():
                self.footprints = load_footprint_data(self.queryarea)
            else:
                self.footprints = fp_source_selector(self)
        elif isinstance(queryarea,tuple):
            self.footprints = fp_source_selector(self)
        elif isinstance(queryarea,list):    
            self.footprints = []
            for query in self.queryarea: 
                self.footprints.extend(fp_source_selector(self))
        else:
            sys.exit('Incorrect location entry. The location entry must be defined as' + 
                     ' 1) a string or a list of strings containing the name(s) of the query areas,' + 
                     ' 2) a string for the name of a GeoJSON file containing footprint data,' +
                     ' 3) or a tuple containing the coordinates for a rectangular' +
                     ' bounding box of interest in (lon1, lat1, lon2, lat2) format.' +
                     ' For defining a bounding box, longitude and latitude values' +
                     ' shall be entered for the vertex pairs of any of the two' +
                     ' diagonals of the rectangular bounding box.')   
             
        self.fpAreas = []
        for fp in self.footprints:
            lons = []
            lats = []
            for pt in fp:
                lons.append(pt[0])
                lats.append(pt[1])        
            self.fpAreas.append(polygon_area(lats, lons))