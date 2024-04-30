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
# 04-30-2024   

import math
import json
import requests
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import groupby
from shapely.geometry import Point, Polygon, LineString, MultiPolygon, box
from shapely.ops import linemerge, unary_union, polygonize
from shapely.strtree import STRtree
from brails.utils.geoTools import *
import concurrent.futures
from requests.adapters import HTTPAdapter, Retry
import unicodedata
import warnings
from brails.EnabledAttributes import BRAILStoR2D_BldgAttrMap

# Set a custom warning message format:
warnings.formatwarning = lambda message, category, filename, lineno, line=None: \
                         f"{category.__name__}: {message}\n"
warnings.simplefilter('always',UserWarning)                                

class FootprintHandler:
    def __init__(self): 
        self.attributes = {}
        self.availableDataSources = ['osm','ms','usastr']
        self.footprints = []
        self.fpSource = 'osm'
        self.lengthUnit = 'ft'
        self.queryarea = []
    
    def __fetch_roi(self,queryarea:str,outfile:str=''):
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
        headers = {'User-Agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1)'+
                                  ' AppleWebKit/537.36 (KHTML, like Gecko)'+
                                  ' Chrome/39.0.2171.95 Safari/537.36')}               
        r = requests.get(nominatimquery, headers=headers)
        datalist = r.json()
        
        areafound = False
        for data in datalist:
            queryarea_osmid = data['osm_id']
            queryarea_name = data['display_name']
            if data['osm_type']=='relation':
                areafound = True
                break
        
        if areafound==True:
            try:
                print(f"Found {queryarea_name}")
            except:
                queryareaNameUTF = unicodedata.normalize(
                    'NFKD', queryarea_name).encode('ascii', 'ignore')
                queryareaNameUTF = queryareaNameUTF.decode("utf-8")
                print(f"Found {queryareaNameUTF}") 
        else:
            sys.exit(f"Could not locate an area named {queryarea}. " + 
                     'Please check your location query to make sure ' +
                     'it was entered correctly.')
        
        queryarea_printname = queryarea_name.split(",")[0]  
        
        url = 'http://overpass-api.de/api/interpreter'
        
        # Get the polygon boundary for the query area:
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
        if outfile:
            write_polygon2geojson(bpoly,outfile)   
        return bpoly, queryarea_printname, queryarea_osmid
    
    def __bbox2poly(self,queryarea:tuple,outfile:str=''):
        # Parse the entered bounding box into a polygon:
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
                raise ValueError('Less than two longitude/latitude pairs were entered to define the bounding box entry. ' + 
                                 'A bounding box can be defined by using at least two longitude/latitude pairs.') 
        else:
                raise ValueError('Incorrect number of elements detected in the tuple for the bounding box. ' 
                                 'Please check to see if you are missing a longitude or latitude value.')  
        if outfile:
            write_polygon2geojson(bpoly,outfile)  
        return bpoly, queryarea_printname
    
    def __write_fp2geojson(self,footprints:list,attributes:dict,
                           outputFilename:str,convertKeys:bool=False):
        attrmap = BRAILStoR2D_BldgAttrMap(); attrmap['lat'] = 'Latitude'; 
        attrmap['lon'] = 'Longitude'; attrmap['fparea'] = 'PlanArea'
        attrkeys = list(attributes.keys())
        geojson = {'type':'FeatureCollection', 
                   "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                   'features':[]}
        for ind,fp in enumerate(footprints):
            feature = {'id': str(ind),
                       'type':'Feature',
                       'properties':{},
                       'geometry':{'type':'Polygon',
                                   'coordinates':[]}}
            feature['geometry']['coordinates'] = [fp]
            for key in attrkeys:
                attr = attributes[key][ind]
                if convertKeys:
                    keyout = attrmap[key]
                else:
                    keyout = key
                feature['properties'][keyout] = 'NA' if attr is None else attr  
            feature['properties']['type'] = 'Building'
            geojson['features'].append(feature)
            
        with open(outputFilename, 'w') as outputFile:
            json.dump(geojson, outputFile, indent=2)    
    
    def fetch_footprint_data(self,queryarea,fpSource:str='osm',attrmap:str='',
                             lengthUnit:str='ft',outputFile:str=''):
        """
        Function that loads footprint data from OpenStreetMap, Microsoft, USA
        Structures, user-defined data
        
        Input: Location entry defined as a string or a list of strings 
               containing the area name(s) or a tuple containing the longitude
               and longitude pairs for the bounding box of the area of interest
               or a GeoJSON file containing footprint (or point location) and
               inventory data 
        Output: Footprint information parsed as a list of lists with each
                coordinate described in longitude and latitude pairs   
        """
        
        def get_osm_footprints(queryarea,lengthUnit='ft'):              
            def cleanstr(inpstr):
                return ''.join(char for char in inpstr if not char.isalpha()
                               and not char.isspace() and 
                               (char == '.' or char.isalnum()))
            
            def yearstr2int(inpstr):
                if inpstr!='NA':
                    yearout =  cleanstr(inpstr)
                    yearout = yearout[:4]
                    if len(yearout)==4:
                        try:
                            yearout = int(yearout)
                        except:
                            yearout = None
                    else:
                        yearout = None
                else:
                    yearout = None
                return yearout
            
            def height2float(inpstr,lengthUnit):    
                if inpstr!='NA':
                    heightout =  cleanstr(inpstr)
                    try:
                        if lengthUnit=='ft':
                            heightout = round(float(heightout)*3.28084,1)
                        else:
                            heightout = round(float(heightout),1)
                    except:
                        heightout = None
                else:
                    heightout = None
                return heightout

            if isinstance(queryarea,str):
                bpoly, queryarea_printname, osmid = self.__fetch_roi(queryarea)
                queryarea_turboid = osmid + 3600000000
                query = f"""
                [out:json][timeout:5000][maxsize:2000000000];
                area({queryarea_turboid})->.searchArea;
                way["building"](area.searchArea);
                out body;
                >;
                out skel qt;
                """
            elif isinstance(queryarea,tuple):
                bpoly, queryarea_printname = self.__bbox2poly(queryarea) 
                if len(queryarea)==4:
                    bbox = [min(queryarea[1],queryarea[3]),
                            min(queryarea[0],queryarea[2]),
                            max(queryarea[1],queryarea[3]),
                            max(queryarea[0],queryarea[2])] 
                    bbox = f'{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}' 
                elif len(queryarea)>4:
                    bbox = 'poly:"'
                    for i in range(int(len(queryarea)/2)):
                        bbox+=f'{queryarea[2*i+1]} {queryarea[2*i]} ' 
                    bbox = bbox[:-1]+'"'
                
                query = f"""
                [out:json][timeout:5000][maxsize:2000000000];
                way["building"]({bbox});
                out body;
                >;
                out skel qt;
                """
            
            url = 'http://overpass-api.de/api/interpreter'   
            r = requests.get(url, params={'data': query})
            
            datalist = r.json()['elements']
            nodedict = {}
            for data in datalist:
                if data['type']=='node':
                   nodedict[data['id']] = [data['lon'],data['lat']]
            
            attrmap = {'start_date':'erabuilt',
                       'building:start_date':'erabuilt',
                       'construction_date':'erabuilt',
                       'roof:shape':'roofshape',
                       'height':'buildingheight',           
                       }
            
            levelkeys = {'building:levels','roof:levels'} # Excluding 'building:levels:underground'
            otherattrkeys = set(attrmap.keys())
            datakeys = levelkeys.union(otherattrkeys)
            
            attrkeys = ['buildingheight','erabuilt','numstories','roofshape']
            attributes = {key: [] for key in attrkeys}
            fpcount = 0
            footprints = []
            for data in datalist:
                if data['type']=='way':
                    nodes = data['nodes']
                    footprint = []
                    for node in nodes:
                        footprint.append(nodedict[node])
                    footprints.append(footprint)
                    fpcount+=1
                    availableTags = set(data['tags'].keys()).intersection(datakeys)
                    for tag in availableTags:
                        nstory = 0
                        if tag in otherattrkeys:
                           attributes[attrmap[tag]].append(data['tags'][tag])
                        elif tag in levelkeys:
                            try:
                                nstory+=int(data['tags'][tag]) 
                            except:
                                pass
                        if nstory>0:
                            attributes['numstories'].append(nstory)
                    for attr in attrkeys:
                        if len(attributes[attr])!=fpcount:
                            attributes[attr].append('NA')
            attributes['buildingheight'] = [height2float(height,lengthUnit)
                                            for height in attributes['buildingheight']]
            attributes['erabuilt'] = [yearstr2int(year) for year in attributes['erabuilt']]            
            attributes['numstories'] = [nstories if nstories!='NA' else None 
                                        for nstories in attributes['numstories']] 
            
            print(f"\nFound a total of {fpcount} building footprints in {queryarea_printname}")           
            return footprints, attributes
        
        def get_ms_footprints(queryarea,lengthUnit='ft'):
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
                

                # Define length unit conversion factor:
                if lengthUnit=='ft':
                    convFactor = 3.28084
                else:
                    convFactor = 1
                 
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
                            height = row['properties']['height']
                            if height!=-1:
                                bldgheights.append(round(height*convFactor,1))
                            else:
                                bldgheights.append(None)
            
                return (footprints, bldgheights)
                        
            if isinstance(queryarea,tuple):
                bpoly, queryarea_printname = self.__bbox2poly(queryarea)   
            elif isinstance(queryarea,str):
                bpoly, queryarea_printname, _ = self.__fetch_roi(queryarea)
            print(f"\nFetching Microsoft building footprints for {queryarea_printname}...")                                                
            
            quadkeys = bbox2quadkeys(bpoly)
            attributes = {'buildingheight':[]}
            (footprints, attributes['buildingheight']) = download_ms_tiles(quadkeys, bpoly)
            
            print(f"\nFound a total of {len(footprints)} building footprints in {queryarea_printname}")
            return footprints,attributes
     
        def get_usastruct_footprints(queryarea,lengthUnit='ft'):          
            def get_usastruct_bldg_counts(bpoly):
                # Get the coordinates of the bounding box for input polygon bpoly:
                bbox = bpoly.bounds 
                
                # Get the number of buildings in the computed bounding box:
                query = ('https://services2.arcgis.com/FiaPA4ga0iQKduv3/ArcGIS/' + 
                         'rest/services/USA_Structures_View/FeatureServer/0/query?' +
                         f'geometry={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}' +
                         '&geometryType=esriGeometryEnvelope&inSR=4326' + 
                         '&spatialRel=esriSpatialRelIntersects' +
                         '&returnCountOnly=true&f=json')
                
                s = requests.Session()
                retries = Retry(total=5, 
                                backoff_factor=0.1,
                                status_forcelist=[500, 502, 503, 504])
                s.mount('https://', HTTPAdapter(max_retries=retries))

                r = s.get(query)
                totalbldgs = r.json()['count']
                
                return totalbldgs
                
            def get_polygon_cells(bpoly, totalbldgs = None, nfeaturesInCell=4000, 
                                  plotfout=False): 
                if totalbldgs is None:
                    # Get the number of buildings in the input polygon bpoly:
                    totalbldgs = get_usastruct_bldg_counts(bpoly)
                
                if totalbldgs>nfeaturesInCell:
                    # Calculate the number of cells required to cover the polygon area with
                    # 20 percent margin of error:
                    ncellsRequired = round(1.2*totalbldgs/nfeaturesInCell)
                    
                    # Get the coordinates of the bounding box for input polygon bpoly:
                    bbox = bpoly.bounds 
                    
                    # Calculate the horizontal and vertical dimensions of the bounding box:
                    xdist = haversine_dist((bbox[0],bbox[1]),(bbox[2],bbox[1]))
                    ydist = haversine_dist((bbox[0],bbox[1]),(bbox[0],bbox[3]))
                    
                    # Determine the bounding box aspect ratio defined (as a number greater
                    # than 1) and the long direction of the bounding box: 
                    if xdist>ydist:
                        bboxAspectRatio = math.ceil(xdist/ydist)
                        longSide = 1
                    else:
                        bboxAspectRatio = math.ceil(ydist/xdist)
                        longSide = 2
                    
                    # Calculate the cells required on the short side of the bounding box (n)
                    # using the relationship ncellsRequired = bboxAspectRatio*n^2:
                    n = math.ceil(math.sqrt(ncellsRequired/bboxAspectRatio))
                    
                    # Based on the calculated n value determined the number of rows and 
                    # columns of cells required:
                    if longSide==1:
                        rows = bboxAspectRatio*n
                        cols = n
                    else:
                        rows = n
                        cols = bboxAspectRatio*n
                    
                    # Determine the coordinates of each cell covering bpoly:    
                    rectangles = mesh_polygon(bpoly, rows, cols)
                else:
                    rectangles = [bpoly.envelope]
                # Plot the generated mesh:
                if plotfout:        
                    plot_polygon_cells(bpoly, rectangles, plotfout)
                
                return rectangles

            def refine_polygon_cells(premCells, nfeaturesInCell=4000):    
                # Download the building count for each cell:
                pbar = tqdm(total=len(premCells), desc='Obtaining the number of buildings in each cell')     
                results = {}             
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_to_url = {
                        executor.submit(get_usastruct_bldg_counts, rect): rect
                        for rect in premCells
                    }
                    for future in concurrent.futures.as_completed(future_to_url):
                        rect = future_to_url[future]
                        pbar.update(n=1)
                        try:
                            results[rect] = future.result()
                        except Exception as exc:
                            results[rect] = None
                            print("%r generated an exception: %s" % (rect, exc))
                
                indRemove = []
                cells2split = []
                cellsKeep = premCells.copy()
                for ind,rect in enumerate(premCells):
                    totalbldgs = results[rect]
                    if totalbldgs is not None:
                        if totalbldgs==0:
                            indRemove.append(ind)
                        elif totalbldgs>nfeaturesInCell:
                            indRemove.append(ind)
                            cells2split.append(rect)
                
                for i in sorted(indRemove, reverse=True):
                    del cellsKeep[i]
                
                cellsSplit = []
                for rect in cells2split:
                    rectangles = get_polygon_cells(rect, totalbldgs=results[rect])
                    cellsSplit+=rectangles     
                
                return cellsKeep, cellsSplit

            def download_ustruct_bldgattr(cell):
                rect = cell.bounds
                s = requests.Session()
                retries = Retry(total=5, 
                                backoff_factor=0.1,
                                status_forcelist=[500, 502, 503, 504])
                s.mount('https://', HTTPAdapter(max_retries=retries))
                query = ('https://services2.arcgis.com/FiaPA4ga0iQKduv3/ArcGIS/' + 
                         'rest/services/USA_Structures_View/FeatureServer/0/query?' +
                         f'geometry={rect[0]},{rect[1]},{rect[2]},{rect[3]}' +
                         '&outFields=BUILD_ID,HEIGHT'+
                         '&geometryType=esriGeometryEnvelope&inSR=4326' + 
                         '&spatialRel=esriSpatialRelIntersects&outSR=4326&f=json')
                
                r = s.get(query)
                datalist = r.json()['features']
                ids = []
                footprints = []
                bldgheight = []
                for data in datalist:
                    footprint = data['geometry']['rings'][0]
                    bldgid = data['attributes']['BUILD_ID']
                    if bldgid not in ids:
                        ids.append(bldgid)
                        footprints.append(footprint)
                        height = data['attributes']['HEIGHT']
                        try:
                            height = float(height)
                        except:
                            height = None
                        bldgheight.append(height)
                
                return (ids, footprints, bldgheight)

            def download_ustruct_bldgattr4region(cellsFinal,bpoly):
                # Download building attribute data for each cell:
                pbar = tqdm(total=len(cellsFinal), desc='Obtaining the building attributes for each cell')     
                results = {}             
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_to_url = {
                        executor.submit(download_ustruct_bldgattr, cell): cell
                        for cell in cellsFinal
                    }
                    for future in concurrent.futures.as_completed(future_to_url):
                        cell = future_to_url[future]
                        pbar.update(n=1)
                        try:
                            results[cell] = future.result()
                        except Exception as exc:
                            results[cell] = None
                            print("%r generated an exception: %s" % (cell, exc))
                pbar.close() 
                
                # Parse the API results into building id, footprints and height
                # information:    
                ids = []
                footprints = []
                bldgheight = []
                for cell in tqdm(cellsFinal):
                    res = results[cell]
                    ids+=res[0]
                    footprints+=res[1]
                    bldgheight+=res[2]

                # Remove the duplicate footprint data by recording the API 
                # outputs to a dictionary:
                data = {}
                for ind,bldgid in enumerate(ids):
                    data[bldgid] = [footprints[ind],bldgheight[ind]]
                
                # Define length unit conversion factor:
                if lengthUnit=='ft':
                    convFactor = 3.28084
                else:
                    convFactor = 1

                # Calculate building centroids and save the API outputs into 
                # their corresponding variables:
                footprints = []
                attributes = {'buildingheight':[]}
                centroids = [] 
                for value in data.values():
                    fp = value[0]
                    centroids.append(Polygon(fp).centroid)
                    footprints.append(fp)
                    heightout = value[1]
                    if heightout is not None:
                        attributes['buildingheight'].append(
                            round(heightout*convFactor,1)) 
                    else:
                        attributes['buildingheight'].append(None)
             
                # Identify building centroids and that fall outside of bpoly:   
                results = {} 
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_to_url = {
                        executor.submit(bpoly.contains, cent): cent
                        for cent in centroids
                    }
                    for future in concurrent.futures.as_completed(future_to_url):
                        cent = future_to_url[future]
                        try:
                            results[cent] = future.result()
                        except Exception as exc:
                            results[cell] = None
                            print("%r generated an exception: %s" % (cent, exc))    
                indRemove = []
                for ind, cent in enumerate(centroids):
                    if not results[cent]:
                       indRemove.append(ind) 
                
                # Remove data corresponding to centroids that fall outside bpoly:
                for i in sorted(indRemove, reverse=True):
                    del footprints[i]
                    del attributes['buildingheight'][i]
                
                return footprints, attributes 
            
            if isinstance(queryarea,tuple):
                bpoly, queryarea_printname = self.__bbox2poly(queryarea)  
            elif isinstance(queryarea,str):
                bpoly, queryarea_printname, _ = self.__fetch_roi(queryarea)
             
            ####################### For debugging only #######################  
            #plotCells = True
            plotCells = False
            if plotCells:
                meshInitialfout = queryarea_printname.replace(' ','_') + '_Mesh_Initial.png'
                meshFinalfout = queryarea_printname.replace(' ','_') + '_Mesh_Final.png'
            else:
                meshInitialfout = False
                meshFinalfout = False
            ##################################################################
                
            print('\nMeshing the defined area...')
            cellsPrem = get_polygon_cells(bpoly, plotfout=meshInitialfout)    

            if len(cellsPrem)>1:
                cellsFinal = []
                cellsSplit = cellsPrem.copy()
                while len(cellsSplit)!=0:
                    cellsKeep, cellsSplit = refine_polygon_cells(cellsSplit)  
                    cellsFinal+=cellsKeep
                print(f'\nMeshing complete. Split {queryarea_printname} into {len(cellsFinal)} cells')
            else:
                cellsFinal = cellsPrem.copy()
                print(f'\nMeshing complete. Covered {queryarea_printname} with a rectangular cell')
            
            ####################### For debugging only #######################     
            if plotCells:
                plot_polygon_cells(bpoly, cellsFinal, meshFinalfout)
            #################################################################
                
            footprints, attributes = download_ustruct_bldgattr4region(cellsFinal,bpoly)
            print(f"\nFound a total of {len(footprints)} building footprints in {queryarea_printname}")      

            return footprints, attributes    
        
        def load_footprint_data(fpfile,fpSource,attrmapfile,lengthUnit):
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

            def get_bbox(points):
                """
                Function that determines the extent of the area covered by a set of points
                as a tight-fit rectangular bounding box
                
                Input:  List of point data defined as a list of coordinates in EPSG 4326,
                        i.e., [longitude,latitude].
                Output: Tuple containing the minimum and maximum latitude and 
                        longitude values 
               """     
               # :
                minlat = points[0][1]
                minlon = points[0][0]
                maxlat = points[0][1]
                maxlon = points[0][0]
                for pt in points:
                    if pt[1]>maxlat:
                        maxlat = pt[1]
                    if pt[0]>maxlon:
                        maxlon = pt[0]
                    if pt[1]<minlat:
                        minlat = pt[1]
                    if pt[0]<minlon:
                        minlon = pt[0]
                return (minlon-0.001,minlat-0.001,maxlon+0.001,maxlat+0.001)
            
            def fp_download(bbox,fpSource):
                if fpSource=='osm':
                    footprints, _ = get_osm_footprints(bbox)
                elif fpSource=='ms':
                     footprints,_ = get_ms_footprints(bbox)
                elif fpSource=='usastr':
                    footprints, _ = get_usastruct_footprints(bbox)
                return footprints
           
            def parse_fp_geojson(data, attrmap, attrkeys, fpfile):     
                # Create the attribute fields that will be extracted from the 
                # GeoJSON file:
                attributes = {}
                attributestr = [attr for attr in attrmap.values() if attr!='']
                for attr in attributestr:
                    attributes[attr] = [] 
                
                footprints_out = []
                discardedfp_count = 0
                correctedfp_count = 0       
                for loc in data:
                    # If the footprint is a polygon:
                    if loc['geometry']['type']=='Polygon':
                        # Read footprint coordinates:
                        temp_fp = loc['geometry']['coordinates']
                        
                        # Check down to two levels deep into extracted JSON
                        # structure to account for inconsistencies in the 
                        # provided footprint data 
                        if len(temp_fp)>1:
                            fp = temp_fp[:] 
                        elif len(temp_fp[0])>1:
                            fp = temp_fp[0][:] 
                        elif len(temp_fp[0][0])>1:
                            fp = temp_fp[0][0][:]
                        
                        # If mutliple polygons are detected for a location, 
                        # take the outermost polygon:
                        if len(fp)==2:
                           list_len = [len(i) for i in fp]
                           fp = fp[list_len.index(max(list_len))]
                           correctedfp_count+=1
                        
                        # Add the footprint and attributes to the output 
                        # variables
                        footprints_out.append(fp)  
                        if attrkeys:
                            for key in attrkeys:
                                try:
                                    attributes[attrmap[key]].append(loc['properties'][key])
                                except:
                                    pass
                    # If the footprint is a multi-polygon, discard the footprint:        
                    elif loc['geometry']['type']=='MultiPolygon':
                        discardedfp_count+=1              

                # Print the results of the footprint extraction:
                if discardedfp_count!=0:   
                    print(f"Corrected {correctedfp_count} building footprint{pluralsuffix(correctedfp_count)} with invalid geometry")
                    print(f"Discarded {discardedfp_count} building footprint{pluralsuffix(discardedfp_count)} with invalid geometry")
                print(f"Extracted a total of {len(footprints_out)} building footprints from {fpfile}\n")
                
                return (footprints_out, attributes)
           
            def parse_pt_geojson(data, attrmap, attrkeys, fpSource):   
                # Create the attribute fields that will be extracted from the 
                # GeoJSON file:
                attributes = {}
                attributestr = [attr for attr in attrmap.values() if attr!='']
                for attr in attributestr:
                    attributes[attr] = []                    
     

                # Write the data in datalist into a dictionary for better data access,
                # and filtering the duplicate entries:
                datadict = {}
                ptcoords = []
                for loc in data:
                    ptcoord = loc['geometry']['coordinates']
                    ptcoords.append(ptcoord)
                    pt = Point(ptcoord)
                    datadict[pt] = loc['properties']
                
                points = list(datadict.keys())
                
                # Determine the coordinates of the bounding box containing the 
                # points:
                bbox = get_bbox(ptcoords)      
                
                # Get the footprint data corresponding to the point GeoJSON
                # input:
                if 'geojson' in fpSource.lower():
                    with open(fpSource) as f:
                        data = json.load(f)['features']
                    (footprints,_) = parse_fp_geojson(data, {}, {}, fpSource)
                else:
                    footprints = fp_download(bbox,fpSource)
                
                # Create an STR tree for efficient parsing of point coordinates:
                pttree = STRtree(points)
                
                # Find the data points that are enclosed in each footprint:
                ress = []
                for fp in footprints:
                    res = pttree.query(Polygon(fp))
                    if res.size!=0:
                        ress.append(res)
                    else:
                        ress.append(np.empty(shape=(0, 0)))       
                    
                # Match point data to each footprint:
                footprints_out = []
                for ind, fp in enumerate(footprints):
                    if ress[ind].size!=0:
                        footprints_out.append(fp) 
                        ptind = ress[ind][0]
                        ptres = datadict[points[ptind]]
                        for key in attrkeys:
                            try:
                                attributes[attrmap[key]].append(ptres[key])
                            except:
                                pass
                
                return (footprints_out, attributes)                

            # Read the GeoJSON file and check if all the data in the file is 
            # point data:
            with open(fpfile) as f:
                data = json.load(f)['features']
            ptdata = all(loc['geometry']['type']=='Point' for loc in data) 
            
            if attrmapfile:
                # Create a dictionary for mapping the attributes in the GeoJSON 
                # file to BRAILS inventory naming conventions:
                with open(attrmapfile) as f:
                    lines = [line.rstrip() for line in f]
                    attrmap = {}
                    for line in lines:
                        lout = line.split(':')
                        if lout[1]!='' and lout[0]!='LengthUnit':
                            attrmap[lout[1]] = lout[0] 
                        else:
                            attrmap[lout[0]] = lout[1]
                            
                # Identify the attribute keys in the GeoJSON file:
                attrkeys0 = list(data[0]['properties'].keys())
                if attrkeys0:
                    print('Building attributes detected in the input GeoJSON: ' +
                          ', '.join(attrkeys0))
                    
                # Check if all of the attribute keys in the GeoJSON have 
                # correspondence in the map. Ignore the keys that do not have 
                # correspondence:
                attrkeys = set()
                for key in attrkeys0:
                    try:
                        attrmap[key]
                        attrkeys.add(key)
                    except:
                        pass            
                ignored_Attr = set(attrkeys0) - attrkeys
                if ignored_Attr:
                    print('\nAttribute mapping does not cover all attributes detected in' 
                          ' the input GeoJSON. Ignoring detected attributes '
                          '(building positions extracted from geometry info): ' +
                          ', '.join(ignored_Attr) + '\n')
            else:
                attrmap = {}
                attrkeys = {}
            
            lengthUnitInp = ''
            if 'LengthUnit' in attrmap.keys():          
                # Get the length unit for the input data:                
                if attrmap['LengthUnit']!='':
                    lengthUnitInp = attrmap['LengthUnit'].lower()[0]
                    if lengthUnitInp=='f': lengthUnitInp='ft'
                del attrmap['LengthUnit']
            
            if ptdata:
                (footprints_out, attributes) = parse_pt_geojson(data, 
                                                                attrmap, 
                                                                attrkeys, 
                                                                fpSource)  
            else:  
                (footprints_out, attributes) = parse_fp_geojson(data, 
                                                                attrmap, 
                                                                attrkeys, 
                                                                fpfile)
                fpSource = fpfile
            
            if 'BldgHeight' in attributes.keys():
                if lengthUnitInp!='' and lengthUnitInp!=lengthUnit:
                    if lengthUnit=='ft':
                        convFactor = 3.28084
                    else:
                        convFactor = 1/3.28084
                elif lengthUnitInp!='' and lengthUnitInp==lengthUnit:
                    convFactor = 1
                else:
                    convFactor = None
                
                if convFactor is not None:
                    bldgheights = attributes['BldgHeight'].copy()
                    bldgheightsConverted = []
                    for height in bldgheights:
                        try:
                            heightFloat = round(height*convFactor,1)
                        except:
                            heightFloat = 0
                        bldgheightsConverted.append(heightFloat)
                    del attributes['BldgHeight']
                    attributes['BldgHeight'] = bldgheightsConverted

            return footprints_out, attributes, fpSource

        def polygon_area(lats,lons,lengthUnit):
        
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
        
            # Area in ratio of sphere total area:
            area = min(area,1-area)
            
            # Area in sqft:
            areaout = area * 4*pi*radius**2
            
            # Area in sqm:
            if lengthUnit=='m': areaout = areaout/(3.28084**2)
            
            return areaout
            
        def fp_source_selector(self):
            if self.fpSource=='osm':
                footprints, attributes = get_osm_footprints(self.queryarea,lengthUnit)
            elif self.fpSource=='ms':
                footprints, attributes = get_ms_footprints(self.queryarea,lengthUnit)
            elif self.fpSource=='usastr':
                footprints, attributes = get_usastruct_footprints(self.queryarea,lengthUnit)
            else:
                warnings.warn('Unimplemented footprint source. Setting footprint source to OSM',
                              UserWarning)
                footprints, attributes = get_osm_footprints(self.queryarea,lengthUnit)
            return footprints, attributes

        self.queryarea = queryarea
        self.fpSource = fpSource
        if isinstance(self.queryarea,str):
            if 'geojson' in queryarea.lower():
                footprints,self.attributes,self.fpSource = load_footprint_data(
                    self.queryarea,
                    self.fpSource,
                    attrmap,
                    lengthUnit)
            else:
                footprints,self.attributes = fp_source_selector(self)
        elif isinstance(queryarea,tuple):
            footprints,self.attributes = fp_source_selector(self)
        elif isinstance(queryarea,list):    
            for query in self.queryarea: 
                (fps, attributes) = fp_source_selector(query)
                attrkeys = list(attributes.keys())
                footprints.extend(fps)
                for key in attrkeys:
                    self.attributes[key].extend(attributes[key])
        else:
            sys.exit('Incorrect location entry. The location entry must be defined as' + 
                     ' 1) a string or a list of strings containing the name(s) of the query areas,' + 
                     ' 2) a string for the name of a GeoJSON file containing footprint data,' +
                     ' 3) or a tuple containing the coordinates for a rectangular' +
                     ' bounding box of interest in (lon1, lat1, lon2, lat2) format.' +
                     ' For defining a bounding box, longitude and latitude values' +
                     ' shall be entered for the vertex pairs of any of the two' +
                     ' diagonals of the rectangular bounding box.')  
           
        self.attributes['fparea'] = []
        for fp in footprints:
            lons = []
            lats = []
            for pt in fp:
                lons.append(pt[0])
                lats.append(pt[1])        
            self.attributes['fparea'].append(int(polygon_area(lats,lons,
                                                               lengthUnit)))

        # Calculate centroids of the footprints and remove footprint data that
        # does not form a polygon:
        self.footprints = []
        self.centroids = []
        indRemove = []
        for (ind,footprint) in enumerate(footprints):
            try:
                self.centroids.append(Polygon(footprint).centroid)
                self.footprints.append(footprint)
            except:
                indRemove.append(ind)
                pass
        
        # Remove attribute corresponding to the removed footprints:
        for i in sorted(indRemove, reverse=True):
            for key in self.attributes.keys():
                del self.attributes[key][i]
                       
        # Write the footprint data into a GeoJSON file:
        if outputFile:
            self.__write_fp2geojson(self.footprints, self.attributes, outputFile)