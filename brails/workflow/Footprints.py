# -*- coding: utf-8 -*-
"""
/*------------------------------------------------------*
|                         BRAILS                        |
|                                                       |
| Author: Charles Wang,  UC Berkeley, c_w@berkeley.edu  |
|                                                       |
| Date:    1/19/2021                                    |
*------------------------------------------------------*/
"""

import os
from pathlib import Path
import wget
import requests 
import json
import shapely
from sys import exit
#from brails.utils.geoDicts import *
from shapely.geometry import Point, Polygon, MultiPolygon
from zipfile import ZipFile
#import geopandas as gpd
import osmnx as ox
import json


#workDir = 'tmp'

#baseurl = 'https://nominatim.openstreetmap.org/search?city={address}&county={county}&state={state}&polygon_geojson=1&format=json'
baseurl = 'https://nominatim.openstreetmap.org/search?city={city}&county={county}&state={state}&country={country}&polygon_geojson=1&format=json'

def getPlaceBoundary(city='',county='',state='',country='',save=True,fileName='',workDir='tmp'):
    """Function for downloading a city boundary.

    Args:
        city (str): Name of the city or county.
        state (str): State of the city.
        save (bool): Save file locally.
        fileName (str): Path of the file.

    Returns:
        boundary (shapely.geometry.multipolygon.MultiPolygon): boundary
        fileName (str): path of the boundary file

    """
    r = requests.get(baseurl.format(city=city,county=county,state=state,country=country))
    if r.status_code==200:
        r = r.json()
        if r:
            btype = r[0]['geojson']['type']

            if btype in ['MultiPolygon']:
                pts_l = r[0]['geojson']['coordinates'][0]
                boundary=[]
                for pts in pts_l:
                    boundary.append(Polygon(pts))
                boundary = MultiPolygon(boundary)
            elif btype in ['Polygon']:
                boundary = r[0]['geojson']['coordinates'][0]
                boundary = Polygon(boundary)
            else:
                pts = r[0]['geojson']['coordinates'][0]
                boundary = Polygon(pts)

            # save file
            if save:
                if fileName=='': 
                    bfilename = f'{city}_{state}_{country}_boundary.geojson'.replace(' ','_')
                    fileName=f'{workDir}/{bfilename}'
                with open(fileName, 'w') as outfile:
                    json.dump(shapely.geometry.mapping(boundary), outfile)
                print('{} saved.'.format(fileName))
            return boundary, fileName

        else:
            print(f'OSM didn\'t find {city} {county} {state} {country}. Exit.')
            return -1, -1
    else: 
        print('Network issue. Exit.')



def getMSFootprintsByPlace(place='',save=True,fileName='',workDir='tmp',GoogleMapAPIKey='',overwrite=False):
    """Function for downloading a building footprints.

    Args:
        place (str): Name of the city or county.
        save (bool): Save file locally.
        fileName (str): Path of the file.

    Returns:
        allFootprints (geopandas dataframe): Footprints
        fileName (str): Path to the footprint file.

    """

    if fileName=='': fileName = Path('{workDir}/{city}_footprints_MS.geojson'.format(workDir = workDir, city=place).replace(' ','_').replace(',','_'))

    if os.path.exists(fileName) and not overwrite:
        print('{} already exists.'.format(fileName))
        allFootprints = gpd.read_file(fileName)

    else:

        #stateName = stateNames[state.upper()]
        city, stateName, country = getStateNameByPlace(place=place,workDir=workDir,GoogleMapAPIKey=GoogleMapAPIKey)
        
        footprintFilename = Path('{}/{}'.format(workDir, footprintFiles[stateName].split('/')[-1]))

        # download
        if not footprintFilename.exists():
            print('Downloading footprints...')
            footprintFilename = wget.download(footprintFiles[stateName], out=workDir)
            print('{} downloaded'.format(footprintFilename))
        else: print('{} exists.'.format(footprintFilename))

        # unzip
        with ZipFile(footprintFilename, 'r') as zip: 
            footprintGeojsonFilename = Path('{}/{}'.format(workDir, zip.namelist()[0]))
            if not footprintGeojsonFilename.exists():

                print('Extracting all files now...') 
                zip.extractall(workDir) 
                print('Extraction done!') 
            else: print('{} exists.'.format(footprintGeojsonFilename))

        # slice
        placeBoundary, boundaryFilename = getPlaceBoundary(city=city,county='',state=stateName,country=country)#(place=place,state=stateName)
        if boundaryFilename==-1 or placeBoundary==-1:
            assert False, f"Couldn't find a boundary file for the place {place} from OSM with the parameters {city} {stateName} {country}"

        print(f'Loading all footprints from {boundaryFilename}...')
        gdf_mask = gpd.read_file(boundaryFilename)
        allFootprints = gpd.read_file(footprintGeojsonFilename,mask=gdf_mask,)

        # save
        if allFootprints.shape[0]<1:
            print('Didn\'t find footprints for this region.')
        else:
            allFootprints.to_file(fileName, driver='GeoJSON')
            print('Footprint saved at {}'.format(fileName))

    return allFootprints, fileName

def getStateNameByPlace(place='',workDir='tmp',GoogleMapAPIKey=''):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={place}&key={GoogleMapAPIKey}"
    r = requests.get(url)
    place = place.replace(' ','_').replace(',','_')
    placeinfo = Path(f'{workDir}/{place}_placeinfo.json')
    f = open(placeinfo, 'wb')
    f.write(r.content)
    f.close()

    j = r.json()
    if j['status']=='OK':
        l = j['results'][0]['formatted_address'].split(',')
        print(l)
        country = l[-1].strip()
        stateCode = l[-2].strip()
        city = l[-3].strip()
        if country=='USA':
            return city, stateNames[stateCode], country
        else: return None,None,None

    else: 
        return None,None,None

def getStateNameByBBox(bbox,workDir='tmp',GoogleMapAPIKey=''):

    if len(bbox)!=4: 
        assert False, f"bbox should be a list with 4 elements. Your bbox {bbox} doesn't work."
        return None
    
    north = bbox[0]
    west = bbox[1]
    south = bbox[2]
    east = bbox[3]
    lon = (west+east)/2.
    lat = (north+south)/2.
    reverseURL = f'https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}&key={GoogleMapAPIKey}'
    #print(reverseURL)

    r = requests.get(reverseURL)
    placeinfo = Path(f'{workDir}/{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}_placeinfo.json')
    f = open(placeinfo, 'wb')
    f.write(r.content)
    f.close()

    j = r.json()
    if j['status']=='OK':
        l = j['plus_code']['compound_code'].split(',')
        country = l[-1].strip()
        stateCode = l[-2].strip()
        if country=='USA':
            return stateNames[stateCode]
        else: return None

    else: 
        return None



def getMSFootprintsByBbox(bbox=[],stateName='',save=True,fileName='',workDir='tmp',overwrite=False,GoogleMapAPIKey=''):
    """Function for downloading a building footprints.

    Args:
        place (str): Name of the city or county.
        state (str): Abbreviation of state name
        save (bool): Save file locally.
        fileName (str): Path of the file.

    Returns:
        allFootprints (geopandas dataframe): Footprints
        fileName (str): Path to the footprint file.

    """

    if fileName=='': fileName = Path(f'{workDir}/{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}_footprints.geojson')

    if os.path.exists(fileName) and not overwrite:
        print('{} already exists.'.format(fileName))
        allFootprints = gpd.read_file(fileName)

    else:

        if stateName == '':
            #stateName = stateNames[state.upper()]
            stateName = getStateNameByBBox(bbox,workDir=workDir,GoogleMapAPIKey=GoogleMapAPIKey)
            if stateName is None:
                print('Didn\'t find the state name by the bbox you provided.')
                return None, None
        
        footprintFilename = Path('{}/{}'.format(workDir, footprintFiles[stateName].split('/')[-1]))

        # download
        if not footprintFilename.exists():
            print('Downloading footprints...')
            footprintFilename = wget.download(footprintFiles[stateName], out=workDir)
            print('{} downloaded'.format(footprintFilename))
        else: print('{} exists.'.format(footprintFilename))

        # unzip
        with ZipFile(footprintFilename, 'r') as zip: 
            footprintGeojsonFilename = Path('{}/{}'.format(workDir, zip.namelist()[0]))
            if not footprintGeojsonFilename.exists():

                print('Extracting all files now...') 
                zip.extractall(workDir) 
                print('Extraction done!') 
            else: print('{} exists.'.format(footprintGeojsonFilename))

        # slice
        print('Reading footprints file ...')
        allFootprints = gpd.read_file(footprintGeojsonFilename,bbox=(bbox[1],bbox[0],bbox[3],bbox[2]))

        # save
        if allFootprints.shape[0]<1:
            print('Didn\'t find footprints for this region.')
        else:
            allFootprints.to_file(fileName, driver='GeoJSON')
            print('Footprint saved at {}'.format(fileName))

    return allFootprints, fileName

def getOSMFootprints(bbox=[],place='',save=True,fileName='',workDir='tmp',overwrite=False):
    """Function for downloading a building footprints.

    Args:
        place (str): Name of the city or county.
        save (bool): Save file locally.
        fileName (str): Path of the file.

    Returns:
        allFootprints (geopandas dataframe): Footprints
        fileName (str): Path to the footprint file.

    """
    
    if fileName=='': 

        if len(bbox)>0: 
            fileName = Path(f'{workDir}/{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}_footprints.geojson')
        else:
            fileName = Path(f'{workDir}/{place}_footprints_OSM.geojson'.replace(' ','_').replace(',','_'))


    if os.path.exists(fileName) and not overwrite:
        print('{} already exists.'.format(fileName))
        allFootprints = gpd.read_file(fileName)

    else:
        if len(bbox)>0:
            allFootprints = ox.geometries_from_bbox(bbox[0],bbox[2],bbox[3],bbox[1],tags = {'building': True})# (north,west,south,east)->(north, south, east, west, tags)
        else:
            allFootprints = ox.geometries_from_place(place, tags = {'building': True})
        
        allFootprints = allFootprints[['geometry']]
        allFootprints = allFootprints[allFootprints['geometry'].type == 'Polygon']

        # save
        if allFootprints.shape[0]<1:
            print('Didn\'t find footprints for this region.')
        else:
            allFootprints.to_file(fileName, driver='GeoJSON')
            print('Footprint saved at {}'.format(fileName))

    return allFootprints, fileName
