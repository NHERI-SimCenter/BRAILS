"""
/*------------------------------------------------------*
|  This script creates a basic BIM file .               |
|                                                       |
| Author: Chaofeng Wang,  UC Berkeley c_w@berkeley.edu  |
|                                                       |
| Date:    06/02/2019                                   |
*------------------------------------------------------*/
"""

import csv
import re
import os
import json
import requests 
import numpy as np
import geopandas as gpd
from scipy import spatial
from matplotlib import pyplot as plt
from shapely.geometry import Point, Polygon
from multiprocessing.dummy import Pool as ThreadPool
import sys
sys.path.append("../.")
# make sure you have keys.py inside src/
# in it you define GoogleMapAPIKey
from configure import * 


# -----------------------------------------------------------------------------
# 1. Users need to provide these files:                                       |
# -----------------------------------------------------------------------------
'''
# geojson file defining the boundary of the interested area
RegionBoundaryFileName = "/Users/simcenter/Files/SimCenter/Wind Storm Surge Workflow/Atlantic City/Data/GIS/AtlanticCostalCities.geojson"
# building footprints obtained from MS: https://github.com/microsoft/USBuildingFootprintsv
BuildingFootPrintsFileName = "/Users/simcenter/Files/SimCenter/Wind Storm Surge Workflow/Atlantic City/Data/AtlanticCostalFootprints.geojson"
# a csv file containing the builidng address and other information. First line is viariable names, first column must be address
cleanedBIMFileName = "/Users/simcenter/Files/SimCenter/Wind Storm Surge Workflow/Atlantic City/BIM/data/Atlantic_Cities_Addrs.csv"
'''
# define columns' types, default is string
intVariables = ['yearBuilt', 'stories']
floatVariables = ['lat', 'lon']


# -----------------------------------------------------------------------------
# 2. Define geocoding API.                                                    |
#    https://developers.google.com/maps/documentation/geocoding/start?utm_    |
# source=google&utm_medium=cpc&utm_campaign=FY18-Q2-global-demandgen-paids    |
# earchonnetworkhouseads-cs-maps_contactsal_saf&utm_content=text-ad-none-n    |
# one-DEV_c-CRE_315916117598-ADGP_Hybrid+%7C+AW+SEM+%7C+BKWS+~+Google+Maps    |
# +Geocoding+API-KWID_43700039136946117-kwd-300650646186-userloc_1013585&u    |
# tm_term=KW_google%20geocoding%20api-ST_google+geocoding+api&gclid=CKvutr    |
# Oi0-ICFR2GxQIdojkNwg                                                        |
#    Once user got API from Google, he/she needs to provide a keys.py file,   |
#    in which write this:                                                     |
#    GoogleMapAPIKey = "replace this with your key"                           |
#                                                                             |
#                                                                             |
# -----------------------------------------------------------------------------

#baseurl_addr_mq_open = "http://www.mapquestapi.com/geocoding/v1/address?key="+MapquestOpenAPIKey+"&location={}" # mapquest
#baseurl_addr_mq = 'http://www.mapquestapi.com/geocoding/v1/address?key='+MapquestAPIKey+'&location={}'
baseurl_addr_google="https://maps.googleapis.com/maps/api/geocode/json?address={}&key="+GoogleMapAPIKey
# which API to use
baseurl_addr = baseurl_addr_google


# -----------------------------------------------------------------------------
# 3. Define paths                                                             |
# -----------------------------------------------------------------------------

'''
# define path for the resulting geojson file that contains BIM for all buildings
resultBIMFileName = "/Users/simcenter/Files/SimCenter/Wind Storm Surge Workflow/Atlantic City/BIM/data/Atlantic_Cities_BIM.geojson"
# define path for a temporary csv file
cleanedBIMFile_w_coord_Name = "/Users/simcenter/Files/SimCenter/Wind Storm Surge Workflow/Atlantic City/BIM/data/Atlantic_Cities_Addrs_Coords.csv"
# where to put json files of addrs' coords
baseDir_addrCoordJson = '/Users/simcenter/Files/SimCenter/Wind Storm Surge Workflow/Atlantic City/BIM/data/geocoding/' 
'''

# -----------------------------------------------------------------------------
# 4. Define functions                                                         |
# -----------------------------------------------------------------------------

def geocode(rows):
    count = 0
    n = len(rows)
    for row in rows:
        addr_raw = row[0]
        addr = addr_raw.replace(' ','-space-').replace('/','-slash-')

        if not count % 500:
            print(count/n*100)
        count += 1 

        geocodingFile = baseDir_addrCoordJson + addr + '.json'
        if not os.path.exists(geocodingFile):
            url = baseurl_addr.format(addr_raw)
            url.replace(' ','+')
            r = requests.get(url)
            f = open(geocodingFile, 'wb')
            f.write(r.content)
            f.close()
            print(url, "           downloaded.")
        else:
            with open(geocodingFile, 'r') as jf:
                #print(geocodingFile)
                j = json.load(jf)
                if j['status']=='OK':
                    d = j['results'][0]
                    coord = d['geometry']['location']
                    lat = coord['lat']
                    lon = coord['lng']
                    l = [lat, lon]
                    [l.append(x) for x in row]
                    coordcsv.writerow(l)
                    #print([lat, lon])
                else:
                    print("No status from API.")

def getBIMIndex(pts):
    #[[-74.45606, 39.371837], [-74.455935, 39.371934], [-74.456037, 39.372013], [-74.456162, 39.371916], [-74.45606, 39.371837]]
    poly = Polygon(pts)
    o = pts[0]
    distance,index = kdTree.query(o,10) # nearest 10 points
    trueIndex = []
    [trueIndex.append(i) for i in index if Point(coordsAll[i]).within(poly)]
    '''
    x,y = poly.exterior.xy
    fig = plt.figure(1, figsize=(5,5), dpi=90)
    ax = fig.add_subplot(111)
    ax.plot(x, y, color='#6699cc', alpha=0.7,
    linewidth=3, solid_capstyle='round', zorder=2)
    ax.set_title('Polygon')
    plt.plot([coordsAll[index[0]][0]],[coordsAll[index[0]][1]],'ro')
    plt.show()
    '''
    if trueIndex:
        theInd = trueIndex[0]
    else:# no point inside polygon
        theInd = index[0]
        
    return theInd

# -----------------------------------------------------------------------------
# 5. Geocode addresses                                                        |
# -----------------------------------------------------------------------------

# query google for coords of addresses
reQuery = True
if reQuery:
    with open(cleanedBIMFileName, 'r') as addrfile, open(cleanedBIMFile_w_coord_Name, 'w+', newline='') as cleanedBIMFile_w_coord_File:

        addrcsv = list(csv.reader(addrfile))

        coordcsv = csv.writer(cleanedBIMFile_w_coord_File)
        firstRow = ['lat','lon']
        [firstRow.append(x) for x in addrcsv[0]]
        coordcsv.writerow(firstRow)

        #geocode(addrcsv[0:2200])
        print(len(addrcsv))

        # divide urls into small chunks
        ncpu = 4
        step = int(len(addrcsv)/ncpu)+1
        chunks = [addrcsv[x:x+step] for x in range(0, len(addrcsv), step)]

        # get some workers
        pool = ThreadPool(ncpu)
        # send job to workers
        results = pool.map(geocode, chunks)
        # jobs are done, clean the site
        pool.close()
        pool.join()


# construct regional BIM file in geojson
with open(cleanedBIMFile_w_coord_Name, 'r') as addrfile, open(resultBIMFileName, 'w+', newline='') as resultBIMFile, open(RegionBoundaryFileName) as RegionBoundaryFile, open(BuildingFootPrintsFileName) as BuildingFootPrintsFile:

    addrcsv = list(csv.reader(addrfile))
    colNames = addrcsv[0][0:]
    addrcsv = addrcsv[1:]
    coordsAll = []
    [coordsAll.append([row[1], row[0]]) for row in addrcsv]
    coordsAll = np.array(coordsAll, dtype=np.float32)
    kdTree = spatial.KDTree(coordsAll)



    boundjsn = json.load(RegionBoundaryFile)
    pts = boundjsn["features"][0]['geometry']['coordinates'][0]
    boundary = Polygon(pts)

    fps = []
    bldgFootPrints = json.load(BuildingFootPrintsFile)
    bldgFootPrintsFeatures = bldgFootPrints["features"]
    bldgFootPrintsFeatures =  bldgFootPrintsFeatures[0:]
    features = []
    for bfpf in bldgFootPrintsFeatures:
        #print(bfpf)
        pts = bfpf['geometry']['coordinates'][0]
        #fps.append(Polygon(pts))
        bimIndex = getBIMIndex(pts)
        info = addrcsv[bimIndex]
        for i in range(len(colNames)):
            vName = colNames[i]
            if info[i] == 'None' or info[i] == '':
                bfpf['properties'][colNames[i]] = None
                continue
            elif vName in intVariables:
                vValue = int(float(info[i]))
            elif vName in floatVariables:
                vValue = float(info[i])
            else:
                vValue = info[i]
            bfpf['properties'][colNames[i]] = vValue
        features.append(bfpf)

    bldgFootPrints['features'] = features
    json.dump(bldgFootPrints, resultBIMFile)
    print("bim has been added to {}".format(resultBIMFileName))

    exit()


