"""
/*------------------------------------------------------*
|  This script downloads roof images from google and    |
|  predicts roof types using CNN.                       |
|                                                       |
| Author: Chaofeng Wang,  UC Berkeley c_w@berkeley.edu  |
|                                                       |
| Date:    07/15/2019                                   |
*------------------------------------------------------*/
"""


import os
import time
import json
import requests 
import random
import numpy as np
import geopandas as gpd
from multiprocessing.dummy import Pool as ThreadPool
import sys
sys.path.append("../.")
# make sure you have keys.py inside src/
# in it you define GoogleMapAPIKey
from configure import * 


outputDir = roofDownloadDir

# API
baseurl_streetview = "https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}&heading=-45&pitch=38&fov=80"
#baseurl_satellite = "https://www.google.com/maps/@?api=1&map_action=map&center={lat},{lon}&zoom=30&basemap=satellite"
#http://maps.googleapis.com/maps/api/streetview?size=1024x768&pitch=30&fov=120&heading=-100&location=300+Cadman+Plaza+West+Brooklyn
#37.880928, -122.293622
baseurl_satellite="https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom=20&scale=1&size=256x256&maptype=satellite&key="+GoogleMapAPIKey+"&format=png&visual_refresh=true"
def capturePic(browser, picname):
    try:
        localurl = browser.save_screenshot(picname)
        print("%s : Success" % localurl)
    except BaseException as msg:
        print("Fail：%s" % msg)

#roofcat = ['gabled']

def download(urls):
    for ls in urls:
        urlTop = ls[0]
        urlStreet = ls[1]
        lon = ls[2]
        lat = ls[3]
        '''
        if rooftype not in roofcat:
            print('not in roofcat')
            continue
        '''
        thisFileDir = roofDownloadDir
        if not os.path.exists(thisFileDir):
            os.makedirs(thisFileDir)
        numoffiles = len(os.listdir(thisFileDir))
        if numoffiles < maxNumofRoofImgs:
            picname = thisFileDir + '/{prefix}x{lon}x{lat}.png'.format(prefix='TopView',lon=lon,lat=lat)
            if not os.path.exists(picname):
                r = requests.get(urlTop)
                f = open(picname, 'wb')
                f.write(r.content)
                f.close()
        else:
            break

# construct urls
cityFile = gpd.read_file(resultBIMFileName).to_json()
footjsons = json.loads(cityFile)['features']
urls = []
for j in footjsons:
    address = j['properties']['address']
    lat = j['properties']['lat']
    lon = j['properties']['lon']
    '''
    coords = j['geometry']['coordinates'][0][0]
    lon, lat = np.mean(np.asarray(coords), axis=0)# centroid
    '''
    #rooftype = j['properties']['roof_shape']

    # a top view
    urlTop = baseurl_satellite.format(lat=lat,lon=lon)
    urlStreet = baseurl_streetview.format(lat=lat,lon=lon)
    urls.append([urlTop,urlStreet,lon,lat])
print('shuffling...')
random.shuffle(urls)
print('shuffled...')


# divide urls into small chunks
ncpu = 4
step = int(len(urls)/ncpu)+1
chunks = [urls[x:x+step] for x in range(0, len(urls), step)]

'''
# test
testUrl = urls[1:10]
print(testUrl)
download(testUrl)
exit()
'''

print('Downloading satellite images of rooves from Google API ...')
# get some workers
pool = ThreadPool(ncpu)
# send job to workers
results = pool.map(download, chunks)
# jobs are done, clean the site
pool.close()
pool.join()
print('Satellite images of rooves downloaded ...')


# predict
