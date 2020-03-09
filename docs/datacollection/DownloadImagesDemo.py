"""
/*------------------------------------------------------*
|  This script downloads images from google API         |
|                                                       |
| Author: Charles Wang,  UC Berkeley c_w@berkeley.edu   |
|                                                       |
| Date:    07/15/2019                                   |
*------------------------------------------------------*/
"""


import os
import requests 
import random
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool


GoogleMapAPIKey = "You need to put your Key here."

outputDir = "./images"

# APIs. notice: you can change pitch and fov to get best capture of the building from the street view. 
#               zoom and sizes can also be changed.
baseurl_streetview = "https://maps.googleapis.com/maps/api/streetview?size=512x512&location={lat},{lon}&pitch=0&fov=30&key="+GoogleMapAPIKey
baseurl_satellite = "https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom=20&scale=1&size=256x256&maptype=satellite&key="+GoogleMapAPIKey+"&format=png&visual_refresh=true"


def download(urls):
    """Function to download images using Google API.
    
    Args:
        urls (List): [[urlTop,urlStreet,lon,lat,BldgID],...]
    """
    for ls in urls:
        urlTop = ls[0]
        urlStreet = ls[1]
        lon = ls[2]
        lat = ls[3]
        BldgID = ls[4]

        roofPicName = outputDir + '/{BldgID}-{prefix}.png'.format(BldgID=BldgID,prefix='TopView')
        if not os.path.exists(roofPicName):
            print(urlTop)
            r = requests.get(urlTop)
            f = open(roofPicName, 'wb')
            f.write(r.content)
            f.close()

        streetPicName = outputDir + '/{BldgID}-{prefix}.png'.format(BldgID=BldgID,prefix='StreetView')
        if not os.path.exists(streetPicName):
            print(urlStreet)
            r = requests.get(urlStreet)
            f = open(streetPicName, 'wb')
            f.write(r.content)
            f.close()


# construct urls
dictFile = pd.read_csv('list.csv')  
urls = []
for index, row in dictFile.iterrows():
    BldgID = row['BldgUniqueID']
    lat = row['CentroidY']
    lon = row['CentroidX']
    #StructureUse = row['StructureUse']

    urlTop = baseurl_satellite.format(lat=lat,lon=lon)
    urlStreet = baseurl_streetview.format(lat=lat,lon=lon)
    urls.append([urlTop,urlStreet,lon,lat,BldgID])
print('shuffling...')
random.shuffle(urls)
print('shuffled...')


# divide urls into small chunks
ncpu = 4
step = int(len(urls)/ncpu)+1
chunks = [urls[x:x+step] for x in range(0, len(urls), step)]


print('Downloading images from Google API ...')
# get some workers
pool = ThreadPool(ncpu)
# send job to workers
results = pool.map(download, chunks)
# jobs are done, clean the site
pool.close()
pool.join()
print('Images downloaded ...')

