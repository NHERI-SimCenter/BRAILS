# -*- coding: utf-8 -*-
"""
/*------------------------------------------------------*
|                         BRAILS                        |
|                                                       |
| Author: NHERI SimCenter                               |
|                                                       |
| Date:    1/10/2021                                    |
*------------------------------------------------------*/
"""

import os
import random
from multiprocessing.dummy import Pool as ThreadPool
import requests 
from pathlib import Path
from functools import lru_cache

@lru_cache(maxsize=None)
def validateGoogleMapsAPI(key: str)->bool:
    """Validate a Google Maps API key.
    
    The `@lru_cache` decorator automatically creates a
    cache for API values so that a validation process
    will only be run the first time the function is 
    called.
    `bool(key)` will be false for both the empty
    string, `''`, and `None` values. This function
    should be expanded.
    """
    return bool(key) and key != 'put-your-key-here'

def capturePic(browser, picname):
    try:
        localurl = browser.save_screenshot(picname)
        print("%s : Success" % localurl)
    except BaseException as msg:
        print("Fail：%s" % msg)


def download(urls):
    xcount = 0
    nlimit = 1e10
    reDownload = urls[0][5]

    for ls in urls:
        urlTop = ls[0]
        urlStreet = ls[1]
        #lon = ls[2]
        #lat = ls[3]
        addr = ls[2]

        cats = ls[3]
        imgDir = ls[4]
        #reDownload = ls[5]

        '''
        if rooftype not in roofcat:
            print('not in roofcat')
            continue
        '''

        #if not os.path.exists(thisFileDir):
        #    os.makedirs(thisFileDir)

        #numoffiles = len(os.listdir(thisFileDir))
        if xcount < nlimit: #numoffiles < maxNumofRoofImgs:

            for cat in cats:
                if cat == 'StreetView': trueURL = urlStreet
                elif cat == 'TopView': trueURL = urlTop

                if type(addr) == str:
                    addrstr = addr.replace(' ','-')
                    picname = Path(f'{imgDir}/{cat}/{cat}x{addrstr}.png')
                else:
                    lon, lat = '%.6f'%addr[0], '%.6f'%addr[1]
                    #picname = thisFileDir + '/{prefix}x{lon}x{lat}.png'.format(prefix='StreetView',lon=lon,lat=lat)
                    picname = Path(f'{imgDir}/{cat}/{cat}x{lon}x{lat}.png')

                exist = os.path.exists(picname)
                if not exist or (exist and reDownload):

                    r = requests.get(trueURL)
                    f = open(picname, 'wb')
                    f.write(r.content)
                    f.close()
                    xcount += 1

                    if os.path.getsize(picname)/1024 < 9: 
                        #print(urlStreet)
                        #print(f"empty image from API: ", addr)
                        pass
                        #exit() # empty image from API

        else:
            break

# construct urls
def getGoogleImages(footprints=None, GoogleMapAPIKey='',imageTypes=['StreetView','TopView'],imgDir='',ncpu=2,fov=60,pitch=0,reDownloadImgs=False):

    if footprints is None:
        raise ValueError('Please provide footprints') 

    if not validateGoogleMapsAPI(GoogleMapAPIKey):
        raise ValueError('Invalid GoogleMapAPIKey.') 

    for imgType in imageTypes:
        tmpImgeDir = os.path.join(imgDir, imgType)
        if not os.path.exists(tmpImgeDir): os.makedirs(tmpImgeDir)

    
    # APIs
    baseurl_streetview = "https://maps.googleapis.com/maps/api/streetview?size=640x640&location={lat},{lon}&fov={fov}&pitch={pitch}&source=outdoor&key="+GoogleMapAPIKey
    # consider using 256x256 to save disk
    
    baseurl_satellite="https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom=20&scale=1&size=256x256&maptype=satellite&key="+GoogleMapAPIKey+"&format=png&visual_refresh=true"

    #footprints = gpd.read_file(BuildingFootPrintsFileName)
    urls = []

    for ind, row in footprints.iterrows():
        o = row['geometry'].centroid
        lon, lat = '%.6f'%o.x, '%.6f'%o.y

        # a top view
        urlTop = baseurl_satellite.format(lat=lat,lon=lon)
        urlStreet = baseurl_streetview.format(lat=lat,lon=lon,fov=fov,pitch=pitch)
        cats = imageTypes
        reDownload = 1 if reDownloadImgs else 0
        urls.append([urlTop,urlStreet,[o.x, o.y],cats,imgDir,reDownload])

    #print('shuffling...')
    #random.shuffle(urls)
    #print('shuffled...')     

    # divide urls into small chunks
    #ncpu = 4
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

def getGoogleImagesByAddrOrCoord(Addrs=None, GoogleMapAPIKey='',imageTypes=['StreetView','TopView'],imgDir='',ncpu=2,fov=60,pitch=0,reDownloadImgs=False):

    if Addrs is None:
        raise ValueError('Please provide Addrs') 

    if not validateGoogleMapsAPI(GoogleMapAPIKey):
        raise ValueError('Invalid GoogleMapAPIKey.') 

    for imgType in imageTypes:
        tmpImgeDir = os.path.join(imgDir, imgType)
        if not os.path.exists(tmpImgeDir): os.makedirs(tmpImgeDir)

    
    # APIs
    baseurl_streetview_addr = "https://maps.googleapis.com/maps/api/streetview?size=640x640&location={addr}&fov={fov}&pitch={pitch}&source=outdoor&key="+GoogleMapAPIKey
    baseurl_streetview_coord = "https://maps.googleapis.com/maps/api/streetview?size=640x640&location={lat},{lon}&fov={fov}&pitch={pitch}&source=outdoor&key="+GoogleMapAPIKey
    # consider using 256x256 to save disk
    
    baseurl_satellite_addr="https://maps.googleapis.com/maps/api/staticmap?center={addr}&zoom=20&scale=1&size=256x256&maptype=satellite&key="+GoogleMapAPIKey+"&format=png&visual_refresh=true"
    baseurl_satellite_coord="https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom=20&scale=1&size=256x256&maptype=satellite&key="+GoogleMapAPIKey+"&format=png&visual_refresh=true"


    #footprints = gpd.read_file(BuildingFootPrintsFileName)
    urls = []

    for addr in Addrs:

        if type(addr) == str:
            urlTop = baseurl_satellite_addr.format(addr=addr)
            urlStreet = baseurl_streetview_addr.format(addr=addr,fov=fov,pitch=pitch)
        else:
            lon, lat = '%.6f'%addr[0], '%.6f'%addr[1]
            urlTop = baseurl_satellite_coord.format(lat=lat,lon=lon)
            urlStreet = baseurl_streetview_coord.format(lat=lat,lon=lon,fov=fov,pitch=pitch)

        cats = imageTypes
        reDownload = 1 if reDownloadImgs else 0
        urls.append([urlTop,urlStreet,addr,cats,imgDir,reDownload])

    #print('shuffling...')
    #random.shuffle(urls)
    #print('shuffled...')     

    # divide urls into small chunks
    #ncpu = 4
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