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
# 04-21-2022    


import os
import requests 
import sys
from math import radians, sin, cos, atan2, sqrt, log, floor
from shapely.geometry import Polygon, MultiPoint



class ImageHandler:
    def __init__(self,apikey: str):        
        # Check if the provided Google API Key successfully obtains street-level
        # and satellite imagery for Doe Memorial Library of University of 
        # California, Berkeley:
        responseStreet = requests.get('https://maps.googleapis.com/maps/api/streetview?' + 
                                      'size=600x600&location=37.87251874078189,' +
                                      '-122.25960286494328&heading=280&fov=120&' +
                                      f"pitch=20&key={apikey}").ok
        responseSatellite = requests.get('https://maps.googleapis.com/maps/api/staticmap?' + 
                                         'maptype=satellite&size=600x600&' + 
                                         'center=37.87251874078189,-122.25960286494328' + 
                                         f"&zoom=20&key={apikey}").ok
        
        # If any of the requested images cannot be downloaded, notify the 
        # user of the error and stop program execution:
        if responseStreet==False and responseSatellite==False:
            error_message = ('Google API key error. Either the API key was entered'
                             + ' incorrectly or both Maps Static API and Street '
                             + 'View Static API are not enabled for the entered '
                             + 'API key.')
        elif responseStreet==False:
            error_message = ('Google API key error. The entered API key is valid '
                             + 'but does not have Street View Static API enabled. ' 
                             + 'Please enter a key that has both Maps Static API '
                             + 'and Street View Static API enabled.')
        elif responseSatellite==False:
            error_message = ('Google API key error. The entered API key is valid '
                             + 'but does not have Maps Static API enabled. Please ' 
                             + 'enter a key that has both Maps Static API '
                             + 'and Street View Static API enabled.')  
        else:
            error_message = None
    
        if error_message!=None:
            sys.exit(error_message)
            
        self.apikey = apikey
        self.footprints = []

    def GetGoogleSatelliteImage(self,fp):
     
        def dist(p1,p2):
            """
            Function that calculates the distance between two points.
            
            Input: Two points with coordinates defined as latitude and longitude
                   with each point defined as a list of two floating-point values
            Output: Distance between the input points in feet
            """

            # Define mean radius of the Earth in kilometers:
            R = 6371.0 
            
            # Convert coordinate values from degrees to radians
            lat1 = radians(p1[0]); lon1 = radians(p1[1])
            lat2 = radians(p2[0]); lon2 = radians(p2[1])
            
            # Compute the difference between latitude and longitude values:
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            
            # Compute the distance between two points as a proportion of Earth's
            # mean radius:
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            
            # Compute distance between the two points in feet:
            distance = R*c*3280.84
            return distance
        
        def compute_zoom(fp):
            """
            Function that computes the zoom level resulting in the Google 
            satellite image that best wraps around the input footprint. Here
            best coverage is defined as the image that results in the highest 
            (Number of building pixels)/(Total number of image pixels) ratio
            when covering almost all of the building footprint.
            
            Input: Building footprint defined as a list of points (i.e., lists
                   of latitude and longitude values defines as floating-point
                   numbers)
            Output: Zoom level that results in the best footprint coverage in
                    a Google satellite image
            """
            
            # Define Earth's circumference along its equator in feet:
            C = 40075*3280.84
            
            # Compute the coordinate extents of the horizontal bounding box 
            # that wraps around the footprint:
            bbox_bnds = MultiPoint(fp).bounds
            
            # Assemble the list of coordinates of the bounding box:
            bbox = [[bbox_bnds[0],bbox_bnds[1]],[bbox_bnds[0],bbox_bnds[3]],
                    [bbox_bnds[2],bbox_bnds[1]],[bbox_bnds[2],bbox_bnds[3]],
                    [bbox_bnds[0],bbox_bnds[1]]]
            
            # Calculate the longest dimension of the bounding box:
            maxdim = max([dist(p1,p2) for p1,p2 in zip(bbox[:-3],bbox[1:-2])])
            
            # Compute the zoom value required to have the entire footprint
            # covered in a satellite image. Assumes a 640x640 square image:
            zoom = (log(C/(maxdim*256/640),2))
        
            # If the calculated zoom value covers most of the footpprint area
            # when it is rounded up, then round it up. Else round it down. 
            # This if-statement is in place to prevent using high zoom values
            # ubless it is absolutely necessary to use them:
            if (zoom - floor(zoom))<0.9: 
                zoom = floor(zoom)
            else:
                zoom = round(zoom)
        
            return zoom



"""

# Download and display the satellite image:
with open('doelibrary_satellite.jpg','wb') as f:
  f.write(requests.get(f"https://maps.googleapis.com/maps/api/staticmap?maptype=satellite&size=600x600&center=37.87251874078189,-122.25960286494328&zoom=20&key={apikey}").content)
display(Image(filename='doelibrary_satellite.jpg'))
    
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

    if not validateGoogleAPIKey(GoogleMapAPIKey):
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
"""