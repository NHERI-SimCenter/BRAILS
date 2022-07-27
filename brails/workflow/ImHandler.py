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
# 05-08-2022    


import os
import requests 
import sys
from math import radians, sin, cos, atan2, sqrt, log, floor
from shapely.geometry import Point, Polygon, MultiPoint

import math

import numpy as np


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
        self.centroids = []
        self.satZoomLevels = []
        self.refLines = []
        self.imagePlanes = []
        self.streetScales = []
        self.streetFOVs = []
        self.streetHeadings = []
        self.satellite_images = []
        self.street_images = []

    def GetGoogleSatelliteImage(self,footprints):
        self.footprints = footprints[:]
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
        
        self.footprints = footprints
        self.satellite_images = []
        os.makedirs('tmp/images/satellite',exist_ok=True)
        for count, fp in enumerate(footprints):
            # Compute the centroid of the footprint polygon: 
            fp_cent = Polygon(fp).centroid
            self.centroids.append([fp_cent.x,fp_cent.y])
            
            # Compute the zoom level required to view the entire footprint in a
            # satellite image:
            zoom = compute_zoom(fp)
            self.satZoomLevels.append(zoom)

            # Download the satellite image that views the building defined by the
            # footprint:
            query_url = ("https://maps.googleapis.com/maps/api/staticmap?center=" + 
                         f"{fp_cent.y},{fp_cent.x}&zoom={zoom}&size=640x640&maptype="+
                         f"satellite&format=png&key={self.apikey}")
            im_name = f"tmp/images/satellite/{count}.png"
            with open(im_name,'wb') as f:
                f.write(requests.get(query_url).content)
            self.satellite_images.append(im_name)

    def GetGoogleStreetImage(self,footprints):
        self.footprints = footprints[:]
        # Function that downloads a file given its URL and the desired path to save it:
        def download_url(url, save_path, chunk_size=128):
            r = requests.get(url, stream=True)
            with open(save_path, 'wb') as fd:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    fd.write(chunk)
    
        # Function that obtains the camera location for a StreetView image given the
        # latitude/longitude and API key for the location:
        def download_metadata(latlon,key):
            metadataBaseURL = "https://maps.googleapis.com/maps/api/streetview/metadata?size=640x640&source=outdoor&location=%s&key=%s"
            metadataURL = metadataBaseURL % (str(latlon)[1:-1].replace(" ", ""), key)
            metadata = requests.get(metadataURL).json()
            if metadata['status']=='OK':
                cameraLoc = Point(metadata['location']['lat'],metadata['location']['lng'])
            else:
                cameraLoc = None
            return cameraLoc
        
        # Function that downloads a StreetView image given the latitude/longitude for
        # the location, camera heading and FOV, API key, and the image path and file 
        # name to save the image: 
        def download_image(latlon,heading,fov,key,imName,im_path):
            os.makedirs(im_path,exist_ok=True)
            streetViewBaseURL = "https://maps.googleapis.com/maps/api/streetview?size=640x640&source=outdoor&location=%s&heading=%s&fov=%s&key=%s"
            image_url = streetViewBaseURL % (str(latlon)[1:-1].replace(" ", ""),str(heading),str(fov),key)
            r = requests.get(image_url)
            if r.status_code == 200:
                with open(os.path.join(im_path,f"{imName}.png"), 'wb') as f:
                    f.write(r.content)
          
        # Function that checks if, given three colinear points p, q, r, point q lies on
        # line segment 'pr': 
        def onSegment(p, q, r):
            if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and 
                   (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
                return True
            return False
        
        # Function that checks the orientation of an ordered triplet (p,q,r):
        def orientation(p, q, r):
            # to find the orientation of an ordered triplet (p,q,r)
            # function returns the following values:
            # 0 : Colinear points
            # 1 : Clockwise points
            # 2 : Counterclockwise
              
            val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
            if (val > 0):
                  
                # Clockwise orientation
                return 1
            elif (val < 0): 
                # Counterclockwise orientation
                return 2
            else:
                # Colinear orientation
                return 0
          
        # The main function that returns true if the line segment 'p1q1' and 'p2q2'
        # intersect:
        def doIntersect(p1,q1,p2,q2):
              
            # Find the 4 orientations required for 
            # the general and special cases
            o1 = orientation(p1, q1, p2)
            o2 = orientation(p1, q1, q2)
            o3 = orientation(p2, q2, p1)
            o4 = orientation(p2, q2, q1)
          
            # General case
            if ((o1 != o2) and (o3 != o4)):
                return True
            # Special Cases
            # p1 , q1 and p2 are colinear and p2 lies on segment p1q1
            if ((o1 == 0) and onSegment(p1, p2, q1)):
                return True
            # p1 , q1 and q2 are colinear and q2 lies on segment p1q1
            if ((o2 == 0) and onSegment(p1, q2, q1)):
                return True
            # p2 , q2 and p1 are colinear and p1 lies on segment p2q2
            if ((o3 == 0) and onSegment(p2, p1, q2)):
                return True
            # p2 , q2 and q1 are colinear and q1 lies on segment p2q2
            if ((o4 == 0) and onSegment(p2, q1, q2)):
                return True
            # If none of the cases
            return False
        
        # Function that returns the midpoint of a line segment given its vertices:
        def midpoint_calc(vert1,vert2):
            q1x = 0.5*(vert1.x + vert2.x)
            q1y = 0.5*(vert1.y + vert2.y)      
            q1 = Point(q1x,q1y)
            return q1        
        
        # Function that calculates the distance between two points.
        # Input: two points with coordinates defined as latitude and longitude
        # Output: distance between the input points in feet
        def dist(p1,p2):
            R = 6371.0 # Mean radius of the earth in kilometers
            
            lat1 = math.radians(p1.x)
            lon1 = math.radians(p1.y)
            lat2 = math.radians(p2.x)
            lon2 = math.radians(p2.y)
            
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            
            distance = R*c*3280.84 # distance between the two points in feet
            return distance
        
        # Function to project the rough reference line onto the frontal line:
        def compute_refLineVert(lineFrontal,refLineVertRough,footprint,linesKeep):
            # Find the index for footprint vertex that matches the rough reference
            # line vertex:    
            indexVertRough = np.where(footprint==refLineVertRough[0])[0].item(0)
        
            # Start looking for the non-visible line segment connecting into the rough
            # reference line vertex by assuming the second vertex of this line is the
            # point before the reference line vertex in the array of footprint 
            # vertices. Here the reason for searching the non-visible line vertex is 
            # to find the vertex that will enable extending the fontal line such that
            # it extends covers the entire front facade:
        
            # If the reference line vertex is the first point in the array of
            # footprint vertices, the point before would be on the row
            # (#Rows of Footprint Array)-2: 
            if indexVertRough==0: 
                n = len(footprint)-2
            # Else the point before would be on the row right before the reference line
            # vertex:
            else:
                n = indexVertRough-1
            
            # If the second vertex for the non-visible line segment connecting into the 
            # rough reference line vertex is the point before the reference line
            # vertex in the array of footprint vertices:
            if len(np.where(linesKeep==np.append(footprint[n,:],
                                                 footprint[indexVertRough,:]))[0])==0:
               # The Boolean above checks if the line formed by [footprint[n,:],
               # footprint[indexVertRough,:]] is a visible line. We not the program 
               # keeps it
                refLineVertConnect =  footprint[n,:]
            # Else:          
            else:
               refLineVertConnect = footprint[indexVertRough+1,:]
            
            # Compute the insersection of the frontal line and the line formed by
            # rough reference line vertex and the connecting vertex on the footprint 
            # that is not a visible line:
            
            m1 = (lineFrontal[3]-lineFrontal[1])/(lineFrontal[2]-lineFrontal[0])        
            m2 = (refLineVertRough[1]-refLineVertConnect[1])/(refLineVertRough[0]
                                                              -refLineVertConnect[0])
            x1 = lineFrontal[0]; y1 = lineFrontal[1]
            x2 = refLineVertConnect[0]; y2 = refLineVertConnect[1]
            
            xRefLineVert = (m1*x1 - m2*x2 - y1 + y2)/(m1 - m2)
            yRefLineVert = m1*(xRefLineVert - x1) + y1
            
            # The computed intersection is one of the vertices of the preliminary
            # reference line: 
            refLineVert = [xRefLineVert, yRefLineVert]
            
            return refLineVert
        
        # Function to compute the preliminary reference line. Here preliminary 
        # reference line is the line segment that spans the extreme frontal edges of 
        # footprint that is snapped on the footprint line segment closest to the camera
        # location:
        def compute_refline(footprint,p1,visibleLineSeg):
            # If there are more than one visible line segments: 
            if visibleLineSeg.shape[0]!=1:
                # Compute to slopes of the visible lines:
                lineSegSlopes = [(visibleLineSeg[k,3]-visibleLineSeg[k,1])/
                                 (visibleLineSeg[k,2]-visibleLineSeg[k,0])
                                 for k in range(visibleLineSeg.shape[0])]
                
                # If all of the calculated line slopes are finite:
                if any(abs(np.array(lineSegSlopes))==float('inf'))==False:
                    
                    # Split the visible lines into two classes based on their 
                    # orientation (i.e., slopes):
                    slopeDiff = (lineSegSlopes-lineSegSlopes[0])/lineSegSlopes[0]*100
                    linesKeep1 = np.where(abs(slopeDiff)<50)[0]
                    linesKeep2 = np.where(abs(slopeDiff)>=50)[0]
                    
                    # Calculate the lengths of the first set of lines:
                    linesKeep1Len = np.zeros(len(linesKeep1))
                    for k in range(len(linesKeep1)): 
                        linesKeep1Len[k] = dist(Point(visibleLineSeg[linesKeep1][k,0:2]),
                                              Point(visibleLineSeg[linesKeep1][k,2:4]))
                    
                    # Calculate the lengths of the second set of lines:
                    linesKeep2Len = np.zeros(len(linesKeep2))
                    for k in range(len(linesKeep2)): 
                        linesKeep2Len[k] = dist(Point(visibleLineSeg[linesKeep2][k,0:2]),
                                              Point(visibleLineSeg[linesKeep2][k,2:4]))
                    
                    # Keep the set of lines that are predominant. Here predominant is 
                    # defined as the set of lines that have the longest cummulative
                    # length:
                    if np.sum(linesKeep1Len)>np.sum(linesKeep2Len):
                        linesKeep = visibleLineSeg[linesKeep1]
                    else:
                        linesKeep = visibleLineSeg[linesKeep2]
                    
                    # Calculate the distance between the camera location and the 
                    # vertices of the set of lines that were kept:
                    linesKeepDist = np.zeros([len(linesKeep),2])
                    for k in range(len(linesKeep)): 
                        linesKeepDist[k,0] = dist(p1,Point(linesKeep[k,0:2]))
                        linesKeepDist[k,1] = dist(p1,Point(linesKeep[k,2:4]))
                    
                    # Find the row index of the vertex closest to the camera location
                    # and get the vertices for the line that contains this vertex. If
                    # orientation based classification above works correctly, there
                    # should be exactly one line that is closest to the camera:
                    lineFrontal = linesKeep[np.where(linesKeepDist==np.min(linesKeepDist))[0].item(0),:]
                    
                    
                    # Create a rough reference line that spans the predominant 
                    # direction of the visible lines. Here the predominant direction
                    # is identified as the direction that spans longest distance
                    # between the extreme vertices of the visible lines (which are 
                    # denoted as left/right or top/bottom points): 
                    vertLat = np.append(linesKeep[:,0],linesKeep[:,2])
                    vertLon = np.append(linesKeep[:,1],linesKeep[:,3])
                    
                    indMinLat = np.argmin(vertLat); bottomPt = Point(vertLat[indMinLat],vertLon[indMinLat])
                    indMaxLat = np.argmax(vertLat); topPt = Point(vertLat[indMaxLat],vertLon[indMaxLat])
                    indMinLon = np.argmin(vertLon); leftPt = Point(vertLat[indMinLon],vertLon[indMinLon])
                    indMaxLon = np.argmax(vertLon); rightPt = Point(vertLat[indMaxLon],vertLon[indMaxLon])
                    
                    distTopBottom = dist(bottomPt,topPt)
                    distLeftRight = dist(leftPt,rightPt)
                    
                    if distTopBottom>distLeftRight:
                        refLineRough = np.array([topPt.x,topPt.y,bottomPt.x,bottomPt.y])
                    else:
                        refLineRough = np.array([leftPt.x,leftPt.y,rightPt.x,rightPt.y])
                    
                    # Case 1: Frontal line is located on the rough reference (i.e., 
                    # the two lines share a vertex):        
                    if len(list(set(refLineRough) - set(lineFrontal)))==2:
                        refLineVert2Rough = list(set(refLineRough) - set(lineFrontal))
                        refLineVert2 = compute_refLineVert(lineFrontal,refLineVert2Rough,
                                                           footprint,linesKeep) 
                        refLineVert1 = list(set(refLineRough) - set(refLineVert2Rough))
                    # Case 2: Frontal line is not located on the rough reference:        
                    elif len(list(set(refLineRough) - set(lineFrontal)))==4:
                        refLineVert1Rough = refLineRough[0:2]
                        refLineVert1 = compute_refLineVert(lineFrontal,refLineVert1Rough,
                                                           footprint,linesKeep) 
                        refLineVert2Rough = refLineRough[2:4]
                        refLineVert2 = compute_refLineVert(lineFrontal,refLineVert2Rough,
                                                           footprint,linesKeep)
                    # Case 3: Frontal line and the rough reference are identical:
                    else:
                        refLineVert1 = refLineRough[0:2]
                        refLineVert2 = refLineRough[2:4]
                    
                    # Define the preliminary reference line:
                    premRefLine = np.array([[refLineVert1[0],refLineVert1[1]],
                                            [refLineVert2[0],refLineVert2[1]]])
                    #refLength = dist(Point(refLineVert1),Point(refLineVert2))
                
                # If not all of the calculated line slopes are finite, set the 
                # preliminary reference line as None:    
                else:
                    premRefLine = None
                    #refLength = None
            
            # If there is only one visible line segment, set it equal to the 
            # preliminary reference line: 
            else:
                premRefLine = np.array([[visibleLineSeg[0,0],visibleLineSeg[0,1]],
                                    [visibleLineSeg[0,2],visibleLineSeg[0,3]]])
                #refLength = dist(Point(visibleLineSeg[0,0:2]),Point(visibleLineSeg[0,2:4]))
            
            if isinstance(premRefLine,np.ndarray):
                slope = abs((premRefLine[0,1] - premRefLine[1,1])/(premRefLine[0,0] - premRefLine[1,0]))
                if np.any(np.isnan(premRefLine)):
                    premRefLine = None 
                elif math.isnan(slope) or math.isinf(slope) or round(slope,4)==0:
                    premRefLine = None    
            return premRefLine
        
        # Function to calculate the bearing of a line given the latitude/longitude
        # values of two points on the line in degrees:
        def get_bearing(lat1, long1, lat2, long2):
            dLon = (long2 - long1)
            x = math.cos(math.radians(lat2))*math.sin(math.radians(dLon))
            y = (math.cos(math.radians(lat1))*math.sin(math.radians(lat2)) - 
                 math.sin(math.radians(lat1))*math.cos(math.radians(lat2))*
                 math.cos(math.radians(dLon)))
            brng = (math.atan2(x,y)*180/math.pi + 360) % 360
            return brng
        
        # Function to compute the camera parameters necessary to retrieve the best
        # image of the building. Here the best image is defined as the image that
        # is primarily filled with a view of the building with very little inclusion
        # of other buildings. The camera parameters computed are the reference line,
        # image plane, image scale, fov, and heading:
        def get_camParam(premRefLine,p1):
            # Camera location:
            x1 = p1.x 
            y1 = p1.y
            
            # One of the end points of the preliminary reference line:
            x2 = premRefLine[0,0]
            y2 = premRefLine[0,1]
        
            # Calculate the slope of the preliminary reference line (since the slope is
            # in proportions, no need to transform coordinates to Cartesian):
            m = (premRefLine[0,1] - premRefLine[1,1])/(premRefLine[0,0] - premRefLine[1,0])
        
            # Compute the point where the line segment that forms the shortest 
            # distance between the camera location and the preliminary reference
            # line segment intersect:
            xint = (m/(1+m**2))*(x1/m + m*x2 + y1 - y2)
            yint = m*(xint - x2) + y2
            pint = Point(xint,yint)
            
            # Calculate the camera distance:
            camDist = dist(p1,pint)
            
            # Measure the distance between the point of intersection and the ends of
            # preliminary reference line to determine the extent of image plane:
            refDist1 = dist(pint,Point(x2,y2))
            refDist2 = dist(pint,Point(premRefLine[1,0],premRefLine[1,1]))
            refDists = [refDist1,refDist2]
            imageExtent = max(refDists)
            max_index = refDists.index(imageExtent)
            
            # Compute the end points of the image plane:
            if max_index==0:
                imagePlane = np.array([[x2,y2],[2*xint - x2, 2*yint - y2]])
            else:
                imagePlane = np.array([[premRefLine[1,0],premRefLine[1,1]],
                                       [2*xint - premRefLine[1,0], 2*yint - premRefLine[1,1]]])
            
            # Compute the FOV required to view the determined image plane. If the FOV
            # exceeds 120 degrees, re-compute the imagePlane for 120 degrees.     
            fov = math.ceil((2*math.atan2(imageExtent,camDist)*180/math.pi)/10)*10
            if fov>120:
                fov = 120
                # Calculate half of the image plane distance resulting from FOV=120
                # degrees:
                d = camDist*math.tan(math.pi/3)
                # Calculate half the length of the previously calculated image plane:
                dimPlane = dist(Point(imagePlane[0,:]),Point(imagePlane[1,:]))/2
                # Scale down the change in x and y by multiplying half the distance
                # between xint and yint and an end of the image plane by d/dimPlane:
                dx = d/dimPlane*(xint - imagePlane[0,0])
                dy = d/dimPlane*(yint - imagePlane[0,1]) 
                # Calculate the new end points of the image plane:
                x0 = xint - dx
                y0 = yint - dy
                x3 = xint + dx
                y3 = yint + dy
                imagePlane = np.array([[x0,y0],[x3,y3]])
        
            ## Determine the endpoints of the reference line:
            # Project the latitude/longitude values to a 1-D coordinate system:  
            lineCoords = np.vstack((imagePlane,premRefLine))
            xLines = np.zeros((len(lineCoords)))
            lon1 = lineCoords[0,0]
            lat1 = lineCoords[0,1]
            for k in range(1,len(xLines)):
                lon2 = lineCoords[k,0]
                lat2 = lineCoords[k,1]
                
                x = (lon1-lon2)*40075000*3.28084*math.cos((lat1+lat2)*math.pi/360)/360
                y = (lat1-lat2)*40075000*3.28084/360
                xLines[k] = np.sign(x)*np.linalg.norm([x,y])
            
            xLines = xLines*np.sign(xLines[1])
            xLines[0] = 0
            xLines.round(decimals=4)
            
            # Needs comments here:
            idx = np.where(np.logical_and(xLines[-2:]>=xLines[0],xLines[-2:]<=xLines[1]))[0]
            
            if len(idx)==2:
                refLine = premRefLine
            elif len(idx)==1:
                pt1 = premRefLine[idx,:]
                if (xLines[3-idx] - xLines[2+idx])<0:
                    pt2 = imagePlane[0,:]
                else:
                    pt2 = imagePlane[1,:]
                refLine = np.vstack((pt1,pt2))
            else:
                refLine = imagePlane
                
            # Compute pixel scale:
            scale = dist(Point([imagePlane[0,0],imagePlane[0,1]]),
                             Point([imagePlane[1,0],imagePlane[1,1]]))/640
           
            # Compute the camera heading by calculating the orientation of the line
            # segment fron camera location (x1,y1) to the point where the line segment
            # from the camera location intersects the preliminary reference line:
            heading = round(get_bearing(x1, y1, xint, yint),2)
            
            return refLine, imagePlane, scale, fov, heading
          
        def image_retrieve(footprint,im_path,imName,key):
            ## Compute the latitude/longitude of footprint centroid:
            latlon = Polygon(footprint).centroid
            latlon = [latlon.x,latlon.y]
            
            ## Extract camera location:
            p1 = download_metadata(latlon,key)
            
            ## Identify the lines that are visible from the extracted camera location:
            
            # If camera location is succesfully extracted and the camera location is 
            # not in within the footprint (i.e., an indoor image was not extracted):
            if (p1 is not None) and (not Polygon(footprint).contains(p1)):
                # Identify the footprint edges visible from the extracted camera
                # location:
                #plt.plot(footprint[:,0],footprint[:,1])
                nVert = footprint.shape[0] - 1
                visibleLineSeg = np.zeros([nVert,4])
                visibleCount = 0
                # Create light rays from the camera location to the midpoint of each 
                # line segment that forms the footprint:
                for k in range(nVert):
                    q1 = midpoint_calc(Point(footprint[k,0],footprint[k,1]),
                                       Point(footprint[k+1,0],footprint[k+1,1]))
                    #plt.plot(q1.x,q1.y,'bo')
                    #plt.plot([p1.x,q1.x],[p1.y,q1.y])
                    counter = 0
                    # If the light ray cast on a line segment from the camera location
                    # intersects with any other line segment that forms the footprint, 
                    # the light ray is obstructed by the intersecting line segment. 
                    # Hence, the line segment on which the ray is cast is not visible
                    # to the camera. Here intersection is defined as the two lines
                    # crossing each other. The situation where a line ends on another
                    # line is not considered line intersection:
                    for m in range(nVert):
                        if m!=k:
                            p2 = Point(footprint[m,0], footprint[m,1])
                            q2 = Point(footprint[m+1,0], footprint[m+1,1])
                            if doIntersect(p1,q1,p2,q2):
                                counter+=1
                    # If the light ray cast on a line segment from the camera location
                    # does not intersect with any other line segment from the footprint 
                    # the line segment on which the ray is cast is visible to the 
                    # camera:
                    if counter==0:
                        visibleLineSeg[visibleCount,:] = [footprint[k,0],
                                                          footprint[k,1],
                                                          footprint[k+1,0],
                                                          footprint[k+1,1]]
                        visibleCount +=1
                
                # Remove the rows of zeros from visibleLineSeg:
                visibleLineSeg = visibleLineSeg[~np.all(visibleLineSeg == 0, axis=1)]
            
                ## Compute the vertices and length of the ideal reference line:
                premRefLine = compute_refline(footprint,p1,visibleLineSeg)  
                if premRefLine is not None:
                    # Compute FOV and heading angles appropriate for the camera/footprint configuration   
                    refLine, imagePlane, scale, fov, heading = get_camParam(premRefLine,p1)
                    
                    # Download image for segmentation:
                    download_image(latlon,heading,fov,key,imName,im_path)
                else:
                    refLine = None
                    imagePlane = None
                    scale = None
                    fov = None
                    heading = None
                    
            else:
                refLine = None
                imagePlane = None
                scale = None
                fov = None
                heading = None
            return refLine, imagePlane, scale, fov, heading

        os.makedirs('tmp/images/street',exist_ok=True)
        self.street_images = []
        for count, footprint in enumerate(footprints):
            fp = np.fliplr(np.squeeze(np.array(footprint)))
            refLine, imagePlane, scale, fov, heading = image_retrieve(fp,'tmp/images/street',count,self.apikey)
            if scale is not None:
                self.street_images.append(f"tmp/images/street/{count}.png")
            else:
                self.street_images.append(None)
            if isinstance(refLine, np.ndarray): 
                self.refLines.append(refLine[:])
            else: 
                self.refLines.append(None)
            if isinstance(imagePlane, np.ndarray): 
                self.imagePlanes.append(imagePlane[:]) 
            else: 
                self.refLines.append(None)
            self.streetScales.append(scale)
            self.streetFOVs.append(fov)
            self.streetHeadings.append(heading)