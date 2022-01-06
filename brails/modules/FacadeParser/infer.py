# -*- coding: utf-8 -*-
"""
author: Barbaros Cetiner
"""
import requests
import math
import torch
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as T
from shapely.geometry import Point, LineString, Polygon
import json
from tqdm import tqdm

key = ""
gpu_enabled = True
model_path = "models/facadeParser.pth"
footprint_file = "Alameda_footprints_400up.geojson"
im_path = "images"
save_segimages = True
segim_path = "segmentedImages"

# Function that downloads a file given its URL and the desired path to save it:
def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

# Function that obtains the camera location for a StreetView image given the
# latitude/longitude and API key for the location:
def download_metadata(latlon,key):
    metadataBaseURL = "https://maps.googleapis.com/maps/api/streetview/metadata?size=640x640&location=%s&key=%s"
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
    streetViewBaseURL = "https://maps.googleapis.com/maps/api/streetview?size=640x640&location=%s&heading=%s&fov=%s&key=%s"
    image_url = streetViewBaseURL % (str(latlon)[1:-1].replace(" ", ""),str(heading),str(fov),key)
    r = requests.get(image_url)
    if r.status_code == 200:
        with open(os.path.join(im_path,f"{imName}.jpg"), 'wb') as f:
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

    # Compute the point where the reference line segment that forms the shortest 
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
    # Convert the  
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
   
    # Compute the camera heading:
    heading = round(get_bearing(x1, y1, xint, yint),2)
    
    plt.plot(footprint[:,0],footprint[:,1])
    plt.plot(premRefLine[:,0],premRefLine[:,1],'r')
    plt.plot(refLine[:,0],refLine[:,1],'rx')
    plt.plot(imagePlane[:,0],imagePlane[:,1],'g')
    plt.plot(p1.x,p1.y,'kx')
    plt.show()
    return refLine, imagePlane, scale, fov, heading
  
def image_retrieve(footprint,im_path,imName,key):
    ## Compute the latitude/longitude of footprint centroid:
    latlon = Polygon(footprint).centroid
    latlon = [latlon.x,latlon.y]
    
    ## Extract camera location:
    p1 = download_metadata(latlon,key)
    
    ## Identify the lines that are visible from the extracted camera location:
    
    # If camera location is succesfully extracted and the camera location is 
    # not in within the footprint (i.e., an indoor was not extracted):
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
            #plt.plot(imagePlane[:,0],imagePlane[:,1],'--r')  
            #plt.plot([refLineRough[0],refLineRough[2]],[refLineRough[1],refLineRough[3]],'k')
            #plt.plot(refLine[:,0],refLine[:,1],'k')
            #plt.show()
            
            # Download image for segmentation:
            download_image(latlon,heading,fov,key,imName,im_path)
            
            #img = Image.open(f"{imName}.jpg")
            #plt.imshow(img)
            #plt.show()
    
        #for k in range(visibleLineSeg.shape[0]):
        #    plt.plot([visibleLineSeg[k,0],visibleLineSeg[k,2]],
        #             [visibleLineSeg[k,1],visibleLineSeg[k,3]])    
        #plt.show()
    else:
        refLine = None
        imagePlane = None
        scale = None
        fov = None
        heading = None
    return refLine, imagePlane, scale, fov, heading

def compute_roofrun(footprint):
    # Find the mimumum area rectangle that fits around the footprint
    # What failed: 1) PCA, 2) Rotated minimum area rectangle 
    # 3) Modified minimum area rectangle
    # Current implementation: Tight-fit bounding box
    
    # Convert lat/lon values to Cartesian coordinates for better coordinate resolution:
    xyFootprint = np.zeros((len(footprint),2))
    for k in range(1,len(footprint)):
        lon1 = footprint[0,0]
        lon2 = footprint[k,0]
        lat1 = footprint[0,1]
        lat2 = footprint[k,1]
        
        xyFootprint[k,0] = (lon1-lon2)*40075000*3.28084*math.cos((lat1+lat2)*math.pi/360)/360
        xyFootprint[k,1] = (lat1-lat2)*40075000*3.28084/360
        
    # Calculate the slope and length of each line segment comprising the footprint polygon:
    slopeSeg = np.diff(xyFootprint[:,1])/np.diff(xyFootprint[:,0])
    
    segments = np.arange(len(slopeSeg))
    lengthSeg = np.zeros(len(slopeSeg))
    for k in range(len(xyFootprint)-1):
        p1 = np.array([xyFootprint[k,0],xyFootprint[k,1]])
        p2 = np.array([xyFootprint[k+1,0],xyFootprint[k+1,1]])
        lengthSeg[k] = np.linalg.norm(p2-p1)
    
    # Cluster the line segments based on their slope values:
    slopeClusters = []
    totalLengthClusters = []
    segmentsClusters = []
    while slopeSeg.size>1:
        ind = np.argwhere(abs((slopeSeg-slopeSeg[0])/slopeSeg[0])<0.3).squeeze()
        slopeClusters.append(np.mean(slopeSeg[ind]))
        totalLengthClusters.append(np.sum(lengthSeg[ind]))
        segmentsClusters.append(segments[ind])
        
        indNot = np.argwhere(abs((slopeSeg-slopeSeg[0])/slopeSeg)>=0.3).squeeze()
        slopeSeg = slopeSeg[indNot]
        lengthSeg = lengthSeg[indNot]
        segments = segments[indNot]
    
    if slopeSeg.size==1:
        slopeClusters.append(slopeSeg)
        totalLengthClusters.append(lengthSeg)
        segmentsClusters.append(segments)
    
    # Mark the two clusters with the highest total segment lengths as the 
    # principal directions of the footprint
    principalDirSlopes = []
    principalDirSegments = []
    for ind in np.flip(np.argsort(totalLengthClusters)[-2:]):
        principalDirSlopes.append(slopeClusters[ind])
        principalDirSegments.append(segmentsClusters[ind])
        
    
    xFootprint = xyFootprint[:,0]
    yFootprint = xyFootprint[:,1]
    slopeSeg = np.diff(xyFootprint[:,1])/np.diff(xyFootprint[:,0])
    
    bndLines = np.zeros((4,4))
    for cno,cluster in enumerate(principalDirSegments):
        xp = np.zeros((2*len(cluster)))
        yp = np.zeros((2*len(cluster)))
        for idx, segment in enumerate(cluster):
            angle = math.pi/2 - math.atan(slopeSeg[segment])
            x = xFootprint[segment:segment+2]
            y = yFootprint[segment:segment+2]
            xp[2*idx:2*idx+2] = x*math.cos(angle) - y*math.sin(angle)
            yp[2*idx:2*idx+2] = x*math.sin(angle) + y*math.cos(angle)
            #plt.plot(xp[2*idx:2*idx+2], yp[2*idx:2*idx+2])
        
        minLineIdx = int(np.argmin(xp)/2)
        maxLineIdx = int(np.argmax(xp)/2)
        
        #plt.plot([xp[2*minLineIdx],xp[2*minLineIdx+1],
        #          xp[2*maxLineIdx],xp[2*maxLineIdx+1]],
        #         [yp[2*minLineIdx],yp[2*minLineIdx+1],
        #          yp[2*maxLineIdx],yp[2*maxLineIdx+1]],'rx')
        #plt.show()
        
        minLineIdx = cluster[int(np.argmin(xp)/2)]
        maxLineIdx = cluster[int(np.argmax(xp)/2)]
        
        bndLines[2*cno:2*cno+2,:] = np.array([[xFootprint[minLineIdx],
                                               yFootprint[minLineIdx],
                                               xFootprint[minLineIdx+1],
                                               yFootprint[minLineIdx+1]],
                                              [xFootprint[maxLineIdx],
                                               yFootprint[maxLineIdx],
                                               xFootprint[maxLineIdx+1],
                                               yFootprint[maxLineIdx+1]]])
    
    
    #plt.plot(xyFootprint[:,0],xyFootprint[:,1],'k',linewidth=5)
    #plt.plot([bndLines[0,0],bndLines[0,2],bndLines[1,0],bndLines[1,2]],
    #         [bndLines[0,1],bndLines[0,3],bndLines[1,1],bndLines[1,3]],'rx')
    #plt.plot([bndLines[2,0],bndLines[2,2],bndLines[3,0],bndLines[3,2]],
    #         [bndLines[2,1],bndLines[2,3],bndLines[3,1],bndLines[3,3]],'bx')
    
    
    bbox = np.zeros((5,2))
    counter = 0
    for k in range(2):
        line1 = bndLines[k,:]
        x1 = line1[0]
        x2 = line1[2]
        y1 = line1[1]
        y2 = line1[3]
        for m in range(2,4):
            line2 = bndLines[m,:]
            x3 = line2[0]
            x4 = line2[2]
            y3 = line2[1]
            y4 = line2[3]
            d = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
            bbox[counter,:] = [((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4))/d, 
                               ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4))/d]
            counter += 1
    bbox[4,:] = bbox[0,:]
    bbox[2:4,:] = np.flipud(bbox[2:4,:])
    #plt.plot(bbox[:,0],bbox[:,1])        
    #plt.show()
    
    sideLengths = np.linalg.norm(np.diff(bbox,axis=0),axis=1)
    roof_run = min(sideLengths)
    
    return roof_run

def install_default_model(model_path):
    if model_path == "models/facadeParser.pth":
        os.makedirs('models',exist_ok=True)
        model_path = os.path.join('models','facadeParser.pth')

        if not os.path.isfile(model_path):
            print('Loading default model file to the models folder...')
            torch.hub.download_url_to_file('https://zenodo.org/record/5809365/files/facadeParser.pth',
                                           model_path, progress=False)
            print('Default model loaded.')
    else:
        print(f'Inferences will be performed using the custom model at {model_path}.')
        
def gen_bbox(roofContour):   
    minx = min(roofContour[:,0])
    maxx = max(roofContour[:,0])
    miny = min(roofContour[:,1])
    maxy = max(roofContour[:,1])
    roofBBoxPoly = Polygon([(minx,miny),(minx,maxy),(maxx,maxy),(maxx,miny)])
    return roofBBoxPoly

def gen_bboxR0(roofBBoxPoly,facadeBBoxPoly):
    x,y = roofBBoxPoly.exterior.xy
    x = np.array(x).astype(int); y = np.array(y).astype(int)
    
    yBottom = max(y)
    ind = np.where(y==yBottom)
    xRoofBottom = np.unique(x[ind])
    yRoofBottom = np.tile(yBottom,len(xRoofBottom))
    
    x,y = facadeBBoxPoly.exterior.xy
    x = np.array(x).astype(int); y = np.array(y).astype(int)
    
    yBottom = max(y)
    ind = np.where(y==yBottom)
    xFacadeBottom = xRoofBottom
    yFacadeBottom = np.tile(yBottom,len(xFacadeBottom))
    
    R0BBoxPoly = Polygon([(xFacadeBottom[0],yFacadeBottom[0]),
                          (xFacadeBottom[1],yFacadeBottom[1]),
                          (xRoofBottom[1],yRoofBottom[1]),
                          (xRoofBottom[0],yRoofBottom[0])])
    return R0BBoxPoly

def decode_segmap(image, nc=5):
    label_colors = np.array([(0, 0, 0),(255, 0, 0), (255, 165, 0), (0, 0, 255),
                             (175,238,238)])
                 # 0=background # 1=roof, 2=facade, 3=window, 4=door

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
  
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    
    rgb = np.stack([r, g, b], axis=2)
    return rgb


# Set the computing environment
if gpu_enabled:
    dev = 'cuda'
else:
    dev = 'cpu'

# Load the trained model and set the model to evaluate mode
install_default_model(model_path)
model = torch.load(model_path)
model.eval()

# Create the output table
dfOut = pd.DataFrame(columns=['img', 'R0', 'R1','pitch'])

# Get the list of footprints from the footprint_file
with open(footprint_file) as f:
    footprintList = json.load(f)["features"]

for polyNo in tqdm(range(4)):#tqdm(range(len(footprintList))):
    # Extract building footprint information and download the image viewing
    # this footprint:
    footprint = np.fliplr(np.squeeze(np.array(footprintList[polyNo]['geometry']['coordinates'])))
    
    refLine, imagePlane, scale, fov, heading = image_retrieve(footprint,im_path,polyNo,key)
    # If ref_length could not be attained, i.e. an image could not be downloaded,
    # skip to the next footprint:
    
    if scale is None:
        continue
    
    # Run image through the segmentation model
    img = Image.open(os.path.join(im_path,f"{polyNo}.jpg"))
    
    trf = T.Compose([T.Resize(640),
                     T.ToTensor(), 
                     T.Normalize(mean = [0.485, 0.456, 0.406], 
                                 std = [0.229, 0.224, 0.225])])
    
    inp = trf(img).unsqueeze(0).to(dev)
    scores = model.to(dev)(inp)['out']
    pred = torch.argmax(scores.squeeze(), dim=0).detach().cpu().numpy()
    
    # Extract component masks
    maskRoof = (pred==1).astype(np.uint8)
    maskFacade = (pred==2).astype(np.uint8)
    maskWin = (pred==3).astype(np.uint8)
    maskDoor = (pred==4).astype(np.uint8)
    
    # Open and close masks
    kernel = np.ones((10,10), np.uint8)
    openedFacadeMask = cv2.morphologyEx(maskFacade, cv2.MORPH_OPEN, kernel)
    maskFacade = cv2.morphologyEx(openedFacadeMask, cv2.MORPH_CLOSE, kernel)
    openedWinMask = cv2.morphologyEx(maskWin, cv2.MORPH_OPEN, kernel)
    maskWin = cv2.morphologyEx(openedWinMask, cv2.MORPH_CLOSE, kernel)
    plt.imshow(maskRoof)
    
    #plt.imshow(maskWin); plt.show()
    
    # Find roof contours
    contours, _ = cv2.findContours(maskRoof,cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    roofContour = max(contours, key = cv2.contourArea).squeeze()
    plt.plot(roofContour[:,0],roofContour[:,1])
    
    # Find the mimumum area rectangle that fits around the primary roof contour
    roofMinRect = cv2.minAreaRect(roofContour)
    roofMinRect = cv2.boxPoints(roofMinRect)
    roofMinRect = np.int0(roofMinRect)
    roofMinRectPoly = Polygon([(roofMinRect[0,0],roofMinRect[0,1]),
                                        (roofMinRect[1,0],roofMinRect[1,1]),
                                        (roofMinRect[2,0],roofMinRect[2,1]),
                                        (roofMinRect[3,0],roofMinRect[3,1])])
    x,y = roofMinRectPoly.exterior.xy
    plt.plot(x,y)
    
    roofBBoxPoly = gen_bbox(roofContour)
    x,y = roofBBoxPoly.exterior.xy
    roofPixHeight = max(y)-min(y)
    
    plt.plot(x,y)
    plt.show()
    
    # Find facade contours
    #plt.imshow(maskFacade)
    contours, _ = cv2.findContours(maskFacade,cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    facadeContour = max(contours, key = cv2.contourArea).squeeze()
    #plt.plot(facadeContour[:,0],facadeContour[:,1])
    
    facadeBBoxPoly = gen_bbox(facadeContour)
    x,y = facadeBBoxPoly.exterior.xy
    #plt.plot(x,y)
    
    R0BBoxPoly = gen_bboxR0(roofBBoxPoly,facadeBBoxPoly)
    x,y = R0BBoxPoly.exterior.xy
    R0PixHeight = max(y)-min(y)
    R1PixHeight = R0PixHeight + roofPixHeight

    #plt.plot(x,y)
    #plt.show()
    
    # Calculate heigths of interest
    R0 = R0PixHeight*scale
    R1 = R1PixHeight*scale
    roof_run = compute_roofrun(footprint) 
    roofPitch = (R1-R0)/roof_run
    dfOut.loc[polyNo] = [polyNo, R0, R1, roofPitch]

    # Save segmented images
    if save_segimages:
        rgb = decode_segmap(pred)
        #plt.imshow(rgb); plt.show()
        
        rgb = Image.fromarray(rgb)
        rgb.save(os.path.join(segim_path,f"{polyNo}_segmented.jpg"))


#dfOut.to_csv('facadeParsingResults.csv')

# Unload the model from GPU
if gpu_enabled:
    del model
    torch.cuda.empty_cache()