# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 The Regents of the University of California
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
# 03-15-2024

from math import radians, sin, cos, atan2, sqrt
from shapely import to_geojson
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import json

def haversine_dist(p1,p2):
    """
    Function that calculates the Haversine distance between two points.
    
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

def mesh_polygon(polygon, rows:int, cols:int):
    """
    Function that splits a polygon into individual rectangular polygons.
    
    Inputs: A Shapely polygon and number of rows and columns of rectangular
            cells requested to mesh the area of the polygon.
            with each point defined as a list of two floating-point values
    Output: Individual rectangular cells stored as Shapely polygons
    """
    
    # Get bounds of the polygon:
    min_x, min_y, max_x, max_y = polygon.bounds

    # Calculate dimensions of each cell:
    width = (max_x - min_x) / cols
    height = (max_y - min_y) / rows

    rectangles = []
    # For each cell:
    for i in range(rows):
        for j in range(cols):
            # Calculate coordinates of the cell vertices: 
            x1 = min_x + j * width
            y1 = min_y + i * height
            x2 = x1 + width
            y2 = y1 + height
            
            # Convert the computed geometry into a polygon:
            poly = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            
            # Check if the obtained cell intersects with the polygon:
            if poly.intersects(polygon):
                poly = poly.intersection(polygon)
                # If the intersection is a finite geometry keep its envelope as
                # a valid cell:
                if poly.area > 0:
                    rectangles.append(poly.envelope)
    return rectangles

def plot_polygon_cells(bpoly, rectangles, fout=False):
    """
    Function that plots the mesh for a polygon and saves the plot into a PNG image
    
    Inputs: A Shapely polygon and a list of rectangular mesh cells saved as
            Shapely polygons. Optional input is a string containing the name of the
            output file
    """
    
    if bpoly.geom_type == 'MultiPolygon':
        for poly in bpoly.geoms:
            plt.plot(*poly.exterior.xy,'k')
    else:
        plt.plot(*bpoly.exterior.xy,'k')
    for rect in rectangles:
        try:
            plt.plot(*rect.exterior.xy)
        except:
            pass        
    if fout:
        plt.savefig(fout, dpi=600, bbox_inches="tight")
    plt.show()

def write_polygon2geojson(poly,outfile):
    """
    Function that writes a single Shapely polygon into a GeoJSON file
    
    Input: A Shapely polygon or multi polygon
    """

    if 'geojson' not in outfile.lower():
        outfile = outfile.replace(outfile.split('.')[-1],'geojson')
    geojson = {'type':'FeatureCollection', 
               "crs": {"type": "name", 
                       "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
               'features':[]}
    if poly.geom_type == 'MultiPolygon':
        polytype = 'MultiPolygon'
    elif poly.geom_type == 'Polygon':
        polytype = 'Polygon'
        
    feature = {'type':'Feature',
               'properties':{},
               'geometry':{'type': polytype,
                           'coordinates':[]}}
    feature['geometry']['coordinates'] = json.loads(
        to_geojson(poly).split('"coordinates":')[-1][:-1])
    geojson['features'].append(feature)
    with open(outfile, 'w') as outputFile:
        json.dump(geojson, outputFile, indent=2)    