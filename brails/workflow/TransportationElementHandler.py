"""Class object for handling elements of transportation networks."""

# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 The Regents of the University of California
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
# 08-19-2024


import copy
import json
from collections import defaultdict
from typing import List, Set, Tuple, Dict, Literal, Union
import requests
import numpy as np
from shapely.ops import split
from shapely.geometry import LineString, Polygon, Point, mapping
from shapely import wkt
from requests.adapters import HTTPAdapter, Retry
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from brails.workflow.FootprintHandler import FootprintHandler
from brails.utils.geoTools import haversine_dist

# The map defines the default values according to MTFCC code
# https://www2.census.gov/geo/pdfs/maps-data/data/tiger/tgrshp2009/TGRSHP09AF.pdf
# May need better models

ROADLANES_MAP = {'S1100': 4, "S1200": 2, "S1400": 1, "S1500": 1, "S1630": 1,
                 "S1640": 1, "S1710": 1, "S1720": 1, "S1730": 1, "S1740": 1,
                 "S1750": 1, "S1780": 1, "S1810": 1, "S1820": 1, "S1830": 1}

ROADSPEED_MAP = {'S1100': 70, "S1200": 55, "S1400": 25, "S1500": 25,
                 "S1630": 25, "S1640": 25, "S1710": 25, "S1720": 25,
                 "S1730": 25, "S1740": 10, "S1750": 10, "S1780": 10,
                 "S1810": 10, "S1820": 10, "S1830": 10}


class TransportationElementHandler:
    """
    BRAILS class for handling elements of transportation networks.

    TransportationElementHandler obtains and handles features of roadways,
    bridges, and tunnels from public databases.

    Attributes__
        requested_elements (list): A list containing the names of requested
                                    elements
        queryarea (str or tuple): Area for which transportation elements
                                    will be obtained
        datasource (str): TIGER or OSM

    Methods__
        fetch_transportation_elements: Methods that extracts transportation
                                        elements for a query area from
                                        available public data sources
    """

    def __init__(self, requested_elements):
        self.queryarea = ""
        if "roads" in requested_elements:
            self.output_files = {"roads": "Roads.geojson"}
        else:
            self.output_files = {}
        self.requested_elements = requested_elements
        self.attribute_maps = {'lanes': ROADLANES_MAP,
                               'speed': ROADSPEED_MAP}

    def fetch_transportation_elements(
        self,
        queryarea: Union[str, tuple],
        datasource: str = "TIGER"
    ):
        """
        Obtain transportation network elements within an area.

        This method gets elements such as roads, bridges, and tunnels
        for an area of interest from a specified data source.

        Arguments__
            queryarea (str or tuple): Area for which transportation elements
                                        will be obtained
            datasource (str, optional): Public dataset used for getting the
                                        transportation elements. Options: TIGER
                                        or OSM. Defaults to TIGER

        Raises__
            NotImplementedError: Raised when an element type or data source is
                                    not yet implemented

        Returns__
            list: A list of transportation elements (e.g., roads, railways)
                    that are withing the query area as obtained from data
                    source.

        Examples__
            >>> fetch_transportation_elements('Berkeley, CA')
            [<Element1>, <Element2>, ...]

            >>> fetch_transportation_elements((-122.2970, 37.8882,
                                               -122.2501, 37.8483),
                                              datasource='OSM'
                                              )
            [<Element1>, <Element2>, ...]

        Notes__
            - If `queryarea` is a string, it is expected to be a place name.
            - If `queryarea` is a tuple, it needs to contain latitude and
                longitude pairs defining a bounding polygon.
        """
        self.queryarea = queryarea
        self.datasource = datasource

        def query_generator(bpoly: Polygon,
                            eltype: str,
                            querytype: str = 'elements'
                            ) -> str:
            # Get the bounds of the entered bounding polygon and lowercase the
            # entered element type:
            bbox = bpoly.bounds
            eltype = eltype.lower()

            # If element type is bridge, generate an NBI query for the bounds
            # of bpoly:
            if eltype == "bridge":
                query = (
                    "https://geo.dot.gov/server/rest/services/Hosted/"
                    + "National_Bridge_Inventory_DS/FeatureServer/0/query?"
                    + "where=1%3D1&outFields=*&geometry="
                    + f"{bbox[0]}%2C{bbox[1]}%2C{bbox[2]}%2C{bbox[3]}"
                    + "&geometryType=esriGeometryEnvelope&inSR=4326"
                    + "&spatialRel=esriSpatialRelIntersects&outSR=4326&f=json"
                )

            # If element type is tunnel, generate an NTI query for the bounds
            # of bpoly:
            elif eltype == "tunnel":
                query = (
                    "https://geo.dot.gov/server/rest/services/Hosted/"
                    + "National_Tunnel_Inventory_DS/FeatureServer/0/query?"
                    + "where=1%3D1&outFields=*"
                    f"&geometry={bbox[0]}%2C{bbox[1]}%2C{bbox[2]}%2C{bbox[3]}"
                    + "&geometryType=esriGeometryEnvelope&inSR=4326"
                    + "&spatialRel=esriSpatialRelIntersects&outSR=4326&f=json"
                )

            # If element type is railroad, generate an NARNL query for the
            # bounds of bpoly:
            elif eltype == "railroad":
                query = (
                    "https://geo.dot.gov/server/rest/services/Hosted/"
                    + "North_American_Rail_Network_Lines_DS/FeatureServer/0/"
                    + "query?where=1%3D1&outFields=*&geometry="
                    + f"{bbox[0]}%2C{bbox[1]}%2C{bbox[2]}%2C{bbox[3]}"
                    + "&geometryType=esriGeometryEnvelope&inSR=4326"
                    + "&spatialRel=esriSpatialRelIntersects&outSR=4326&f=json"
                )

            # If element type is primary_road, generate a TIGER query for the
            # bounds of bpoly:
            elif eltype == "primary_road":
                query = (
                    "https://tigerweb.geo.census.gov/arcgis/rest/services/"
                    + "TIGERweb/Transportation/MapServer/2/query?where=&text="
                    + "&outFields=OID,NAME,MTFCC"
                    f"&geometry={bbox[0]}%2C{bbox[1]}%2C{bbox[2]}%2C{bbox[3]}"
                    + "&geometryType=esriGeometryEnvelope&inSR=4326"
                    + "&spatialRel=esriSpatialRelIntersects&outSR=4326&f=json"
                )

            # If element type is secondary_road, generate a TIGER query for the
            # bounds of bpoly:
            elif eltype == "secondary_road":
                query = (
                    "https://tigerweb.geo.census.gov/arcgis/rest/services/"
                    + "TIGERweb/Transportation/MapServer/6/query?where=&text="
                    + "&outFields=OID,NAME,MTFCC"
                    f"&geometry={bbox[0]}%2C{bbox[1]}%2C{bbox[2]}%2C{bbox[3]}"
                    + "&geometryType=esriGeometryEnvelope&inSR=4326"
                    + "&spatialRel=esriSpatialRelIntersects&outSR=4326&f=json"
                )

            # If element type is local_road, generate a TIGER query for the
            # bounds of bpoly:
            elif eltype == "local_road":
                query = (
                    "https://tigerweb.geo.census.gov/arcgis/rest/services/"
                    + "TIGERweb/Transportation/MapServer/8/query?where=&text="
                    + "&outFields=OID,NAME,MTFCC"
                    f"&geometry={bbox[0]}%2C{bbox[1]}%2C{bbox[2]}%2C{bbox[3]}"
                    + "&geometryType=esriGeometryEnvelope&inSR=4326"
                    + "&spatialRel=esriSpatialRelIntersects&outSR=4326&f=json"
                )

            # Otherwise, raise a NotImplementedError:
            else:
                raise NotImplementedError("Element type not implemented")

            # If the query is intended for obtaining element counts, modify
            # the obtained query accordingly:
            if querytype == 'counts':
                query.replace("outSR=4326", "returnCountOnly=true")
            elif querytype == 'maxcounts':
                query = query.split("/query?")
                query = query[0] + "?f=pjson"
            return query

        def create_pooling_session():
            # Define a retry stratey for common error codes to use when
            # downloading data:
            session = requests.Session()
            retries = Retry(
                total=10,
                backoff_factor=0.1,
                status_forcelist=[500, 502, 503, 504],
            )
            session.mount("https://", HTTPAdapter(max_retries=retries))

            return session

        def get_el_counts(bpoly: Polygon, eltype: str) -> int:
            # Create a persistent requests session:
            s = create_pooling_session()

            # Create the query required to get the element counts:
            query = query_generator(bpoly, eltype, 'counts')

            # Download the element count for the bounding polygon using the
            # defined retry strategy:
            print("Querying element count for the bounding polygon")
            r = s.get(query)
            elcount = r.json()["count"]
            print(f"Querying finished with count {elcount}")

            return elcount

        def get_max_el_count(eltype: str) -> int:
            # Create a persistent requests session:
            s = create_pooling_session()

            # Create the query required to get the maximum element count:
            query = query_generator(bpoly, eltype, 'maxcounts')

            # Download the maximum element count for the bounding polygon using
            # the defined retry strategy:
            r = s.get(query)
            maxelcount = r.json()["maxRecordCount"]

            return maxelcount

        def list2geojson(datalist: list, eltype: str, bpoly: Polygon) -> dict:
            # Lowercase the entered element type string:
            eltype = eltype.lower()

            # Populate the geojson header:
            geojson = {
                "type": "FeatureCollection",
                "crs": {
                    "type": "name",
                    "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
                },
                "features": [],
            }

            # Scroll through each item in the datalist:
            for item in datalist:
                # If the element is a bridge or tunnel, parse its geometry as a
                # point and check if the extracted point is within the bounding
                # polygon:
                if eltype in ["bridge", "tunnel"]:
                    geometry = [item["geometry"]["x"], item["geometry"]["y"]]
                    if bpoly.contains(Point(geometry)):
                        feature = {
                            "type": "Feature",
                            "properties": {},
                            "geometry": {"type": "Point", "coordinates": []},
                        }
                    else:
                        continue
                else:
                    # If the element is a road segment, parse it as a
                    # MultiLineString and check if the extracted segment is
                    # within the bounding polygon:
                    geometry = item["geometry"]["paths"]
                    if bpoly.intersects(LineString(geometry[0])):
                        feature = {
                            "type": "Feature",
                            "properties": {},
                            "geometry": {
                                "type": "MultiLineString",
                                "coordinates": [],
                            },
                        }
                    else:
                        continue

                # Copy the obtained geometry in a feature dictionary:
                feature["geometry"]["coordinates"] = geometry.copy()

                # Read item attributes
                properties = item["attributes"]

                # For each attribute:
                for prop in properties.keys():
                    # Clean up the property name from redundant numeric text:
                    if "_" in prop:
                        strsegs = prop.split("_")
                        removestr = ""
                        for seg in strsegs:
                            if any(char.isdigit() for char in seg):
                                removestr = "_" + seg
                        propname = prop.replace(removestr, "")
                    else:
                        propname = prop

                    # Write the property in a feature dictionary:
                    feature["properties"][propname] = properties[prop]

                # Add the feature in the geojson dictionary:
                geojson["features"].append(feature)
            return geojson

        def print_el_counts(datalist: list, eltype: str):
            nel = len(datalist)
            eltype_print = eltype.replace("_", " ")

            if eltype in ["bridge", "tunnel"]:
                elntype = "node"
            else:
                elntype = "edge"

            if nel == 1:
                suffix = ""
            else:
                suffix = "s"

            print(f"Found {nel} {eltype_print} {elntype}{suffix}")

        def write2geojson(bpoly: Polygon, eltype: str) -> dict:
            # nel = get_el_counts(bpoly, eltype)

            # Create a persistent requests session:
            s = create_pooling_session()

            # Download element data using the defined retry strategy:
            query = query_generator(bpoly, eltype)
            r = s.get(query)

            # Check to see if the data was successfully downloaded:
            if "error" in r.text:
                print(
                    f"Data server for {eltype.replace('_',' ')}s is currently"
                    + "unresponsive. Please try again later."
                )
                datalist = []
            else:
                datalist = r.json()["features"]

            # If road data convert it into GeoJSON format:
            jsonout = list2geojson(datalist, eltype, bpoly)

            # If not road data convert it into GeoJSON format and write it into
            # a file:
            if "_road" not in eltype:
                print_el_counts(jsonout["features"], eltype)
                if len(datalist) != 0:
                    output_filename = f"{eltype.title()}s.geojson"
                    with open(output_filename, "w") as output_file:
                        json.dump(jsonout, output_file, indent=2)
                else:
                    jsonout = ""
            return jsonout

        def find(s, ch):
            return [i for i, ltr in enumerate(s) if ltr == ch]

        def lstring2xylist(lstring):
            coords = lstring.xy
            coordsout = []
            for i in range(len(coords[0])):
                coordsout.append([coords[0][i], coords[1][i]])
            return coordsout

        def combine_write_roadjsons(roadjsons, bpoly):
            roadjsons_combined = {
                "type": "FeatureCollection",
                "crs": {
                    "type": "name",
                    "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
                },
                "features": [],
            }
            rtype = {
                "primary_road": "P",
                "secondary_road": "S",
                "local_road": "L",
            }
            for key, roadjson in roadjsons.items():
                for item in roadjson["features"]:
                    # Get into the properties and add RTYPE into edge property
                    item["properties"]["RTYPE"] = rtype[key]
                    # Get into the geometry and intersect it with the bounding
                    # polygon:
                    lstring = LineString(item["geometry"]["coordinates"][0])
                    coords = lstring.intersection(bpoly)
                    # If output is LineString write the intersecting geometry
                    # in item and append it to the combined the json file:
                    if coords.geom_type == "LineString":
                        coordsout = lstring2xylist(coords)
                        item["geometry"]["coordinates"] = [coordsout]
                        roadjsons_combined["features"].append(item)
                    # If not, create multiple copies of the same json for each
                    # linestring:
                    else:
                        mlstring_wkt = coords.wkt
                        inds = find(mlstring_wkt, "(")[1:]
                        nedges = len(inds)
                        for i in range(nedges):
                            if i + 1 != nedges:
                                edge = wkt.loads(
                                    "LINESTRING "
                                    + mlstring_wkt[inds[i]: inds[i + 1] - 2]
                                )
                            else:
                                edge = wkt.loads(
                                    "LINESTRING " + mlstring_wkt[inds[i]: -1]
                                )
                            coordsout = lstring2xylist(edge)
                            newitem = copy.deepcopy(item)
                            newitem["geometry"]["coordinates"] = [coordsout]
                            roadjsons_combined["features"].append(newitem)

            print_el_counts(roadjsons_combined["features"], "road")
            with open("Roads.geojson", "w") as output_file:
                json.dump(roadjsons_combined, output_file, indent=2)

        # Initialize FootprintHandler:
        fp_handler = FootprintHandler()

        # Run FootprintHandler to get the boundary for the entered location:
        if isinstance(queryarea, tuple):
            bpoly, _ = fp_handler._FootprintHandler__bbox2poly(queryarea)
        else:
            bpoly, _, _ = fp_handler._FootprintHandler__fetch_roi(queryarea)

        # Define supported element types:
        eltypes = self.requested_elements.copy()
        roadjsons = {
            "primary_road": [],
            "secondary_road": [],
            "local_road": [],
        }
        if "roads" in eltypes:
            eltypes.remove("roads")
            eltypes += ["primary_road", "secondary_road", "local_road"]

        # Write the GeoJSON output for each element:
        for eltype in eltypes:
            if "_road" not in eltype:
                jsonout = write2geojson(bpoly, eltype)
                if jsonout != "":
                    self.output_files[eltype + "s"] = (
                        eltype.capitalize() + "s.geojson"
                    )
            else:
                print(
                    f"Fetching {eltype.replace('_',' ')}s, may take some "
                    + "time..."
                )
                roadjsons[eltype] = write2geojson(bpoly, eltype)
        if "roads" in self.requested_elements:
            combine_write_roadjsons(roadjsons, bpoly)

    def get_graph_network(self, inventory_files: List[str],
                          output_files: List[str] = ['edges.csv', 'nodes.csv'],
                          save_additional_attributes:
                              Literal['', 'residual_demand'] = ''
                          ) -> Tuple[Dict, Dict]:
        """
        Create a graph network from inventory data .

        This function processes inventory files containing road and structural
        data, constructs a graph network with nodes and edges, and optionally
        saves additional attributes such as residual demand. The node and edge
        features are saved to specified output files.

        Args__
            inventory_files (List[str]): A list of file paths to inventory data
                files used to create the graph network.
            output_files (List[str]): A list of file paths where the node and
                edge features will be saved. The first file in the list is used
                for edges and the second for nodes.
            save_additional_attributes (Literal['', 'residual_demand']):
                A flag indicating whether to save additional attributes.
                The only supported additional attribute is 'residual_demand'.

        Returns__
            Tuple[Dict, Dict]: A tuple containing two dictionaries:
                - The first dictionary contains the edge features.
                - The second dictionary contains the node features.

        Example__
            >>> inventory_files = ['Roads.geojson', 'Bridges.geojson']
            >>> output_files = ['edges.csv', 'nodes.csv']
            >>> save_additional_attributes = 'residual_demand'
            >>> edges, nodes = get_graph_network(inventory_files,
                                                 output_files,
                                                 save_additional_attributes)
            >>> print(edges)
            >>> print(nodes)
        """

        def create_circle_from_lat_lon(lat, lon, radius_ft, num_points=100):
            """
            Create a circle polygon from latitude and longitude.

            Args__
                lat (float): Latitude of the center.
                lon (float): Longitude of the center.
                radius_km (float): Radius of the circle in kilometers.
                num_points (int): Number of points to approximate the circle.

            Returns__
                Polygon: A Shapely polygon representing the circle.
            """
            # Earth's radius in kilometers
            earth_radius_ft = 20925721.784777

            # Convert radius from kilometers to radians
            radius_rad = radius_ft / earth_radius_ft

            # Convert latitude and longitude to radians
            lat_rad = np.radians(lat)
            lon_rad = np.radians(lon)

            # Generate points around the circle
            angle = np.linspace(0, 2 * np.pi, num_points)
            lat_points = lat_rad + radius_rad * np.cos(angle)
            lon_points = lon_rad + radius_rad * np.sin(angle)

            # Convert radians back to degrees
            lat_points_deg = np.degrees(lat_points)
            lon_points_deg = np.degrees(lon_points)

            # Create a polygon from the points
            points = list(zip(lon_points_deg, lat_points_deg))
            return Polygon(points)

        def parse_element_geometries_from_geojson(
                output_files: List[str], save_additional_attributes: str
        ) -> Tuple[List[Dict], Tuple[List[LineString], List[Dict]]]:
            """
            Parse geometries from a list of GeoJSON files.

            This function reads GeoJSON files specified in `output_files`. It
            extracts and parses `LineString` geometries from files that contain
            "road" in their name. For other files, it accumulates point
            features.

            Args__
                output_files (list of str): List of file paths to GeoJSON
                    files. Each file is read  to extract geometries. The
                    function distinguishes between files containing road data
                    and other transportation elements based on whether "road"
                    is in the file name.

            Returns__
                tuple: A tuple containing two elements:
                    - ptdata (list of dict): List of point features parsed from
                        GeoJSON files for bridges and tunnels
                    - road_data (tuple): A tuple containing:
                        - road_polys (list of LineString): List of `LineString`
                            objects created from the road geometries in GeoJSON
                            files that contain "road" in their name.
                        - road_features (list of dict): List of road features
                            as dictionaries from GeoJSON files that contain
                            "roads" in their name.

            Raises__
                FileNotFoundError: If any of the specified files in
                    `output_files` do not exist.
                json.JSONDecodeError: If a file cannot be parsed as valid JSON.
            """
            ptdata = []
            for file_name in output_files:
                with open(file_name, "r") as file:
                    if "road" in file_name.lower():
                        temp = json.load(file)
                        road_features = temp["features"]
                        road_polys = []
                        for ind, road in enumerate(road_features):
                            road_polys.append(
                                LineString(road["geometry"]["coordinates"][0])
                            )
                            road_features[ind]['properties']['asset_type'] = \
                                'road'
                            if save_additional_attributes:
                                mtfcc = road_features[ind]['properties']['MTFCC']
                                road_features[ind]['properties']['lanes'] = \
                                    self.attribute_maps['lanes'][mtfcc]
                                road_features[ind]['properties']['maxspeed'] = \
                                    self.attribute_maps['speed'][mtfcc]
                                road_features[ind]['properties']['capacity'] = \
                                    road_features[ind]['properties']['lanes']*1800
                    else:
                        temp = json.load(file)
                        if 'structure_number' in temp["features"][0]['properties']:
                            asset_type = 'bridge'
                        else:
                            asset_type = 'tunnel'
                        for index, _ in enumerate(temp["features"]):
                            temp["features"][index]['properties']['asset_type'] = \
                                asset_type

                        ptdata += temp["features"]

            return ptdata, (road_polys, road_features)

        def find_intersections(lines: List[LineString]) -> Set[Point]:
            """
            Find intersection points between pairs of LineString geometries.

            This function takes a list of `LineString` objects and identifies
            points where any two lines intersect. The intersections are
            returned as a set of `Point` objects.

            Args__
                lines (List[LineString]): A list of `LineString` objects. The
                function computes intersections between each pair of
                `LineString` objects in this list.

            Returns__
                Set[Point]: A set of `Point` objects representing the
                intersection points between the `LineString` objects. If
                multiple intersections occur at the same location, it will
                only be included once in the set.

            Example__
                >>> from shapely.geometry import LineString
                >>> line1 = LineString([(0, 0), (1, 1)])
                >>> line2 = LineString([(0, 1), (1, 0)])
                >>> find_intersections([line1, line2])
                {<shapely.geometry.point.Point object at 0x...>}

            Notes__
                - The function assumes that all input geometries are valid
                `LineString`
                    objects.
                - The resulting set may be empty if no intersections are found.

            Raises__
                TypeError: If any element in `lines` is not a `LineString`
                object.
            """
            intersections = set()
            for i, line1 in enumerate(lines):
                for line2 in lines[i + 1:]:
                    if line1.intersects(line2):
                        inter_points = line1.intersection(line2)
                        if inter_points.geom_type == "Point":
                            intersections.add(inter_points)
                        elif inter_points.geom_type == "MultiPoint":
                            intersections.update(inter_points.geoms)
            return intersections

        def cut_lines_at_intersections(lines: List[LineString],
                                       line_features: List[Dict],
                                       intersections: List[Point]
                                       ) -> List[LineString]:
            """
            Cut LineStrings at intersection points & return resulting segments.

            This function takes a list of `LineString` objects and a list of
            `Point` objects representing intersection points. For each
            `LineString`, it splits the line at each intersection point. The
            resulting segments are collected and returned.

            Args__
                lines (List[LineString]): A list of `LineString` objects to be
                    cut at the intersection points.
                line_features (List[Dict]): List of features for the
                    `LineString` objects in lines.
                intersections (List[Point]): A list of `Point` objects where
                    the lines are intersected and split.

            Returns__
                List[LineString]: A list of `LineString` objects resulting from cutting
                                    the original lines at the intersection points.

            Notes__
                - The `split` function from `shapely.ops` is used to perform the
                    cutting of lines at intersection points.
                - The function handles cases where splitting results in a
                    `GeometryCollection` by extracting only `LineString` geometries.

            Example__
                >>> from shapely.geometry import LineString, Point
                >>> from shapely.ops import split
                >>> lines = [
                ...     LineString([(0, 0), (2, 2)]),
                ...     LineString([(2, 0), (0, 2)])
                ... ]
                >>> intersections = [
                ...     Point(1, 1)
                ... ]
                >>> cut_lines_at_intersections(lines, intersections)
                [<shapely.geometry.linestring.LineString object at 0x...>,
                 <shapely.geometry.linestring.LineString object at 0x...>]
            """
            new_lines = []
            new_line_features = []

            for ind_line, line in enumerate(lines):
                segments = [line]  # Start with the original line
                for point in intersections:
                    new_segments = []
                    features = []
                    for segment in segments:
                        if segment.intersects(point):
                            split_result = split(segment, point)
                            if split_result.geom_type == "GeometryCollection":
                                new_segments.extend(
                                    geom
                                    for geom in split_result.geoms
                                    if geom.geom_type == "LineString"
                                )
                                features.extend([copy.deepcopy(line_features[ind_line])
                                                 for _ in range(len(split_result.geoms)
                                                                )])
                            elif split_result.geom_type == "LineString":
                                segments.append(split_result)
                                features.append(line_features[ind_line])
                        else:
                            new_segments.append(segment)
                            features.append(line_features[ind_line])
                    segments = new_segments

                # Add remaining segments that were not intersected by any points
                new_lines.extend(segments)
                new_line_features.extend(features)

            return (new_lines, new_line_features)

        def save_cut_lines_and_intersections(lines: List[LineString],
                                             points: List[Point]) -> None:
            """
            Save LineString and Point objects to separate GeoJSON files.

            This function converts lists of `LineString` and `Point` objects to GeoJSON
            format and saves them to separate files. The `LineString` objects are saved
            to "lines.geojson", and the `Point` objects are saved to "points.geojson".

            Args__
                lines (List[LineString]): A list of `LineString` objects to be
                                                saved in GeoJSON format.
                intersections (List[Point]): A list of `Point` objects to be saved in
                                                GeoJSON format.

            Returns__
                None: This function does not return any value. It writes GeoJSON data
                        to files.

            Notes__
                - The function uses the `mapping` function from `shapely.geometry` to
                    convert geometries to GeoJSON format.
                - Two separate GeoJSON files are created: one for lines and one for
                    points.
                - The output files are named "lines.geojson" and "points.geojson"
                    respectively.

            Example__
                >>> from shapely.geometry import LineString, Point
                >>> lines = [
                ...     LineString([(0, 0), (1, 1)]),
                ...     LineString([(1, 1), (2, 2)])
                ... ]
                >>> points = [
                ...     Point(0.5, 0.5),
                ...     Point(1.5, 1.5)
                ... ]
                >>> save_cut_lines_and_intersections(lines, points)
                # This will create "lines.geojson" and "points.geojson" files with the
                    corresponding GeoJSON data.
            """
            # Convert LineString objects to GeoJSON format
            features = []
            for line in lines:
                features.append(
                    {"type": "Feature", "geometry": mapping(
                        line), "properties": {}}
                )

            # Create a GeoJSON FeatureCollection
            geojson = {"type": "FeatureCollection", "features": features}

            # Save the GeoJSON to a file
            with open("lines.geojson", "w") as file:
                json.dump(geojson, file, indent=2)

            # Convert Point objects to GeoJSON format
            features = []
            for point in points:
                features.append(
                    {"type": "Feature", "geometry": mapping(
                        point), "properties": {}}
                )

            # Create a GeoJSON FeatureCollection
            geojson = {"type": "FeatureCollection", "features": features}

            # Save the GeoJSON to a file
            with open("points.geojson", "w") as file:
                json.dump(geojson, file, indent=2)

        def find_repeated_line_pairs(lines: List[LineString]) -> Set[Tuple]:
            """
            Find and groups indices of repeated LineString objects from a list.

            This function processes a list of `LineString` objects to identify and
            group all unique index pairs where LineString objects are repeated. The
            function converts each `LineString` to its Well-Known Text (WKT)
            representation to identify duplicates.

            Args__
                lines (List[LineString]): A list of `LineString` objects to be analyzed
                                            for duplicates.

            Returns__
                Set[Tuple]: A set of tuples, where each tuple contains indices for
                LineString objects that are repeated.

            Raises__
                TypeError: If any element in the `lines` list is not an instance of
                            `LineString`.

            Example__
                >>> from shapely.geometry import LineString
                >>> lines = [
                ...     LineString([(0, 0), (1, 1)]),
                ...     LineString([(0, 0), (1, 1)]),
                ...     LineString([(1, 1), (2, 2)]),
                ...     LineString([(0, 0), (1, 1)]),
                ...     LineString([(1, 1), (2, 2)])
                ... ]
                >>> find_repeated_line_pairs(lines)
                [{0, 1, 3}, {2, 4}]
            """
            line_indices = defaultdict(list)

            for index, line in enumerate(lines):
                if not isinstance(line, LineString):
                    raise TypeError(
                        "All elements in the input list must be LineString objects.")

                # Convert LineString to its WKT representation to use as a unique
                # identifier:
                line_wkt = line.wkt
                line_indices[line_wkt].append(index)

            repeated_pairs = set()
            for indices in line_indices.values():
                if len(indices) > 1:
                    # Create pairs of all indices for the repeated LineString
                    for i, _ in enumerate(indices):
                        for j in range(i + 1, len(indices)):
                            repeated_pairs.add((indices[i], indices[j]))

            repeated_pairs = list(repeated_pairs)
            ind_matched = []
            repeated_polys = []
            for index_p1, pair1 in enumerate(repeated_pairs):
                pt1 = set(pair1)
                for index_p2, pair2 in enumerate(repeated_pairs[index_p1+1:]):
                    if (index_p1 + index_p2 + 1) not in ind_matched:
                        pt2 = set(pair2)
                        if bool(pt1 & pt2):
                            pt1 |= pt2
                            ind_matched.append(index_p1 + index_p2 + 1)
                if pt1 not in repeated_polys and index_p1 not in ind_matched:
                    repeated_polys.append(pt1)

            return repeated_polys

        def match_edges_to_points(ptdata: List[Dict],
                                  road_polys: List[LineString],
                                  road_features: List[Dict]) -> List[List[int]]:
            """
            Match points to the closest road polylines based on name similarity.

            This function takes a list of points and a list of road polylines. For each
            point, it searches for intersecting road polylines within a specified
            radius and calculates the similarity between the point's associated
            facility name and the road's name. It returns a list of lists where each
            sublist contains indices of the road polylines that best match the point
            based on the similarity score.

            Args__
                ptdata (List[Dict[str, Any]]): A list of dictionaries where each
                                                dictionary represents a point with its
                                                geometry and properties. The 'geometry'
                                                key should contain 'coordinates', and
                                                the 'properties' key should contain
                                                'tunnel_name' or 'facility_carried'.
                road_polys (List[LineString]): A list of `LineString` objects
                                                representing road polylines.

            Returns__
                List[List[int]]: A list of lists where each sublist contains the
                                    indices of road polylines that have the highest
                                    textual similarity to the point's facility name.
                                    If no similarity is found, the sublist is empty.

            Notes__
                - The function uses a search radius of 100 feet to find intersecting
                    road polylines.
                - TF-IDF vectors are used to compute the textual similarity between the
                    facility names and road names.
                - Cosine similarity is used to determine the best matches.

            Example__
                >>> from shapely.geometry import Point, LineString
                >>> ptdata = [
                ...     {"geometry": {"coordinates": [1.0, 1.0]},
                         "properties": {"tunnel_name": "Tunnel A"}},
                ...     {"geometry": {"coordinates": [2.0, 2.0]},
                         "properties": {"facility_carried": "Road B"}}
                ... ]
                >>> road_polys = [
                ...     LineString([(0, 0), (2, 2)]),
                ...     LineString([(1, 1), (3, 3)])
                ... ]
                >>> match_edges_to_points(ptdata, road_polys)
                [[0], [1]]
            """
            edge_matches = []
            for point in ptdata:
                (lon, lat) = point["geometry"]["coordinates"]
                search_radius = create_circle_from_lat_lon(lat, lon, 100)
                # Check for intersections:
                intersecting_polylines = [
                    ind
                    for (ind, poly) in enumerate(road_polys)
                    if poly.intersects(search_radius)
                ]
                try:
                    facility = point["properties"]["tunnel_name"].lower()
                except Exception:
                    facility = point["properties"]["facility_carried"]
                similarities = []
                for polyline in intersecting_polylines:
                    roadway = road_features[polyline]["properties"]["NAME"]
                    if roadway:
                        # Create TF-IDF vectors:
                        vectorizer = TfidfVectorizer()
                        tfidf_matrix = vectorizer.fit_transform(
                            [facility, roadway.lower()]
                        )

                        # Calculate cosine similarity:
                        similarity = cosine_similarity(
                            tfidf_matrix[0:1], tfidf_matrix[1:2]
                        )
                    else:
                        similarity = -1
                    similarities.append(similarity)
                max_similarity = max(similarities)
                if max_similarity == 0:
                    max_similarity = -1
                indices_of_max = [
                    intersecting_polylines[index]
                    for index, value in enumerate(similarities)
                    if value == max_similarity
                ]
                edge_matches.append(indices_of_max)

            return edge_matches

        def merge_brtn_features(ptdata: List[Dict], road_features_brtn: List[Dict],
                                edge_matches: List[List],
                                save_additional_attributes: str) -> List[Dict]:
            """
            Merge bridge or tunnel features into road features based on edge matches.

            This function updates road features with additional attributes derived from
            bridge or tunnel point data. It uses edge matches to determine how to
            distribute lane and capacity information among road features.

            Args__
                ptdata (List[Dict]): A list of dictionaries where each dictionary
                    contains properties of bridge or tunnel features. Each dictionary
                    should have 'asset_type', 'traffic_lanes_on', 'structure_number',
                    'total_number_of_lanes', and 'tunnel_number' as keys.
                road_features_brtn (List[Dict]): A list of dictionaries representing
                    road features that will be updated. Each dictionary should
                    have a 'properties' key where attributes are stored.
                edge_matches (List[List[int]]): A list of lists, where each sublist
                    contains indices that correspond to `road_features_brtn` and
                    represent which features should be updated together.
                save_additional_attributes (str): A flag indicating whether to save
                    additional attributes like 'lanes' and 'capacity'. If non-empty,
                    additional attributes will be saved.

            Returns__
                List[Dict]: The updated list of road features with merged attributes.

            Example__
                >>> ptdata = [
                ...     {'properties': {'asset_type': 'bridge', 'traffic_lanes_on': 4,
                                        'structure_number': '12345'}},
                ...     {'properties': {'asset_type': 'tunnel',
                                        'total_number_of_lanes': 6,
                                        'tunnel_number': '67890'}}
                ... ]
                >>> road_features_brtn = [{} for _ in range(4)]  # List of empty
                    dictionaries for demonstration
                >>> edge_matches = [[0, 1], [2, 3]]
                >>> save_additional_attributes = 'yes'
                >>> updated_features = merge_brtn_features(ptdata, road_features_brtn,
                                                           edge_matches,
                                                           save_additional_attributes)
                >>> print(updated_features)
            """
            poly_index = 0
            for item_index, edge_indices in enumerate(edge_matches):
                nitems = len(edge_indices)
                features = ptdata[item_index]['properties']
                asset_type = features['asset_type']
                if asset_type == 'bridge':
                    total_nlanes = features['traffic_lanes_on']
                    struct_no = features['structure_number']
                else:
                    total_nlanes = features['total_number_of_lanes']
                    struct_no = features['tunnel_number']

                lanes_per_item = round(int(total_nlanes)/nitems)

                for index in range(poly_index, poly_index + nitems):
                    road_features_brtn[index]['properties']['asset_type'] = asset_type
                    road_features_brtn[index]['properties']['OID'] = struct_no
                    if save_additional_attributes:
                        road_features_brtn[index]['properties']['lanes'] =  \
                            lanes_per_item
                        road_features_brtn[index]['properties']['capacity'] = \
                            lanes_per_item*1800

                poly_index += nitems
            return road_features_brtn

        def get_nodes_edges(lines: List[LineString],
                            length_unit: Literal['ft', 'm'] = 'ft'
                            ) -> Tuple[Dict, Dict]:
            """
            Extract nodes and edges from a list of LineString objects.

            This function processes a list of `LineString` geometries to generate
            nodes and edges. Nodes are defined by their unique coordinates, and
            edges are represented by their start and end nodes, length, and geometry.

            Args__
                lines (List[LineString]): A list of `LineString` objects representing
                    road segments.
                length_unit (Literal['ft', 'm']): The unit of length for the edge
                    distances. Defaults to 'ft'. Acceptable values are 'ft' for feet
                    and 'm' for meters.

            Returns__
                Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, Any]]]:
                    - A dictionary where keys are node IDs and values are dictionaries
                        with node attributes:
                        - 'lon': Longitude of the node.
                        - 'lat': Latitude of the node.
                        - 'geometry': WKT representation of the node.
                    - A dictionary where keys are edge IDs and values are dictionaries
                        with edge attributes:
                        - 'start_nid': ID of the start node.
                        - 'end_nid': ID of the end node.
                        - 'length': Length of the edge in the specified unit.
                        - 'geometry': WKT representation of the edge.

            Raises__
                TypeError: If any element in the `lines` list is not an instance of
                    `LineString`.

            Example__
                >>> from shapely.geometry import LineString
                >>> lines = [
                ...     LineString([(0, 0), (1, 1)]),
                ...     LineString([(1, 1), (2, 2)])
                ... ]
                >>> nodes, edges = get_nodes_edges(lines, length_unit='m')
                >>> print(nodes)
                >>> print(edges)
            """
            node_list = []
            edges = {}
            node_counter = 0
            for line_counter, line in enumerate(lines):
                # Ensure the object is a LineString
                if not isinstance(line, LineString):
                    raise TypeError(
                        "All elements in the list must be LineString objects.")

                # Extract coordinates
                coords = list(line.coords)
                ncoord_pairs = len(coords)
                if ncoord_pairs > 0:
                    start_node_coord = coords[0]
                    end_node_coord = coords[-1]

                    if start_node_coord not in node_list:
                        node_list.append(start_node_coord)
                        start_nid = node_counter
                        node_counter += 1
                    else:
                        start_nid = node_list.index(start_node_coord)

                    if end_node_coord not in node_list:
                        node_list.append(end_node_coord)
                        end_nid = node_counter
                        node_counter += 1
                    else:
                        end_nid = node_list.index(end_node_coord)

                    length = 0
                    (lon, lat) = line.coords.xy
                    for pair_no in range(ncoord_pairs - 1):
                        length += haversine_dist([lat[pair_no], lon[pair_no]],
                                                 [lat[pair_no+1],
                                                  lon[pair_no+1]])
                    if length_unit == 'm':
                        length = 0.3048*length

                    edges[line_counter] = {'start_nid': start_nid,
                                           'end_nid': end_nid,
                                           'length': length,
                                           'geometry': line.wkt}

                nodes = {}
                for node_id, node_coords in enumerate(node_list):
                    nodes[node_id] = {'lon': node_coords[0],
                                      'lat': node_coords[1],
                                      'geometry': f'POINT ({node_coords[0]:.7f} ' +
                                                  f'{node_coords[1]:.7f})'}

            return (nodes, edges)

        def get_node_edge_features(updated_road_polys: List[LineString],
                                   updated_road_features: List[Dict],
                                   output_files: List[str]
                                   ) -> Tuple[Dict, Dict]:
            """
            Extract and write node and edge features from updated road polygons.

            This function processes road polygon data to generate nodes and edges,
            then writes the extracted features to specified output files.

            Args__
                updated_road_polys (List[LineString]): A list of LineString objects
                    representing updated road polygons.
                updated_road_features (List[Dict]): A list of dictionaries containing
                    feature properties for each road segment.
                output_files (List[str]): A list of two file paths where the first path
                    is for edge data and the second for node data.

            Returns__
                Tuple[Dict, Dict]: A tuple containing two dictionaries:
                    - The first dictionary contains edge data.
                    - The second dictionary contains node data.

            Raises__
                TypeError: If any input is not of the expected type.

            Example__
                >>> from shapely.geometry import LineString
                >>> road_polys = [LineString([(0, 0), (1, 1)]),
                                  LineString([(1, 1), (2, 2)])]
                >>> road_features = [{'properties': {'OID': 1, 'asset_type': 'road',
                                                     'lanes': 2, 'capacity': 2000,
                                                     'maxspeed': 30}}]
                >>> output_files = ['edges.csv', 'nodes.csv']
                >>> get_node_edge_features(road_polys, road_features, output_files)
            """
            self.graph_network = {'edges': dict,
                                  'nodes': dict,
                                  'output_files': list}

            (nodes, edges) = get_nodes_edges(
                updated_road_polys, length_unit='m')

            with open(output_files[1], 'w') as nodes_file:
                nodes_file.write('node_id, lon, lat, geometry\n')
                for key in nodes:
                    nodes_file.write(f'{key}, {nodes[key]["lon"]}, '
                                     f'{nodes[key]["lat"]}, '
                                     f'{nodes[key]["geometry"]}\n')

            with open(output_files[0], 'w') as edge_file:
                edge_file.write('uniqueid, start_nid, end_nid, osmid, length, type, '
                                'lanes, maxspeed, capacity, fft, geometry\n')

                for key in edges:
                    features = updated_road_features[key]['properties']
                    edges[key]['osmid'] = features['OID']
                    edges[key]['type'] = features['asset_type']
                    edges[key]['lanes'] = features['lanes']
                    edges[key]['capacity'] = features['capacity']
                    maxspeed = features['maxspeed']
                    edges[key]['maxspeed'] = maxspeed
                    free_flow_time = edges[key]['length'] / \
                        (maxspeed*1609.34/3600)
                    edges[key]['fft'] = free_flow_time
                    edge_file.write(f'{key}, {edges[key]["start_nid"]}, '
                                    f'{edges[key]["end_nid"]}, {edges[key]["osmid"]}, '
                                    f'{edges[key]["length"]}, {edges[key]["type"]}, '
                                    f'{edges[key]["lanes"]}, {maxspeed}, '
                                    f'{edges[key]["capacity"]}, {free_flow_time}, '
                                    f'{edges[key]["geometry"]}\n')

            return (edges, nodes)

        print('Getting graph network for elements in '
              f'{", ".join(inventory_files)}...')

        # Read inventory data:
        ptdata, (road_polys, road_features) = \
            parse_element_geometries_from_geojson(
            inventory_files,
            save_additional_attributes=save_additional_attributes)

        # Find edges that match bridges and tunnels:
        edge_matches = match_edges_to_points(ptdata, road_polys, road_features)

        # Get the indices for bridges and tunnels:
        brtn_poly_idx = [item for sublist in edge_matches for item in sublist]

        # Detect repeated edges and save their indices:
        repeated_edges = find_repeated_line_pairs(road_polys)

        edges_remove = []
        for edge_set in repeated_edges:
            bridge_poly = set(brtn_poly_idx)
            if edge_set & bridge_poly:
                remove = list(edge_set - bridge_poly)
            else:
                temp = list(edge_set)
                remove = temp[1:].copy()
            edges_remove.extend(remove)

        # Save polygons that are not bridge or tunnel edges or marked for
        # removal in road polygons:
        road_polys_local = [poly for (ind, poly) in enumerate(road_polys) if
                            ind not in brtn_poly_idx + edges_remove]
        road_features_local = [feature for (ind, feature) in
                               enumerate(road_features)
                               if ind not in brtn_poly_idx + edges_remove]

        # Save polygons that are not bridge or tunnel edges:
        road_polys_brtn = [poly for (ind, poly) in enumerate(road_polys)
                           if ind in brtn_poly_idx]
        road_features_brtn = [feature for (ind, feature)
                              in enumerate(road_features)
                              if ind in brtn_poly_idx]
        road_features_brtn = merge_brtn_features(ptdata,
                                                 road_features_brtn,
                                                 edge_matches,
                                                 save_additional_attributes)

        # Compute the intersections of road polygons:
        intersections = find_intersections(road_polys_local)

        # Cut road polygons at intersection points:
        cut_lines, cut_features = \
            cut_lines_at_intersections(road_polys_local,
                                       road_features_local,
                                       intersections)
        # Come back and update cut_lines_at_intersections to not intersect
        # lines within a certain diameter of a bridge point

        # Combine all polygons and their features:
        updated_road_polys = cut_lines + road_polys_brtn
        updated_road_features = cut_features + road_features_brtn

        # Save created polygons (for debugging only)
        # save_cut_lines_and_intersections(updated_road_polys, intersections)

        # Get nodes and edges of the final set of road polygons:
        (edges, nodes) = get_node_edge_features(updated_road_polys,
                                                updated_road_features,
                                                output_files)
        self.graph_network['edges'] = edges
        self.graph_network['nodes'] = nodes
        self.graph_network['output_files'] = output_files

        print('Edges and nodes of the graph network are saved in '
              f'{", ".join(output_files)}')
