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
# Frank McKenna
# Jinyan Zhao

#
# Last updated:
# 01-29-2024


import geopandas as gpd
import pandas as pd
import momepy
import os
import shapely
import json
import warnings
import numpy as np
from datetime import datetime
from importlib.metadata import version
from brails.workflow.TransportationElementHandler import TransportationElementHandler

# The map defines the default values according to MTFCC code
# https://www2.census.gov/geo/pdfs/maps-data/data/tiger/tgrshp2009/TGRSHP09AF.pdf
# May need better models
ROADTYPE_MAP = {'S1100':"primary", "S1200":"secondary", "S1400":"residential",
                "S1500":"unclassified", "S1630":"unclassified", "S1640":"unclassified",
                "S1710":"unclassified", "S1720":"unclassified", "S1730":"unclassified",
                "S1740":"unclassified", "S1750":"unclassified", "S1780":"unclassified",
                "S1810":"unclassified", "S1820":"unclassified", "S1830":"unclassified"}
ROADLANES_MAP = {'S1100':4, "S1200":2, "S1400":1, "S1500":1, "S1630":1, "S1640":1, 
                 "S1710":1, "S1720":1, "S1730":1, "S1740":1, "S1750":1, "S1780":1,
                 "S1810":1, "S1820":1, "S1830":1}
        # speedMap = {'S1100':70,"S1200":55,"S1400":25,"S1500":25,"S1630":25,"S1640":25,"S1710":25,"S1720":25,"S1730":25}
ROADCAPACITY_MAP = {'S1100':70, "S1200":55, "S1400":25, "S1500":25, "S1630":25, "S1640":25,
                    "S1710":25, "S1720":25, "S1730":25, "S1740":10, "S1750":10, "S1780":10,
                    "S1810":10, "S1820":10, "S1830":10}

class TranspInventoryGenerator:

    def __init__(self, location='Berkeley, CA'):                
        self.enabledElements = ['roads','bridges','tunnels','railroads']
        self.location = location
        self.workDir = 'tmp'
        self.modelDir = 'tmp/models'
        self.inventory_files = ''
    
    def enabled_elements(self):
        print('Here is the list of elements currently enabled in TranspInventoryGenerator:\n')
        for element in self.enabledElements:
            print(f'       {element}')

    def generate(self):
        tphandler = TransportationElementHandler()
        tphandler.fetch_transportation_elements(self.location) 
        
        self.inventory_files = tphandler.output_files
        
        outfiles = ", ".join(value for value in tphandler.output_files.values())
        print(f'\nTransportation inventory data available in {outfiles} in {os.getcwd()}')
    
    def combineAndFormat_HWY(self, minimumHAZUS=True, connectivity=False, maxRoadLength=100, lengthUnit='m'):
        outfiles = ", ".join(value for value in self.inventory_files.values())
        print(f"\nReformatting and combining the data in {outfiles}")

        # convert maxRoadLength to a unit of m, which is used in the cartesian 
        # coordinate epsg:6500
        maxRoadLength = convertUnits(maxRoadLength, lengthUnit, 'm')
        # Format bridges
        bridgesFile = self.inventory_files.get("bridges", None)
        if bridgesFile is not None:
            bridges_gdf = gpd.read_file(bridgesFile)
            bnodeDF, bridgesDict = formatBridges(minimumHAZUS, connectivity,\
                                                 bridges_gdf, lengthUnit)
        else:
            bnodeDF = gpd.GeoDataFrame(columns = ["nodeID", "geometry"], crs = "epsg:4326")
            bridgesDict = {'type':'FeatureCollection', 
                           'generated':str(datetime.now()),
                           'brails_version': version('BRAILS'),
                           "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                           'units': {"length": lengthUnit},
                           'features':[]}

        # Format roadways
        roadsFile = self.inventory_files.get("roads", None)
        if roadsFile is not None:
            roads_gdf = gpd.read_file(roadsFile).explode(index_parts = False)
            rnodeDF, roadsDict = formatRoads(minimumHAZUS, connectivity,\
                                             maxRoadLength, roads_gdf)
            formattedRoadsFile = "ProcessedRoadNetworkRoads.geojson"
            with open(formattedRoadsFile, 'w') as f:
                json.dump(roadsDict, f, indent = 2)
            if connectivity:
                rnodeDF.to_file('ProcessedRoadNetworkNodes.geojson',driver='GeoJSON')
        else:
            rnodeDF = gpd.GeoDataFrame(columns = ["nodeID", "geometry"], crs = "epsg:4326")
            roadsDict = {'type':'FeatureCollection', 
                        'generated':str(datetime.now()),
                        'brails_version': version('BRAILS'),
                        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                        'units': {"length": lengthUnit},
                        'features':[]}
        
        # Format tunnels
        tunnelsFile = self.inventory_files.get("tunnels", None)
        if tunnelsFile is not None:
            tunnels_gdf = gpd.read_file(tunnelsFile).explode(index_parts = False)
            tnodeDF, tunnelsDict = formatTunnels(minimumHAZUS, connectivity, tunnels_gdf)
        else:
            tnodeDF = gpd.GeoDataFrame(columns = ["nodeID", "geometry"], crs = "epsg:4326")
            tunnelsDict = {'type':'FeatureCollection',
                           'generated':str(datetime.now()),
                           'brails_version': version('BRAILS'),
                           "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                           'units': {"length": lengthUnit},
                           'features':[]}

        # Combine nodes and update dicts
        combinedGeoJSON = combineDict(bnodeDF, bridgesDict, rnodeDF, roadsDict,
                                      tnodeDF, tunnelsDict, connectivity,
                                      lengthUnit)
        # Dump to json file
        with open("hwy_inventory.geojson", "w") as f:
            json.dump(combinedGeoJSON, f, indent = 2)
            print('Combined transportation inventory saved in hwy_inventory.geojson.'
                  f' This file is suitable for R2D use and is available in {os.getcwd()}')
        return

# Convert common length units
def convertUnits(value, unit_in, unit_out):
    aval_types = ['m', 'mm', 'cm', 'km', 'inch', 'ft', 'mile']
    m = 1.
    mm = 0.001 * m
    cm = 0.01 * m
    km = 1000 * m
    inch = 0.0254 * m
    ft = 12. * inch
    mile = 5280. * ft
    scale_map = {'m':m, 'mm':mm, 'cm':cm, 'km':km, 'inch':inch, 'ft':ft,\
                  'mile':mile}
    if (unit_in not in aval_types) or (unit_out not in aval_types):
        print(f"The unit {unit_in} or {unit_out} are used in BRAILS but not supported")
        return
    value = value*scale_map[unit_in]/scale_map[unit_out]
    return value

# Break down long roads according to delta
def breakDownLongEdges(edges, delta, nodes = None, tolerance = 10e-3):
    dropedEdges = []
    newEdges = []
    crs = edges.crs
    edges["SegID"] = 0
    edges_dict = edges.reset_index().to_crs("epsg:6500")
    edges_dict = edges_dict.to_dict(orient='records')
    if nodes is not None:
        newNodes = []
        nodeCount = nodes.loc[:,"nodeID"].max()
    for row_ind in range(len(edges_dict)):
        LS = edges_dict[row_ind]["geometry"]
        num_seg = int(np.ceil(LS.length/delta))
        if num_seg == 1:
            continue
        distances = np.linspace(0, LS.length, num_seg+1)
        points = shapely.MultiPoint([LS.interpolate(distance) for distance in \
                                     distances[:-1]] + [LS.coords[-1]])
        LS = shapely.ops.snap(LS, points, tolerance)
        with warnings.catch_warnings(): #Suppress the warning of points not on 
            # LS. Shaply will first project the point to the line and then split
            warnings.simplefilter("ignore")
            splittedLS = shapely.ops.split(LS,points).geoms
        currentEdge = edges_dict[row_ind].copy()
        for sLS_ind, sLS in enumerate(splittedLS):
            newGeom = sLS
            newEdge = currentEdge.copy()
            newEdge.update({"geometry":newGeom,\
                            "SegID":sLS_ind})
            if nodes is not None:
                if sLS_ind != len(splittedLS)-1:
                    newNode = {"geometry":shapely.Point(sLS.coords[-1]),\
                               "nodeID":nodeCount+1}
                    newNodes.append(newNode)
                    nodeCount += 1
                if sLS_ind == 0:
                    newEdge.update({'node_end':nodeCount})
                elif sLS_ind==len(splittedLS)-1:
                    newEdge.update({'node_start':nodeCount})
                else:
                    newEdge.update({'node_start':nodeCount-1,\
                                    'node_end':nodeCount})
            newEdges.append(newEdge)            
        dropedEdges.append(edges_dict[row_ind]["index"])
    edges = edges.drop(dropedEdges)
    if len(newEdges)>0:
        newEdges = gpd.GeoDataFrame(newEdges, crs="epsg:6500").to_crs(crs)
        edges = pd.concat([edges, newEdges], ignore_index=True)
    edges = edges.reset_index(drop=True).drop(columns = 'index', axis = 1)
    if nodes is not None:
        edges = edges.sort_values(by=['OID','ExplodeID','SegID']).reset_index(drop=True)
    else:
        edges = edges.sort_values(by=['OID','SegID']).reset_index(drop=True)
    if nodes is not None and len(newNodes)>0:
        newNodes = gpd.GeoDataFrame(newNodes, crs="epsg:6500").to_crs(crs)
        nodes = pd.concat([nodes, newNodes], ignore_index=True)
    return edges, nodes        
    
def formatBridges(minimumHAZUS, connectivity, bridges_gdf, lengthUnit):
    ## Format bridge nodes
    if connectivity:
        bnodeDF = bridges_gdf["geometry"].reset_index().rename(columns = {"index":"nodeID"})
        bridges_gdf = bridges_gdf.reset_index().rename(columns={"index":"Location"})
    else:
        bnodeDF = gpd.GeoDataFrame(columns = ["nodeID", "geometry"], crs=bridges_gdf.crs)
    ## Format bridge items
    bridges_gdf["BridgeClass"] = bridges_gdf['structure_kind'].apply(int)*100+\
                        bridges_gdf['structure_kind'].apply(int)
    bridges_gdf = bridges_gdf.rename(columns = {'structure_number':"StructureNumber",\
        "year_built":"YearBuilt", "main_unit_spans":"NumOfSpans",\
        'max_span_len_mt':"MaxSpanLength","state_code":"StateCode",\
        'degrees_skew':"Skew","deck_width_mt":"DeckWidth"})
    # bridges_gdf["StructureNumber"] = bridges_gdf["StructureNumber"].\apply(lambda x: x.replace(" ",""))
    bridges_gdf["DeckWidth"] = bridges_gdf["DeckWidth"].apply(lambda x :\
                                convertUnits(x, "m", lengthUnit))
    bridges_gdf["MaxSpanLength"] = bridges_gdf["MaxSpanLength"].apply(lambda x :\
                                convertUnits(x, "m", lengthUnit))
    if minimumHAZUS:
        columnsNeededByHAZUS = ["StructureNumber", "geometry", "BridgeClass", "YearBuilt",\
                                "NumOfSpans", "MaxSpanLength", "StateCode", "Skew",\
                                "DeckWidth"]
        if connectivity:
            columnsNeededByHAZUS.append('Location')
        bridges_gdf = bridges_gdf.loc[:,columnsNeededByHAZUS]
    ## Format the hwy_bridges geojson
    bridges_gdf["type"] = "Bridge"
    bridges_gdf["assetSubtype"] = "HwyBridge"
    bridgeDict = json.loads(bridges_gdf.to_json())
    return bnodeDF, bridgeDict

# Remove the nodes with 2 neibours 
# https://stackoverflow.com/questions/56380053/combine-edges-when-node-degree-is-n-in-networkx
# Needs parallel
def remove2neighborEdges(nodes, edges, graph):
    import datetime
    print(f"Initialized roadway data correction at {datetime.datetime.now()}")
    ### Some edges has start_node as the last point in the geometry and end_node
    #  as the first point, check and reorder
    for ind in edges.index:
        start = nodes.loc[edges.loc[ind, "node_start"],"geometry"]
        end = nodes.loc[edges.loc[ind, "node_end"],"geometry"]
        first = shapely.geometry.Point(edges.loc[ind,"geometry"].coords[0])
        last = shapely.geometry.Point(edges.loc[ind,"geometry"].coords[-1])
        #check if first and last are the same
        if (start == first and end == last):
            continue
        elif (start == last and end == first):
            newStartID = edges.loc[ind, "node_end"]
            newEndID = edges.loc[ind, "node_start"]
            edges.loc[ind,"node_start"] = newStartID
            edges.loc[ind,"node_end"] = newEndID
        else:
            print(ind, "th row of edges has wrong start/first, end/last pairs, likely a bug of momepy.gdf_to_nx function")
    # Find nodes with only two neighbor
    nodesID_to_remove = [i for i, n in enumerate(graph.nodes) if len(list(graph.neighbors(n))) == 2]
    nodes_to_remove = [n for i, n in enumerate(graph.nodes) if len(list(graph.neighbors(n))) == 2]
    # For each of those nodes
    removedID_list = [] # nodes with two neighbors. Removed from graph
    skippedID_list = [] # nodes involved in loops. Skipped removing.
    error_list = [] #nodes run into error. Left the node in the graph as is. 
    # import time
    # timeList = np.zeros([5,1])
    # start = time.process_time_ns()
    edges_end_index = edges.reset_index().set_index("node_end")
    edges_start_index = edges.reset_index().set_index("node_start")
    # timeList[4] += time.process_time_ns() - start
    # for i in range(2000):
    for i in range(len(nodesID_to_remove)):
        # start = time.process_time_ns()
        nodeid = nodesID_to_remove[i]
        node = nodes_to_remove[i]
        #Option 1, 6.9+e9
        edge1 = edges[edges["node_end"] == nodeid]
        edge2 = edges[edges["node_start"] == nodeid]
        # timeList[0] += time.process_time_ns() - start
        # start = time.process_time_ns()

        if (edge1.shape[0]==1 and edge2.shape[0]==1 and 
            edge1["node_start"].values[0]!= edge2["node_end"].values[0]):
            pass # Do things after continue
        elif(edge1.shape[0]==0 and edge2.shape[0]==2):
            ns = edges.loc[edge2.index[0],"node_start"]
            ne = edges.loc[edge2.index[0],"node_end"]
            edges.loc[edge2.index[0],"node_start"] = ne
            edges.loc[edge2.index[0],"node_end"] = ns
            # edges.loc[edge2.index[0],"geometry"] = shapely.LineString(list(edges.loc[edge2.index[0],"geometry"].coords)[::-1])
            edges.loc[edge2.index[0],"geometry"] = edges.loc[edge2.index[0],"geometry"].reverse()
            edge1 = edges[edges["node_end"] == nodeid]
            edge2 = edges[edges["node_start"] == nodeid]
        elif(edge1.shape[0]==2 and edge2.shape[0]==0):
            ns = edges.loc[edge1.index[1],"node_start"]
            ne = edges.loc[edge1.index[1],"node_end"]
            edges.loc[edge1.index[1],"node_start"] = ne
            edges.loc[edge1.index[1],"node_end"] = ns
            # edges.loc[edge1.index[1],"geometry"] = shapely.LineString(list(edges.loc[edge1.index[1],"geometry"].coords)[::-1])
            edges.loc[edge1.index[1],"geometry"] = edges.loc[edge1.index[1],"geometry"].reverse()
            edge1 = edges[edges["node_end"] == nodeid]
            edge2 = edges[edges["node_start"] == nodeid]
        else:
            skippedID_list.append(nodeid)
            continue

        # timeList[1] += time.process_time_ns() - start
        # start = time.process_time_ns()
        
        try:
            removedID_list.append(nodeid)
            newLineCoords = list(edge1["geometry"].values[0].coords)+list(edge2["geometry"].values[0].coords[1:])
            # newLineCoords.append(edge2["geometry"].values[0].coords[1:])
            edges.loc[edge1.index, "geometry"] = shapely.LineString(newLineCoords)
            edges.loc[edge1.index, "node_end"] = edge2["node_end"].values[0]
            edges.loc[edge1.index, "ExplodeID"] = min(edge1["ExplodeID"].values[0],\
                                                      edge2["ExplodeID"].values[0])
            edges.drop(edge2.index, axis = 0, inplace=True)

            # timeList[2] += time.process_time_ns() - start
            # start = time.process_time_ns()

            newEdge = list(graph.neighbors(node))
            graph.add_edge(newEdge[0], newEdge[1])
            # And delete the node
            graph.remove_node(node)

            # timeList[3] += time.process_time_ns() - start
            # start = time.process_time_ns()
        except:
            error_list.append(nodeid)
    
    remainingNodesOldID = list(set(edges["node_start"].values.tolist() + edges["node_end"].values.tolist()))
    nodes = nodes.loc[remainingNodesOldID,:].sort_index()
    nodes = nodes.reset_index(drop=True).reset_index().rename(columns={"index":"nodeID", "nodeID":"oldNodeID"})
    edges = edges.merge(nodes[["nodeID", "oldNodeID"]], left_on="node_start",
            right_on = "oldNodeID", how="left").drop(["node_start", "oldNodeID"], axis=1).rename(columns = {"nodeID":"node_start"})
    edges = edges.merge(nodes[["nodeID", "oldNodeID"]], left_on="node_end",
            right_on = "oldNodeID", how="left").drop(["node_end", "oldNodeID"], axis=1).rename(columns = {"nodeID":"node_end"})
    nodes = nodes.drop(columns = ["oldNodeID"])
    # Reset explode ID
    edges = edges.sort_values(by=["OID", 'ExplodeID'])
    edges['ExplodeID'] = edges.groupby('OID').cumcount()
    # print(f"timeList in remove neighbor {timeList}")
    print(f"Completed roadway data correction at {datetime.datetime.now()}")
    return nodes, edges

def explodeLineString(roads_gdf):
    expandedRoads = []
    crs = roads_gdf.crs
    roads_gdf_dict = roads_gdf.to_dict(orient = 'records')
    for rd in roads_gdf_dict:
        LS_list = []
        multiseg_line = rd["geometry"]
        for pt1, pt2 in zip(multiseg_line.coords, multiseg_line.coords[1:]):
            LS_list.append(shapely.LineString([pt1, pt2]))
        for ind_i in range(len(LS_list)):
            newRoad = rd.copy()
            newRoad.update({"geometry":LS_list[ind_i],\
                                      "ExplodeID":int(ind_i)})
            expandedRoads.append(newRoad)
    expandedRoads = gpd.GeoDataFrame(expandedRoads, crs=crs)
    return expandedRoads
    
def formatRoads(minimumHAZUS, connectivity, maxRoadLength, roads_gdf):
    if roads_gdf.shape[0] == 0:
        rnodeDF = gpd.GeoDataFrame(columns = ["nodeID", "geometry"], crs=roads_gdf.crs)
        edgesDict = json.loads(roads_gdf.to_json())
        return rnodeDF, edgesDict
    roads_gdf = roads_gdf.sort_values(by="OID").reset_index(drop=True)
    if connectivity:
        ## Convert to graph to find the intersection nodes
        expandedRoads = explodeLineString(roads_gdf)
        graph = momepy.gdf_to_nx(expandedRoads.to_crs("epsg:6500"), approach='primal')
        with warnings.catch_warnings(): #Suppress the warning of disconnected components in the graph
            warnings.simplefilter("ignore")
            nodes, edges, sw = momepy.nx_to_gdf(graph, points=True, lines=True,
                                                spatial_weights=True)
        # The CRS of SimCenter is CRS:84 (equivalent to EPSG:4326)
        # The CRS of US Census is NAD83, which is https://epsg.io/4269
        nodes, edges = remove2neighborEdges(nodes, edges, graph)
        ### Some edges are duplicated, keep only the first one
        # edges = edges[edges.duplicated(['node_start', 'node_end'], keep="first")==False]
        # edges = edges.reset_index(drop=True)
        ## Break long roads into multiple roads
        if maxRoadLength is not None:
            segmentedRoads, nodes = breakDownLongEdges(edges, maxRoadLength, nodes)
        else:
            segmentedRoads = edges
        nodes = nodes.to_crs("epsg:4326")
        segmentedRoads = segmentedRoads.to_crs("epsg:4326")
        segmentedRoads = segmentedRoads.rename(columns={'node_start': 'StartNode', 'node_end': 'EndNode'})
        segmentedRoads = segmentedRoads.drop(columns="mm_len", axis=1)
        rnodeDF = nodes
    else:
        rnodeDF = gpd.GeoDataFrame(columns = ["nodeID", "geometry"], crs=roads_gdf.crs)
        edges = roads_gdf
        ## Break long roads into multiple roads
        if maxRoadLength is not None:
            segmentedRoads, _ = breakDownLongEdges(edges, maxRoadLength)
        else:
            segmentedRoads = edges
    ## Format roadways
    ### Format and clean up roadway edges
    road_type = []
    lanes = []
    capacity = []
    edge_id = []
    for row_ind in segmentedRoads.index:
        mtfcc = segmentedRoads.loc[row_ind,"MTFCC"]
        road_type.append(ROADTYPE_MAP[mtfcc])
        lanes.append(ROADLANES_MAP[mtfcc])
        capacity.append(ROADCAPACITY_MAP[mtfcc])
        edge_id.append(segmentedRoads.loc[row_ind,"OID"])
    segmentedRoads["TigerOID"] = edge_id
    segmentedRoads["RoadType"] = road_type
    segmentedRoads["NumOfLanes"] = lanes
    segmentedRoads["MaxMPH"] = capacity
    if minimumHAZUS:
        columnsNeededByHAZUS=['TigerOID','RoadType','NumOfLanes','MaxMPH', 'geometry']
        if connectivity:
            columnsNeededByHAZUS+=["StartNode", "EndNode","ExplodeID"]
        if maxRoadLength is not None:
            columnsNeededByHAZUS+=["SegID"]
        
        segmentedRoads = segmentedRoads[columnsNeededByHAZUS]

    segmentedRoads['type'] = "Roadway"
    segmentedRoads['assetSubtype'] = "Roadway"
    edgesDict = json.loads(segmentedRoads.to_json())
    return rnodeDF, edgesDict

def formatTunnels(minimumHAZUS, connectivity, tunnels_gdf):
    ## Format tunnel nodes
    if connectivity:
        tnodeDF = tunnels_gdf["geometry"].reset_index().rename(columns = {"index":"nodeID"})
        tunnels_gdf = tunnels_gdf.reset_index().rename(columns={"index":"Location"})
    else:
        tnodeDF = gpd.GeoDataFrame(columns = ["nodeID", "geometry"], crs=tunnels_gdf.crs)
    ## Format tunnel items
    if "cons_type" not in tunnels_gdf.columns:
        print("Construction type data could not be obtained for tunnels. Construction type for tunnels was set as unclassified")
        tunnels_gdf["cons_type"] = "unclassified"
    tunnels_gdf = tunnels_gdf.rename(columns = {"tunnel_number":"TunnelNumber", 
                                                "cons_type":"ConstructType"})
    if minimumHAZUS:
        columnsNeededByHAZUS = ["TunnelNumber", "ConstructType", "geometry"]
        if connectivity:
            columnsNeededByHAZUS.append('Location')
        tunnels_gdf = tunnels_gdf.loc[:,columnsNeededByHAZUS]
    # tunnels_gdf["ID"] = tunnels_gdf["ID"].apply(lambda x: x.replace(" ",""))
    ## Format the hwy_tunnels dict array
    tunnels_gdf["type"] = "Tunnel"
    tunnels_gdf["assetSubtype"] = "HwyTunnel"
    tunnelDict = json.loads(tunnels_gdf.to_json())
    return tnodeDF, tunnelDict   

def combineDict(bnodeDF, bridgesDict, rnodeDF, roadsDict, tnodeDF, tunnelsDict,
                connectivity, lengthUnit, crs="epsg:4326"):
    combinedDict = {'type':'FeatureCollection', 
                    'generated':str(datetime.now()),
                    'brails_version': version('BRAILS'),
                    "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                    'units': {"length": lengthUnit},
                    'features':[]}
    combinedDict["features"] += bridgesDict['features']
    combinedDict["features"] += tunnelsDict['features']
    combinedDict["features"] += roadsDict['features']
    if connectivity:
        NumOfBridgeNodes = bnodeDF.shape[0]
        NumOfRoadwayNode = rnodeDF.shape[0]
        NumOfTunnelNodes = tnodeDF.shape[0]
        # Append tunnels to bridges
        tnodeDF["nodeID"] = tnodeDF["nodeID"].apply(lambda x:x + NumOfBridgeNodes)
        for tunnel in tunnelsDict['features']:
            tunnel["properties"]["Location"] = tunnel["properties"]["Location"] + NumOfBridgeNodes
        # Append roadways to tunnels and bridges
        rnodeDF["nodeID"] = rnodeDF["nodeID"].apply(lambda x:x + NumOfBridgeNodes + NumOfTunnelNodes)
        for road in roadsDict['features']:
            road["properties"]["StartNode"] = road["properties"]["StartNode"] + NumOfBridgeNodes + NumOfTunnelNodes
            road["properties"]["EndNode"] = road["properties"]["EndNode"] + NumOfBridgeNodes + NumOfTunnelNodes
    # Create the combined dic
        allNodeDict = pd.concat([bnodeDF, tnodeDF, rnodeDF], axis=0, ignore_index=True)
        allNodeDict.to_file('hwy_inventory_nodes.geojson',driver='GeoJSON')
        print(f"\nCombined inventory data available in hwy_inventory_nodes.geojson in {os.getcwd()}")
        # allNodeDict["type"] = "TransportationNode"
        # allNodeDict = json.loads(allNodeDict.to_json())
        # combinedDict["features"]+=allNodeDict['features']
    return combinedDict