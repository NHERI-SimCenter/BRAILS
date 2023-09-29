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
# 08-29-2023 

from brails.workflow.TransportationElementHandler import TransportationElementHandler
import geopandas as gpd
import pandas as pd
import momepy
from shapely.geometry import MultiLineString
import shapely
import gc
import json
import warnings

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
        print('Here is the list of attributes currently enabled in InventoryGenerator:\n')
        for element in self.enabledElements:
            print(f'       {element}')

    def generate(self):
        tphandler = TransportationElementHandler()
        tphandler.fetch_transportation_elements(self.location) 
        
        self.inventory_files = tphandler.output_files
        
        outfiles = ", ".join(value for value in tphandler.output_files.values())
        print(f'\nTransportation inventory data available in {outfiles}')
    
    def combineAndFormat_HWY(self):
        print(f"Formatting and combining fetched data in {self.inventory_files}")
        # Format bridges
        bridgesFile = self.inventory_files.get("bridges", None)
        if bridgesFile is not None:
            bridges_gdf = gpd.read_file(bridgesFile)
            bnodeDF, bridgesDict = formatBridges(bridges_gdf)
        else:
            bnodeDF = pd.DataFrame(columns = ["nodeID", "lat", "lon"])
            bridgesDict = []

        # Format roadways
        roadsFile = self.inventory_files.get("roads", None)
        if roadsFile is not None:
            roads_gdf = gpd.read_file(roadsFile).explode(index_parts = False)
            rnodeDF, roadsDict = formatRoads(roads_gdf)
        else:
            rnodeDF = pd.DataFrame(columns = ["nodeID", "lat", "lon"])
            roadsDict = []
        
        # Format tunnels
        tunnelsFile = self.inventory_files.get("tunnels", None)
        if tunnelsFile is not None:
            tunnels_gdf = gpd.read_file(tunnelsFile).explode(index_parts = False)
            tnodeDF, tunnelsDict = formatTunnels(tunnels_gdf)
        else:
            tnodeDF = pd.DataFrame(columns = ["nodeID", "lat", "lon"])
            tunnelsDict = []

        # Combine nodes and update dicts
        combinedDict = combineDict(bnodeDF, bridgesDict, rnodeDF, roadsDict, tnodeDF, tunnelsDict, )
        # Dump to json file
        with open("hwy_inventory.json", "w") as f:
            json.dump(combinedDict, f, indent = 4)
        
        
    
def formatBridges(bridges_gdf):
    ## Format bridge nodes
    bnodeDF = pd.DataFrame({"geometry":bridges_gdf["geometry"]}).reset_index().rename(columns = {"index":"nodeID"})
    bnodeDF["lat"] = bnodeDF["geometry"].apply(lambda pt:pt.y)
    bnodeDF["lon"] = bnodeDF["geometry"].apply(lambda pt:pt.x)
    bnodeDF.drop("geometry", axis=1, inplace=True)
    ## Format bridge items
    bridges_gdf["bridge_class"] = bridges_gdf["STRUCTURE_KIND"].apply(int)*100+bridges_gdf["STRUCTURE_TYPE"].apply(int)
    bridges_gdf = bridges_gdf.rename(columns = {"STRUCTURE_NUMBER":"ID","geometry":"location", 
        "YEAR_BUILT":"year_built", "MAIN_UNIT_SPANS":"nspans", "MAX_SPAN_LEN_MT":"lmaxspan","STATE_CODE":"state_code"})
    columnsNeeded = ["ID", "bridge_class", "year_built", "nspans", "lmaxspan", "state_code"]
    bridges_gdf = bridges_gdf.loc[:,columnsNeeded]
    bridges_gdf["ID"] = bridges_gdf["ID"].apply(lambda x: x.replace(" ",""))
    ## Format the hwy_bridges array
    bridgeDict = pd.DataFrame(bridges_gdf)
    bridgeDict = bridgeDict.reset_index().rename(columns={"index":"location"})
    bridgeDict = bridgeDict[["ID", "location", "bridge_class", "year_built", "nspans", "lmaxspan", "state_code"]]
    bridgeDict = bridgeDict.sort_values(by = 'ID') 
    bridgeDict = bridgeDict.to_dict("records")
    return bnodeDF, bridgeDict
    
def formatRoads(roads_gdf):
    ## Expand roads_gdf geometry from multi-segment to singel segment
    expandedRoads = []
    for row_ind in roads_gdf.index:
        LS_list = []
        multiseg_line = roads_gdf.loc[row_ind,"geometry"]
        for pt1, pt2 in zip(multiseg_line.coords, multiseg_line.coords[1:]):
            LS_list.append(shapely.LineString([pt1, pt2]))
        gdf_i = pd.concat([roads_gdf.loc[row_ind,:].to_frame().T]*len(LS_list), ignore_index=True)
        for ind_i in range(len(LS_list)):
            numDigit = len(str(len(LS_list)))
            suffixID = str(ind_i).zfill(numDigit)        
            gdf_i.loc[ind_i,"OID"] = gdf_i.loc[ind_i,"OID"]+"_"+suffixID
        gdf_i["geometry"] = LS_list
        expandedRoads.append(gdf_i)
    expandedRoads = pd.concat(expandedRoads, ignore_index=True)
    expandedRoads = gpd.GeoDataFrame(expandedRoads, crs=roads_gdf.crs)
    ## Convert to graph to find the intersection nodes
    graph = momepy.gdf_to_nx(expandedRoads.to_crs("epsg:32610"), approach='primal')
    with warnings.catch_warnings(): #Suppress the warning of disconnected components in the graph
        warnings.simplefilter("ignore")
        nodes, edges, sw = momepy.nx_to_gdf(graph, points=True, lines=True,
                                            spatial_weights=True)
    # The CRS of SimCenter is CRS:84 (equivalent to EPSG:4326)
    # The CRS of US Census is NAD83, which is https://epsg.io/4269
    nodes = nodes.to_crs("epsg:4326")
    edges = edges.to_crs("epsg:4326")
    rnodeDF = pd.DataFrame(nodes)
    ### Some edges has start_node as the last point in the geometry and end_node as the first point, check and reorder
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
            print(ind, "th row of roadway has wrong start/first, end/last pairs, likely a bug of momepy.gdf_to_nx function")
    ### Some edges are duplicated, keep only the first one
    # edges = edges[edges.duplicated(['node_start', 'node_end'], keep="first")==False]
    # edges = edges.reset_index(drop=True)
    ## Format roadways
    ### Format and clean up roadway edges
    road_type = []
    lanes = []
    capacity = []
    edge_id = []
    for row_ind in edges.index:
        mtfcc = edges.loc[row_ind,"MTFCC"]
        road_type.append(ROADTYPE_MAP[mtfcc])
        lanes.append(ROADLANES_MAP[mtfcc])
        capacity.append(ROADCAPACITY_MAP[mtfcc])
        edge_id.append(edges.loc[row_ind,"OID"])
    edges["ID"] = edge_id
    edges["road_type"] = road_type
    edges["lanes"] = lanes
    edges["capacity"] = capacity
    edges = edges.rename(columns={'node_start': 'start_node', 'node_end': 'end_node', 'mm_len':"length"})
    columnsNeeded=['ID','length','start_node','end_node','road_type','lanes','capacity']
    edges = edges[columnsNeeded]
    edges = edges.sort_values(by="ID")
    edgesDict = edges.to_dict(orient='records')

    ## Format roadway nodes
    rnodeDF["lat"] = rnodeDF["geometry"].apply(lambda pt:pt.y)
    rnodeDF["lon"] = rnodeDF["geometry"].apply(lambda pt:pt.x)
    rnodeDF = rnodeDF.drop("geometry", axis=1)

    return rnodeDF, edgesDict

def formatTunnels(tunnels_gdf):
    ## Format tunnel nodes
    tnodeDF = pd.DataFrame({"geometry":tunnels_gdf["geometry"]}).reset_index().rename(columns = {"index":"nodeID"})
    tnodeDF["lat"] = tnodeDF["geometry"].apply(lambda pt:pt.y)
    tnodeDF["lon"] = tnodeDF["geometry"].apply(lambda pt:pt.x)
    tnodeDF.drop("geometry", axis=1, inplace=True)
    ## Format tunnel items
    if "cons_type" not in tunnels_gdf.columns:
        tunnels_gdf["cons_type"] = "unclassified"
    tunnels_gdf = tunnels_gdf.rename(columns = {"tunnel_number":"ID"})
    columnsNeeded = ["ID", "cons_type"]
    tunnels_gdf = tunnels_gdf.loc[:,columnsNeeded]
    tunnels_gdf["ID"] = tunnels_gdf["ID"].apply(lambda x: x.replace(" ",""))
    ## Format the hwy_tunnels dict array
    tunnelDict = pd.DataFrame(tunnels_gdf)
    tunnelDict = tunnelDict.reset_index().rename(columns={"index":"location"})
    tunnelDict = tunnelDict[["ID", "location", "cons_type"]]
    tunnelDict = tunnelDict.sort_values("ID")
    tunnelDict = tunnelDict.to_dict("records")
    return tnodeDF, tunnelDict
    

def combineDict(bnodeDF, bridgesDict, rnodeDF, roadsDict, tnodeDF, tunnelsDict, crs="epsg:4326"):
    NumOfBridgeNodes = bnodeDF.shape[0]
    NumOfRoadwayNode = rnodeDF.shape[0]
    NumOfTunnelNodes = tnodeDF.shape[0]
    # Append tunnels to bridges
    tnodeDF["nodeID"] = tnodeDF["nodeID"].apply(lambda x:x + NumOfBridgeNodes)
    for tunnel in tunnelsDict:
        tunnel["location"] = tunnel["location"] + NumOfBridgeNodes
    # Append roadways to tunnels and bridges
    rnodeDF["nodeID"] = rnodeDF["nodeID"].apply(lambda x:x + NumOfBridgeNodes + NumOfTunnelNodes)
    for road in roadsDict:
        road["start_node"] = road["start_node"] + NumOfBridgeNodes + NumOfTunnelNodes
        road["end_node"] = road["end_node"] + NumOfBridgeNodes + NumOfTunnelNodes
    # Create the combined dic
    allNodeDict = pd.concat([bnodeDF, tnodeDF, rnodeDF], axis=0, ignore_index=True)
    allNodeDict = allNodeDict.set_index('nodeID').to_dict('index')
    combinedDict = {
        "crs":crs,
        "hwy_bridges":bridgesDict,
        "hwy_tunnels":tunnelsDict,
        "roadways":roadsDict,
        "nodes":allNodeDict,
    }
    return combinedDict
