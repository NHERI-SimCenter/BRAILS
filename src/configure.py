import os
from keys import * 

# ---------------------------------
# define dirs
# ---------------------------------

srcDir = os.path.dirname(os.path.realpath(__file__))
dataDir = srcDir + "/../data/preparedata"



# ---------------------------------
# define file names here:
# ---------------------------------

# for MOD4BIM-Get-Uniq-Addr.py
stateName = 'NJ'
MOD4FilePath = dataDir+"/AtlanticMOD4.csv"
cleanedBIMFileName = dataDir+"/Atlantic_Cities_Addrs.csv"
OutputBuildingsPath = dataDir+"/AtlanticBuildings.csv"
BIMFileName = dataDir+"/Addrs_tmp.csv"

# for geocoding_addr.py
# geojson file defining the boundary of the interested area
RegionBoundaryFileName = dataDir+"/AtlanticCoastalCities_Boundary.geojson"
# building footprints obtained from MS: https://github.com/microsoft/USBuildingFootprintsv
BuildingFootPrintsFileName = dataDir+"/AtlanticCoastalCities_Footprints.geojson"
## a csv file containing the builidng address and other information. First line is viariable names, first column must be address
#cleanedBIMFileName = dataDir+"/Atlantic_Cities_Addrs.csv"

# define path for the resulting geojson file that contains BIM for all buildings
resultBIMFileName = dataDir+"/Atlantic_Cities_BIM.geojson"
# define path for a temporary csv file
cleanedBIMFile_w_coord_Name = dataDir+"/Atlantic_Cities_Addrs_Coords.csv"
# where to put json files of addrs' coords
baseDir_addrCoordJson = dataDir+"/geocoding/"

roofDownloadDir = dataDir+"/roof/"

# maximum number of images to be downloaded from google
maxNumofRoofImgs = 10000 # 2.00 USD per 1000
