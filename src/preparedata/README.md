<!-- originally from: /Users/simcenter/Files/SimCenter/Wind Storm Surge Workflow/Atlantic City/BIM -->

# Prepare the data

### 0. Get google map api key
Define google api key in src/keys.py file like this:   
``` 
GoogleMapAPIKey = "replace this with your key" 
```

### 1. Prepare a list of building addresses in csv format
For example, Atlantic_Cities_Addrs.csv looks like this
![img](../../docs/images/addressList.png)

These basic information are scraped from tax websites. 
For some buildings, some information may be missing from the websites, just leave them blank in csv file.
Once Atlantic_Cities_Addrs.csv is prepared, define the path of it in src/confiugre.py like this:

```
cleanedBIMFileName = dataDir+"/Atlantic_Cities_Addrs.csv"
```
<!--
To get Atlantic_Cities_Addrs.csv, with AtlanticMOD4.csv, run this
1. python MOD4BIM-Get-Uniq-Addr.py
2. python prepareRaw.py
3. python geocoding_addr.py
-->
### 2. Prepare a boundary file of the region of interest in geojson format

Define the path in src/configure.py like this:
```
RegionBoundaryFileName = dataDir+"/AtlanticCoastalCities_Boundary.geojson"
```

### 3. Prepare building footprints in geojson format
Download computer generated building footprints -> [USBuildingFootprints](https://github.com/microsoft/USBuildingFootprints)

Define the path of this footprints file in src/configure.py like this:
```
BuildingFootPrintsFileName = dataDir+"/AtlanticCoastalCities_Footprints.geojson"
```

### 4. Geocode buildings and create a basic BIM file for this region.
Define the file path to store BIM for all buildings in src/configure.py like this:
```
resultBIMFileName = dataDir+"/Atlantic_Cities_BIM.geojson"
```
Then run the following command from src/preparedata

```
python geocoding_addr.py
```

This will create a BIM file Atlantic_Cities_BIM.geojson containing basic building information within the interested region.
The generated BIM file can be visulized in a GIS software, such as QGIS.
```
{
	"type": "FeatureCollection",
	"features": [{
			"type": "Feature",
			"id": "8460",
			"properties": {
				"id": "8460",
				"lat": 39.371879,
				"lon": -74.456126,
				"address": "1970 W RIVERSIDE DR, ATLANTIC CITY,NJ",
				"stories": 2,
				"yearBuilt": 2006,
				"occupancy": "Residential",
				"structureType": "Frame",
				"buildingDescription": "2SF",
				"city": "ATLANTIC CITY CITY"
			},
			"geometry": {
				"type": "Polygon",
				"coordinates": [
					[
						[-74.45606, 39.371837],
						[-74.455935, 39.371934],
						[-74.456037, 39.372013],
						[-74.456162, 39.371916],
						[-74.45606, 39.371837]
					]
				]
			}
		},
		{
			"type": "Feature",
			"id": "8461",
			"properties": {
				"id": "8461",
				"lat": 39.3716807,
				"lon": -74.4513949,
				"address": "1619 COLUMBIA AVE, ATLANTIC CITY,NJ",
				"stories": 2,
				"yearBuilt": 1979,
				"occupancy": "Residential",
				"structureType": "Frame",
				"buildingDescription": "2SF",
				"city": "ATLANTIC CITY CITY"
			},
			"geometry": {
				"type": "Polygon",
				"coordinates": [
					[
						[-74.451353, 39.371717],
						[-74.451493, 39.371755],
						[-74.451526, 39.37168],
						[-74.451386, 39.371643],
						[-74.451353, 39.371717]
					]
				]
			}
		}
	]
}
``` 

