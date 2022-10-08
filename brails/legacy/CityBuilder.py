# -*- coding: utf-8 -*-
"""
/*------------------------------------------------------*
|                         BRAILS                        |
|                                                       |
| Authors: Barbaros Cetiner                             |
|          Frank McKenna                                |
|          Chaofeng Wang                                |
|                                                       |
| Date:         01/05/2021                              |
| Last revised: 12/12/2021                              |
*------------------------------------------------------*/
"""

import argparse
import os
from pathlib import Path
import random
import numpy as np
from glob import glob
import pandas as pd
import geopandas as gpd
import json


from brails.modules import (RoofClassifier, OccupancyClassifier, 
                            SoftstoryClassifier, FoundationHeightClassifier, 
                            YearBuiltClassifier, NFloorDetector, 
                            GarageDetector, ChimneyDetector)


from .workflow.Footprints import getMSFootprintsByPlace, getStateNameByBBox, getOSMFootprints, getMSFootprintsByBbox
from .workflow.Images import getGoogleImages



class CityBuilder:
    """Class for creating city-scale BIM."""

    def __init__(self, attributes=['numstories','occupancy','roofshape'], 
                 numBldg=10, random=True,bbox=[], place='', 
                 footPrints='OSM', save=True, fileName='', 
                 workDir='tmp',GoogleMapAPIKey='', overwrite=False, 
                 reDownloadImgs=False):
        """init function for CityBuilder class.

        Args:

            attributes (list): The list of requested building attributes, such as ['numstories', 'occupancy', 'roofshape']
            numBldg (int): Number of buildings to generate.
            random (bool): Randomly select numBldg buildings from the database if random is True
            bbox (list): [north, west, south, east]
            place (str): The region of interest, e.g., Berkeley, California
            footPrints (str): The footprint provide, choose from OSM or Microsoft. The default value is OSM.
            save (bool): Save temporary files. Default value is True.
            fileName (str): Name of the generated BIM file. Default value will be generated if not provided by the user.
            workDir (str): Work directory where all files will be saved. Default value is ./tmp
            GoogleMapAPIKey (str): Google API Key
            overwrite (bool): Overwrite existing tmp files. Default value is False.
            reDownloadImgs (bool): Re-download if an image exists locally. Default value is False.

        Returns:
            BIM (geopandas dataframe): BIM
            fileName (str): Path to the BIM file.

        """

        if GoogleMapAPIKey == '':
            print('Please provide GoogleMapAPIKey') 
            return None

        if not os.path.exists(workDir): os.makedirs(workDir)

        #if fileName=='': fileName=Path(f'{place} {state}_BIM.geojson'.replace(' ','_').replace(',','_'))
        if fileName=='': 

            if len(bbox)>0: 
                fileName = Path(f'/{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}_BIM.geojson')
            else:
                fileName = Path(f'/{place}_BIM.geojson'.replace(' ','_').replace(',','_'))

        

        bimFile = Path(f'{workDir}/{fileName}')
        
        self.attributes = attributes
        self.numBldg = numBldg
        self.random = random
        self.place = place
        self.save = save
        self.fileName = fileName
        self.workDir = workDir
        self.overwrite =  overwrite
        self.bimFile = bimFile
        self.GoogleMapAPIKey = GoogleMapAPIKey
        self.reDownloadImgs = reDownloadImgs
        self.footPrints = footPrints

        self.bimExist = False
        if os.path.exists(self.bimFile) and not overwrite:
            print('{} already exists, nothing to do. \n Set overwrite=True in CityBuilder to overwrite.'.format(self.bimFile))
            self.BIM = gpd.read_file(self.bimFile)
            self.bimExist = True
            return None



        if len(bbox)>0: # bbox provided
            self.getFootprintByBbox(bbox=bbox,workDir=workDir,GoogleMapAPIKey=GoogleMapAPIKey,save=save,overwrite=overwrite)

        elif len(place)>0:
            print(place)
            self.getFootprintByPlace(place=place,workDir=workDir,GoogleMapAPIKey=GoogleMapAPIKey,save=save,overwrite=overwrite)

        else:
            assert False, f"Please provide a bbox or place name."
        
        print(f'{self.BIM.shape[0]} buildings found.')
        if numBldg < self.BIM.shape[0]: 
            if random: self.BIM = self.BIM.sample(n=numBldg)
            else: self.BIM = self.BIM[:numBldg]
            print(f'{self.BIM.shape[0]} buildings selected.')
        
    def getFootprintByBbox(self,bbox=[],workDir='',GoogleMapAPIKey='',save=True,overwrite=True):
        #BIM = getCityBIM()

        stateName = getStateNameByBBox(bbox=bbox,workDir=workDir,GoogleMapAPIKey=GoogleMapAPIKey) # isNone means didn't find in the U.S.
        if stateName:# bbox, in the US
            #self.BIM, footprintFilename = getMSFootprintsByBbox(bbox=bbox,stateName=stateName,save=save,workDir=workDir,overwrite=overwrite,GoogleMapAPIKey=GoogleMapAPIKey)
            if self.footPrints=='OSM':
                print(f"Trying to get OSM footprints by the bbox {bbox}.")
                self.BIM, footprintFilename = getOSMFootprints(bbox=bbox,save=save,workDir=workDir,overwrite=overwrite)
            elif self.footPrints=='Microsoft':
                # MS
                print(f"Trying to get MS footprints by the bbox {bbox}.")
                self.BIM, footprintFilename = getMSFootprintsByBbox(bbox=bbox,stateName=stateName,save=save,workDir=workDir,overwrite=overwrite,GoogleMapAPIKey=GoogleMapAPIKey)
            else:
                assert False, f"footPrints must be OSM or Microsoft in CityBuilder. Your value is {self.footPrints}."
        else: # out of US, use OSM no matter what
                print(f"The bbox seems to be outside of the US. Footprints probably only exist in OSM.  Trying to get OSM footprints by the bbox {bbox}.")
                self.BIM, footprintFilename = getOSMFootprints(bbox=bbox,save=save,workDir=workDir,overwrite=overwrite)
            

    def getFootprintByPlace(self,place='',workDir='',GoogleMapAPIKey='',save=True,overwrite=True):
        if self.footPrints=='OSM':
            print(f"Trying to get OSM footprints by the place name {place}.")
            self.BIM, footprintFilename = getOSMFootprints(place=place,save=save,workDir=workDir,overwrite=True)
        elif self.footPrints=='Microsoft':
            #assert False, f"Trying to get Microsoft footprints by the place name {place}."
            self.BIM, footprintFilename = getMSFootprintsByPlace(place=place,save=save,workDir=workDir,GoogleMapAPIKey=GoogleMapAPIKey,overwrite=overwrite)
        else: 
            assert False, f"footPrints must be OSM or Microsoft in CityBuilder. Your value is {self.footPrints}."

    def downloadImgs(self,fov=60,pitch=0): 

        if os.path.exists(self.bimFile) and not self.overwrite:
            print('{} already exists.'.format(self.bimFile))
            BIM = gpd.read_file(self.bimFile)
            return BIM 
        
        if self.BIM.shape[0] < 1:
            print('Can not build because the Footprints module didn\'t find footprints for this region.')
            return None

        print('Starting downloading images.')
        imgDir = os.path.join(self.workDir, 'images')

        #imageTypes = ['StreetView','TopView']
        imageTypes = []
        if 'roofshape' in self.attributes:
            imageTypes.append('TopView')

        svAttrs = ['occupancy','softstory','elevated','numstories','year']
        usvAttrs = [x for x in svAttrs if x in set(self.attributes)]
        if len(usvAttrs)> 0: imageTypes.append('StreetView')

        self.imageTypes = imageTypes
        self.imgDir = imgDir

        getGoogleImages(self.BIM,GoogleMapAPIKey=self.GoogleMapAPIKey, imageTypes=imageTypes, imgDir=imgDir, ncpu=2,fov=fov,pitch=pitch,reDownloadImgs=self.reDownloadImgs)


    def build(self): 

        if self.bimExist:
            print('{} already exists, nothing to do. \n Set overwrite=True in CityBuilder to overwrite.'.format(self.bimFile))
            return self.BIM

        self.downloadImgs()

        imageTypes = self.imageTypes
        imgDir = self.imgDir

        #BIM = allFootprints
        self.BIM['Lon'] = self.BIM['geometry'].centroid.x#.round(decimals=6)
        self.BIM['Lat'] = self.BIM['geometry'].centroid.y#.round(decimals=6)
        for cat in imageTypes:
            self.BIM[cat] = self.BIM.apply(lambda row: Path(f"{imgDir}/{cat}/{cat}x{'%.6f'%row['Lon']}x{'%.6f'%row['Lat']}.png"), axis=1)

        self.BIM.reset_index(inplace = True) 
        self.BIM['ID'] = self.BIM.index
    

        for attr in self.attributes:
            if attr.lower()=='roofshape':

                # initialize a roof classifier
                roofModel = RoofClassifier(workDir=self.workDir,printRes=False)

                # use the roof classifier 
                roofShape_df = roofModel.predict(self.BIM['TopView'].tolist())

                roofShape = roofShape_df['prediction'].to_list()
                roofShapeProb = roofShape_df['probability'].to_list()
                self.BIM['roofShape'] = self.BIM.apply(lambda x: roofShape[x['ID']], axis=1)
                self.BIM['roofShapeProb'] = self.BIM.apply(lambda x: roofShapeProb[x['ID']], axis=1)

            elif attr.lower()=='occupancy':
                # initialize an occupancy classifier
                occupancyModel = OccupancyClassifier(workDir=self.workDir,printRes=False)
                # use the occupancy classifier 
                occupancy_df = occupancyModel.predict(self.BIM['StreetView'].tolist())

                occupancy = occupancy_df['prediction'].to_list()
                occupancyProb = occupancy_df['probability'].to_list()
                self.BIM['occupancy'] = self.BIM.apply(lambda x: occupancy[x['ID']], axis=1)
                self.BIM['occupancyProb'] = self.BIM.apply(lambda x: occupancyProb[x['ID']], axis=1)

            elif attr.lower()=='softstory':
                # initialize a soft-story classifier
                ssModel = SoftstoryClassifier(workDir=self.workDir,printRes=False)
                # use the softstory classifier 
                softstory_df = ssModel.predict(self.BIM['StreetView'].tolist())

                softstory = softstory_df['prediction'].to_list()
                softstoryProb = softstory_df['probability'].to_list()
                self.BIM['softStory'] = self.BIM.apply(lambda x: softstory[x['ID']], axis=1)
                self.BIM['softStoryProb'] = self.BIM.apply(lambda x: softstoryProb[x['ID']], axis=1)

            elif attr.lower()=='elevated':
                # initialize a foundation classifier
                elvModel = FoundationHeightClassifier(workDir=self.workDir,printRes=False)

                # use the classifier 
                elv_df = elvModel.predict(self.BIM['StreetView'].tolist())

                elv = elv_df['prediction'].to_list()
                elvProb = elv_df['probability'].to_list()
                self.BIM['elevated'] = self.BIM.apply(lambda x: elv[x['ID']], axis=1)
                self.BIM['elevatedProb'] = self.BIM.apply(lambda x: elvProb[x['ID']], axis=1) 

            elif attr.lower()=='numstories':
                # Initialize the floor detector object
                storyModel = NFloorDetector()

                # Call the floor detector to determine number of floors
                story_df = storyModel.predict(self.BIM['StreetView'].tolist())

                # Write the results to a dataframe
                story = story_df['prediction'].to_list()
                self.BIM['numStories'] = self.BIM.apply(lambda x: story[x['ID']], axis=1)
                
            elif attr.lower()=='garage':
                # Initialize the garage detector object
                garageModel = GarageDetector()

                # Call the floor detector to determine number of floors
                garage_df = garageModel.predict(self.BIM['StreetView'].tolist())

                # Write the results to a dataframe
                garage = garage_df['prediction'].to_list()
                self.BIM['garageExist'] = self.BIM.apply(lambda x: story[x['ID']], axis=1)

            elif attr.lower()=='chimney':
                # Initialize the floor detector object
                chimneyModel = ChimneyDetector()

                # Call the floor detector to determine number of floors
                chimney_df = chimneyModel.predict(self.BIM['StreetView'].tolist())

                # Write the results to a dataframe
                chimney = chimney_df['prediction'].to_list()
                self.BIM['chimneyExist'] = self.BIM.apply(lambda x: story[x['ID']], axis=1)
                
            elif attr.lower()=='year':
                yearModel = YearBuiltClassifier(workDir=self.workDir,printRes=False)

                year_df = yearModel.predict(self.BIM['StreetView'].tolist())

                year = year_df['prediction'].to_list()
                yearProb = year_df['probability'].to_list()
                self.BIM['year'] = self.BIM.apply(lambda x: year[x['ID']], axis=1)
                self.BIM['yearProb'] = self.BIM.apply(lambda x: yearProb[x['ID']], axis=1) 

            else:
                assert False, "attributes can only contain roofshape, occupancy, numstories, garage, chimney, year, elevated, softstory. Your % caused an error." % attr

        # delete columns
        self.BIM.drop(columns=['Lat','Lon','index'], axis=1, inplace=True)
        for c in imageTypes:
            #self.BIM.drop(columns=[c], axis=1, inplace=True)
            self.BIM[c] = self.BIM[c].astype(str)

        # save
        self.BIM.to_file(self.bimFile, driver='GeoJSON')
        print('BIM saved at {}'.format(self.bimFile))

        #print(self.BIM)
        return self.BIM








def main():
    pass

if __name__ == '__main__':
    main()




