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
# Yunhui Guo
# Sascha Hornauer
# Frank McKenna
# Satish Rao
#
# Last updated:
# 05-08-2022 

import random
import sys
import pandas as pd
from shapely.geometry import Polygon
#import os
#import shutil

#import brails.models as models
from brails.modules import (ChimneyDetector, FacadeParser, GarageDetector, 
                            NFloorDetector, PytorchRoofClassifier, 
                            PytorchOccupancyClassifier, RoofCoverClassifier, 
                            YearBuiltClassifier)
from .workflow.ImHandler import ImageHandler
from .workflow.FootprintHandler import FootprintHandler

class InventoryGenerator:

    def __init__(self, location='Berkeley', nbldgs=10, randomSelection=True,
                 GoogleAPIKey=''):                
        self.apiKey = GoogleAPIKey
        """
        self.enabledAttributes = ['buildingheight','chimney','erabuilt',
                                  'garage','numstories','occupancy',
                                  'roofcover','roofeaveheight','roofshape',
                                  'roofpitch']
        """
        self.enabledAttributes = ['buildingheight','chimney','erabuilt',
                                  'garage','numstories','roofeaveheight',
                                  'roofshape','roofpitch']

        self.inventory = None
        self.location = location
        self.nbldgs = nbldgs
        self.randomSelection = randomSelection
        self.workDir = 'tmp'
        self.modelDir = 'tmp/models'

        """
        # Copy the model files to the current directory:
        os.makedirs(self.modelDir,exist_ok=True)
        model_dir_org = models.__path__[0]
        model_files = [file for file in os.listdir(model_dir_org) if file.endswith(('pth','pkl'))]
        for model_file in model_files:
            shutil.copyfile(os.path.join(model_dir_org,model_file),
                            os.path.join(self.modelDir,model_file))
        """
        
        # Get footprint data for the defined location:
        fpHandler = FootprintHandler()
        if 'geojson' in location:
            fpHandler.load_footprint_data(self.location)
        else:
            fpHandler.fetch_footprint_data(self.location)

        # Randomly select nbldgs from the footprint data if randomSelection is 
        # set to True:          
        if self.randomSelection==True:
            footprints = random.sample(fpHandler.footprints, nbldgs)
            print(f'Randomly selected {nbldgs} buildings')
        else: 
            footprints = fpHandler.footprints[:nbldgs]
            print(f'Selected the first {nbldgs} buildings')
        
        # Initialize the inventory DataFrame with the obtained footprint data:
        self.inventory = pd.DataFrame(pd.Series(footprints,name='footprint'))
        
        # Initialize the image handler class  to check if the provided API key
        # is valid:
        image_handler = ImageHandler(self.apiKey)
    
    def enabled_attributes(self):
        print('Here is the list of attributes currently enabled in InventoryGenerator:\n')
        for attribute in self.enabledAttributes:
            print(f'       {attribute}')
        print('\nIf you want to get all of these attributes when you run '
              "InventoryGenerator.generate, simply set attributes='all'")

    def generate(self,attributes=['numstories','occupancy','roofshape']):
        
        def write_to_dataframe(df,predictions,column,imtype='street_images'):
            """
            Function that writes predictions of a model by going through the 
            DataFrame, df, and finding the image name matches between the 
            list of images corresponding to the predictions, i.e., imagelist,
            then assigning the predictions  to df. The function only searches 
            the column named im_type for image names in imagelist.
            
            Inputs: 
                df: DataFrame
                    A DataFrame containing at least a column of names of source
                    images named either 'satellite_images' or 'street_images',
                    depending on the value of im_type
                predictions: list or DataFrame
                    A list consisting of two lists: 1) list of image names. 
                    These names need to correspond to the predictions in 
                    predictionlist 2) list of predictions. These predictions
                    need to correspond to the image names in imagelist.
                    Alternately, a Pandas DataFrame with a column titled image 
                    for image names, and a column for the predictions titled
                    predictions
                imtype: string
                    'satellite_images' or 'street_images' depending on the
                    type of source images
                column: Name of the column where the predictions will be written
            
            Output: An updated version of df that includes an extra column for 
                    the predictions
            """
            if isinstance(predictions,list):
                imagelist = predictions[0][:]
                predictionlist = predictions[1][:]
                for ind, im in enumerate(imagelist):
                    df.loc[df.index[df[imtype] == im],column] = predictionlist[ind]
            else:    
                for index, row in predictions.iterrows():
                    df.loc[df.index[df[imtype] == row['image']],column] = row['prediction']
                          
            return df
        
        # Pre-process the attribute entries such that incorrect entries are 
        # removed:
        if isinstance(attributes,str) and attributes=='all':    
            self.attributes = self.enabledAttributes[:]
        elif isinstance(attributes,list):
            self.attributes = [attribute.lower() for attribute in attributes]
            ignore_entries = []
            for attribute in self.attributes:
                if attribute not in self.enabledAttributes:
                    ignore_entries.append(attribute)
                    self.attributes.remove(attribute)
            if len(ignore_entries)==1:
                print('Found an entry in attributes that was not enabled in ' + 
                      'InventoryGenerator.\nIgnoring entry: ' +
                      ', '.join(ignore_entries))
            elif len(ignore_entries)>1:
                print('Found entries in attributes that were not enabled in ' + 
                      'InventoryGenerator.\nIgnoring entries: ' +
                      ', '.join(ignore_entries))
        
        if len(self.attributes)==0:
            sys.exit('Defined list of attributes does not contain a ' + 
                     'correct attribute entry. Attribute entries enabled' +
                     ' are: ' + ', '.join(self.enabledAttributes))
        
        # Remove duplicate attribute entries:
        self.attributes = sorted(list(set(self.attributes)))
        
        # Create a list of footprints for easier module calls:
        footprints = self.inventory['footprint'].values.tolist()
        
        # Download the images required for the requested attributes:
        image_handler = ImageHandler(self.apiKey)
        
        if 'roofshape' in self.attributes: #or 'roofcover' in self.attributes:
            image_handler.GetGoogleSatelliteImage(footprints)
            imsat = [im for im in image_handler.satellite_images if im is not None]
            self.inventory['satellite_images'] = image_handler.satellite_images
        
        streetAttributes = self.enabledAttributes[:]
        streetAttributes.remove('roofshape')
        #streetAttributes.remove('roofcover')
        if set.intersection(set(streetAttributes),set(self.attributes))!=set():
            image_handler.GetGoogleStreetImage(footprints)
            imstreet = [im for im in image_handler.street_images if im is not None]
            self.inventory['street_images'] = image_handler.street_images
        
        for attribute in self.attributes:
 
            if attribute=='chimney':
                # Initialize the chimney detector object:
                chimneyModel = ChimneyDetector()

                # Call the chimney detector to determine the existence of chimneys:
                chimneyModel.predict(imstreet)

                # Write the results to the inventory DataFrame:
                self.inventory = write_to_dataframe(self.inventory,
                                   [chimneyModel.system_dict['infer']['images'],
                                   chimneyModel.system_dict['infer']['predictions']],
                                   'chimneyExists')
                self.inventory['chimneyExists'] = self.inventory['chimneyExists'].astype(dtype="boolean")
                
            elif attribute=='erabuilt':
                # Initialize the era of construction classifier:
                yearModel = YearBuiltClassifier()
                
                # Call the classifier to determine the era of construction for
                # each building:
                yearModel.predict(imstreet)
                
                # Write the results to the inventory DataFrame:
                self.inventory = write_to_dataframe(self.inventory,yearModel.results_df,'eraBuilt')
                self.inventory['eraBuilt'] = self.inventory['eraBuilt'].fillna('N/A')

            elif attribute=='garage':
                # Initialize the garage detector object:
                garageModel = GarageDetector()

                # Call the garage detector to determine the existence of garages:
                garageModel.predict(imstreet)

                # Write the results to the inventory DataFrame:
                self.inventory = write_to_dataframe(self.inventory,
                                   [garageModel.system_dict['infer']['images'],
                                   garageModel.system_dict['infer']['predictions']],
                                   'garageExists')
                self.inventory['garageExists'] = self.inventory['garageExists'].astype(dtype="boolean")
                
            elif attribute=='numstories' and 'storyModel' not in locals():
                # Initialize the floor detector object:
                storyModel = NFloorDetector()

                # Call the floor detector to determine number of floors of 
                # buildings in each image:
                storyModel.predict(imstreet)

                # Write the results to the inventory DataFrame:
                self.inventory = write_to_dataframe(self.inventory,
                                   [storyModel.system_dict['infer']['images'],
                                   storyModel.system_dict['infer']['predictions']],
                                   'nstories')
                self.inventory['nstories'] = self.inventory['nstories'].astype(dtype='Int64')
                
            elif attribute=='occupancy':
                # Initialize the occupancy classifier object:
                occupancyModel = PytorchOccupancyClassifier(modelName='transformer_occupancy_v1',
                                                            download=True)
                
                # Call the occupancy classifier to determine the occupancy 
                # class of each building:
                occupancy = occupancyModel.predictMultipleImages(imstreet)
                
                # Write the results to the inventory DataFrame:
                self.inventory = write_to_dataframe(self.inventory,occupancy,
                                                    'occupancy')
                self.inventory['occupancy'] = self.inventory['occupancy'].fillna('N/A')
            
            elif attribute=='roofcover':
                # Initialize the roof cover classifier object:
                roofCoverModel = RoofCoverClassifier()

                # Call the roof cover classifier to classify roof cover type of
                # each building:                
                roofCoverModel.predict(imsat)

                # Write the results to the inventory DataFrame:
                self.inventory = write_to_dataframe(self.inventory,
                                                    roofCoverModel.system_dict['infer']['predictions'],
                                                    'roofCover',
                                                    'satellite_images')
            
            elif attribute=='roofshape':
                # Initialize the roof type classifier object:
                roofModel = PytorchRoofClassifier(modelName='transformer_rooftype_v1',
                                                  download=True)

                # Call the roof type classifier to determine the roof type of
                # each building:                
                roofShape = roofModel.predictMultipleImages(imsat)
                
                # Write the results to the inventory DataFrame:
                roofShape['prediction'] = roofShape['prediction'].replace({'flat':'Flat', 'hipped':'Hip', 'gabled':'Gable'})
                self.inventory = write_to_dataframe(self.inventory,roofShape,
                                                    'roofshape',
                                                    'satellite_images')
                
            elif attribute in ['buildingheight','roofeaveheight','roofpitch']:
                if 'storyModel' not in locals():
                    # Initialize the floor detector object:
                    storyModel = NFloorDetector()

                    # Call the floor detector to determine number of floors of 
                    # buildings in each image:
                    storyModel.predict(imstreet)
                    
                    # Write the results to the inventory DataFrame:
                    self.inventory = write_to_dataframe(self.inventory,
                                       [storyModel.system_dict['infer']['images'],
                                       storyModel.system_dict['infer']['predictions']],
                                       'nstories')
                    self.inventory['nstories'] = self.inventory['nstories'].astype(dtype='Int64')
                
                if 'facadeParserModel' not in locals():   
                    # Initialize the facade parser object:
                    facadeParserModel = FacadeParser()              
                    
                    # Call the facade parser to determine the requested 
                    # attribute for each building:
                    facadeParserModel.predict(image_handler,storyModel)
                    
                self.inventory = write_to_dataframe(self.inventory,
                                                    [facadeParserModel.predictions['image'].to_list(),
                                                     facadeParserModel.predictions[attribute].to_list()],
                                                     attribute)    
        
        dfout = self.inventory.copy(deep=True)
        for index, row in self.inventory.iterrows():
            dfout.loc[index, 'footprint'] = Polygon(row['footprint']).wkt
        dfout = dfout.drop(columns=['satellite_images', 'street_images'], 
                           errors='ignore')
        dfout.to_csv('inventory.csv', index=False) 
        print('Final inventory available in inventory.csv')