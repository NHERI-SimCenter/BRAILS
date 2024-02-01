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
# Sascha Hornauer
# Frank McKenna
# Satish Rao
#
# Last updated:
# 02-01-2024 

import random
import sys
import pandas as pd
import os
import json
from shapely.geometry import Polygon
from datetime import datetime
from importlib.metadata import version

#import brails.models as models
from brails.modules import (ChimneyDetector, FacadeParser, GarageDetector, 
                            NFloorDetector, RoofClassifier, 
                            OccupancyClassifier, RoofCoverClassifier, 
                            YearBuiltClassifier)
from .workflow.ImHandler import ImageHandler
from .workflow.FootprintHandler import FootprintHandler
from brails.workflow.NSIParser import NSIParser

class InventoryGenerator:

    def __init__(self, location='Berkeley', nbldgs=10, randomSelection=True,
                 fpSource='osm', GoogleAPIKey=''):                
        self.apiKey = GoogleAPIKey
        self.enabledAttributes = ['buildingheight','chimney','erabuilt',
                                  'garage','numstories','occupancy',
                                  'roofeaveheight','roofshape','roofpitch',
                                  'constype'] #roofcover,'constype']

        self.inventory = None
        self.incompleteInventory = None
        self.location = location
        self.nbldgs = nbldgs
        self.randomSelection = randomSelection
        self.workDir = 'tmp'
        self.modelDir = 'tmp/models'
        self.fpSource = fpSource

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
        fpHandler.fetch_footprint_data(self.location,self.fpSource)

        # Randomly select nbldgs from the footprint data if randomSelection is 
        # set to True:          
        if nbldgs!='all':
            if self.randomSelection==False:
                footprints = fpHandler.footprints[:nbldgs]
                fpareas = fpHandler.fpAreas[:nbldgs]
                print(f'Selected the first {nbldgs} buildings')
            elif self.randomSelection==True: 
                inds = random.sample(list(range(len(fpHandler.footprints))),nbldgs); inds.sort()
                footprints = [fpHandler.footprints[ind] for ind in inds]
                fpareas = [fpHandler.fpAreas[ind] for ind in inds]
                print(f'Randomly selected {nbldgs} buildings')
            else:
                random.seed(self.randomSelection)
                inds = random.sample(list(range(len(fpHandler.footprints))),nbldgs); inds.sort()
                footprints = [fpHandler.footprints[ind] for ind in inds]
                fpareas = [fpHandler.fpAreas[ind] for ind in inds]
                print(f'Randomly selected {nbldgs} buildings using the seed {self.randomSelection}')
        else:	
                footprints = fpHandler.footprints[:]
                fpareas = fpHandler.fpAreas[:]
                print(f'Selected all {len(footprints)} buildings')
        
        # Initialize the inventory DataFrame with the footprint data obtained for nbldgs:
        self.inventory = pd.DataFrame(pd.Series(footprints,name='Footprint'))
        self.inventory['PlanArea'] = fpareas
        
        # Initialize a seperate inventory DataFrame including all footprints for
        # the defined region for use with a data imputation application:
        self.incompleteInventory = pd.DataFrame(pd.Series(fpHandler.footprints[:],name='Footprint'))
        self.incompleteInventory['PlanArea'] = fpHandler.fpAreas[:]                
        
        # Initialize the image handler class  to check if the provided API key
        # is valid:
        image_handler = ImageHandler(self.apiKey)
    
    def enabled_attributes(self):
        print('Here is the list of attributes currently enabled in InventoryGenerator:\n')
        for attribute in self.enabledAttributes:
            print(f'       {attribute}')
        print('\nIf you want to get all of these attributes when you run '
              "InventoryGenerator.generate, simply set attributes='all'")

    def generate(self,attributes=['numstories','occupancy','roofshape'],
                 useNSIBaseline= True, outFile='inventory.csv', 
                 lengthUnit='ft'):
        
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
        
        
        def parse_attribute_input(attrIn,attrEnabled):             
            """
            Function that pre-processes the user attribute entries to 
            InventoryGenerator().generate method such that pre-defined 
            attribute sets are correctly parsed and incorrect entries are 
            removed:
            
            Inputs: 
                df: list
                    A list of string descriptions for the user requested 
                    building attributes
                attrEnabled: list 
                    A list of string descriptions for the building attributes
                    currently enabled in InventoryGenerator().generate method
            
            Output: A list of string descriptions for the user requested 
                    building attributes that were determined to valid for use
                    with InventoryGenerator().generate method
            """       
            
            if isinstance(attrIn,str) and attrIn=='all':    
                attrOut = attrEnabled[:]
            elif isinstance(attrIn,str) and attrIn=='hazuseq':
                attrOut = ['numstories']
            elif isinstance(attrIn,list):
                attrOut = [attribute.lower() for attribute in attrIn]
                ignore_entries = []
                for attribute in attrOut:
                    if attribute not in attrEnabled:
                        ignore_entries.append(attribute)
                        attrOut.remove(attribute)
                if len(ignore_entries)==1:
                    print('An entry in attributes is not enabled.'
                          f'\nIgnoring entry: {ignore_entries[0]}') 
                elif len(ignore_entries)>1:
                    print('Several entries in attributes are not enabled.' 
                          '\nIgnoring entries: ' + ', '.join(ignore_entries))
                    
            if len(attrOut)==0:
                sys.exit('Defined list of attributes does not contain a ' + 
                         'correct attribute entry. Attribute entries enabled' +
                         ' are: ' + ', '.join(attrEnabled))
            
            # Remove duplicate attribute entries:
            attrOut = sorted(list(set(attrOut)))
            
            return attrOut
 
        def write_inventory_output(inventorydf,outFile):          
            """
            Function that writes the data in inventorydf DataFrame into a CSV 
            or GeoJSON file based on the file name defined in the outFile.  
            
            Inputs: 
                df: DataFrame
                    A DataFrame containing building attributes. If this 
                    DataFrame contains columns for satellite_images and 
                    street_images, these columns are removed before the 
                    output file is created.
                outFile: string
                    String for the name of the output file, e.g., out.geojson. 
                    This function writes data only in CSV and GeoJSON formats.
                    If an output format different than these two formats is 
                    defined in the file extension, the function defaults to 
                    GeoJSON format.
            """   
            
            # Create a new table with does not list the image names 
            # corresponding to each building but includes building ID, latitude, 
            # and longitudecolumns added:
            dfout = inventorydf.copy(deep=True)
            dfout = dfout.drop(columns=['satellite_images', 'street_images'], 
                               errors='ignore')
    
          
            for index, row in inventorydf.iterrows():            
                dfout.loc[index, 'Footprint'] = ('{"type":"Feature","geometry":' + 
                '{"type":"Polygon","coordinates":[' + 
                f"""{row['Footprint']}""" + 
                ']},"properties":{}}')
                centroid = Polygon(row['Footprint']).centroid
                dfout.loc[index, 'Latitude'] = centroid.y
                dfout.loc[index, 'Longitude'] = centroid.x 
            
            # Rearrange the column order of dfout such that the Footprint field is 
            # the last:
            cols = [col for col in dfout.columns if col!='Footprint'] 
            new_cols = ['Latitude','Longitude'] + cols + ['Footprint']
            dfout = dfout[new_cols]
            
            # If the inventory is desired in CSV format, write dfout to a CSV:
            if '.csv' in outFile.lower():
                dfout.to_csv(outFile, index=True, index_label='id') 

            # Else write the inventory into a GeoJSON file:
            else:
                if '.geojson' not in outFile.lower():
                    print('Output format unimplemented!')
                    outFile = outFile.replace(outFile.split('.')[-1],'geojson')
                geojson = {'type':'FeatureCollection', 
                            'generated':str(datetime.now()),
                            'brails_version': version('BRAILS'),
                            "crs": {"type": "name", 
                                    "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84" }},
                            'units': {"length": lengthUnit},
                            'features':[]}
    
                attrs = dfout.columns.values.tolist()
                attrs.remove('Footprint')
                for index, row in dfout.iterrows(): 
                    feature = {'type':'Feature',
                               'properties':{},
                               'geometry':{'type':'Polygon',
                                           'coordinates':[]}}
                    fp = dfout.loc[index, 'Footprint'].split('"coordinates":')[-1]
                    fp = fp.replace('},"properties":{}}','')
                    feature['geometry']['coordinates'] = json.loads(fp)
                    feature['properties']['id'] = index
                    for attr in attrs:
                        feature['properties'][attr] = row[attr]
                    geojson['features'].append(feature)
    
                with open(outFile, 'w') as output_file:
                    json.dump(geojson, output_file, indent=2)
            
            print(f'\nFinal inventory data available in {outFile} in {os.getcwd()}')
        # Parse/correct the list of user requested building attributes:
        self.attributes = parse_attribute_input(attributes, self.enabledAttributes)      
        
        # Create a list of footprints for easier module calls:
        footprints = self.inventory['Footprint'].values.tolist()

        if useNSIBaseline:
            nsi = NSIParser()
            nsi.GenerateBldgInventory(footprints)
            attributes_process = set(self.attributes) - set(nsi.attributes)
            self.inventory = nsi.inventory.copy(deep=True)
            footprints = self.inventory['Footprint'].values.tolist()
        else:
            attributes_process = self.attributes.copy()
        
        # Download the images required for the requested attributes:
        image_handler = ImageHandler(self.apiKey)
        
        if 'roofshape' in attributes_process: #or 'roofcover' in attributes_process:
            image_handler.GetGoogleSatelliteImage(footprints)
            imsat = [im for im in image_handler.satellite_images if im is not None]
            self.inventory['satellite_images'] = image_handler.satellite_images
        
        streetAttributes = self.enabledAttributes[:]
        streetAttributes.remove('roofshape')
        #streetAttributes.remove('roofcover')
        if set.intersection(set(streetAttributes),set(attributes_process))!=set():
            image_handler.GetGoogleStreetImage(footprints)
            imstreet = [im for im in image_handler.street_images if im is not None]
            self.inventory['street_images'] = image_handler.street_images
        
        for attribute in attributes_process:
 
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
                self.inventory = write_to_dataframe(self.inventory,yearModel.results_df,'YearBuilt')
                self.inventory['YearBuilt'] = self.inventory['YearBuilt'].fillna('N/A')

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
                
            elif attribute=='numstories':
                # Initialize the floor detector object:
                storyModel = NFloorDetector()

                # Call the floor detector to determine number of floors of 
                # buildings in each image:
                storyModel.predict(imstreet)

                # Write the results to the inventory DataFrame:
                self.inventory = write_to_dataframe(self.inventory,
                                   [storyModel.system_dict['infer']['images'],
                                   storyModel.system_dict['infer']['predictions']],
                                   'NumberOfStories')
                self.inventory['NumberOfStories'] = self.inventory['NumberOfStories'].astype(dtype='Int64')
                
            elif attribute=='occupancy': 
                # Initialize the occupancy classifier object:
                occupancyModel = OccupancyClassifier()
                
                # Call the occupancy classifier to determine the occupancy 
                # class of each building:
                occupancyModel.predict(imstreet)
                
                # Write the results to the inventory DataFrame:
                occupancy = [[im for (im,_) in occupancyModel.preds],
                             [pred for (_,pred) in occupancyModel.preds]]
                self.inventory = write_to_dataframe(self.inventory,occupancy,
                                                    'OccupancyClass')
                self.inventory['OccupancyClass'] = self.inventory['OccupancyClass'].fillna('N/A')
            
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
                roofModel = RoofClassifier()

                # Call the roof type classifier to determine the roof type of
                # each building:                
                roofModel.predict(imsat)
                
                # Write the results to the inventory DataFrame:
                roofShape = [[im for (im,_) in roofModel.preds],
                             [pred for (_,pred) in roofModel.preds]]
                self.inventory = write_to_dataframe(self.inventory,roofShape,
                                                    'roofshape',
                                                    'satellite_images')
                
            elif attribute in ['buildingheight','roofeaveheight','roofpitch']:
                if 'facadeParserModel' not in locals():                
                    # Initialize the facade parser object:
                    facadeParserModel = FacadeParser()              
                    
                    # Call the facade parser to determine the requested 
                    # attribute for each building:
                    facadeParserModel.predict(image_handler)
                    
                self.inventory = write_to_dataframe(self.inventory,
                                                    [facadeParserModel.predictions['image'].to_list(),
                                                     facadeParserModel.predictions[attribute].to_list()],
                                                     attribute)    
        
            elif attribute=='constype':
                self.inventory['StructureType'] = ['W1' for ind in range(len(self.inventory.index))] 
        
        # Bring the attribute values to the desired length unit:
        if lengthUnit.lower()=='m':
            self.inventory['PlanArea'] = self.inventory['PlanArea'].apply(lambda x: x*0.0929)
            if 'buildingheight' in attributes_process:
                self.inventory['buildingheight'] = self.inventory['buildingheight'].apply(lambda x: x*0.3048)
            if 'roofeaveheight' in attributes_process:
                self.inventory['roofeaveheight'] = self.inventory['roofeaveheight'].apply(lambda x: x*0.3048)

        # Write the genereated inventory in outFile:
        write_inventory_output(self.inventory,outFile)   
            
        # Merge the DataFrame of predicted attributes with the DataFrame of
        # incomplete inventory and print the resulting table to the output file
        # titled IncompleteInventory.csv:  
        # dfout2merge = dfout.copy(deep=True) 
        # dfout2merge['fp_as_string'] = dfout2merge['Footprint'].apply(lambda x: "".join(str(x)))
            
        # dfout_incomp = self.incompleteInventory.copy(deep=True)
        # dfout_incomp['fp_as_string'] = dfout_incomp['Footprint'].apply(lambda x: "".join(str(x)))
        
        # dfout_incomp = pd.merge(left=dfout_incomp, 
        #                         right=dfout2merge.drop(columns=['Footprint'], errors='ignore'),
        #                         how='left', left_on=['fp_as_string','PlanArea'], 
        #                         right_on=['fp_as_string','PlanArea'],
        #                         sort=False)
        
        # dfout_incomp = dfout2merge.append(dfout_incomp[dfout_incomp.roofshape.isnull()])
        # dfout_incomp = dfout_incomp.reset_index(drop=True).drop(columns=['fp_as_string'], errors='ignore')

        # self.incompleteInventory = dfout_incomp.copy(deep=True)
        
        # dfout_incomp4print = dfout_incomp.copy(deep=True)         
        # for index, row in dfout_incomp.iterrows():            
        #     dfout_incomp4print.loc[index, 'Footprint'] = ('{"type":"Feature","geometry":' + 
        #     '{"type":"Polygon","coordinates":[' + 
        #     f"""{row['Footprint']}""" + 
        #     ']},"properties":{}}')
        #     centroid = Polygon(row['Footprint']).centroid
        #     dfout_incomp4print.loc[index, 'Latitude'] = centroid.y
        #     dfout_incomp4print.loc[index, 'Longitude'] = centroid.x 
        
        # cols = [col for col in dfout_incomp4print.columns if col!='Footprint'] 
        # new_cols = ['Latitude','Longitude'] + cols[:-2] + ['Footprint']
        # dfout_incomp4print = dfout_incomp4print[new_cols]
         
        # dfout_incomp4print.to_csv('IncompleteInventory.csv', index=True, index_label='id', na_rep='NA') 
        # print('Incomplete inventory data available in IncompleteInventory.csv')        