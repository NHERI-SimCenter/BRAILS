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
# Frank McKenna
#
# Last updated:
# 04-01-2024 

from datetime import datetime
from importlib.metadata import version
import json
import numpy as np
import os
import pandas as pd
import random
import sys
import warnings

#import brails.models as models
from brails.EnabledAttributes import BldgAttributes, BRAILStoR2D_BldgAttrMap
from brails.modules import (ChimneyDetector, FacadeParser, GarageDetector, 
                            NFloorDetector, RoofClassifier, 
                            OccupancyClassifier, RoofCoverClassifier, 
                            YearBuiltClassifier)
from brails.workflow.FootprintHandler import FootprintHandler
from brails.workflow.ImHandler import ImageHandler
from brails.workflow.NSIParser import NSIParser

# Set a custom warning message format:
warnings.formatwarning = lambda message, category, filename, lineno, line=None: \
                         f"{category.__name__}: {message}\n"
warnings.simplefilter('always')                         

class InventoryGenerator:
    def __init__(self, location='Berkeley California', fpSource:str='osm', 
                 baselineInv:str='', attrmap:str='', lengthUnit:str='ft',
                 outputFile:str=''):                
        
        # Define class variables:
        self.attributes = []
        self.apiKey = ''
        self.baselineInventory = pd.DataFrame()
        self.baselineInventoryOutputFile = outputFile
        self.enabledAttributes = BldgAttributes()
        self.fpSource = fpSource        
        self.inventory = pd.DataFrame()
        self.inventoryInventoryOutputFile = ''
        self.lengthUnit = lengthUnit
        self.location = location
        self.modelDir = 'tmp/models'
        self.nbldgs = 10
        self.randomSeed = 0
        self.workDir = 'tmp'
        
        # Get footprint and building attribute data:
        fpHandler = FootprintHandler()
        # If a baseline inventory is not specified:
        if not baselineInv:
            # If location entry is a string containing geojson or csv file 
            # extension and an attribute mapping file is specified:
            if ('csv' in location.lower() or 'geojson' in location.lower()) and attrmap:
                fpHandler.fetch_footprint_data(location,
                                               fpSource=fpSource,
                                               attrmap=attrmap,
                                               lengthUnit=lengthUnit)
            # If location entry is a string or tuple. String may contain geojson 
            # or csv file extension:
            else:
                fpHandler.fetch_footprint_data(location,fpSource=fpSource,lengthUnit=lengthUnit)
            
        # A baseline inventory is specified:    
        else:
            # If NSI is defined as the base inventory:
            if baselineInv.lower()=='nsi':
                fpHandler.fetch_footprint_data(location,fpSource=fpSource,lengthUnit=lengthUnit)
                
            # If a user-specified inventory is defined and is accompanied by an
            # attribute mapping file:
            elif attrmap:
                if ('csv' in location.lower() or 'geojson' in location.lower()):
                    self.fpSource = 'user-specified'
                    fpInp = location
                else:
                    fpInp = fpSource
                fpHandler.fetch_footprint_data(baselineInv,
                                               fpSource=fpInp,
                                               attrmap=attrmap,
                                               lengthUnit=lengthUnit)
                
                    
            # If a user-specified baseline inventory is defined but is not 
            # accompanied by an attribute mapping file, ignore the entered:
            else:
                warnings.warn('Missing attribute mapping file. ' +
                              'Ignoring the user-specified baseline inventory')
                fpHandler.fetch_footprint_data(location,fpSource=fpSource,lengthUnit=lengthUnit)              
 
        # Write geometry information from footprint data into a DataFrame:
        fpDataDf = pd.DataFrame(pd.Series(fpHandler.footprints,name='Footprint'))
        fpDataDf.Footprint = fpDataDf.Footprint.apply(str)
        lon = []; lat = []
        for pt in fpHandler.centroids:
            lon.append(pt.x)
            lat.append(pt.y)
        fpDataDf['Latitude'] = lat
        fpDataDf['Longitude'] = lon
        fpDataDf['PlanArea'] = fpHandler.attributes['fparea']
        
        # Get the dictionary that maps between BRAILS and R2D attribute names:
        brails2r2dmap = BRAILStoR2D_BldgAttrMap()
        
        # Get a list of the fphandler attributes that remain to be merged:
        attr2merge = list(fpHandler.attributes.keys())
        attr2merge.remove('fparea')
        
        if baselineInv.lower()=='nsi':
            # Read NSI data data corresponding to the extracted footprint polygons:
            nsiParser = NSIParser()
            nsiParser.GetNSIData(fpHandler.footprints,
                                 lengthUnit=lengthUnit)   
            
            # Write building attributes extracted from NSI into a DataFrame and merge with
            # the footprint data:
            nsiDataDf = pd.DataFrame(pd.Series(nsiParser.footprints,name='Footprint'))
            nsiDataDf.Footprint = nsiDataDf.Footprint.apply(str)
            for attr in nsiParser.attributes.keys():
                if attr not in ['fparea','lat','lon']:
                    nsiDataDf[brails2r2dmap[attr]] = nsiParser.attributes[attr]
                    
            self.baselineInventory = fpDataDf.merge(nsiDataDf, how='left', on=['Footprint']) 
            
            # Check if the attributes columns in fpHandler exists in inventoryBaseline:
            for attr in attr2merge:
                col = brails2r2dmap[attr]
                # If a column does not exist, add it to inventoryBaseline:
                if col not in self.baselineInventory:
                    self.baselineInventory[col] = pd.DataFrame(fpHandler.attributes[attr]).replace('NA',np.nan)
                # If the column exists, merge it so that it overrides the existing values 
                # on that column, unless a row in fpHandler contains a None value:  
                else:
                    df = pd.DataFrame(fpHandler.attributes[attr], columns=[attr])
                    df.replace('NA',np.nan)
                    self.baselineInventory[col] = df[attr].fillna(self.baselineInventory[col])
            self.baselineInventory.Footprint = fpDataDf.Footprint.apply(json.loads)
        else:
            for attr in attr2merge:
                col = brails2r2dmap[attr]
                fpDataDf[col] = pd.DataFrame(fpHandler.attributes[attr]).replace('NA',np.nan)
            fpDataDf.Footprint = fpDataDf.Footprint.apply(json.loads)
            self.baselineInventory = fpDataDf.copy(deep=True)
        
        # Write the genereated inventory in outFile:
        if outputFile:    
            self.__write_inventory_output(self.baselineInventory,
                                          lengthUnit=lengthUnit,
                                          outputFile=outputFile)

    def __write_inventory_output(self,inventorydf:pd.DataFrame,
                                 lengthUnit:str='ft',outputFile:str=''):          
        """
        Method that writes the data in an inventory DataFrame into a CSV 
        or GeoJSON file based on the file name defined in the outputFile. 
        lengthUnit is for informational purposes only and no unit conversion
        is performed in this function.
        
        Inputs: 
            df: DataFrame
                A DataFrame containing building attributes. If this 
                DataFrame contains columns for satellite_images and 
                street_images, these columns are removed before the 
                output file is created.
            outputFile: string
                String for the name of the output file, e.g., out.geojson. 
                This function writes data only in CSV and GeoJSON formats.
                If an output format different than these two formats is 
                defined in the file extension, the function defaults to 
                GeoJSON format.
            lengthUnit: string
                String defining the length units used to report the inventory
                data. Available options are ft or m.
        """   
        
        # Create a new table that does not list the image names 
        # corresponding to each building but includes building ID:
        dfout = inventorydf.copy(deep=True)
        
        imColumns2Remove = []
        if 'satellite_images' in dfout: imColumns2Remove.append('satellite_images')
        if 'street_images' in dfout: imColumns2Remove.append('street_images')
        
        if imColumns2Remove:
            dfout = dfout.drop(columns=imColumns2Remove,errors='ignore')
        
        # Rewrite the footprints in a format compatible with R2D:
        for index, row in inventorydf.iterrows():            
            dfout.loc[index, 'Footprint'] = ('{"type":"Feature","geometry":' + 
            '{"type":"Polygon","coordinates":[' + 
            f"""{row['Footprint']}""" + 
            ']},"properties":{}}')
        
        # Rearrange the column order of dfout such that the Footprint field is 
        # the last:
        cols = [col for col in dfout.columns if col not in ['Footprint','Latitude','Longitude']] 
        new_cols = ['Latitude','Longitude'] + cols + ['Footprint']
        dfout = dfout[new_cols]
        
        # If the inventory is desired in CSV format, write dfout to a CSV:
        if '.csv' in outputFile.lower():
            dfout.to_csv(outputFile, index=True, index_label='id') 
    
        # Else write the inventory into a GeoJSON file:
        else:
            # If the extension of outputFile is not CSV or GeoJSON, write the 
            # inventory output in GeoJSON format:
            if '.geojson' not in outputFile.lower():
                warnings.warn('Output format unimplemented! '
                              'Writing the inventory output in GeoJSON format.')
                outputFile = outputFile.replace(outputFile.split('.')[-1],'geojson')
            
            # Define GeoJSON file dictionary including basic metadata:
            geojson = {'type':'FeatureCollection', 
                        'generated':str(datetime.now()),
                        'brails_version': version('BRAILS'),
                        "crs": {"type": "name", 
                                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84" }},
                        'units': {"length": lengthUnit},
                        'features':[]}
            
            # Get the list of attributes available in dfout:
            attrs = dfout.columns.values.tolist()
            attrs.remove('Footprint')
            
            # Run through each row of dfout and write the extracted data into
            # the GeoJSON dictionary:
            for index, row in dfout.iterrows(): 
                feature = {'type':'Feature',
                           'properties':{},
                           'geometry':{'type':'Polygon',
                                       'coordinates':[]}}
                fp = dfout.loc[index, 'Footprint'].split('"coordinates":')[-1]
                fp = fp.replace('},"properties":{}}','')
                feature['geometry']['coordinates'] = json.loads(fp)
                feature['properties']['id'] = index
                # Read attribute values on the current row, filling null values
                # as NA:
                for attr in attrs:
                    res = row[attr]
                    if pd.isnull(res):
                        feature['properties'][attr] = 'NA'
                    else:
                        feature['properties'][attr] = res
                geojson['features'].append(feature)
            
            # Write the created GeoJSON dictionary into a GeoJSON file:
            with open(outputFile, 'w') as output_file:
                json.dump(geojson, output_file, indent=2)
        
        print(f'\nInventory data available in {outputFile} in {os.getcwd()}')        
  
    def enabled_attributes(self)->None:
        """
        Method that prints the building attributes that can be obtained by 
        BRAILS
        """   
        
        print('Here is the list of attributes currently enabled in ' +
              'InventoryGenerator:\n')
        for attribute in self.enabledAttributes:
            print(f'       {attribute}')
        print('\nIf you want to get all of these attributes when you run '
              "InventoryGenerator.generate, simply set attributes='all'")

    def generate(self,attributes=['numstories','occupancy','roofshape'],
                 GoogleAPIKey:str='',nbldgs=10,outputFile:str='', 
                 randomSelection:int=0):

        """
        Method that generates the building inventory for the location entered
        when InventoryGenerator class is initialized.
        
        Inputs: 
            attributes: list or string
                Argument containing a description of the attributes requested
                as a part of inventory generation. If set to 'all', all 
                attributes in self.enabledAttributes are included. If set to
                'hazuseq', numstories, erabuilt, repaircost, constype, and 
                occupancy are included. If defined as a list, all valid 
                attribute labels in the list are included in the inventory 
                output. 
            GoogleAPIKey: string
                String containing a valid Google API key that has Street View 
                Static API enabled
            nbldgs: int or string
                If set to 'all' runs BRAILS models on all detected buildings. 
                If set to an integer value, runs BRAILS models on only the
                number of building defined by the set integer value
            randomSelection: int
                Random seed for arbitrary selection of nbldgs from the detected
                buildings
        """  
        
        def write_to_dataframe(df:pd.DataFrame,predictions,column:str,
                               imtype:str='street_images')->pd.DataFrame:
            """
            Function that writes predictions of a model by going through the 
            DataFrame, df, and finding the image name matches between the 
            list of images corresponding to the predictions, i.e., imagelist,
            then assigning the predictions to df. The function only searches 
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
                column: string 
                    Name of the column where the predictions will be written
            
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
        
        def parse_attribute_input(attrIn:list,attrEnabled:list)->list:             
            """
            Function that pre-processes the user attribute entries to 
            InventoryGenerator().generate method such that pre-defined 
            attribute sets are correctly parsed and incorrect entries are 
            removed.
            
            Inputs: 
                attrIn: list
                    A list of string descriptions for the user-requested 
                    building attributes
                attrEnabled: list 
                    A list of string descriptions for the building attributes
                    currently enabled in InventoryGenerator().generate method
            
            Output: A list of string descriptions for the user-requested 
                    building attributes that were determined to be valid for use
                    with InventoryGenerator().generate method
            """       
            
            # If all attributes are requested:
            if isinstance(attrIn,str) and attrIn=='all':    
                attrOut = attrEnabled[:]
            # If only the attributes required for HAZUS seismic analysis are 
            # requested:
            elif isinstance(attrIn,str) and attrIn=='hazuseq':
                attrOut = ['numstories', 'erabuilt', 'repaircost', 'constype',
                           'occupancy']
            # If a custom list of attributes is requested:
            elif isinstance(attrIn,list):
                attrOut = [attribute.lower() for attribute in attrIn]
                # Get the attributes that are not in enabled attributes and 
                # remove them from the requested list of attributes:
                ignore_entries = []
                for attribute in attrOut:
                    if attribute not in attrEnabled:
                        ignore_entries.append(attribute)
                        attrOut.remove(attribute)
                # Display the ignored (removed) attribute entries:
                if len(ignore_entries)==1:
                    print('An entry in attributes is not enabled.'
                          f'\nIgnoring entry: {ignore_entries[0]}') 
                elif len(ignore_entries)>1:
                    print('Several entries in attributes are not enabled.' 
                          '\nIgnoring entries: ' + ', '.join(ignore_entries))
            
                # If no valid attribute is detected, stop code execution:
                if len(attrOut)==0:
                    sys.exit('Entered list of attributes does not contain a ' + 
                             'correct attribute entry. Attribute entries enabled' +
                             ' are: ' + ', '.join(attrEnabled))
            
                # Remove duplicate attribute entries:
                attrOut = sorted(list(set(attrOut)))
                
            else:
                warnings.warn('Incorrect attributes entry. Supported attributes' +
                              ' entries are a list containing the string labels ' +
                              " for requested attributes, 'all' or 'hazuseq'. " +
                              ' Running BRAILS for roof shape detection only...')
                attrOut=['roofshape']
            return attrOut

        self.apiKey = GoogleAPIKey  
        self.nbldgs = nbldgs
        self.inventoryInventoryOutputFile = outputFile
        
        # Initialize the image handler class  to check if the provided API key
        # is valid:
        image_handler = ImageHandler(self.apiKey)
        
        if nbldgs!='all':
            # If randomSelection is set to a non-zero value, randomly select 
            # nbldgs from the inventory data with the specified seed. Else,
            # pull nbldgs at random with an arbitrary seed:  
            if randomSelection<0:
                randomSelection = random.randint(0,1e8)
                print(f'Randomly selected {nbldgs} buildings')
            else:
                print(f'Randomly selected {nbldgs} buildings using the seed {randomSelection}')
            self.inventory = self.baselineInventory.sample(n=nbldgs, 
                                                           random_state=randomSelection)
            self.randomSelection = randomSelection
        else:
            print(f'Selected all {len(self.baselineInventory.index)} buildings')
            self.inventory = self.baselineInventory.copy(deep=True)
        
        # Parse/correct the list of user requested building attributes:
        self.attributes = parse_attribute_input(attributes, self.enabledAttributes)      
        
        # Create a list of footprints for easier module calls:
        footprints = self.inventory['Footprint'].values.tolist()

        # Get the list of attributes that will be processed:
        attributes_process = [attr for attr in self.attributes if attr not in 
                              ['occupancy','constype','repaircost','roofcover']]
        
        # Download the images required for the requested attributes:
        if 'roofshape' or 'roofcover' in attributes_process:
            image_handler.GetGoogleSatelliteImage(footprints)
            imsat = [im for im in image_handler.satellite_images if im is not None]
            self.inventory['satellite_images'] = image_handler.satellite_images
        
        streetAttributes = self.enabledAttributes[:]
        streetAttributes.remove('roofshape'); streetAttributes.remove('roofcover')
        if set.intersection(set(streetAttributes),set(attributes_process))!=set():
            image_handler.GetGoogleStreetImage(footprints)
            imstreet = [im for im in image_handler.street_images if im is not None]
            self.inventory['street_images'] = image_handler.street_images

        # Run the obtained images through BRAILS computer vision models:
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
                
            elif attribute in ['buildingheight','roofeaveheight','roofpitch','winarea']:
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
        if self.lengthUnit.lower()=='m':
            self.inventory['PlanArea'] = self.inventory['PlanArea'].apply(lambda x: x*0.0929)
            if 'buildingheight' in attributes_process:
                self.inventory['buildingheight'] = self.inventory['buildingheight'].apply(lambda x: x*0.3048)
            if 'roofeaveheight' in attributes_process:
                self.inventory['roofeaveheight'] = self.inventory['roofeaveheight'].apply(lambda x: x*0.3048)

        # Write the genereated inventory in outFile:
        if outputFile:
            self.__write_inventory_output(self.inventory,
                                          lengthUnit=self.lengthUnit,
                                          outputFile=outputFile)   

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