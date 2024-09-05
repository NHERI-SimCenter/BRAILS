"""Class object for creating inventories of buildings."""
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
# 09-04-2024

import os
from datetime import datetime
from importlib.metadata import version
import json
import random
import sys
import warnings
from typing import List, Optional, Union
import numpy as np
import pandas as pd
# import brails.models as models
from brails.EnabledAttributes import BldgAttributes, BRAILStoR2D_BldgAttrMap
from brails.modules import (ChimneyDetector, FacadeParser, GarageDetector,
                            NFloorDetector, RoofClassifier,
                            OccupancyClassifier, RoofCoverClassifier,
                            YearBuiltClassifier)
from brails.workflow.FootprintHandler import FootprintHandler
from brails.workflow.ImHandler import ImageHandler
from brails.workflow.NSIParser import NSIParser

# Set a custom warning message format:
warnings.formatwarning = lambda message, category, filename, lineno, \
    line=None: f"{category.__name__}: {message}\n"
warnings.simplefilter('always', UserWarning)
warnings.filterwarnings("ignore",
                        message='Implicit dimension choice for softmax has '
                        'been deprecated.*')


class InventoryGenerator:
    """
    A class for generating and managing building inventory data.

    A class for generating and managing building inventory data. The class
    supports getting building footprint data, generating building attributes
    using various AI models, and saving the results in different formats
    (CSV or GeoJSON). It integrates with external APIs and services to gather
    and process data, providing functionality for analyzing building
    attributes and managing inventory.

    Attributes__
        location (str): The geographical location for fetching building data.
            Defaults to 'Berkeley California'.
        fpSource (str): The source of the footprint data. Defaults to 'osm'.
        baselineInv (str): Path to the baseline inventory file or a special
            identifier (e.g., 'nsi'). Defaults to ''.
        attrmap (str): Path to the attribute mapping file. Defaults to ''.
        lengthUnit (str): The unit of length for reporting inventory data.
            Defaults to 'ft'. Can be 'ft' or 'm'.
        outputFile (str): The name of the file to save the output.
            Defaults to ''.
        attributes (list): List of building attributes to be processed.
        apiKey (str): Google API key for accessing image metadata
        baselineInventory (pd.DataFrame): DataFrame holding the baseline
            inventory data.
        baselineInventoryOutputFile (str): File name for saving the baseline
            inventory data.
        enabledAttributes (BldgAttributes): An instance of the BldgAttributes
            class holding enabled attributes.
        fpSource (str): Source of the footprint data.
        inventory (pd.DataFrame): DataFrame holding the generated inventory
            data.
        inventoryInventoryOutputFile (str): File name for saving the generated
            inventory data.
        lengthUnit (str): Unit of length for reporting inventory data.
        location (str): Geographical location for fetching data.
        modelDir (str): Directory path for storing models. Defaults to
            'tmp/models'.
        nbldgs (int): Number of buildings to process. Defaults to 10.
        randomSeed (int): Seed for random selection of buildings. Defaults
            to 0.
        workDir (str): Directory for temporary files. Defaults to 'tmp'.

    Methods__
        __write_inventory_output: Writes the inventory data to a specified
            file in CSV or GeoJSON format.
        enabled_attributes: Prints the list of currently enabled attributes in
            InventoryGenerator.
        generate: Generates building inventory data based on requested
            attributes and processes it using various models.

    Usage__
        Create an instance of InventoryGenerator with the desired
            configuration.
        Use the `generate` method to process building data and generate the
            inventory,
        specifying attributes, number of buildings, and output file options.
        Use `enabled_attributes` to list all available attributes for inventory
            generation.

    Example__
        >>> generator = InventoryGenerator(location='San Francisco',
                                           lengthUnit='m',
                                           outputFile='output.geojson')
        >>> generator.generate(attributes=['roofshape', 'numstories'],
                               nbldgs=20,
                               outputFile='inventory.geojson')
        >>> generator.enabled_attributes()
    """

    def __init__(self, location='Berkeley California',
                 fpSource: str = 'osm',
                 baselineInv: str = '',
                 attrmap: str = '',
                 lengthUnit: str = 'ft',
                 outputFile: str = ''):

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
        fp_handler = FootprintHandler()

        # If a baseline inventory is not specified:
        if not baselineInv:
            # If location entry is a string containing GeoJSON file extension
            # and an attribute mapping file is specified:
            if (location is str and 'geojson' in location.lower()) and attrmap:
                fp_handler.fetch_footprint_data(location,
                                                fpSource=fpSource,
                                                attrmap=attrmap,
                                                lengthUnit=lengthUnit)
            # If location entry is a string or tuple. String may contain
            # geojson or csv file extension:
            else:
                fp_handler.fetch_footprint_data(
                    location, fpSource=fpSource, lengthUnit=lengthUnit)

        # A baseline inventory is specified:
        else:
            # If NSI is defined as the base inventory:
            if baselineInv.lower() == 'nsi':
                fp_handler.fetch_footprint_data(
                    location, fpSource=fpSource, lengthUnit=lengthUnit)

            # If a user-specified inventory is defined and is accompanied by an
            # attribute mapping file:
            elif attrmap:
                if location is str and ('csv' in location.lower() or
                                        'geojson' in location.lower()):
                    self.fpSource = 'user-specified'
                    fpInp = location
                else:
                    fpInp = fpSource
                fp_handler.fetch_footprint_data(baselineInv,
                                                fpSource=fpInp,
                                                attrmap=attrmap,
                                                lengthUnit=lengthUnit)

            # If a user-specified baseline inventory is defined but is not
            # accompanied by an attribute mapping file, ignore the entered:
            else:
                warnings.warn('Missing attribute mapping file. ' +
                              'Ignoring the user-specified baseline inventory',
                              UserWarning)
                fp_handler.fetch_footprint_data(
                    location, fpSource=fpSource, lengthUnit=lengthUnit)

        # Write geometry information from footprint data into a DataFrame:
        fp_data_df = pd.DataFrame(
            pd.Series(fp_handler.footprints, name='Footprint'))
        fp_data_df.Footprint = fp_data_df.Footprint.apply(str)
        lon = []
        lat = []
        for pt in fp_handler.centroids:
            lon.append(pt.x)
            lat.append(pt.y)
        fp_data_df['Latitude'] = lat
        fp_data_df['Longitude'] = lon
        fp_data_df['PlanArea'] = fp_handler.attributes['fparea']

        # Get the dictionary that maps between BRAILS and R2D attribute names:
        brails2r2dmap = BRAILStoR2D_BldgAttrMap()

        # Get a list of the fp_handler attributes that remain to be merged:
        attr2merge = list(fp_handler.attributes.keys())
        attr2merge.remove('fparea')

        if baselineInv.lower() == 'nsi':
            # Read NSI data data corresponding to the extracted footprint
            # polygons:
            nsi_parser = NSIParser()
            nsi_parser.GetNSIData(fp_handler.footprints,
                                  lengthUnit=lengthUnit)

            # Write building attributes extracted from NSI into a DataFrame
            # and merge with the footprint data:
            nsi_data_df = pd.DataFrame(
                pd.Series(nsi_parser.footprints, name='Footprint'))
            nsi_data_df.Footprint = nsi_data_df.Footprint.apply(str)
            for attr in nsi_parser.attributes.keys():
                if attr not in ['fparea', 'lat', 'lon']:
                    nsi_data_df[brails2r2dmap[attr]
                                ] = nsi_parser.attributes[attr]

            self.baselineInventory = fp_data_df.merge(
                nsi_data_df, how='left', on=['Footprint'])

            # Check if the attributes columns in fp_handler exists in
            # inventoryBaseline:
            for attr in attr2merge:
                col = brails2r2dmap[attr]
                # If a column does not exist, add it to inventoryBaseline:
                if col not in self.baselineInventory:
                    self.baselineInventory[col] = pd.DataFrame(
                        fp_handler.attributes[attr]).replace('NA', np.nan)
                # If the column exists, merge it so that it overrides the
                # existing values on that column, unless a row in fp_handler
                # contains a None value:
                else:
                    df = pd.DataFrame(
                        fp_handler.attributes[attr], columns=[attr])
                    df.replace('NA', np.nan)
                    self.baselineInventory[col] = df[attr].fillna(
                        self.baselineInventory[col])
            self.baselineInventory.Footprint = fp_data_df.Footprint.apply(
                json.loads)
        else:
            for attr in attr2merge:
                col = brails2r2dmap[attr]
                fp_data_df[col] = pd.DataFrame(
                    fp_handler.attributes[attr]).replace('NA', np.nan)
            fp_data_df.Footprint = fp_data_df.Footprint.apply(json.loads)
            self.baselineInventory = fp_data_df.copy(deep=True)

        # Write the generated inventory in outFile:
        if outputFile:
            self.__write_inventory_output(self.baselineInventory,
                                          lengthUnit=lengthUnit,
                                          outputFile=outputFile)

    def __write_inventory_output(self,
                                 inventorydf: pd.DataFrame,
                                 lengthUnit: str = 'ft',
                                 outputFile: str = ''):
        """
        Write inventory data from a DataFrame to a CSV or GeoJSON file.

        This method saves the data in the `inventorydf` DataFrame to a file.
        The format of the file is determined by the file extension in
        `output_file`. If the extension is `.csv`, the data is saved in CSV
        format; if it is `.geojson`, the data is saved in GeoJSON format. If
        the file extension is neither `.csv` nor `.geojson`, the data is saved
        in GeoJSON format by default. The `length_unit` parameter is for
        informational purposes only and does not affect the data saved.

        Parameters__
        - inventorydf (pd.DataFrame):
            A DataFrame containing the inventory data. If this DataFrame
            includes columns named `satellite_images` or `street_images`,
            these columns are removed before saving the file.

        - length_unit (str, optional):
            The length unit used to report the inventory data. This parameter
            is not used for unit conversion but is included for informational
            purposes. Options are 'ft' (feet) or 'm' (meters). Default is 'ft'.

        - output_file (str):
            The path and name of the output file, including the file extension.
            Supported extensions are '.csv' for CSV format and '.geojson' for
            GeoJSON format. If an unsupported extension is provided, the
            default format is GeoJSON.

        Example__
        >>> df = pd.DataFrame({'column1': [1, 2], 'column2': [3, 4]})
        >>> self.__write_inventory_output(df,
                                          length_unit='m',
                                          output_file='output.geojson')
        """
        # Create a new table that does not list the image names
        # corresponding to each building but includes building ID:
        dfout = inventorydf.copy(deep=True)

        imColumns2Remove = []
        if 'satellite_images' in dfout:
            imColumns2Remove.append('satellite_images')
        if 'street_images' in dfout:
            imColumns2Remove.append('street_images')

        if imColumns2Remove:
            dfout = dfout.drop(columns=imColumns2Remove, errors='ignore')

        # Rewrite the footprints in a format compatible with R2D:
        for index, row in inventorydf.iterrows():
            dfout.loc[index, 'Footprint'] = ('{"type":"Feature","geometry":'
                                             '{"type":"Polygon","coordinates"'
                                             f""":[{row['Footprint']}""" +
                                             ']},"properties":{}}')

        # Rearrange the column order of dfout such that the Footprint field is
        # the last:
        cols = [col for col in dfout.columns if col not in [
            'Footprint', 'Latitude', 'Longitude']]
        new_cols = ['Latitude', 'Longitude'] + cols + ['Footprint']
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
                              'Writing the inventory output in GeoJSON format',
                              UserWarning)
                outputFile = outputFile.replace(
                    outputFile.split('.')[-1], 'geojson')

            # Define GeoJSON file dictionary including basic metadata:
            geojson = {'type': 'FeatureCollection',
                       'generated': str(datetime.now()),
                       'brails_version': version('BRAILS'),
                       "crs": {"type": "name",
                               "properties": {
                                   "name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                       'units': {"length": lengthUnit},
                       'features': []}

            # Get the list of attributes available in dfout:
            attrs = dfout.columns.values.tolist()
            attrs.remove('Footprint')

            # Run through each row of dfout and write the extracted data into
            # the GeoJSON dictionary:
            for index, row in dfout.iterrows():
                feature = {'type': 'Feature',
                           'properties': {},
                           'geometry': {'type': 'Polygon',
                                        'coordinates': []}}
                fp = dfout.loc[index, 'Footprint'].split('"coordinates":')[-1]
                fp = fp.replace('},"properties":{}}', '')
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
            with open(outputFile, 'w', encoding='utf-8') as output_file:
                json.dump(geojson, output_file, indent=2)

        print(f'\nInventory data available in {outputFile} in {os.getcwd()}')

    def enabled_attributes(self) -> None:
        """
        Print the building attributes that can be obtained by BRAILS.

        This method outputs to the console the attributes that are currently
        enabled and available for use within the InventoryGenerator. It lists
        each enabled attribute and provides a brief instruction on how to
        include all attributes when generating inventory data.

        Example__
        >>> obj.enabled_attributes()
        Here is the list of attributes currently enabled in InventoryGenerator:
               attribute1
               attribute2
               ...

        Note__
        To include all available attributes in the generated inventory data,
        set the `attributes` parameter to 'all' when calling the
        `InventoryGenerator.generate` method.
        """
        print('Here is the list of attributes currently enabled in ' +
              'InventoryGenerator:\n')
        for attribute in self.enabledAttributes:
            print(f'       {attribute}')
        print('\nIf you want to get all of these attributes when you run '
              "InventoryGenerator.generate, simply set attributes='all'")

    def generate(self,
                 attributes: Optional[Union[List[str], str]] = '',
                 GoogleAPIKey: str = '',
                 nbldgs: Union[int, str] = 10,
                 outputFile: str = '',
                 randomSelection: int = 0):
        """
        Create the building inventory for the location specified.

        Inputs__
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
        if attributes == '':
            attributes = ['numstories', 'occupancy', 'roofshape']

        def merge_results2inventory(inventory: pd.DataFrame,
                                    predictions: pd.DataFrame,
                                    attrname: str,
                                    imtype: str = 'street_images'
                                    ) -> pd.DataFrame:
            """
            Merge predictions of a model to an inventory DataFrame.

            Function that merges predictions of a model to an inventory by
            finding the image name matches between the inventory and
            predictions. The function only searches the column named imtype for
            image names

            Inputs__
                inventory: DataFrame
                    A DataFrame containing at least a column of names of source
                    images named either 'satellite_images' or 'street_images',
                    depending on the value of imtype
                predictions: DataFrame
                    A DataFrame containing two columns for 1) image names
                    and 2) model predictions. Columns of this DataFrame are
                    titled with the strings contained in imtype and attrname
                imtype: string
                    'satellite_images' or 'street_images' depending on the
                    type of source images
                column: string
                    Name of the column where the predictions are stored and
                    will be written

            Output__ inventory expanded such that it includes a new or updated
                    column for the predictions
            """
            # If inventory does not contain a column for attrname:
            if attrname not in inventory:
                # Merge the prediction results to inventory DataFrame:
                inventory = inventory.merge(predictions,
                                            how='left',
                                            on=[imtype])

            else:
                # Merge the prediction results to inventory DataFrame:
                mergedInv = inventory.merge(predictions,
                                            how='left',
                                            on=[imtype])

                # Augment prediction results by merging values from the
                # first attribute column only if the corresponding value
                # in the second (predictions) column is nan:
                mergedInv[attrname] = mergedInv[attrname+'_y'].fillna(
                    mergedInv[attrname+'_x'])

                # Remove the two attribute columns, keeping the combined
                # column, and assign the resulting DataFrame to inventory:
                inventory = mergedInv.drop(columns=[attrname+'_x',
                                                    attrname+'_y'],
                                           errors='ignore')
            return inventory

        def parse_attribute_input(attr_in: list[str],
                                  attr_enabled: list[str]
                                  ) -> list[str]:
            """
            Process user-requested attribute entries for BRAILS use.

            This function takes a list of user-requested attributes and
            compares them against a list of attributes that are currently
            enabled in the `InventoryGenerator.generate` method. It returns a
            list of valid attributes that are recognized and enabled, removing
            any invalid or unsupported entries.

            Args__
                attr_in (list of str): A list of attribute names requested by
                    the user.
                attr_enabled (list of str): A list of attribute names that are
                    currently enabled and available in the
                    `InventoryGenerator.generate` method.

            Returns__
                list of str: A list of valid attribute names from `attr_in`
                    that are also present in `attr_enabled`.

            Example__
                >>> parse_attribute_input(['height', 'width', 'color'],
                                          ['height', 'color'])
                ['height', 'color']

            Note__
                Attributes in `attr_in` that are not present in `attr_enabled`
                will be filtered out. This ensures that only valid attributes
                are processed.
            """
            # If all attributes are requested:
            if isinstance(attr_in, str) and attr_in == 'all':
                attr_out = attr_enabled[:]
            # If only the attributes required for HAZUS seismic analysis are
            # requested:
            elif isinstance(attr_in, str) and attr_in == 'hazuseq':
                attr_out = ['numstories', 'erabuilt', 'repaircost', 'constype',
                            'occupancy']
            # If a custom list of attributes is requested:
            elif isinstance(attr_in, list):
                attr_out = [attribute.lower() for attribute in attr_in]
                # Get the attributes that are not in enabled attributes and
                # remove them from the requested list of attributes:
                ignore_entries = []
                for attribute in attr_out:
                    if attribute not in attr_enabled:
                        ignore_entries.append(attribute)
                        attr_out.remove(attribute)
                # Display the ignored (removed) attribute entries:
                if len(ignore_entries) == 1:
                    print('An entry in attributes is not enabled.'
                          f'\nIgnoring entry: {ignore_entries[0]}')
                elif len(ignore_entries) > 1:
                    print('Several entries in attributes are not enabled.'
                          '\nIgnoring entries: ' + ', '.join(ignore_entries))

                # If no valid attribute is detected, stop code execution:
                if len(attr_out) == 0:
                    sys.exit('Entered list of attributes does not contain a '
                             'correct attribute entry. Attribute entries '
                             'enabled are: ' + ', '.join(attr_enabled))

                # Remove duplicate attribute entries:
                attr_out = sorted(list(set(attr_out)))

            else:
                warnings.warn(
                    'Incorrect attributes entry. Supported attributes'
                    ' entries are a list containing the string labels '
                    " for requested attributes, 'all' or 'hazuseq'. "
                    ' Running BRAILS for roof shape detection only...',
                    UserWarning)
                attr_out = ['roofshape']
            return attr_out

        self.apiKey = GoogleAPIKey
        self.nbldgs = nbldgs
        self.inventoryInventoryOutputFile = outputFile

        # Initialize the image handler class  to check if the provided API key
        # is valid:
        image_handler = ImageHandler(self.apiKey)

        if nbldgs != 'all':
            # If randomSelection is set to a non-zero value, randomly select
            # nbldgs from the inventory data with the specified seed. Else,
            # pull nbldgs at random with an arbitrary seed:
            if randomSelection < 0:
                randomSelection = random.randint(0, 1e8)
                print(f'\nRandomly selected {nbldgs} buildings\n')
            else:
                print(f'\nRandomly selected {nbldgs} buildings using the seed '
                      f'{randomSelection}\n')
            self.inventory = self.baselineInventory.sample(
                n=nbldgs, random_state=randomSelection)
            self.randomSeed = randomSelection
        else:
            print(
                f'\nSelected all {len(self.baselineInventory.index)} '
                'buildings\n')
            self.inventory = self.baselineInventory.copy(deep=True)

        # Parse/correct the list of user requested building attributes:
        self.attributes = parse_attribute_input(
            attributes, self.enabledAttributes)

        # Create a list of footprints for easier module calls:
        footprints = self.inventory['Footprint'].values.tolist()

        # Get the list of attributes that will be processed:
        attributes_process = [attr for attr in self.attributes if attr not in
                              ['occupancy', 'constype', 'repaircost',
                               'roofcover']]

        # Download the images required for the requested attributes:
        if 'roofshape' in attributes_process or \
                'roofcover' in attributes_process:
            image_handler.GetGoogleSatelliteImage(footprints)
            imsat = [im for im in image_handler.satellite_images
                     if im is not None]
            self.inventory['satellite_images'] = image_handler.satellite_images

        street_attributes = self.enabledAttributes[:]
        street_attributes.remove('roofshape')
        street_attributes.remove('roofcover')
        if set.intersection(set(street_attributes),
                            set(attributes_process)) != set():
            image_handler.GetGoogleStreetImage(footprints)
            imstreet = [
                im for im in image_handler.street_images if im is not None]
            self.inventory['street_images'] = image_handler.street_images
        print('')

        # Get the dictionary that maps between BRAILS and R2D attribute names:
        brails2r2dmap = BRAILStoR2D_BldgAttrMap()

        # Run the obtained images through BRAILS computer vision models:
        for attribute in attributes_process:
            # Define the column name:
            colname = brails2r2dmap[attribute]

            if attribute == 'chimney':
                # Initialize the chimney detector object:
                chimneyModel = ChimneyDetector()

                # Call the chimney detector to determine the existence of
                # chimneys:
                chimneyModel.predict(imstreet)

                # Write the results to the inventory DataFrame:
                predResults = pd.DataFrame(
                    list(zip(chimneyModel.system_dict['infer']['images'],
                             chimneyModel.system_dict['infer']['predictions'])
                         ), columns=['street_images', colname])
                self.inventory = merge_results2inventory(self.inventory,
                                                         predResults,
                                                         colname)
                self.inventory[colname] = self.inventory[colname].astype(
                    dtype="boolean")

            elif attribute == 'erabuilt':
                # Initialize the era of construction classifier:
                yearModel = YearBuiltClassifier()

                # Call the classifier to determine the era of construction for
                # each building:
                yearModel.predict(imstreet)

                # Write the results to the inventory DataFrame:
                predResults = yearModel.results_df.copy(deep=True)
                predResults = predResults.rename(
                    columns={'prediction': colname,
                             'image': 'street_images'}).drop(
                                 columns=['probability'])
                self.inventory = merge_results2inventory(self.inventory,
                                                         predResults,
                                                         colname)

            elif attribute == 'garage':
                # Initialize the garage detector object:
                garageModel = GarageDetector()

                # Call the garage detector to determine the existence of
                # garages:
                garageModel.predict(imstreet)

                # Write the results to the inventory DataFrame:
                predResults = pd.DataFrame(
                    list(zip(garageModel.system_dict['infer']['images'],
                             garageModel.system_dict['infer']['predictions'])),
                    columns=['street_images', colname])
                self.inventory = merge_results2inventory(self.inventory,
                                                         predResults,
                                                         colname)
                self.inventory[colname] = self.inventory[colname].astype(
                    dtype="boolean")

            elif attribute == 'numstories':
                # Initialize the floor detector object:
                storyModel = NFloorDetector()

                # Call the floor detector to determine number of floors of
                # buildings in each image:
                storyModel.predict(imstreet)

                # Write the results to the inventory DataFrame:
                predResults = pd.DataFrame(
                    list(zip(storyModel.system_dict['infer']['images'],
                             storyModel.system_dict['infer']['predictions'])),
                    columns=['street_images', colname])
                self.inventory = merge_results2inventory(self.inventory,
                                                         predResults,
                                                         colname)
                self.inventory[colname] = self.inventory[colname].astype(
                    dtype='Int64')

            elif attribute == 'occupancy':
                # Initialize the occupancy classifier object:
                occupancyModel = OccupancyClassifier()

                # Call the occupancy classifier to determine the occupancy
                # class of each building:
                occupancyModel.predict(imstreet)

                # Write the prediction results to a DataFrame:
                predResults = pd.DataFrame(occupancyModel.preds, columns=[
                                           'street_images', colname])
                self.inventory = merge_results2inventory(self.inventory,
                                                         predResults,
                                                         colname)

            elif attribute == 'roofcover':
                # Initialize the roof cover classifier object:
                roofCoverModel = RoofCoverClassifier()

                # Call the roof cover classifier to classify roof cover type of
                # each building:
                roofCoverModel.predict(imsat)

                # Write the prediction results to a DataFrame:
                predResults = pd.DataFrame(roofCoverModel.preds, columns=[
                                           'satellite_images', colname])
                self.inventory = merge_results2inventory(self.inventory,
                                                         predResults,
                                                         colname,
                                                         'satellite_images')

            elif attribute == 'roofshape':
                # Initialize the roof type classifier object:
                roofModel = RoofClassifier()

                # Call the roof type classifier to determine the roof type of
                # each building:
                roofModel.predict(imsat)

                # Write the prediction results to a DataFrame:
                predResults = pd.DataFrame(roofModel.preds, columns=[
                                           'satellite_images', colname])
                self.inventory = merge_results2inventory(self.inventory,
                                                         predResults,
                                                         colname,
                                                         'satellite_images')

            elif attribute in ['buildingheight', 'roofeaveheight',
                               'roofpitch', 'winarea']:
                if 'facadeParserModel' not in locals():
                    # Initialize the facade parser object:
                    facadeParserModel = FacadeParser()

                    # Call the facade parser to determine the requested
                    # attribute for each building:
                    facadeParserModel.predict(image_handler)

                # Get the relevant subset of the prediction results:
                predResults = facadeParserModel.predictions[[
                    'image', attribute]]

                # Bring the results to the desired length unit:
                if self.lengthUnit.lower() == 'm' and (attribute in [
                        'buildingheight', 'roofeaveheight']):
                    predResults[colname] = predResults[colname]*0.3048
                predResults = predResults.rename(columns={
                    'image': 'street_images', attribute: colname})

                # Write the results to the inventory DataFrame:
                self.inventory = merge_results2inventory(self.inventory,
                                                         predResults,
                                                         colname)

        # Write the generated inventory in outFile:
        if outputFile:
            self.__write_inventory_output(self.inventory,
                                          lengthUnit=self.lengthUnit,
                                          outputFile=outputFile)
        """
        # Merge the DataFrame of predicted attributes with the DataFrame of
        # incomplete inventory and print the resulting table to the output file
        # titled IncompleteInventory.csv:
        dfout2merge = dfout.copy(deep=True)
        dfout2merge['fp_as_string'] = dfout2merge['Footprint'].apply(
            lambda x: "".join(str(x)))

        dfout_incomp = self.incompleteInventory.copy(deep=True)
        dfout_incomp['fp_as_string'] = dfout_incomp['Footprint'].apply(
            lambda x: "".join(str(x)))

        dfout_incomp = pd.merge(left=dfout_incomp,
                                right=dfout2merge.drop(
                                    columns=['Footprint'], errors='ignore'),
                                how='left', left_on=[
                                    'fp_as_string', 'PlanArea'],
                                right_on=['fp_as_string', 'PlanArea'],
                                sort=False)

        dfout_incomp = dfout2merge.append(
            dfout_incomp[dfout_incomp.roofshape.isnull()])
        dfout_incomp = dfout_incomp.reset_index(drop=True).drop(
            columns=['fp_as_string'], errors='ignore')

        self.incompleteInventory = dfout_incomp.copy(deep=True)

        dfout_incomp4print = dfout_incomp.copy(deep=True)
        for index, row in dfout_incomp.iterrows():
            dfout_incomp4print.loc[index, 'Footprint'] = (
                '{"type":"Feature","geometry":'
                '{"type":"Polygon","coordinates":['
                f"{row['Footprint']}"
                ']},"properties":{}}')
            centroid = Polygon(row['Footprint']).centroid
            dfout_incomp4print.loc[index, 'Latitude'] = centroid.y
            dfout_incomp4print.loc[index, 'Longitude'] = centroid.x

        cols = [col for col in dfout_incomp4print.columns if
                col != 'Footprint']
        new_cols = ['Latitude', 'Longitude'] + cols[:-2] + ['Footprint']
        dfout_incomp4print = dfout_incomp4print[new_cols]

        dfout_incomp4print.to_csv(
            'IncompleteInventory.csv', index=True, index_label='id',
            na_rep='NA')
        print('Incomplete inventory data available in '
              'IncompleteInventory.csv')
        """
