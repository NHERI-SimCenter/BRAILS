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
#
# Last updated:
# 04-25-2022 

import random
import sys
import os
from pathlib import Path
import pandas as pd

from brails.modules import (PytorchRoofClassifier, PytorchOccupancyClassifier, 
                            SoftstoryClassifier, FoundationHeightClassifier, 
                            YearBuiltClassifier, NFloorDetector, 
                            GarageDetector, ChimneyDetector)
from .workflow.ImHandler import ImageHandler
from .workflow.FootprintHandler import FootprintHandler

class InventoryGenerator:

    def __init__(self, location='', nbldgs=10, randomSelection=True,
                 GoogleAPIKey=''):                
        self.workdir = 'tmp'
        self.location = location
        self.nbldgs = nbldgs
        self.randomSelection = randomSelection
        self.apikey = GoogleAPIKey
        
        # Get footprint data:
        fpHandler = FootprintHandler()
        if 'geojson' in location:
            fpHandler.load_footprint_data(self.location)
        else:
            fpHandler.fetch_footprint_data(self.location)
            
        if self.randomSelection==True:
            self.footprints = random.sample(fpHandler.footprints, nbldgs)
        else: 
            self.footprints = fpHandler.footprints[:nbldgs]
            
        # Initialize the image handler to check if the provided API key is valid:
        image_handler = ImageHandler(self.apikey)

    def generate(self,attributes=['numstories','occupancy','roofshape']):
        self.attributes = attributes

        # Download required images
        image_handler = ImageHandler(self.apikey)
        if 'roofshape' in attributes:
            image_handler.GetGoogleSatelliteImage(self.footprints)
            imsat = ['tmp/images/satellite/' + im for im in os.listdir('tmp/images/satellite')]
        elif set.intersection(set(['chimney','elevated','garage','numstories',
                                   'occupancy','year']),attributes)!=set():            
            image_handler.GetGoogleStreetImage(self.footprints)
            imstreet = ['tmp/images/street/' + im for im in os.listdir('tmp/images/street')]
        else:
            sys.exit('Defined list of attributes does not contain a correct' +
                     'attribute entry. Attribute entries enabled are: chimney, ' + 
                     'elevated, garage, numstories, occupancy, roofshape, year')
        
        for attribute in self.attributes:
            if attribute.lower()=='roofshape':
                roofModel = PytorchRoofClassifier(modelName='transformer_rooftype_v1', download=True)
                roofShape = roofModel.predictMultipleImages(imsat)
                self.BIM['roofShape'] = self.BIM.apply(lambda x: roofShape[x['ID']], axis=1)

            elif attribute.lower()=='occupancy':
                occupancyModel = PytorchRoofClassifier(modelName='transformer_occupancy_v1', download=True)
                occupancy = occupancyModel.predictMultipleImages(imstreet)
                self.BIM['occupancy'] = self.BIM.apply(lambda x: occupancy[x['ID']], axis=1) 
           
            elif attribute.lower()=='elevated':
                # initialize a foundation classifier
                elvModel = FoundationHeightClassifier(workDir=self.workDir,printRes=False)

                # use the classifier 
                elv_df = elvModel.predict(imstreet)

                elv = elv_df['prediction'].to_list()
                elvProb = elv_df['probability'].to_list()
                self.BIM['elevated'] = self.BIM.apply(lambda x: elv[x['ID']], axis=1)

            elif attribute.lower()=='numstories':
                # Initialize the floor detector object
                storyModel = NFloorDetector()

                # Call the floor detector to determine number of floors
                story_df = storyModel.predict(imstreet)

                # Write the results to a dataframe
                story = story_df['prediction'].to_list()
                self.BIM['numStories'] = self.BIM.apply(lambda x: story[x['ID']], axis=1)
                
            elif attribute.lower()=='garage':
                # Initialize the garage detector object
                garageModel = GarageDetector()

                # Call the garage detector to determine the existence of garages
                garage_df = garageModel.predict(imstreet)

                # Write the results to a dataframe
                garage = garage_df['prediction'].to_list()
                self.BIM['garageExist'] = self.BIM.apply(lambda x: garage[x['ID']], axis=1)

            elif attribute.lower()=='chimney':
                # Initialize the chimney detector object
                chimneyModel = ChimneyDetector()

                # Call the chimney detector to existence of chimneys
                chimney_df = chimneyModel.predict(imstreet)

                # Write the results to a dataframe
                chimney = chimney_df['prediction'].to_list()
                self.BIM['chimneyExist'] = self.BIM.apply(lambda x: chimney[x['ID']], axis=1)
                
            elif attribute.lower()=='year':
                yearModel = YearBuiltClassifier(workDir=self.workDir,printRes=False)

                year_df = yearModel.predict(imstreet)

                year = year_df['prediction'].to_list()
                yearProb = year_df['probability'].to_list()
                self.BIM['year'] = self.BIM.apply(lambda x: year[x['ID']], axis=1)
                self.BIM['yearProb'] = self.BIM.apply(lambda x: yearProb[x['ID']], axis=1)                 
                