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



import os
from pathlib import Path
import pandas as pd

from brails.modules import (PytorchRoofClassifier, PytorchOccupancyClassifier, 
                            SoftstoryClassifier, FoundationHeightClassifier, 
                            YearBuiltClassifier, NFloorDetector, 
                            GarageDetector, ChimneyDetector)


from .workflow.Footprints import getMSFootprintsByPlace, getStateNameByBBox, getOSMFootprints, getMSFootprintsByBbox
from .workflow.Images import getGoogleImages


class InventoryGenerator:

    def __init__(self, location='', nbldgs=10, random=True, footprints='OSM',
                 GoogleAPIKey=''):
        def getFootprintByPlace(self,place='',workDir='',GoogleAPIKey='',save=True,overwrite=True):
            if self.footPrints=='OSM':
                print(f"Trying to get OSM footprints by the place name {place}.")
                self.BIM, footprintFilename = getOSMFootprints(place=place,save=True,workDir=workDir,overwrite=True)
            elif self.footPrints=='Microsoft':
                #assert False, f"Trying to get Microsoft footprints by the place name {place}."
                self.BIM, footprintFilename = getMSFootprintsByPlace(place=location,save=True,workDir='tmp',GoogleMapAPIKey=GoogleAPIKey,overwrite=False)
            else: 
                assert False, f"footPrints must be OSM or Microsoft in CityBuilder. Your value is {self.footPrints}."
                
        self.workdir = 'tmp'
        self.footprints = footprints
        self.bimExist = False
        
        if len(location)>0:
            print(location)
            bimout = Path(f'/{location}_BIM.geojson'.replace(' ','_').replace(',','_'))
            bimFile = Path(f'tmp/{bimout}')
            self.BIM = pd.read_file(self.bimFile)
            self.bimExist = True
            getFootprintByPlace(place=location,workDir='tmp',GoogleMapAPIKey=GoogleAPIKey,save=True,overwrite=True)
        else:
            assert False, f"footprint source must be OSM or Microsoft. Your value is {self.footPrints}."


    def generate(self,attributes=['numstories','occupancy','roofshape']):
        self.attributes = attributes
        
        
        
        for attribute in self.attributes:
            if attribute.lower()=='roofshape':
                roofModel = PytorchRoofClassifier(modelName='transformer_rooftype_v1', download=True)
                roofShape = roofModel.predictMultipleImages('images/satellite')
                self.BIM['roofShape'] = self.BIM.apply(lambda x: roofShape[x['ID']], axis=1)
            if attribute.lower()=='occupancy':
                occupancyModel = PytorchRoofClassifier(modelName='transformer_occupancy_v1', download=True)
                occupancy = occupancyModel.predictMultipleImages('images/satellite')
                self.BIM['occupancy'] = self.BIM.apply(lambda x: occupancy[x['ID']], axis=1) 
            elif attribute.lower()=='elevated':
                # initialize a foundation classifier
                elvModel = FoundationHeightClassifier(workDir=self.workDir,printRes=False)

                # use the classifier 
                elv_df = elvModel.predict(self.BIM['StreetView'].tolist())

                elv = elv_df['prediction'].to_list()
                elvProb = elv_df['probability'].to_list()
                self.BIM['elevated'] = self.BIM.apply(lambda x: elv[x['ID']], axis=1)
                self.BIM['elevated  Prob'] = self.BIM.apply(lambda x: elvProb[x['ID']], axis=1) 

            elif attribute.lower()=='numstories':
                # Initialize the floor detector object
                storyModel = NFloorDetector()

                # Call the floor detector to determine number of floors
                story_df = storyModel.predict(self.BIM['StreetView'].tolist())

                # Write the results to a dataframe
                story = story_df['prediction'].to_list()
                self.BIM['numStories'] = self.BIM.apply(lambda x: story[x['ID']], axis=1)
                
            elif attribute.lower()=='garage':
                # Initialize the garage detector object
                garageModel = GarageDetector()

                # Call the floor detector to determine number of floors
                garage_df = garageModel.predict(self.BIM['StreetView'].tolist())

                # Write the results to a dataframe
                garage = garage_df['prediction'].to_list()
                self.BIM['garageExist'] = self.BIM.apply(lambda x: story[x['ID']], axis=1)

            elif attribute.lower()=='chimney':
                # Initialize the floor detector object
                chimneyModel = ChimneyDetector()

                # Call the floor detector to determine number of floors
                chimney_df = chimneyModel.predict(self.BIM['StreetView'].tolist())

                # Write the results to a dataframe
                chimney = chimney_df['prediction'].to_list()
                self.BIM['chimneyExist'] = self.BIM.apply(lambda x: story[x['ID']], axis=1)
                
            elif attribute.lower()=='year':
                yearModel = YearBuiltClassifier(workDir=self.workDir,printRes=False)

                year_df = yearModel.predict(self.BIM['StreetView'].tolist())

                year = year_df['prediction'].to_list()
                yearProb = year_df['probability'].to_list()
                self.BIM['year'] = self.BIM.apply(lambda x: year[x['ID']], axis=1)
                self.BIM['yearProb'] = self.BIM.apply(lambda x: yearProb[x['ID']], axis=1)                 
                