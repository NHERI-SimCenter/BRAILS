# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 The Regents of the University of California
#
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


from brails.modules.PytorchOccupancyClassClassifier.OccupancyModelZoo import zoo
from brails.modules.PytorchGenericModelClassifier.GenericImageClassifier import *

#from OccupancyModelZoo import zoo
#import sys
#sys.path.insert(0,'..')
#from PytorchGenericModelClassifier.GenericImageClassifier import *

import wget 
import os
from sys import exit

class PytorchOccupancyClassifier(PytorchImageClassifier):
    """
    An Occupancy Classifier. Classes: 'OTH', 'RRE'
    
    Parameters
    ----------
    modelName: architecture of the model. Please refer to https://github.com/rwightman/pytorch-image-models for supported models.
    download: dowbload the pre-trained occupancy classifier
    imgDir: directories for training data
    resultFile: name of the result file for predicting multple images.
    workDir: the working directory
    printRes: show the probability and prediction
    """

    def __init__(self, 
            modelName=None, 
            imgDir='',
            valimgDir = '',
            download=True, 
            resultFile='Occupancy_preds.csv', 
            workDir='./tmp',
            printRes=False
    ):

        if not modelName:

            modelName = 'transformer_occupancy_v1'
            self.arch = 'transformer'
            print('Default occupancy classifier will be used: {}.'.format(modelName))

        else:

            self.arch, self.task, self.version = modelName.split("_")


        print('\nDetermining the occupancy class for each building...')
        if download:

            if self.arch != 'transformer':
                print ('Please download pre-trained model. Currently only the'+
                       'model titled transformer_occupancy_v1 is supported')
                exit()

            os.makedirs(workDir + '/models/', exist_ok=True)

            modelFile = os.path.join(workDir + '/models/', '{}.pkl'.format(modelName))

            if not os.path.exists(modelFile):

                print('Loading default occcupancy classifier model file to tmp/models folder...')
                
                self.download_model(modelFile)

            else:
                print('Default occupancy classifier model at tmp/models loaded')

            self.classNames = zoo['Occupancy']['classNames']

        else:

            if imgDir == "":
                print('Pre-trained will not be downloaded. You need to provide training data for training the model')
                exit()


        PytorchImageClassifier.__init__(self,
            modelName=modelName,
            imgDir=imgDir,
            valimgDir=valimgDir,
            download=download,
            resultFile=resultFile,
            workDir=workDir,
            printRes=printRes
        )

    def download_model(self, modelFile):

        fileURL = zoo['Occupancy']['fileURL']

        wget.download(fileURL, out=modelFile)


if __name__ == '__main__':
    
    work = PytorchOccupancyClassifier(modelName='transformer_occupancy_v1', 
                                      download=True, imgDir="./occupancy_val/")

    #work.train(lr=0.01, batch_size=16, epochs=5)
    work.predictOneDirectory("./occupancy_val/OTH/")