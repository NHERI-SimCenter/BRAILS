
# -*- coding: utf-8 -*-
"""
/*------------------------------------------------------*
|                         BRAILS                        |
|                                                       |
| Author: Charles Wang,  UC Berkeley, c_w@berkeley.edu  |
|                                                       |
| Date:    10/15/2020                                   |
*------------------------------------------------------*/
"""

from brails.modules.ModelZoo import zoo
from brails.modules.GenericImageClassifier.GenericImageClassifier import *
import wget 
import os

class OccupancyClassifier(ImageClassifier):
    """ Occupancy Class Classifier. """


    def __init__(self, modelName=None, classNames=None, resultFile='occupancy_preds.csv', workDir='tmp', printRes=True):
        '''
        modelFile: path to the model
        classNames: a list of classnames
        '''

        if not os.path.exists(workDir): os.makedirs(workDir)

        fileURL = zoo['occupancyClass']['fileURL']
        
        if not classNames:
            classNames = zoo['occupancyClass']['classNames']

        if not modelName:
            #modelName = 'occupancy_InceptionV3_V0.2'
            #modelName = 'occupancy_ResNet50_V0.1' # r
            modelName = 'occupancy_ResNet50_V0.2' # r
            print('A default occupancy model will be used: {}.'.format(modelName))

        modelFile = os.path.join(workDir,'{}.h5'.format(modelName))


        if not os.path.exists(modelFile): # download
            print('Downloading the model ...')
            downloadedModelFile = wget.download(fileURL, out=modelFile)
        
        # Call parent constructor
        ImageClassifier.__init__(self, 
            modelName=modelName, 
            classNames=classNames, 
            resultFile=resultFile, 
            workDir=workDir,
            printRes=printRes
        )



if __name__ == '__main__':
    main()








