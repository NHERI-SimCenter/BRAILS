
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

from ModelZoo import zoo
from GeneralImageClassifier import *
import wget 
import os

class OccupancyClassifier:
    """ Occupancy Class Classifier. """


    def __init__(self, resultFile='occupancy_preds.csv'):
        '''
        modelFile: path to the model
        classNames: a list of classnames
        '''

        modelDir = 'tmp'
        if not os.path.exists(modelDir): os.makedirs(modelDir)

        modelFile = os.path.join(modelDir,'occupancy-78-78-79.h5')
        
        fileURL = zoo['residentialOccupancyClass']['fileURL']
        classNames = zoo['residentialOccupancyClass']['classNames']

        if not os.path.exists(modelFile): # download
            print('Downloading the model ...')
            downloadedModelFile = wget.download(fileURL, out=modelFile)
 

        roofClassifier = ImageClassifier(modelFile, classNames=classNames, resultFile=resultFile)

        self.predict = roofClassifier.predict



if __name__ == '__main__':
    main()





