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

class SoftstoryClassifier:
    """ Softstory Image Classifier. """


    def __init__(self, resultFile='softstory_preds.csv'):
        '''
        modelFile: path to the model
        classNames: a list of classnames
        '''

        modelDir = 'tmp'
        if not os.path.exists(modelDir): os.makedirs(modelDir)

        modelFile = os.path.join(modelDir,'softstory-80-81-87.5-v0.1.h5')
        
        fileURL = zoo['softstory']['fileURL']
        classNames = zoo['softstory']['classNames']

        if not os.path.exists(modelFile): # download
            print('Downloading the model ...')
            downloadedModelFile = wget.download(fileURL, out=modelFile)
 

        roofClassifier = ImageClassifier(modelFile, classNames=classNames, resultFile=resultFile)

        self.predict = roofClassifier.predict



if __name__ == '__main__':
    main()





