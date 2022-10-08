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

class SoftstoryClassifier(ImageClassifier):
    """ Softstory Image Classifier. """


    def __init__(self, modelName=None, classNames=None, resultFile='softstory_preds.csv', workDir='tmp', printRes=True):
        '''
        modelFile: path to the model
        classNames: a list of classnames
        '''

        if not os.path.exists(workDir): os.makedirs(workDir)

        fileURL = zoo['softstory']['fileURL']
        
        if not classNames:
            classNames = zoo['softstory']['classNames']

        if not modelName:
            modelName = 'softstory_ResNet50_V0.1'  # good  softstory_ResNet50_V0.1_r
            print('A default softstory model will be used: {}.'.format(modelName))

        modelFile = os.path.join(workDir,'{}.h5'.format(modelName))


        if not os.path.exists(modelFile): # download
            print('Downloading the model ...')
            downloadedModelFile = wget.download(fileURL, out=modelFile)

        ImageClassifier.__init__(self,
            modelName=modelName,
            classNames=classNames,
            resultFile=resultFile,
            workDir=workDir,
            printRes=printRes
        )



if __name__ == '__main__':
    main()






