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

class RoofClassifier(ImageClassifier):
    """ Roof Image Classifier. """


    def __init__(self, modelName=None, classNames=None, resultFile='roofType_preds.csv', workDir='tmp', printRes=True):
        '''
        modelFile: path to the model
        classNames: a list of classnames
        '''

        if not os.path.exists(workDir): os.makedirs(workDir)

        fileURL = zoo['roofType']['fileURL']
        
        if not classNames:
            classNames = zoo['roofType']['classNames']

        if not modelName:
            #modelName = 'roof_classifier_v0.1'
            modelName = 'rooftype_ResNet50_V0.2'
            print('A default roof type model will be used: {}.'.format(modelName))

        modelFile = os.path.join(workDir,'{}.h5'.format(modelName))


        if not os.path.exists(modelFile): # download
            print('Downloading the model ...')
            downloadedModelFile = wget.download(fileURL, out=modelFile)

        ImageClassifier.__init__(self, modelName=modelName, classNames=classNames, resultFile=resultFile, printRes=printRes)



if __name__ == '__main__':
    main()



'''
Potential errors:
https://stackoverflow.com/questions/52805115/certificate-verify-failed-unable-to-get-local-issuer-certificate
'''

