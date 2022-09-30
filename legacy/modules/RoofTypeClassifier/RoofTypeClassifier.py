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
#
# Contributors:
# Yunhui Guo


from brails.modules.ModelZoo import zoo
from brails.modules.GenericImageClassifier.GenericImageClassifier import *
import wget 
import os

class RoofClassifier(ImageClassifier):
    """
    A Roof Type Classifier. 

    
    Parameters
    ----------
    modelName: architecture of the model. Please refer to https://github.com/rwightman/pytorch-image-models for supported models.
    imgDir: directories for training data
    valimgDir: directories for validation data
    random_split: ratio to split the data into a training set and validation set if validation data is not provided.
    resultFile: name of the result file for predicting multple images.
    workDir: the working directory
    printRes: show the probability and prediction

    """

    def __init__(self, 
            modelName=None, 
            classNames=None, 
            resultFile='roofType_preds.csv', 
            workDir='tmp', 
            printRes=True
    ):
        '''
        modelFile: path to the model
        classNames: a list of classnames
        '''

        if not os.path.exists(workDir):
            os.makedirs(workDir)

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

        ImageClassifier.__init__(self,
            modelName=modelName,
            classNames=classNames,
            resultFile=resultFile,
            workDir=workDir,
            printRes=printRes
        )



if __name__ == '__main__':
    main()



'''
Potential errors:
https://stackoverflow.com/questions/52805115/certificate-verify-failed-unable-to-get-local-issuer-certificate
'''

