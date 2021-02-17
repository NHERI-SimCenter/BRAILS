Year Built Classification
=================================================

.. contents:: Table of Contents

What is Year Built Classification
------------------------------------------
With this module images of houses can be classifier into categories, 
where each represents a range of years in which the house was built. 
For classification, the path of a folder holding the images has to be supplied. 
The result will be a comma separated value file in that folder, 
listing the filenames and classification result.

The code is optimized to extract suitable features from Google Streetview
images to classify houses into decades. However, the code does not make 
assumptions about the semantic meaning of the provided folders. They
have to be consistent between the training, validation and test folders but
can be decades or shorter or longer and even uneven timespans.

Copyright
~~~~~~~~~
::

    Copyright (c) 2018, The Regents of the University of California
    Contact: Sascha Hornauer  - sascha.hornauer@uni-oldenburg.de


BSD 3-Caluse license
~~~~~~~~~~~~~~~~~~~~
::

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Installation
---------------------------
The following commands clone the BRAILS repository and enters the year built classification module.
Requirements are installed using pip. Weight files of models will be downloaded automatically upon
first run and saved in subfolders. The final command adds the current folder to the PYTHONPATH
environment variable which is necessary for training.

::

    git clone https://github.com/NHERI-SimCenter/BRAILS.git BRAILS
    cd BRAILS/brails/modules/Year_Built_Classifier
    python3 -m pip install -r requirements.txt
    export PYTHONPATH=$PYTHONPATH:`pwd`

How to use
---------------------------

Input Data Format for Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Training, validation and test folders should be separate. A top folder can contain the 
whole dataset as in the following structure. However, subfolders to train on can be chosen separately.
::

    IMG_FOLDER
    ├── train
    │   └── class 1
    │       ├── image_00x.jpg
    │       └── image_00x.jpg
    │   └── class 2
    │       └── image_00x.jpg
    ├── val
    │   └── <structure identical to train>
    └── test
        └── <structure identical to train>

Within each folder, each subfolder signifies one class. The user is free in how many classes should exist or
how their year range should be. So e.g. if the user would like to have three classes, one for buildings
older than 1970, one for 1970-2000 and one for 2000 onwards, three folders should be created: '1970','1970-2000','2000'
and the desired building files have to be distributed accordingly by the user.

Execute with pretrained model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detection on a single image or all images in a folder is started in the same way. The default will load
a trained checkpoint and classify all images or a single image, provided with the --image-path argument.

::

    python3 detect.py --image-path <IMG_FOLDER or IMAGE PATH>  --checkpoint <PATH TO CHECKPOINT>

The result will be a comma separated value file *<IMG_FOLDER>_prediction_results.csv* which contains in each
row a filename for each image and the predicted year built range class as well as the ground truth, as taken 
from the folder names.
If the image path points to a folder in the structure, described at `Input Data Format for Training` then the
::

	--calc-perf
	
command line argument can be set. This will evaluate the performance, using the folder structure as labels.
Therefore, the label structure has to be the same as it was when the checkpoint was trained. It will show
performance measures after testing all images and save a confusion matrix to show class-wise performance. 

The prediction falls in 6 categories:
::

    0 : before 1969
    1 : 1970-1979
    2 : 1980-1989
    3 : 1990-1999
    4 : 2000-2009
    5 : after 2010

Cleaning the Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~

Many datasets have outliers which do not show buildings but empty properties or
objects covering the camera. It is suggested to use the cleaning method described
and provided in detail in the foundation classification module.

Training the Model
~~~~~~~~~~~~~~~~~~~~~~~~~~
Training can happen with or without soft labels. This is an experimental variation which
supports the notion that the limits of the year built labels may not reflect actual visual
differences in houses well. E.g. a separation by decades may lead to a training regime that
tries to put the same architectural style in two different classes. 


Foundation Classification Model Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model can be trained on existing data. The folder structure shown in `Input Data Format for Training`
has to be observed so the right labels are assigned. The most important command line parameters are:


.. parsed-literal::

    --image-path Folder on which to train. This can be e.g. the train folder as in the input data format example.

    --checkpoint Load a checkpoint to continue training

    --exp-name Folder name to save the results to
    
    --soft-labels Activate use of soft labels
    
    --gaussian-std Choose standard deviation of the gaussian
    
    --epochs Epochs to train


All parameters can be seen by just parsing the -h parameter. The command to train 100 epochs is therefore:

::

    python3 train.py
        --image-path <TRAINING IMAGE FOLDER>
        --exp-name <NAME FOR LOGFILES AND CHECKPOINTS>
        --epochs 100

This will create a folder with the experiment name and save epoch-wise training results in it, along
with the checkpoints containing the weights. Note that this does only evaluate the results on the
training set, with and without random pertubations. That means to investigate the 
validation and test performance, the detect script with activated --calc-perf option has to be used.
