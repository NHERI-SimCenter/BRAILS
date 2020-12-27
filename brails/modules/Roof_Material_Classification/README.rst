Roof Material Classification
=================================================

.. contents:: Table of Contents

What is Roof Material Classification
------------------------------------------
With this module, satellite images of houses can be classified into categories, 
where each represents a roof material, such as shingles or tiles. 
For classification, the path of a folder holding the images has to be supplied. 
The result will be a comma separated value file, either in the root folder or in
the folder provided via the result-file parameter, listing files and predictions.

The code does not make assumptions about the semantic meaning of the provided folders. 
They have to be consistent between the training and test but can contain arbitrary 
roof types. During training the code saves the class names in the checkpoint and 
loads the names from there. 

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
The following commands clone the BRAILS repository and enters the roof material module.
Requirements are installed using pip. Weight files of models will be downloaded automatically upon
first run and saved in subfolders. The final command adds the current folder to the PYTHONPATH
environment variable which is necessary for training.

::

    git clone https://github.com/NHERI-SimCenter/BRAILS.git BRAILS
    cd BRAILS/brails/modules/Roof_Material_Classification
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
how their year range should be. So e.g. if the user would like to have three classes, one for shingles and
one for concrete, the folders could be 'shingles' and 'concrete'.

Execute with pretrained model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detection on a single image or all images in a folder is started in the same way. The default evaluation will download
a trained checkpoint from Zenodo and classify all images or a single image, provided with the --val-data argument.

::

    python3 main.py --val-data <IMG_FOLDER or IMAGE PATH> --evaluate

The result will be a comma separated value file *results.csv* which contains in each
row a filename for each image and the predicted roof type.

If the image path points to a folder in the structure, described at `Input Data Format for Training` then 
a tensorboard entry will be added in the *runs* folder, where a confusion matrix can be found, showing how
well the evaluation went. If the folder structure is different the algorithm will still write all predictions
into the csv file but the confusion matrix may not make any sense.



Training the Model
~~~~~~~~~~~~~~~~~~~~~~~~~~
Training the model works by providing data as described in `Input Data Format for Training` and not using the *--evaluate*
command line parameter. Here, the tensorboard entry in the runs folder shows also classification scores and loss functions
over epochs.

For a complete list of command line parameters for evaluation and training use the *--help* parameter.

The command to train 100 epochs and saving checkpoints in the folder *testrun* is therefore:

::

    python3 main.py
     --train-data <TRAINING DATA>
     --val-data <VALIDATION DATA>
     --epochs 100 
     --checkpoint-dir testrun

