Raised Foundation Classification
=================================================


What is Raised Foundation Classification
------------------------------------------
The code in this package enables to see if a building is on piles piers or posts (PPP). For classification, the path of a folder holding the images has to be supplied. The result will be a comma separated value file in that folder, listing the filenames and classification result.

There is further optional code to improve the quality and speed of the classification. At the current moment, classification reaches an F1-score of 72% on a random test set, holding out 20% of the data.

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
The following commands clone the BRAILS repository and enter the foundation classification module.
Requirements are installed using pip. The final 
command adds the current folder to the PYTHONPATH environment variable which is necessary to train.

::

    git clone https://github.com/NHERI-SimCenter/BRAILS.git BRAILS
    cd BRAILS/brails/modules/Foundation_Classification
    python3 -m pip install -r requirements.txt
    export PYTHONPATH=$PYTHONPATH:`pwd`

How to use
---------------------------

Execute with pretrained model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The detect code will download approximately 300 mb in required model weight files on its first start. 
Detection on a single image or all images in a folder is started in the same way. The default classifier
with the provided checkpoint expects as input, given with --image-path, either a single image or
a folder which will be searched for images. One way of improving detection is to mask the parts
of the image which are not a building. That is done with --mask-buildings.
Please see `Pre-Saving Masked Images` about how to create masks. If masks should not be used remove the
--mask-buildings flag from the command line.
If masks should be generated on the fly remove the --load-masks flag, however this will
take more time and use more GPU memory.

::

    python3 detect.py --image-path <IMG_FOLDER or IMAGE PATH> --mask-buildings --load-masks

The result will be a comma separated value file *<IMG_FOLDER>_prediction_results.csv* which contains in each row a filename for each image and a 1 if the building is higher than 8ft or 0 otherwise.


Optional Improvements
--------------------------

Pre-Saving Masked Images
~~~~~~~~~~~~~~~~~~~~~~~~~~
One step of the detection is masking out the background from the buildings.
This step can take up some time so for re-training of the model, it is suggested
to pre-compute the masks with this command:
::

    python3 save_masked_images.py --image-path <IMG_FOLDER or IMAGE PATH>

Masks of images will be saved and expected at the same folder as the original images.

Input Data Format for Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Training, validation and test folders should be separate. The name is arbitrary and each path can
be stated separately with command line parameters during training. Within each folder there
has to be one folder for each label, named *ppp_buildings*, containing images of buildings from the PPP category
and other containing buildings which are not in that category. The same structure holds for the val and test folders.
::

    IMG_FOLDER
    ├── train
    │   └── ppp_buildings
    │       ├── image_00x.jpg
    │       └── image_00x-mask.png
    │   └── other
    │       ├── image_00x.jpg
    │       └── image_00x-mask.png
    ├── val
    │   └── <structure identical to train>
    └── test
        └── <structure identical to train>

For training it is necessary to create the masks beforehand, as described in the previous section.

Cleaning the Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~

Many datasets have outliers which do not show buildings but empty properties or
objects covering the camera. A half-automatic way of finding and removing noisy images from the
dataset is described in the following

Training the Model
~~~~~~~~~~~~~~~~~~~~~~~~~~
Training can happen in two different ways: With an attention based
network or with a standard Resnet 50, with or without unsupervised pretraining.

Unsupervised Pretraining
~~~~~~~~~~~~~~~~~~~~~~~~~~
The network used for foundation classification can be pre-trained on a different task to improve
the results. Because of the unsupervised nature of this pre-training the whole dataset can be used,
including validation and test data. It is also possible to set different sub-sets with the
command line preferences. This step will produce a checkpoint which can be loaded for later
foundation detection, called in the following <NPID CHECKPOINT>.

The most important command line parameters are:

.. parsed-literal::

    --train-data, --val-data for setting the folders in which images are kept for training and validation.
    Validation here is a kNN step which will judge the quality of the feature embedding of the network. The
    reported performance is correlated with the foundation detection performance but not the same

    --resume Optionally, a checkpoint can be loaded which was pretrained on ImageNet to further improve training. The same
    parameter is used if training is interrupted and should be continued

    --name Logfiles and checkpoints will be created with this name as prefix.

    --mask-buildings Masks are used to mask out buildings. See `Pre-Saving Masked Images`_ for how to create masks from images.

    --epochs How many epochs should be trained.

    --low-dim The embedded dimension of the approach. This is a hyperparameter which can be
    optimized. For most purposes the default of 128 will suffice. Smaller values can be
    chosen for datasets, significantly smaller than 1000 training images.

Further parameters can be seen by just parsing the -h parameter. The command to train 100 epochs is therefore:

::

    python3 npid/main.py
        --train-data <TRAINING IMAGE FOLDER>
        --val-data <VALIDATION IMAGE FOLDER>
        --resume <CHECKPOINT TO LOAD>
        --name <NAME FOR LOGFILES AND CHECKPOINTS>
        --mask-buildings
        --epochs 100

During training two checkpoints will be saved in the same folder, one which is always the checkpoint of the
latest epoch and one which is the best, according to the internal quality measure.

Foundation Classification Model Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model can be trained on existing data. The folder structure shown in `Input Data Format for Training`_
has to be observed so the right labels are assigned. The most important command line parameters are:

.. parsed-literal::

    --epochs Amount of epochs to train

    --train-data, --val-data, --test-data for setting the folders in which images are kept for training and validation.
    These folders should contain separate data. --test-data is only needed in combination with the --eval flag to
    check the performance on the test data.

    --eval Evaluate the trained model on the test set. A model should be loaded with the --checkpoint flag.

    --checkpoint Load a checkpoint to continue training or evaluate the performance on the test set.

    --mask-buildings Mask the buildings. Warning: Prior masking is mandatory. On the fly generation does not work for training.
    See `Pre-Saving Masked Images`_ for how to create masks from images.

    --freeze-layers When loading from a checkpoint, all layers apart from the final fully connected layer can be frozen
    for finetuning.

    --pretrained Removes saved classifier weights from a checkpoint and uses the remaining for pretraining. Load the checkpoint via --checkpoint

    --exp-name  Prefix for logfiles and checkpoints


Further parameters can be seen by just parsing the -h parameter. The command to train 100 epochs is therefore:

::

    python3 train.py
        --train-data <TRAINING IMAGE FOLDER>
        --val-data <VALIDATION IMAGE FOLDER>
        --exp-name <NAME FOR LOGFILES AND CHECKPOINTS>
        --mask-buildings
        --epochs 100

