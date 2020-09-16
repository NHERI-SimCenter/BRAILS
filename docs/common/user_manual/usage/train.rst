.. _lbl-train:


Train a model
================

There are pretrained ConvNets released with |brailsName| that can be used out of the box.

However, if the user is interested in training his / her own ConvNets, the following is an example demonstrating
how to train a classifier (for roof shape), and how to use the trained model in the |brailsName| application.

Train
---------------

The image data set for training has been prepared by SimCenter and can be download from here:

Charles Wang. (2019). Random satellite images of buildings (Version v1.0) [Data set]. Zenodo. `http://doi.org/10.5281/zenodo.3521067 <http://doi.org/10.5281/zenodo.3521067>`_.

To train on the downloaded data, run the following command in the terminal

.. code-block:: none 

    cd src/training/
    python train_classifier.py --img_dir <IMAGE_DIRECTORY> --model_dir <MODEL_DIRECTORY>

IMAGE_DIRECTORY is the directory where you have your images for training

::

    IMAGE_DIRECTORY
    │── class_1
    │       └── *.png
    │── class_2
    |      └── *.png
    │── ...
    |
    └── class_n
           └── *.png

MODEL_DIRECTORY is the directory where the trained model will be saved. 

.. 
    Commented
    The training takes a long time on laptops. 
    If you don't want to run the training process, we have a CNN trained on TACC and can be downloaded from `here <https://berkeley.box.com/shared/static/awyyc22sjwknn9xg3p7wru4v5zwnlkjp.zip>`_.
    Put the downloaded file inside src/training/roof/tmp/roof-traindir/ and unzip it.

It is better to run the above code on a GPU machine.


Predict
---------------

Now we use the trained model to predict roof types based on satellite images.

Firstly we need to download those images by calling Google API (will cost $).

.. code-block:: none 

    cd src/predicting
    python downloadRoofImages.py


To save $, instead of running the above command, you can just download them from
`here <https://berkeley.box.com/shared/static/n8l9kusi9eszsnnkefq37fofz22680t2.zip>`_.


Now predictions can be made using the trained model:

.. code-block:: none 

    cd src/predicting
    python predict.py --image_dir <IMAGE_DIRECTORY> --model_path <MODEL_PATH>

IMAGE_DIRECTORY is the directory of your images. MODEL_PATH is the folder where you have your trained model saved.

.. 
    Commented
    This script will look into the BIM file and call the ConvNet to predict the roof type of a building if the image is downloaded.
    If the image is not downloaded, it will assign a null value for the roof type in the new BIM file.
    By running the above commands, the original BIM file will be enriched with a new building property, roof type, which is added for each building.