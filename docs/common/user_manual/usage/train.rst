.. _lbl-train:


Train your model
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

    # train (better to run on a GPU machine)
    cd src/training/roof/2_train
    
    sh finetune_inception_v3_on_roof_train.sh


The training takes a long time on laptops. 
If you don't want to run the training process, we have a CNN trained on TACC and can be downloaded from `here <https://berkeley.box.com/shared/static/awyyc22sjwknn9xg3p7wru4v5zwnlkjp.zip>`_.
Put the downloaded file inside src/training/roof/tmp/roof-traindir/ and unzip it.




Test
---------------

SimCenter prepared some labeled images for the users to test. 

These images can be downloaded from `here <https://berkeley.box.com/shared/static/wfwf4ku9561lcytldy1p7vkjoobgv9sz.zip>`_.

Make sure the following variables have been set correctly in *"finetune_inception_v3_on_roof_eval.sh"* :

.. code-block:: none

    checkpoint_file = the-path-of-your-checkpoint-file
    TEST_DIR = the-path-of-your-testing-images-dir


Now the test can be performed to see if the trained ConvNet works or not:

.. code-block:: none 

    cd src/training/roof/2_train
    sh finetune_inception_v3_on_roof_eval.sh




Predict
---------------

Now we use the ConvNet to predict roof types based on satellite images.

Firstly we need to download those images by calling Google API (will cost $).

.. code-block:: none 

    cd src/predicting
    python downloadRoofImages.py


To save $, instead of running the above command, you can just download them from
`here <https://berkeley.box.com/shared/static/n8l9kusi9eszsnnkefq37fofz22680t2.zip>`_.


Now make predictions can be made using the trained ConvNet:

.. code-block:: none 

    cd src/predicting
    sh classifyRoof.sh

This script will look into the BIM file and call the ConvNet to predict the roof type of a building if the image is downloaded.

If the image is not downloaded, it will assign a null value for the roof type in the new BIM file.

By running the above commands, the original BIM file will be enriched with a new building property, roof type, which is added for each building.