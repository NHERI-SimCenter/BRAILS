.. _lbl-train:


Make predictions
================


Predict
---------------

Now we can use the trained model to predict on new images. 

We show a example here: predict roof type based on satellite images.

Firstly we need to download those satellite images by calling Google API (will cost $).

.. code-block:: none 

    cd src/predicting
    python downloadRoofImages.py


To save $, instead of running the above command, you can just get them from
`here <https://berkeley.box.com/shared/static/n8l9kusi9eszsnnkefq37fofz22680t2.zip>`_. that are prepared by SimCenter.


Predictions can now be made using the trained model:

.. code-block:: none 

    cd src/predicting
    python predict.py --image_dir <IMAGE_DIRECTORY> --model_path <MODEL_PATH>

IMAGE_DIRECTORY is the directory of your images. MODEL_PATH is the folder where you have your trained model saved.

.. 
    Commented
    This script will look into the BIM file and call the ConvNet to predict the roof type of a building if the image is downloaded.
    If the image is not downloaded, it will assign a null value for the roof type in the new BIM file.
    By running the above commands, the original BIM file will be enriched with a new building property, roof type, which is added for each building.