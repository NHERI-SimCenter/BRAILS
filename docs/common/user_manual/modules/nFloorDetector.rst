.. _lbl-nFloorDetector:

Number of Floors Detector
===========================

The module is bundled with BRAILS, hence its use does not require a separate installation if BRAILS was installed following the :ref:`lbl-install` instructions. 

This module enables automated detection of number of floors in a building from image input. It takes the directory for an image or folder of images as input and writes the number of floor detections for each images into a CSV file.

Use the module
-----------------
.. code-block:: none 

    # Import the module
    from brails.modules import NFloorDetector

    # Initialize the detector
    nfloorDetector = NFloorDetector()

    # Define the path of the images:
    imDir = "datasets/test/"

    # Detect the number of floors in each image inside imDir and write them in a 
    # CSV file. The prediction can be also assigned to DataFrame variable:
    predictions = nfloorDetector.predict(imDir)

    # Train a new detector using EfficientDet-D7 for 50 epochs
    nfloorDetector.load_train_data(rootDir="datasets/")
    nfloorDetector.train(compCoeff=7,numEpochs=50)