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

Floor detection through Object Detection 
-------------------------------------------	
The number of floor detections performed by the number of floors detector are based on image-based detections of visible floor locations (as in rows of windows) from street-level images. The current pretrained model that comes with this module was trained on the EfficientDet-D4 architecture with a dataset of 60,000 building images retrieved from all counties of New Jersey, excluding Atlantic county. In training, validation and testing of the model, 80%, 15%, and 5% of this dataset were randomly selected and utilized, respectively. In training the model, to ensure faster model convergence, initial weights of the model were set to model weights of the (pretrained) object detection model that, at the time, achieved state-of-the-art performance on the 2017 COCO Detection set. For this specific implementation, the peak model performance was achieved using the Adam optimizer at a learning rate of 0.0001 (batch size: 2) after 50 epochs. Figure :numref:`fig_FloorDetections` shows examples of the floor detections performed by the model.

.. _fig_FloorDetections:
.. figure:: ../../../images/image_examples/nFloor/sampleModelOutputs.gif
   :width: 70 %
   :alt: Sample model floor detections

   Sample floor detections of the pretrained model provided with this module, shown by bright green bounding boxes. The percentage value shown on the top right corner of each bounding box indicates the model's confidence level associated with that prediction.

For an image, the described floor detection model generates the bounding box output for its detections and calculates the confidence level associated with each detection. As a part of this module, a post-processor that converts stacks of neighboring bounding boxes into floor counts was developed to convert bounding box output into floor counts. Recognizing an image may contain multiple buildings at a time, this post-processor is designed to perform counts at the individual building level. 