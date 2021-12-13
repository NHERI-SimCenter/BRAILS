.. _lbl-ChimneyDetector:

Chimney Detector
===========================

The module is bundled with BRAILS, hence its use does not require a separate installation if BRAILS was installed following the :ref:`lbl-install` instructions. 

This module enables automated detection of masonry or concrete building chimneys from image input. It takes the directory for an image or folder of images as input and writes the existence of chimneys in each image into a CSV file. As the module is developed for performing predictions on only street-level building imagery, meaningful model outputs for other classes of images shall not be expected.

Use the module
--------------------
.. code-block:: none 

    # Import the module
    from brails.modules import ChimneyDetector

    # Initialize the detector
    chimneyDetector = ChimneyDetector()

    # Define the path of the images:
    imDir = "datasets/test/"

    # Detect if there are chimneys in each image inside imDir and write results in a 
    # CSV file. The prediction can be also assigned to DataFrame variable:
    predictions = chimneyDetector.predict(imDir)

    # Train a new detector using EfficientDet-D7 for 50 epochs
    chimneyDetector.load_train_data(rootDir="datasets/")
    chimneyDetector.train(compCoeff=7,numEpochs=50)

Chimney detection using Object Detection 
-------------------------------------------------
The chimney detections performed by this module are based on image-based detections of visible chimneys from street-level images. The current pretrained model that comes with this module was trained on the  `EfficientDet-D4 architecture
<https://arxiv.org/abs/1911.09070>`_ using a dataset of 1,703 building images retrieved from California, New Jersey, and Louisiana. 80%, 10%, and 10% of the samples in dataset were used for training, validation, and testing, respectively. All three sets were formed to be disjoint from each other to eliminate data contamination. In training the model, to ensure faster model convergence, initial weights of the model were set to model weights of the (pretrained) object detection model that, at the time, achieved state-of-the-art performance on the 2017 COCO Detection set. For this specific implementation, the peak model performance was achieved using the Adam optimizer at a learning rate of 0.0001 (batch size: 2) after 50 epochs. Figure :numref:`_fig_ChimneyDetections` shows examples of chimney detections performed by the model.

.. _fig_ChimneyDetections:
.. figure:: ../../../images/image_examples/Chimney/sampleChimneyDetections.gif
   :width: 70 %
   :alt: Sample chimney detections

Sample chimney detections of the pretrained model provided with this module, shown by bright green bounding boxes. The percentage value shown on the top right corner of each bounding box indicates the model's confidence level associated with that prediction.

For a given image, the described chimney detection model generates the bounding box output for its detections and calculates the confidence level associated with each detection. A post-processor that converts bounding boxes detections into chimney existence information is provided as a part of this module. Recognizing an image may contain more than one building at a time, this post-processor is capable of detecting chimneys for multiple building instances in an input image. 