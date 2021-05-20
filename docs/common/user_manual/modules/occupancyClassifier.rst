.. _lbl-occupancyClassifier:

Occupancy Classifier
========================

The Occupancy Classifier is a module built upon the :ref:`lbl-genericImageClassifier` module. 

The module is shipped with BRAILS, 
so you don't have to install it standalone if you've installed BRAILS following the :ref:`lbl-install` instruction. 

It takes a list of street view images of residential buildings as the input, and classify the buildings into three categories: 
RES1 (single family building), RES3 (multi-family building), COM(Commercial building).


Use the module
-----------------

A pretrained model is shipped with BRAILS. So you can use it directly without training your own model.

The first time you initialize this model, it will download the model from the internet to your local computer.

The images used in the example can be downloaded from `here <https://zenodo.org/record/4627958/files/image_examples.zip>`_.

.. code-block:: none 

    # import the module
    from brails.modules import OccupancyClassifier

    # initialize an occupancy classifier
    occupancyModel = OccupancyClassifier()

    # define the paths of images in a list
    imgs = ['image_examples/Occupancy/RES1/36887.jpg',
        'image_examples/Occupancy/RES3/37902.jpg',
        'image_examples/Occupancy/COM/42915.jpg']
    
    # use the model to predict
    predictions = occupancyModel.predict(imgs)


The predictions look like this:

.. code-block:: none 

    Image :  image_examples/Occupancy/RES1/36887.jpg     Class : RES1 (100.0%)
    Image :  image_examples/Occupancy/RES3/37902.jpg     Class : RES3 (100.0%)
    Image :  image_examples/Occupancy/COM/42915.jpg     Class : COM (100.0%)
    Results written in file tmp/occupancy_preds.csv

Sample images used in this example are:

.. list-table::

    * - .. figure:: ../../../images/image_examples/Occupancy/RES1/36887.jpg

           Predicted as Single-family Building

      - .. figure:: ../../../images/image_examples/Occupancy/RES3/37902.jpg

           Predicted as Multi-family Building

      - .. figure:: ../../../images/image_examples/Occupancy/COM/42915.jpg

           Predicted as Commercial Building
    
.. note:: 

   The classifier takes an image as the input and will always produce a prediction. 
   Since the classifier is trained to classify only a specific category of images, 
   its prediction is meaningful only if the input image belongs to the category the model is trained for.

   
Retrain the model
------------------

You can retrain the existing model with your own data.

.. code-block:: none 

    # Load images from a folder
    occupancyModel.loadData('folder-of-images')

    # Re-train it for only 1 epoch for this demo. You can increase it.
    occupancyModel.retrain(initial_epochs=1)

    # Test the re-trained model
    predictions = occupancyModel.predict(imgs)

    # Save the re-trained model
    occupancyModel.save('myCoolNewModelv0.1')