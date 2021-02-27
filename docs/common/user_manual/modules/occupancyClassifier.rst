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

The images used in the example can be downloaded from `here <https://zenodo.org/record/4562949/files/image_examples.zip>`_.

.. code-block:: none 

    # import the module
    from brails.modules import OccupancyClassifier

    # initialize an occupancy classifier
    occupancyModel = OccupancyClassifier()

    # define the paths of images in a list
    imgs = ['image_examples/Occupancy/RES1/51563.png',
            'image_examples/Occupancy/RES3/65883.png']
    
    # use the model to predict
    predictions = occupancyModel.predict(imgs)


The predictions look like this:

.. code-block:: none 

    Image :  image_examples/Occupancy/RES1/51563.png     Class : RES1 (66.41%)
    Image :  image_examples/Occupancy/RES3/65883.png     Class : RES1 (49.51%)
    Results written in file occupancy_preds.csv 

The images used in this example are:

.. list-table::

    * - .. figure:: ../../../images/image_examples/Occupancy/RES1/51563.png 

           images/image_examples/Occupancy/RES1/51563.png Single-family Building

      - .. figure:: ../../../images/image_examples/Occupancy/RES3/65883.png

           images/image_examples/Occupancy/RES3/65883.png Multi-family Building
    

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