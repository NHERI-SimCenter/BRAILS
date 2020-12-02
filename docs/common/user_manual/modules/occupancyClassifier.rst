.. _lbl-occupancyClassifier:

Occupancy Classifier
========================

The Occupancy Classifier is a module that wraps the :ref:`lbl-generalImageClassifier`. 

The module is shipped with BRAILS, 
so you don't have to install it standalone if you've installed BRAILS following the :ref:`lbl-install` instruction. 

It takes a list of street view images of residential buildings as the input, and classify the buildings into two categories: single family building and  multi-family building.


Use the module
-----------------

A pretrained model is shipped with BRAILS. So you can use it directly without training your own model.

The first time you initialize this model, it will download the model from the internet to your local computer.

.. code-block:: none 

    # import the module
    from brails.OccupancyClassClassifier import OccupancyClassifier

    # initialize an occupancy classifier
    occupancyModel = OccupancyClassifier()

    # define the paths of images in a list
    imgs = ['image_examples/Occupancy/RES1/51563.png',
            'image_examples/Occupancy/RES3/65883.png']
    
    # use the model to predict
    predictions = occupancyModel.predict(imgs)


The predictions look like this:

.. code-block:: none 

    Image :  image_examples/Occupancy/RES1/51563.png     Class : RES1 (99.99%)
    Image :  image_examples/Occupancy/RES3/65883.png     Class : RES3 (98.67%)
    Results written in file occupancy_preds.csv 

The images used in this example are:

.. list-table::

    * - .. figure:: ../../../images/image_examples/Occupancy/RES1/51563.png 

           images/image_examples/Occupancy/RES1/51563.png Single-family Building

      - .. figure:: ../../../images/image_examples/Occupancy/RES3/65883.png

           images/image_examples/Occupancy/RES3/65883.png Multi-family Building
    