.. _lbl-softstoryClassifier:

Soft-story Building Classifier
===============================

The Soft-story Building Classifier is a module that wraps the :ref:`lbl-genericImageClassifier`. 

The module is shipped with BRAILS, 
so you don't have to install it standalone if you've installed BRAILS following the :ref:`lbl-install` instruction. 

It takes a list of street view images of buildings as the input, and classify the buildings into two categories: soft-story building and  other building.


Use the module
-----------------

A pretrained model is shipped with BRAILS. So you can use it directly without training your own model.

The first time you initialize this model, it will download the model from the internet to your local computer.

.. code-block:: none 

    # import the module
    from brails.SoftstoryClassifier import SoftstoryClassifier

    # initilize a soft-story classifier
    ssModel = SoftstoryClassifier()

    # define the paths of images in a list
    imgs = ['image_examples/Softstory/Others/3110.jpg',
            'image_examples/Softstory/Softstory/901.jpg']
    
    # use the model to predict
    predictions = ssModel.predict(imgs)


The predictions look like this:

.. code-block:: none 

    Image :  image_examples/Softstory/Others/3110.jpg     Class : others (96.13%)
    Image :  image_examples/Softstory/Softstory/901.jpg     Class : softstory (96.31%)
    Results written in file softstory_preds.csv

The images used in this example are:

.. list-table::

    * - .. figure:: ../../../images/image_examples/Softstory/Others/3110.jpg

           image_examples/Softstory/Others/3110.jpg Non-Soft-story Building

      - .. figure:: ../../../images/image_examples/Softstory/Softstory/901.jpg 

           image_examples/Softstory/Softstory/901.jpg Soft-story Building