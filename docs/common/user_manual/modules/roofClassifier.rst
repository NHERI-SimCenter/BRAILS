.. _lbl-roofClassifier:

Roof Classifier
========================

The Roof Classifier is a module that wraps the :ref:`lbl-generalImageClassifier`. 

The module is shipped with BRAILS, 
so you don't have to install it standalone if you've installed BRAILS following the :ref:`lbl-install` instruction. 

It takes a list of satellite images of roof tops as the input, and classify the roof types into three categories: gabled, hipped, and flat.



Use the module
-----------------

A pretrained model is shipped with BRAILS. So you can use it directly without training your own model.

The first time you initialize this model, it will download the model from the internet to your local computer.

.. code-block:: none 

    # import the module
    from brails.RoofTypeClassifier import RoofClassifier

    # initialize a roof classifier
    roofModel = RoofClassifier()

    # define the paths of images in a list
    imgs = ['image_examples/Roof/gabled/76.png',
            'image_examples/Roof/hipped/54.png',
            'image_examples/Roof/flat/94.png']
    
    # use the model to predict
    predictions = roofModel.predict(imgs)


The predictions look like this:

.. code-block:: none 

    Image :  image_examples/Roof/gabled/76.png     Class : gabled (83.21%)
    Image :  image_examples/Roof/hipped/54.png     Class : hipped (100.0%)
    Image :  image_examples/Roof/flat/94.png     Class : flat (97.68%)
    Results written in file roofType_preds.csv

The images used in this example are:

.. list-table::

    * - .. figure:: ../../../images/image_examples/Roof/gabled/76.png

           image_examples/Roof/gabled/76.png Gabled

      - .. figure:: ../../../images/image_examples/Roof/hipped/54.png 

           image_examples/Roof/hipped/54.png  Hipped

      - .. figure:: ../../../images/image_examples/Roof/flat/94.png 

           image_examples/Roof/flat/94.png  Flat



.. note:: 

   The classifier takes an image as the input and will always produce a prediction. 
   Since the classifier is trained to classify only a specific category of images, 
   its prediction is meaningful only if the input image belongs to the category the model is trained for.
