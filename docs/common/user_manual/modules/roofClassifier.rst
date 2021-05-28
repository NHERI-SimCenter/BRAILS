.. _lbl-roofClassifier:

Roof Shape Classifier
=====================

The Roof Shape Classifier is a module built upon the :ref:`lbl-genericImageClassifier` module. It's purpose is to classify the roof shape of a building.  The roof classifier takes a list of satellite images as the input and will classify the roof type into one of three categories: gabled, hipped, and flat. 

.. _roof_types:
.. list-table:: Roof prototypes

    * - .. figure:: ../../../images/technical/flat.jpg

           Flat

      - .. figure:: ../../../images/technical/gable.jpg

           Gabled
      - .. figure:: ../../../images/technical/hip.jpg

           Hipped

.. note::

   #. The Roof shape classifier will only identify one roof type per image. As a consequence, the image you provide should only contain the building whose roof type is to be determined.

   #. Many roof shapes found these days are classified as `complex`.  The categories `hipped`, `flat`, and `gabled` were chosen as these are the classifications used in Hazus.

.. warning:: 

   The classifier takes an image as the input and will always produce a prediction. 
   Since the classifier is trained to classify only a specific category of images, 
   its prediction is meaningful only if the input image belongs to the category the model is trained for.

   
Use the module
--------------

Suppose you had a number of images in a folder named image_examples/Roof.


.. list-table::

    * - .. figure:: ../../../images/image_examples/Roof/gabled/76.png

           image_examples/Roof/gabled/76.png Gabled

      - .. figure:: ../../../images/image_examples/Roof/hipped/54.png 

           image_examples/Roof/hipped/54.png  Hipped

      - .. figure:: ../../../images/image_examples/Roof/flat/94.png 

           image_examples/Roof/flat/94.png  Flat


The following is a python script you would create to use the RoofClassifier to predict the roof shapes for these images:
		   
.. code-block:: none 

    # import the module
    from brails.modules import RoofClassifier

    # initialize a roof classifier
    roofModel = RoofClassifier()

    # define the paths of images in a list
    imgs = ['image_examples/Roof/gabled/76.png',
            'image_examples/Roof/hipped/54.png',
            'image_examples/Roof/flat/94.png']
    
    # use the model to predict
    predictions = roofModel.predict(imgs)


The predictions obtained when the script runs will look like following:

.. code-block:: none 

    Image :  image_examples/Roof/gabled/76.png     Class : gabled (83.21%)
    Image :  image_examples/Roof/hipped/54.png     Class : hipped (100.0%)
    Image :  image_examples/Roof/flat/94.png       Class : flat (97.68%)
    Results written in file roofType_preds.csv


.. note::

   #. The reference to a pretrained model is in a file shipped with BRAILS and the first time you use this module, it will download that model from the internet to your local computer. This will allow you to use it directly without training your own model.

   #. The current pretrained model was trained with 6,000 labeled images utilizing ResNet50. To perform the training a number of buildings with the roof classifications desired were identified utilizing OpenStreetMaps, including 2,000 'flat', 'gabled', and 'hipped', respectively. Satellite images for those buildings were obtained using Google Maps, and these images were placed into one of three folders as discussed in :ref:`lbl-genericImageClassifier` module. Prior to training the model, the images in the folders were reviewed to ensure they contained an image of a roof and were of the correct roof shape. The noise in this dataset is negligible.

   #. As mentioned in the introduction, SimCenter is constantly updating these trained models. The simplest way to get the latest model is to update your BRAILS installation. This can be done by issuing the following in a terminal/powershell window:
   
      .. code-block:: none 

	  pip install -U BRAILS --upgrade

   #. The images used in the example can be downloaded from `Zenodo <https://zenodo.org/record/4562949/files/image_examples.zip>`_.

Retrain the model
-----------------

You can retrain the existing model with your own data. To do so, you would place each of your labeled images (images of type .png) into one three seperate folders.

.. code-block:: none 

    my_roof_shapes
    │── flat
    │       └── *.png
    │── hipped
    |      └── *.png
    └── gabled
           └── *.png


Then you would create a python script as shown below and run finally run that script to train the model.

.. code-block:: none 

    # Load images from a folder
    roofModel.loadData('my_roof_shapes')

    # Re-train it for only 1 epoch for this demo. You can increase it.
    roofModel.retrain(initial_epochs=1)

    # Test the re-trained model
    predictions = roofModel.predict(imgs)

    # Save the re-trained model
    roofModel.save('myCoolNewRoofModelv0.1')


To use your newly trained model with the Roof Shape classifier, you would include in the RoofClassifier's constructor the name of the trained model as shown in the following script.

.. code-block:: none 

    # import the module
    from brails.modules import RoofClassifier

    # initialize a roof classifier
    roofModel = RoofClassifier('myCoolNewRoofModelv0.1')

    # define the paths of images in a list
    imgs = ['image_examples/Roof/gabled/76.png',
            'image_examples/Roof/hipped/54.png',
            'image_examples/Roof/flat/94.png']
    
    # use the model to predict
    predictions = roofModel.predict(imgs)
