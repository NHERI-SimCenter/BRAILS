.. _lbl-occupancyClassifier:

Occupancy Classifier
========================

The Occupancy Classifier is a module built upon the :ref:`lbl-genericImageClassifier` module. It's purpose is to label a building based on occupancy class. The occupancy classifier takes a list of street view images as the input and will classify the building into one of three categories: RES1, RES3, COM1.

    #. RES1 - Single Family

    #. RES2 - Multi-Familiy

    #. COM1 - Commercial

.. note::

   #. The Occupancy class classifier will only identify one class per image. As a consequence, the image you provide should only contain the building whose type is to be determined.

   #. There are many possible divisions for buildings other than those presented here, e.g. schools, hospitals, industrial. The ones used here are a consequence of the types required for SimCenter's Atlantic City testbed and the availability of data needed to train a model for use by the classifier.

.. warning:: 

   The classifier takes an image as the input and will always produce a prediction. 
   Since the classifier is trained to classify only a specific category of images, 
   its prediction is meaningful only if the input image belongs to the category the model is trained for.      

Use the module
--------------

Suppose you had a number of images in a folder named image_examples/Occupancy.

.. note::

   The images used in this example can be downloaded from `Zenodo <https://zenodo.org/record/4627958/files/image_examples.zip>`_.

.. list-table::

    * - .. figure:: ../../../images/image_examples/Occupancy/RES1/36887.jpg

           Single-family Building (36887.png)

      - .. figure:: ../../../images/image_examples/Occupancy/RES3/37902.jpg

           Multi-family Building (37902.jpg)

      - .. figure:: ../../../images/image_examples/Occupancy/COM/42915.jpg

           Commercial Building (42915.jpg)

The following python script can be used to classify these images using the Occupancy Classifier:
	   
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


The predictions obtained when the script runs will look like following:    

.. code-block:: none 

    Image :  image_examples/Occupancy/RES1/36887.jpg     Class : RES1 (100.0%)
    Image :  image_examples/Occupancy/RES3/37902.jpg     Class : RES3 (100.0%)
    Image :  image_examples/Occupancy/COM/42915.jpg     Class : COM (100.0%)
    Results written in file tmp/occupancy_preds.csv

.. note::
   
    #. The reference to a pretrained model is in a file that is shipped with BRAILS and the first time you use this module, it will download that model from the internet to your local computer. This will allow you to use it directly without training your own model.
 
    #. The current pretrained model was trained with 15,743 labeled images utilizing ResNet50. To perform the training a number of buildings with the occupancy classifications desired were identified from OpenStreetMaps and a dataset provided by New Jersey Department of Environmental Protection (NJDEP), including 7,868 RES1 (2,868 'detached' from OpenStreetMap + 4,999 'RES1' from NJDEP), 5,074 RES3 (2,207 'apartment' from OpenStreetMap and 2,867 'RES3' from NJDEP), and 2,804 COM (2,418 'commercial' from OpenStreetMap + 386 'COM' from NJDEP), respectively. Satellite images for those buildings were obtained using Google Maps, and these images were placed into one of three folders as discussed in :ref:`lbl-genericImageClassifier` module.

    #. **NJDEP**: is a dateset that SimCenter uses as part of it's Atlantic City Testbed. Information on the data and it's contents is explained in the `documentation for that testbed <https://nheri-simcenter.github.io/R2D-Documentation/common/testbeds/atlantic_city/index.html>`_. The NJDEP information was made possible through the ongoing collaboration between the University of Notre dame and the New Jersey Department of Community Affairs (NJ DCA) through the NJcoast project. NJ DCA’s Keith Henderson’s sustained support and collaboration is greatly appreciated.
 
    #. As mentioned in the introduction, SimCenter is constantly updating these trained models. The simplest way to get the latest model is to update your BRAILS installation. This can be done by issuing the following in a terminal/powershell window:
    
       .. code-block:: none 
 
          pip install -U BRAILS --upgrade


    #. As will be shown in :ref:`lbl-understand`, it is important to review sample results and possible retrain the model yourself.


   
Retrain the model
------------------

You can retrain the existing model with your own data. To do so, you would place each of your labeled images (images of type .png) into one three seperate folders.

.. code-block:: none 

    my_occupancy_images
    │── RES1
    │       └── *.png
    │── RES3
    |      └── *.png
    └── COM1
           └── *.png


Then you would create a python script as shown below and run finally run that script to train the model.

.. code-block:: none 

    # Load images from a folder
    occupancyModel.loadData('my_occupancy_images')

    # Re-train it for only 1 epoch for this demo. You can increase it.
    occupancyModel.retrain(initial_epochs=1)

    # Test the re-trained model
    predictions = occupancyModel.predict(imgs)

    # Save the re-trained model
    occupancyModel.save('myCoolNewOccupancyModelv0.1')

To use your newly trained model with the Occupancy type classifier, you would include in the OccupancyModels's constructor the name of the trained model as shown in the following script.


.. code-block:: none 

    # import the module
    from brails.modules import OccupancyClassifier

    # initialize an occupancy classifier
    occupancyModel = OccupancyClassifier('myCoolNewOccupancyModelv0.1')

    # define the paths of images in a list
    imgs = ['image_examples/Occupancy/RES1/36887.jpg',
        'image_examples/Occupancy/RES3/37902.jpg',
        'image_examples/Occupancy/COM/42915.jpg']
    
    # use the model to predict
    predictions = occupancyModel.predict(imgs)
