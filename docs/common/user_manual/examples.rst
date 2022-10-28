.. _lbl-examples:

.. role:: python(code)
   :language: python

********
Examples
********

Example 1: Inventory Generation Using BRAILS
================================================

This example shows you how to use the BRAILS InventoryGenerator workflow to obtain the building inventory for an area. At the minimum, :python:`InventoryGenerator` requires the user to provide location information and a Google API key using :python:`location` and :python:`GoogleAPIKey` input parameters. Each time :python:`InventoryGenerator` is executed, it will use Google APIs to access satellite and street-level imagery of the buildings in the defined area. Given that accessing these images requires using Google credits, it is advised to experiment with :python:`InventoryGenerator` for a few buildings using the :python:`nbldgs` input parameter. 

The code snippet below runs :python:`InventoryGenerator` for San Rafael, CA, and obtains the number of floors, roof shape, and building height information for a randomly selected 20 buildings from this area. Once the code is executed, it prints the obtained inventory and saves it in an inventory file titled inventory.csv in the directory the code is executed.

.. literalinclude:: ./sample_scripts/ex1.py
   :language: python
   :linenos:

The contents of inventory.csv file will appear as follows. Given :python:`randomSelection` was set to :python:`True`, it is expected that the information stored in inventory.csv will be different each time this code is executed.

.. literalinclude:: ./sample_outputs/inventory.csv
   :language: csv
   :linenos:
      
.. warning::

    1. To successfully run this example, you need to replace the text on line 6 "PROVIDE_YOUR_KEY" with your Google API key. Instructions on obtaining a Google API key can be found here: `<https://developers.google.com/maps/documentation/embed/get-api-key>`_. BRAILS uses Maps Static API and Street View Static API to access satellite and street-level imagery, so please make sure to enable these APIs with your Google API key.   
    2. Once the code is executed, it will create tmp folder in the directory it is executed. This folder will contain the pretrained BRAILS models (which may take a moment to download the first time you run the example) in "tmp/models" and images accessed from Google APIs in "tmp/images" folder. :python:`InventoryGenerator` will download satellite images for the roof shapes and place them in "tmp/images/satellite" folder. It will download street-level images for calculating the building heights and number of stories and save them in the "tmp/images/street" folder. The folders for satellite and street-level imagery are required to run Example 2.
    3. For best results with BRAILS classification models, it is essential to retrain a pretrained model with data specific to the region for which inventory information is required. The procedure for retraining existing BRAILS modules is discussed in Example 3.


Example 2: Calling Individual Modules of BRAILS
=================================================

This example demonstrates the use of specific modules within BRAILS to predict certain building attributes from a provided set of images. The code below assumes that the folders of images on which the inferences will be performed, i.e., "tmp/images/satellite" and "tmp/images/street", are located in the directory where this code is executed. If this example is run after Example 1, these images will already be located in the specified directories. 

The following code runs BRAILS roof shape and occupancy type prediction modules for satellite and street-level imagery located in "tmp/images/satellite" and "tmp/images/street" folders.

.. literalinclude:: ./sample_scripts/ex2.py
   :language: python
   :linenos:

Once the code is executed, two CSV files will be created in the tmp folder, Occupancy_preds.csv, and roofType_preds.csv. These files contain the information deduced by each module for the input images. For example, the roofType_preds.csv file for the set of building images obtained after Example 1 results in the following:

.. literalinclude:: ./sample_outputs/roofType_preds.csv
   :language: python
   :linenos:
   
.. warning::

   1. This example assumes the inference images are stored in tmp/images/satellite and tmp/images/street folders. If you ran Example 1 beforehand, these folders should already be populated as expected.
   2. As mentioned in Example 1, for best results, it is essential to retrain a pretrained classification model with data specific to the region for which inventory information is required. The steps to retrain a BRAILS model are illustrated in Example 3.

Example 3: Retraining Existing Modules of BRAILS
=================================================
    
This example shows the steps required to retrain an existing BRAILS module using a user-provided training dataset. The script below illustrates this process for BRAILS roof shape classifier module only, but the same steps are applicable to all other BRAILS modules. In this specific implementation, the roof classifier model is retrained for a learning rate (:python:`lr`) of 0.001, batch size of 64 (:python:`batch_size`), and the number of epochs of 5 (:python:`epochs`). These input parameters are training hyperparameters that may depend on the application. At the minimum, retraining a deep learning module typically requires more epochs than in this example.

.. literalinclude:: ./sample_scripts/ex3.py
   :language: python
   :linenos:
   
.. warning::

   1. This example assumes that the user has the training data stored in a folder name roofType in the directory where this code is executed. If the training data is stored elsewhere, please change the input for :python:`imgDir` accordingly.
   2. BRAILS, by default, expects the training data folder to contain three subfolders named flat, gabled, and hipped, with the images for each category saved under their respective folder.   

