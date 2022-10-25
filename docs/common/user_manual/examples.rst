.. _lbl-examples:

********
Examples
********

Example 1: Inventory Generator
==============================

This code snippet shows you how to use BRAILS InventoryGenerator module to generate a building inventory for an area. You provide the town, in this case San Rafael, CA, the number of buildings to sample from the town (remember it will start to cost you money to download images from Google and thus the nblgs option on line 6), and a Google Maps API key (line 7). The code when it runs will print the inventory and also produce an inventory file, inventory.csv.

.. literalinclude:: ./ex1.py
   :language: python
   :linenos:

The inventory.csv file will look something like the following:

.. literalinclude:: ./inventory.csv
   :language: csv
   :linenos:
      
.. warning::

    1. You need to replace the text on line 7 "PROVIDE_YOUR_KEY" with your Google API key. The examples will fail otherwise. Instructions on obtaining your Google API key can be found here: `<https://developers.google.com/maps/documentation/embed/get-api-key>`_.   
    2. The code when it runs will place additional data in the tmp folder. Additional data will include the trained models (which might take awhile the first time you run the example) and images downloaded from the www, which are placed in "tmp/images" folder.  In the images folder the script will download images for the roofshapes and place them in "tmp/images/satellite" folder> It will download other images for calculating the building heights and number of stories, which will be placed placed in the "tmp/images/street" folder. These folders are required to run Example 2, and this example will create and populate these folder with images.
   3. For the classifiers, it is important to retrain a trained model with local data first to get decent results. How to do this will be shown in an upcoming example, Example 3.


Example 2: Modules
==================

This second example will use specific modules within BRAILS to label a set of images. For this example it assumes images are located in two folders in the tmp folder, which is located in the current folder. These folders are "tmp/images/satellite" and "tmp/images/street". Note: If this example is run after running Example 1, it will use the images already there.

.. literalinclude:: ./ex2.py
   :language: python
   :linenos:

When the example runs, two files will be created in the tmp folder, Occupancy_preds.csv and roofType_preds.csv, containing the information deduced by each module for the particular images. The roofType_preds.csv file for the sample of buildings obtained from my running of Example 1 results in the following:

.. literalinclude:: ./tmp/roofType_preds.csv
   :language: python
   :linenos:
   
.. warning::

   1. This example assumes images exist in the tmp/images/satellite and tmp/images/street folders. If you ran example 1, data will exist in these folders.
   2. As mentioned in Example 1, for the classifiers it is important to retrain with local data first to get decent results. The steps to do this will be demonstrated in an upcomng Example 3.


    
    


