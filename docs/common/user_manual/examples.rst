.. _lbl-examples:

********
Examples
********

Example 1: Inventory Generator
==============================

This code snippet shows you how to use BRAILS to generate a building inventory for a area. You provide the town, in this case Tiburon, CA, the number of buildings to sample from the town (remember it will start to cost you money to download images from Google), and a Google Maps API key. The code when it runs will print the inventory and also produce an inventory file, inventory.csv.

.. literalinclude:: ./ex1.py
   :language: python
   :linenos:

The inventory.csv file will look something like the following:

.. literalinclude:: ./inventory.csv
   :language: csv
   :linenos:
      
.. warning::

    1. The code when it runs will place additional data in the tmp folder. Additional data will include the trained modules and images downloaded from the www. In the images folder it will download images for the roofshapes and place them in "tmp/images/satellite" folder, and other images for calculating the heights will bee placed placed in the "tmp/images/street folder". These are required to run example 2.
    2. You need to replace the text PROVIDE_YOUR_KEY with your oogle API key. The examples will fail otherwisee. Instructions on obtaining your Google API key can be found here: `<https://developers.google.com/maps/documentation/embed/get-api-key>`_.
   2. For the classifiers, it is important to pretrain with local data first to get decent results.       


Example 2: Modules
==================

This second example will use specific modules within BRAILS to label a set of images. For this example it assumes images are located in the current folder in the "tmp/images/satellite" and "tmp/images/street". If this example is run after running Example 1, it will use the images already there.

.. literalinclude:: ./ex2.py
   :language: python
   :linenos:

When the example runs, two files will be created containg the information deduced for each module.      
   
.. warning::

   1. This example assumes images exist in the tmp/images/satellite and tmp/images/street folders. If you ran example 1, data will exist in these folders.
   2. As for example 1, for the classifiers it is important to pretrain with local data first to get decent results.


    
    


