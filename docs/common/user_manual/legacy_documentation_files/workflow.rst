.. _lbl-workflow-tutorial:

Workflow
================

The workflow is designed to facilitate the creation of regional building inventories. 

It is implemented in a class, CityBuilder. 

  .. code-block:: python

    from brails.CityBuilder import CityBuilder

    cityBuilder = CityBuilder(attributes, 
                    numBldg, 
                    random,
                    bbox, 
                    place, 
                    footPrints, 
                    save, 
                    fileName, 
                    workDir,
                    GoogleMapAPIKey, 
                    overwrite, 
                    reDownloadImgs)


:attributes (list):     
    A list of building attributes, such as [':ref:`roofshape<lbl-roofClassifier>`', ':ref:`occupancy<lbl-occupancyClassifier>`', ':ref:`softstory<lbl-softstoryClassifier>`', ':ref:`elevated<lbl-foundationElevationClassifier>`', ':ref:`year<lbl-yearClassifier>`', ':ref:`numstories<lbl-nFloorDetector>`'], which are available in the current version.
:numBldg (int):         
    Number of buildings to generate.
:random (bool):         
    Randomly select numBldg buildings from the database if random is True.
:bbox (list):           
    [north, west, south, east], which are latitudes and longitudes of two corners that define a region of interest. 
:place (str):           
    The region of interest, e.g., 'Berkeley, California'.
:footPrints (str):      
    The footprint provide, choose from 'OSM' or 'Microsoft'. The default value is 'OSM'.
:save (bool):           
    Save temporary files. Default value is True.
:fileName (str):        
    Name of the generated BIM file. Default value will be generated if not provided by the user.
:workDir (str):         
    Work directory where all files will be saved. Default value is './tmp'
:GoogleMapAPIKey (str): 
    Google API Key. Must be provided to use the workflow.
:overwrite (bool):      
    Overwrite existing tmp files. Default value is False.
:reDownloadImgs (bool): 
    Re-download even an image exists locally. Default value is False.



Specify attributes to be collected
-----------------------------------

Use the key 'attributes' to specify a list of attributes you intend to collect for each building. 
Available ones in the current version include: 
['roofshape', 
'occupancy', 
'softstory',
'elevated',
'year',
'numstories']. 
These attributes will be inferred from images using specific :ref:`modules <lbl-modules>`.

* roofshape is the roof class, details can be found in :ref:`lbl-roofClassifier`.

* occupancy is the occupancy class, details can be found in :ref:`lbl-occupancyClassifier`. 

* softstory is the soft-story attribute, details can be found in :ref:`lbl-softstoryClassifier`.

* elevated is the foundation elevation attribute, details can be found in :ref:`lbl-foundationElevationClassifier`.

* year is the year built, details can be found in :ref:`lbl-yearClassifier`. 

* numstories is the number of stories, details can be found in :ref:`lbl-nFloorDetector`. 



.. _limitthenumber:

Limit the number of buildings to be collected
-----------------------------------------------
The workflow will download a street view image and a satellite view image for each building.
The images are downloaded from Google Maps using  your personal `Google API Keys <https://developers.google.com/maps/documentation/embed/get-api-key>`_.
The price of the API calls can be found `here <https://cloud.google.com/maps-platform/pricing>`_. 
As of February 3, 2021, $7 per 1,000 street view images and $2 per 1,000 satellite images.
Each Google account has $200 free monthly usage. Exceeding that limit will result in being charged by Google. 

You can use the key 'numBldg' to limit the number of buildings to be generated. 


Control the selection randomness 
-----------------------------------
In a region, numBldg of buildings will be generated. 
You can use the key 'random' to specify if you want to randomly select numBldg buildings from the database.
If its value is False, BIM will be generated for the first numBldg buildings found in the footprint database. 

Define the region of interest
-------------------------------
There are two options to define the region of interest: 'bbox' or 'place'.

If 'bbox' is provided, the workflow will retrieve numBldg buildings within the bounding box defined by 'bbox'.

If 'bbox' is empty and 'place' is provided, the workflow will search the database based on 'place'.

Footprints options
----------------------------

Use the key 'footPrints' to specify the source of building footprints to be used in the workflow.
Currently, the workflow supports '
`OSM <https://www.openstreetmap.org/>`_' and '`Microsoft <https://github.com/microsoft/USBuildingFootprints>`_'.


Examples
----------------------------

Check the :ref:`lbl-examples`.

