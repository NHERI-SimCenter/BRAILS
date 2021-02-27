.. _lbl-examples:

********
Examples
********

Example 1: Modules
===================

The following is an example showing how to call pretrained models to predict on images. 

You can run this example on your computer or in this notebook on `Google Colab <https://colab.research.google.com/drive/1zspDwK-rGA1gYcHZDnrQr_3Z27JL-ooS?usp=sharing>`_.

The images used in the example can be downloaded from here: `image_examples.zip <https://zenodo.org/record/4095668/files/image_examples.zip>`_.

  .. code-block:: python

    # import modules
    from brails.modules import RoofClassifier, OccupancyClassifier, SoftstoryClassifier

    # initilize a roof classifier
    roofModel = RoofClassifier()

    # initilize an occupancy classifier
    occupancyModel = OccupancyClassifier()

    # initilize a soft-story classifier
    ssModel = SoftstoryClassifier()

    # use the roof classifier 

    imgs = ['image_examples/Roof/gabled/76.png',
            'image_examples/Roof/hipped/54.png',
            'image_examples/Roof/flat/94.png']

    predictions = roofModel.predict(imgs)

    # use the occupancy classifier 

    imgs = ['image_examples/Occupancy/RES1/51563.png',
            'image_examples/Occupancy/RES3/65883.png']

    predictions = occupancyModel.predict(imgs)

    # use the softstory classifier 

    imgs = ['image_examples/Softstory/Others/3110.jpg',
            'image_examples/Softstory/Softstory/901.jpg']

    predictions = ssModel.predict(imgs)


Example 2: Workflow
=====================

The following is an example showing how to create a building inventory for a city.

You can run this example on your computer or in this notebook on `Google Colab <https://colab.research.google.com/drive/1tG6xVRCmDyi6K8TWgoNd_31vV034VcSO?usp=sharing>`_.

You need to provide the Google maps API key for downloading street view and satellite images.

Instructions on obtaining the API key can be found here: `<https://developers.google.com/maps/documentation/embed/get-api-key>`_.

Use should limit the number of buildings (numBldg) because of :ref:`this <limitthenumber>`. 

  .. code-block:: python

    from brails.CityBuilder import CityBuilder

    cityBuilder = CityBuilder(attributes=['occupancy','roofshape'], 
                   numBldg=1000,random=False, place='Lake Charles, LA', 
                   GoogleMapAPIKey='put-your-API-key-here',
                   overwrite=True)

    BIM = cityBuilder.build()


:attributes (list):     
    A list of building attributes, such as ['story', 'occupancy', 'roofshape'], which are available in the current version.
:numBldg (int):         
    Number of buildings to generate.
:random (bool):         
    Randomly select numBldg buildings from the database if random is True.
:place (str):           
    The region of interest, e.g., Berkeley, California.
:GoogleMapAPIKey (str): 
    Google API Key.
:overwrite (bool):      
    Overwrite existing tmp files. Default value is False.


.. figure:: figures/Berkeley.png
   :name: num_building_city
   :align: center
   :figclass: align-center
   :figwidth: 90%

   Generated Buildings

.. csv-table:: Generated BIM File
   :name: bldg_inv
   :file: data/Berkeley.csv
   :header-rows: 1
   :align: center


    
Example 3: Workflow
======================

The following is an example showing how to create a building inventory for a region defined using a bounding box.

You can run this example on your computer or in this notebook on `Google Colab <https://colab.research.google.com/drive/1tG6xVRCmDyi6K8TWgoNd_31vV034VcSO?usp=sharing>`_.

You need to provide the Google maps API key for downloading street view and satellite images.

Instructions on obtaining the API key can be found here: `<https://developers.google.com/maps/documentation/embed/get-api-key>`_.

Use should limit the number of buildings (numBldg) because of :ref:`this <limitthenumber>`. 

  .. code-block:: python

    from brails.CityBuilder import CityBuilder

    cityBuilder = CityBuilder(attributes=['softstory','occupancy','roofshape'], 
                   numBldg=100,random=False, bbox=[37.872187, -122.282178,37.870629, -122.279765], 
                   GoogleMapAPIKey='put-your-API-key-here',
                   overwrite=True)

    BIM = cityBuilder.build()

    
:attributes (list):     
    A list of building attributes, such as ['story', 'occupancy', 'roofshape'], which are available in the current version.
:numBldg (int):         
    Number of buildings to generate.
:random (bool):         
    Randomly select numBldg buildings from the database if random is True.
:bbox (list):           
    [north, west, south, east], which defines a region of interest.
:GoogleMapAPIKey (str): 
    Google API Key.
:overwrite (bool):      
    Overwrite existing tmp files. Default value is False.

.. figure:: figures/Christchurch.png
   :name: num_building_city
   :align: center
   :figclass: align-center
   :figwidth: 90%

   Generated Buildings

.. csv-table:: Generated BIM File
   :name: bldg_inv
   :file: data/Christchurch.csv
   :header-rows: 1
   :align: center

    


    


    
    


