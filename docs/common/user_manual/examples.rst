.. _lbl-examples:

********
Examples
********

The following is an example showing how to call pretrained models to predict on images.

The images used in the example can be downloaded from here: `image_examples.zip <https://zenodo.org/record/4095668/files/image_examples.zip>`_.

  .. code-block:: python

    # import modules
    from brails.RoofTypeClassifier import RoofClassifier
    from brails.OccupancyClassClassifier import OccupancyClassifier
    from brails.SoftstoryClassifier import SoftstoryClassifier

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





