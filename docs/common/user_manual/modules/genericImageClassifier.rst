.. _lbl-genericImageClassifier:

Generic Image Classifier
========================

The Generic Image Classifier is a module that can be used for creating user defined classifier.

The user provides categorized images to this module. 

An image classifier will be built automatically based on the images provided.

The classifier is then trained and saved locally.

The trained classifier can be used for inference readily and can be shared with other users.

During the inference stage, the classifier takes a list of images as the input, and predicts the classes of the images. 



The following is an example, in which a classifier is created and trained.

The image dataset for this example contains street view images categorized according to construction materials.

The dataset can be downloaded `here <https://zenodo.org/record/4416845/files/building_materials.zip>`_.

When unzipped, the file gives the 'building_materials' which is a directory where you have your images for training:


.. code-block:: none 

    building_materials
    │── class_1
    │       └── *.png
    │── class_2
    |      └── *.png
    │── ...
    |
    └── class_n
           └── *.png



Construct the image classifier 
------------------------------------


.. code-block:: none 

    # import the module
    from brails.modules import ImageClassifier

    # initialize the classifier, give it a name
    materialClassifier = ImageClassifier(modelName='materialClassifierV0.1')

    # load data
    materialClassifier.loadData('building_materials')






Train the model
---------------------

.. code-block:: none 

    # train the base model for 50 epochs and then fine tune for 200 epochs
    materialClassifier.train(baseModel='InceptionV3', initial_epochs=50,fine_tune_epochs=200)


It is better to run the above example on a GPU machine.

You can choose from the following options for the baseModel key:

.. code-block:: none 

    'Xception',
    'VGG16',
    'VGG19',
    'ResNet50',
    'ResNet101',
    'ResNet152',
    'ResNet50V2',
    'ResNet101V2',	
    'ResNet152V2',	
    'InceptionV3',	
    'InceptionResNetV2',
    'MobileNet',
    'MobileNetV2',	
    'DenseNet121',	
    'DenseNet169',	
    'DenseNet201',	
    'NASNetMobile',
    'NASNetLarge',	
    'EfficientNetB0',	
    'EfficientNetB1',	
    'EfficientNetB2',	
    'EfficientNetB3',	
    'EfficientNetB4',	
    'EfficientNetB5',	
    'EfficientNetB6',	
    'EfficientNetB7'



Use the model
---------------------


Now you can use the trained model to predict on given images.

.. code-block:: none 

    # If you are running the inference from another place, you need to initialize the classifier firstly:
    from brails.GenericImageClassifier import ImageClassifier
    materialClassifier = ImageClassifier(modelName='materialClassifierV0.1')
                                            
    # define the paths of images in a list
    imgs = ['building_materials/concrete/469 VAN BUREN AVE Oakland2.jpg',
            'building_materials/masonry/101 FAIRMOUNT AVE Oakland2.jpg',
            'building_materials/wood/41 MOSS AVE Oakland2.jpg']

    # use the model to predict
    predictions = materialClassifier.predict(imgs)


The predictions will be written in preds.csv under the current directory.






