.. _lbl-genericImageClassifier:

Generic Image Classifier
========================

The Generic Image Classifier is a class that can be used for creating user defined classifier. 

.. container:: toggle
	       
   .. container:: header

       **Methods**

   #. **init**
      
      #. **modelName** Path to model
      #. **classNames** List of class names
      #. **resultFile** default = preds.csv
      #. **workDir** default = tmp
      #. **printRes** default=True

   #.  train

      #. **baseModel**: default='InceptionV3'
      #. **lr1**: default=0.0001
      #. **initial_epochs**: default==10,
      #. **fine_tune_at**: default==300,
      #. **lr2**: default=0.00001,
      #. **fine_tune_epochs:** default==50,
      #. **color_mode default**: ='rgb',
      #. **horizontalFlip**: default=False,
      #. **verticalFlip**: default=False,
      #. **dropout**: default=0.6
      #. **randomRotation**: default=0.0,
      #. **callbacks**: default==[],
      #. **plot**: default==True

   #. **predict**

      #. **image**: single image or list of images
      #. **color_mode**: default='rgb'

   #.  loadData

      #. **imgDir**:
      #. **valimgDir**: default=''
      #. **randomseed**: default=1993,
      #. **color_mode** default='rgb',
      #. **image_size** default=(256, 256),
      #. **batch_size**: default = 32,
      #. **split**: default=[0.8,0.2]):   	       	 


Decription
----------

This class implements the abstraction of an image classifier, it can be first used to train the classifier and save the data needed by the classifier locally. Once trained, the classifier can be used to predict the class of each image given a set of images. The user provides categorized images to the classifier so that it can be initially trained.

Example
-------

The following is an example, in which a classifier is created and trained.

The image dataset for this example contains street view images categorized according to construction materials.

The dataset can be downloaded `here <https://zenodo.org/record/4416845/files/building_materials.zip>`_.

When unzipped, the file gives the 'building_materials' which is a directory that contains the images for training:


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
-------------------------------

.. code-block:: none 

    # import the module
    from brails.modules import ImageClassifier

    # initialize the classifier, give it a name
    materialClassifier = ImageClassifier(modelName='materialClassifierV0.1')

    # load data
    materialClassifier.loadData('building_materials')



Train the model
---------------

.. code-block:: none 

    # train the base model for 50 epochs and then fine tune for 200 epochs
    materialClassifier.train(baseModel='InceptionV3', initial_epochs=50,fine_tune_epochs=200)


It is recommended to run the above example on a GPU machine.

The following ML model training options are available for selection as the baseModel key:

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



Classify Images Based on Model
------------------------------

Now you can use the trained model to predict the (building materials) class for a given image.

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

.. note::
    The generic image classifier is intended to illustrate the overall process of model training and prediction.
    The classifier takes an image as the input and will always produce a prediction. 
    Since the classifier is trained to classify only a specific category of images, its prediction is meaningful only if 
    the input image belongs to the category the model is trained for.



