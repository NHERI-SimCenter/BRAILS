.. _lbl-yearClassifier:

Year Built Classification
=================================================


With this module images of houses can be classifier into categories, 
where each represents a range of years in which the house was built. 
For classification, the path of a folder holding the images has to be supplied. 
The result will be a comma separated value file in that folder, 
listing the filenames and classification result.

The code is optimized to extract suitable features from Google Streetview
images to classify houses into decades. However, the code does not make 
assumptions about the semantic meaning of the provided folders. They
have to be consistent between the training, validation and test folders but
can be categorized by decades, or short or longer than a decade, and any specified timespans.

The predictions fall in 6 categories:
::

    0 : before 1969
    1 : 1970-1979
    2 : 1980-1989
    3 : 1990-1999
    4 : 2000-2009
    5 : after 2010

Use the module
---------------------------

A pretrained model is shipped with BRAILS. So you can use it directly without training your own model.

The first time you initialize this model, it will download the model from the internet to your local computer.

.. code-block:: none 

    # import the module
    from brails.modules import YearBuiltClassifier

    # initialize a year classifier
    model = YearBuiltClassifier()

    # define the paths of images in a list
    from glob import glob
    imgs = glob('image_examples/Year/*/*.jpg')
    
    # use the model to predict
    predictions = model.predict(imgs)

The module is currently under active development and testing.
More details about the training, modification, improvement of this module can be found `here <https://github.com/NHERI-SimCenter/BRAILS/tree/master/brails/modules/Year_Built_Classifier>`_.

.. note:: 

   The classifier takes an image as the input and will always produce a prediction. 
   Since the classifier is trained to classify only a specific category of images, 
   its prediction is meaningful only if the input image belongs to the category the model is trained for.
