.. _lbl-foundationElevationClassifier:

Raised Foundation Classification
=================================================


What is Raised Foundation Classification
------------------------------------------
The code in this package enables to see if a building is elevated on piles piers or posts (PPP). 

For classification, the path of a folder holding the images has to be supplied. The result will be a comma separated value file in that folder, listing the filenames, classification (1: elevated, 0: not elevated), and the confidence of the prediction.


Use the module
---------------------------

A pretrained model is shipped with BRAILS. So you can use it directly without training your own model.

The first time you initialize this model, it will download the model from the internet to your local computer.

The images used in the example can be downloaded from `here <https://zenodo.org/record/4562949/files/image_examples.zip>`_.

.. code-block:: none 

    # import the module
    from brails.modules import FoundationHeightClassifier

    # initialize a roof classifier
    model = FoundationHeightClassifier()

    # define the paths of images in a list
    from glob import glob
    imgs = glob('image_examples/Foundation/*/*.jpg')
    
    # use the model to predict
    predictions = model.predict(imgs)


The predictions look like this:

.. code-block:: none 

    Image :  a.jpg     Class : 1 (52.5%)
    Image :  b.jpg     Class : 1 (56.36%)
    Image :  c.jpg     Class : 0 (83.4%)
    Image :  d.jpg     Class : 0 (63.43%)
    Results written in file tmp/FoundationElevation.csv

The images used in this example are:

.. list-table::

    * - .. figure:: ../../../images/image_examples/Foundation/Elevated/a.jpg

           Elevated

      - .. figure:: ../../../images/image_examples/Foundation/Elevated/b.jpg 

           Elevated

      - .. figure:: ../../../images/image_examples/Foundation/NotElevated/c.jpg 

           Not Elevated

      - .. figure:: ../../../images/image_examples/Foundation/NotElevated/d.jpg 

           Not Elevated

This module is currently under active development and testing.
Currently, for the data set used, classification reaches an F1-score of 72% on a random test set that holds 20% of the data.
Further optional code to improve the quality and speed of the classification is available.
More details about the training, modification, improvement of this module can be found `here <https://github.com/NHERI-SimCenter/BRAILS/tree/master/brails/modules/Foundation_Classification>`_.

.. note:: 

   The classifier takes an image as the input and will always produce a prediction. 
   Since the classifier is trained to classify only a specific category of images, 
   its prediction is meaningful only if the input image belongs to the category the model is trained for.


