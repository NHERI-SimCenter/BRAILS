.. _lbl-pytorchRoofTypeClassifier:

Pytorch Roof Type Classifier
========================

The Pytorch roof type Image Classifier is a subclass of Pytorch generic image classifier. It can be used for inference and fine-tuning.

.. container:: toggle
         
   .. container:: header

       **Methods**

   #. **init**
      
      #. **modelName** Name of the model default = 'rooftype_resnet18_v1'
      #. **imgDir** Directories for training data
      #. **resultFile** Name of the result file for predicting multple images. default = preds.csv
      #. **workDir** The working directory default = tmp
      #. **printRes** Show the probability and prediction default=True      

   #.  **train**

      #. **lr1**: default=0.01
      #. **epochs**: default==10
      #. **batch_size**: default==64
      #. **plot**: default==False
     
   #. **predictOneImage**
   
      #. **imagePath**: Path to a single image

   #. **predictMultipleImages**
  
      #. **imagePathList**: A list of image paths
      #. **resultFile**: The name of the result filename default=None
                   
   #. **predictOneDirectory**

      #. **directory_name**: Directory for saving all the images
      #. **resultFile**: The name of the result filename default=None
                   

Description
----------

This class implements a roof type classifier which is a subsclass of Pytorch generic classifier. It will download the pretrained model for inference. It can also be used for fine-tuning the pre-trained model if training data is provided.

Example
-------

The following is an example, in which a roof type classifier is created.

The image dataset for this example contains satellite images categorized according to roof type.

The dataset can be downloaded `here <https://zenodo.org/record/6231341/files/roofType.zip>`_.

When unzipped, the file gives the 'roofType'. You need to set ''imgDir'' to the corresponding directory.  The roofType directory contains the images for training:


.. code-block:: none 

    roofType
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
    from brails.modules.PytorchRoofTypeClassifier import PytorchRoofClassifier

    # initialize the classifier, give it a name and a directory
    roofClassifier = PytorchRoofClassifier(modelName='transformer_rooftype_v1')


Classify Images Based on Model
------------------------------

Now you can use the trained model to predict the (roofType) class for a given image.

.. code-block:: none 

    # If you are running the inference from another place, you need to initialize the classifier firstly:
    from brails.modules.PytorchRoofTypeClassifier import PytorchRoofClassifier
    roofClassifier = PytorchRoofClassifier(modelName='transformer_rooftype_v1')
                                            
    # define the paths of images in a list
    imgs = ["/home/yunhui/SimCenter/train_BRAILS_models/datasets/roofType/flat/TopViewx-76.84779286x38.81642318.png",   
         "/home/yunhui/SimCenter/train_BRAILS_models/datasets/roofType/flat/TopViewx-76.96240924000001x38.94450328.png"]

    # use the model to predict
    predictions_dataframe = roofClassifier.predictMultipleImages(imgs)


The predictions will be written in preds.csv under the working directory.


Fine-tune the model for transfer learning. You need to provide the training data.
---------------

.. code-block:: none 

    from brails.modules.PytorchRoofTypeClassifier import PytorchRoofClassifier
    roofClassifier = PytorchRoofClassifier(modelName='transformer_rooftype_v1', imgDir='/home/yunhui/SimCenter/train_BRAILS_models/datasets/roofType/')

    # train the base model for 5 epochs with an initial learning rate of 0.01. 
    
    roofClassifier.train(lr=0.01, batch_size=64, epochs=5)


It is recommended to run the above example on a GPU machine.

Please refer to https://github.com/rwightman/pytorch-image-models for supported models. You may need to first install timm via pip: pip install timm. Currently, BRAILS only provides pre-trained roof type model based on Transformer.


