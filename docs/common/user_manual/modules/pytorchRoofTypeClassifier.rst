.. _lbl-pytorchRoofTypeClassifier:

PyTorch Roof Type Classifier
========================

The PyTorch Roof Type Image Classifier is a subclass of PyTorch generic image classifier. It can be used for inference and fine-tuning.

.. container:: toggle
         
   .. container:: header

       **Methods**

   #. **init**
      
      #. **modelName** Name of the model default = 'transformer_rooftype_v1'
      #. **imgDir** Directories for training data
      #. **download** Dowbload the pre-trained roof type classifier
      #. **resultFile** Name of the result file for predicting multple images. default = preds.csv
      #. **workDir** The working directory default = tmp
      #. **printRes** Show the probability and prediction default=True      

   #.  **train**

      #. **lr1**: default=0.01
      #. **epochs**: default==10
      #. **batch_size**: default==64
      #. **plot**: default==False
     
   #.  **fine_tuning*

      #. **lr1**: default=0.001
      #. **epochs**: default==10
      #. **batch_size**: default==32
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

This class implements a roof type classifier which is a subsclass of PyTorch generic classifier. It will download the pretrained model for inference. It can also be used for fine-tuning the pre-trained model if training data is provided. 

Example
-------

The following is an example, in which a roof type classifier is created.

The image dataset for this example contains satellite images categorized according to roof type.

The dataset can be downloaded `here <https://zenodo.org/record/6231341/files/roofType.zip>`_.

When unzipped, the file gives the 'roofType'. You need to set ''imgDir'' to the corresponding directory. The roofType directory contains the images for training or inference:


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
    
    from brails.modules.PytorchRoofTypeClassifier.RoofTypeClassifier import PytorchRoofClassifier

    # initialize the classifier, give it a name
    
    roofClassifier = PytorchRoofClassifier(modelName='transformer_rooftype_v1', download=True))


Classify Images Based on Model
------------------------------

Now you can use the trained model to predict the (roofType) class for a given image.

.. code-block:: none 

    # If you are running the inference from another place, you need to initialize the classifier firstly:
    
    from brails.modules.PytorchRoofTypeClassifier.RoofTypeClassifier import PytorchRoofClassifier

    roofClassifier = PytorchRoofClassifier(modelName='transformer_rooftype_v1', download=True)
                                            
    # define the paths of images in a list
    
    imgs = ["./roofType/flat/TopViewx-76.84779286x38.81642318.png", "./roofType/flat/TopViewx-76.96240924000001x38.94450328.png"]

    # use the model to predict
    predictions_dataframe = roofClassifier.predictMultipleImages(imgs)


The predictions will be written in preds.csv under the working directory.


Fine-tune the model for transfer learning. Transfer learning is a technique to overcome the distribution shift and adapt the model for the new task (https://ftp.cs.wisc.edu/machine-learning/shavlik-group/torrey.handbook09.pdf). You need to provide the training data.
---------------

.. code-block:: none 

    from brails.modules.PytorchRoofTypeClassifier.RoofTypeClassifier import PytorchRoofClassifier

    roofClassifier = PytorchRoofClassifier(modelName='transformer_rooftype_v1', download=True, imgDir='./roofType/')

    # fine-tune the base model for 5 epochs with an initial learning rate of 0.001. 
    
    roofClassifier.fine_tuning(lr=0.001, batch_size=64, epochs=5)


It is recommended to run the above example on a GPU machine.

Please refer to https://github.com/rwightman/pytorch-image-models for supported models. You may need to first install timm via pip: pip install timm. Currently, BRAILS only provides pre-trained roof type model based on Transformer.