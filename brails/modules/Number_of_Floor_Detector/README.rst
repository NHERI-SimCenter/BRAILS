Number of Floor Detector
=================================================

Purpose of the Module
------------------------------------------
This module enables automated detection of number of floors in a building from image input.

.. image:: images/sampleModelOutputs.gif

Copyright
~~~~~~~~~
::

    Copyright (c) 2020, The Regents of the University of California
    Contact: Barbaros Cetiner at bacetiner@ucla.edu


BSD 3-Caluse license
~~~~~~~~~~~~~~~~~~~~~
::

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Installation
---------------------------
The following commands clone the BRAILS repository and install the number of floor detection module. Requirements are installed using pip and weights of all used models are downloaded. Make sure to run the last line to add the current folder to the PYTHONPATH variable to avoid issues in training.

::

    git clone https://github.com/NHERI-SimCenter/BRAILS.git BRAILS
    cd BRAILS/brails/modules/NumberOfFloorDetection
    python3 -m pip install -r requirements.txt
    export PYTHONPATH=$PYTHONPATH:`pwd`

Program 
---------------------------

Input Data Format for Training and Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Training, validation, and test folders should be separate. All three folders must be stored in the COCO format and follow the convention defined below. For training a model using a custom dataset, training, validation, and annotations folders must exist. Bounding box annotations for the training and validation folders shall be placed under the annotations folder. The current version of the module only takes horizontal bounding box input. 
::


    IMG_FOLDER
    ├── train
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── .......... (and so on)
    ├── valid
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── .......... (and so on)
    ├── test
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── .......... (and so on)
    └── annotations 
        ├── instances_train.json
        ├── instances_valid.json
        └── classes.txt


Running the Module Using the Pretrained Floor Detection Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The module is bundled with a pretrained model, trained on 80,000 training samples. This model can be called out-of-the-box via ``infer.py``, a powerful post-processor custom-tailored to convert bounding box detections to floor counts. The basic syntax to perform inferences on a set of images requires defining the path for the images and the type of computational environment (i.e., use of CPU or GPU units for inference) by the user as follows.

::

    python3 infer.py
	--im_path "/path/to/images"
        --gpu_enabled True

Using the command line option ``--model_path``, ``infer.py`` can be called with a custom model trained by the user. For a brief description of all the options built into ``infer.py``, please use the ``infer.py --help`` syntax. Below is a complete list of these options.

.. parsed-literal::

    --im_path (default: "VOC/test/") Path for the building images that will be inferred by module 

    --model_path (default: "models/efficientdet-d4_trained.pth") Path for the pretrained inference model.
                                                                 Do NOT define this argument if the pretrained model bundled with the module will be used

    --gpu_enabled (default: True) Enable GPU processing (Enter False for CPU-based inference)

    --csv_out (default: "nFloorPredict.csv") Name of the CSV output file where the inference results will be written


Model Training
~~~~~~~~~~~~~~~

If the user wishes to further train the pretrained model that is bundled with this module, or train a separate model from scratch, using custom data; the folder structure shown in `Input Data Format for Training and Testing`_ shall be strictly followed. Model training is performed using `train.py
<brails/modules/Number_of_Floor_Detector/train.py>`_. 

Following is an comprehensive list of the available command line parameters. The user may also use the ``train.py --help`` syntax to view a brief version of the list below.

.. parsed-literal::

    -c (default: 4) Compund coefficient for the EfficientDet backbone, e.g., enter 7 for EfficientDet-D7 

    -n (default: 0) Number of loader processes to use with Pytorch DataLoader

    --top_only (default: False) True if desired to finetune the regressor and the classifier (head) only. 
                                False if desired to finetune the entire network

    --num_gpus (default: 1) Number of GPUs available for training. Enter 0 for CPU-based training

    --optim (default: "adamw") Optimizer used for training. Available options: AdamW and SGD. 
                               Use of AdamW until the last stage of training then switching to SGD recommended

    --lr (default: 0.0001) Optimizer learning rate

    --batch_size (default: 2) The number of images used per training step

    --num_epochs (default: 25) Number of training epochs

    --data_path (default: "datasets/") Path for the root folder of dataset

    --val_interval (default: 1) Number of epoches between model validating. Enter 1 for validating at the end of each epoch

    --save_interval (default: 5) Number of epoches between model saving. Enter 1 for saving at the end of each epoch

    --es_min_delta (default: 0.0) Early stopping parameter: Minimum change in loss to qualify as an improvement

    --es_patience (default: 0) Number of epochs with no improvement after which training will be stopped. 
                               Set to 0 to disable early stopping

    --customModel_path (default: "models/efficientdet-d4_trained.pth") Path for the custom pretrained model desired to be used in training. 
                               This option is meant for continued training of an existing model. 
                               It can be used for models trained on an EfficientDet backbone only

For example, the command to train a floor detection model **on CPU** by **fine-tuning the full EfficientDet-D4 backbone trained on COCO dataset** for **25 epochs** using a **learning rate of 0.0001**:

::

    python3 train.py
	--num_gpus 0
        --head_only False
        --num_epochs 25
	--lr 0.0001
