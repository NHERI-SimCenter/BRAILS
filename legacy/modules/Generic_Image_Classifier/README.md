
## Get the code


```bash
git clone https://github.com/NHERI-SimCenter/BRAILS.git
cd BRAILS/brails/modules/Generic_Image_Classifier
pip install -r requirements.txt
```

## Train a model

```bash
python train.py --imgDir <IMAGE_DIRECTORY> --modelDir <MODEL_DIRECTORY>
```

IMAGE_DIRECTORY is the directory where you have your images for training


```
    IMAGE_DIRECTORY
    │── class_1
    │       └── *.png
    │── class_2
    |      └── *.png
    │── ...
    |
    └── class_n
           └── *.png
```

An image dataset for test that contains images of different rooves can be downloaded here:
```
https://zenodo.org/record/3986721/files/Roof_Satellite_Images.zip
```

MODEL_DIRECTORY is the directory where the trained model will be saved. 

It is better to run the above code on a GPU machine.




## Predict


Now you can use the trained model to predict on given images.

```
python predict.py --imageDir <IMAGE_DIRECTORY> --modelDir <MODEL_DIRECTORY> 
```

The predictions will be written in preds.csv under the current directory.

IMAGE_DIRECTORY is the directory of your images. 

MODEL_DIRECTORY is the folder where you have your trained model saved.

You need also to specify the name of your model file by --modelFile <MODEL_FILE_NAME> if the name of your model file is not the default "classifier.h5".

You can specify the name of classes like this --classNames <CLASS_A> <CLASS_B> ...


### Example 1: Roof type classification

A pre-trained model for roof type classification can be downloaded from [here](https://doi.org/10.5281/zenodo.4059083). 

This model takes a satellite image as input and predicts roof types (flat/gabled/hipped). To test it:

```
wget https://zenodo.org/record/4059084/files/roof_classifier_v0.1.h5
python predict.py 
        --imageDir <IMAGE_DIRECTORY> 
        --modelDir . 
        --modelFile roof_classifier_v0.1.h5 
        --classNames flat gabled hipped
```





