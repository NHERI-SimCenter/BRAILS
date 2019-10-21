# <img src="https://raw.githubusercontent.com/NHERI-SimCenter/BRAILS/master/docs/Logo/Logo.png" alt="logo" height="80"/> Building Recognition using AI at Large-Scale.

<img src="https://raw.githubusercontent.com/NHERI-SimCenter/BRAILS/master/docs/images/brails-demo.gif" alt="BRAILS" height="200"/>


## 1. What is BRAILS?

BRAILS is the acronym for Building Recognition using AI at Large-Scale, 
which is an AI-Based pipeline for city-scale building information modeling (BIM).

## 2. How to use?


### a. Data preparation 

<a href="#preparedata">Prepare data</a>


### b. Train

#### Roof
##### training
```
cd src/training/roof/2_train
# train (better to run on a GPU machine)
sh finetune_inception_v3_on_roof_train.sh
```

The training takes a long time on laptops. 
If you don't want to run the training process, we have a CNN trained on TACC and can be downloaded [here](https://berkeley.box.com/shared/static/awyyc22sjwknn9xg3p7wru4v5zwnlkjp.zip).
Put the downloaded file inside src/training/roof/tmp/roof-traindir/ and unzip it.

##### testing

 Before your run the test *"sh finetune_inception_v3_on_roof_eval.sh"*, make sure you set the following variables correctly in *"finetune_inception_v3_on_roof_eval.sh"*:
 ```
 checkpoint_file = the-path-of-your-checkpoint-file
 TEST_DIR = the-path-of-your-testing-images-dir
 ```
 We've prepared some labeld images for you to test. These images can be downloaded [here](https://berkeley.box.com/shared/static/wfwf4ku9561lcytldy1p7vkjoobgv9sz.zip).

Now you can test if the trained CNN works or not:
 ```
cd src/training/roof/2_train
sh finetune_inception_v3_on_roof_eval.sh
```

`If you have your own data, you can train your own neural nets following the above procedures.`

### c. Predict

#### Roof
Now we use the CNN to predict roof types based on satellite images.

Firstly we need to download those images by calling Google API (will cost $).

```
cd src/predicting
python downloadRoofImages.py
```

To save $, instead of running the above python, you can just download them 
[here](https://berkeley.box.com/shared/static/n8l9kusi9eszsnnkefq37fofz22680t2.zip).


Now we can make predictions:

```
cd src/predicting
sh classifyRoof.sh
```
This script will look into the BIM file and call CNN to predict the roof type of a building if the image is downloaded.

If the image is not downloaded, it will assign a null value for the roof type in the new BIM file.


### d. Enhance

Use [*SURF*](https://github.com/charlesxwang/SURF) to predict missing building information.


#### Year built

<img src="https://raw.githubusercontent.com/NHERI-SimCenter/BRAILS/master/docs/images/yearBuilt-prediction-error.png" width="700">


#### Number of stories 

<img src="https://raw.githubusercontent.com/NHERI-SimCenter/BRAILS/master/docs/images/stories_Predictions_classification_error.png" width="700">

#### Structure type

<img src="https://raw.githubusercontent.com/NHERI-SimCenter/BRAILS/master/docs/images/structureType_Predictions_classification_error.png" width="700">

#### Occupancy

<img src="https://raw.githubusercontent.com/NHERI-SimCenter/BRAILS/master/docs/images/occupancy_Predictions_classification_error.png" width="700">


## 3. Release of BIM data
SimCenter will post obtained data here.
### City-scale BIM for Atlantic coastal cities, NJ -> [Download](https://berkeley.box.com/shared/static/5tb6gszbbyj35bgpypk1gsdem0ntt5ca.geojson)
<img src="https://raw.githubusercontent.com/NHERI-SimCenter/BRAILS/master/docs/images/AtlanticCities.png" width="700">
<img src="https://raw.githubusercontent.com/NHERI-SimCenter/BRAILS/master/docs/images/BIM-demo.png" width="700">

## 4. Release of trained CNN  
Trained CNNs in the format of Tensorflow checkpoint will be posted here.
### Roof type classifier ->[Download](https://berkeley.box.com/shared/static/awyyc22sjwknn9xg3p7wru4v5zwnlkjp.zip)


## 5. How to Cite
Charles Wang, Qian Yu, Frank McKenna, Barbaros Cetiner, Stella X. Yu, Ertugrul Taciroglu & Kincho H. Law. (2019, October 11). NHERI-SimCenter/BRAILS: v1.0.1 (Version v1.0.1). Zenodo. http://doi.org/10.5281/zenodo.3483208



## 6. Acknowledgement
This material is based upon work supported by the National Science Foundation under Grant No. 1612843.

## 7. Contact
Charles Wang, NHERI SimCenter, UC Berkeley, c_w@berkeley.edu


