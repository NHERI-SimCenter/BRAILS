# BIM.AI

<img src="docs/images/pipeline.png" width="800">


AI-Based Pipeline for Regional Building Inventory Procurement

# 

###### Tested with Tensorflow in Python 3.6.6 

### 1. Data preparation 


[Prepare data](src/preparedata/README.md)


### 2. Train

##### Roof
```
cd src/training/roof/2_train
# train (better to run on a GPU machine)
sh finetune_inception_v3_on_roof_train.sh
# test
sh finetune_inception_v3_on_roof_eval.sh
```
 Before your run the test *"sh finetune_inception_v3_on_roof_eval.sh"*, make sure you set the following variables correctly in *"finetune_inception_v3_on_roof_eval.sh"*:
 ```
 checkpoint_file = the-path-of-your-checkpoint-file
 TEST_DIR = the-path-of-your-testing-images-dir
 ```


### 3. Predict

Use trained NN to predict building properties.
##### Roof
```
cd src/predicting
sh classifyRoof.sh
```



### 4. Enhance

Use [*SURF*](https://github.com/charlesxwang/SURF) to predict missing building information:


##### Year built
```
cd src/training/yearBuilt
python yearBuiltNNSpatial.py
```
<img src="docs/images/yearBuilt-prediction-error.png" width="700">

##### Number of stories 

<img src="docs/images/stories_Predictions_classification_error.png" width="700">

##### Structure type

<img src="docs/images/structureType_Predictions_classification_error.png" width="700">

##### Occupancy

<img src="docs/images/occupancy_Predictions_classification_error.png" width="700">


### 5. BIM data release
SimCenter will post obtained data here.
##### Data 1. Atlantic coastal cities, NJ -> [Download](https://berkeley.box.com/shared/static/5tb6gszbbyj35bgpypk1gsdem0ntt5ca.geojson)
<img src="docs/images/AtlanticCities.png" width="700">
<img src="docs/images/BIM-demo.png" width="700">

