# BIM.AI

###### Tested with Python 3.6.6 

### 1. Data preparation 

Prepare a list of building addresses.

Prepare basic building information dataset based on tax information.

Download computer generated building footprints -> [USBuildingFootprints](https://github.com/microsoft/USBuildingFootprints)




### 2. Train

###### Roof
```
cd src/training/roof/2_train
# train
sh finetune_inception_v3_on_roof_train.sh
# test
sh finetune_inception_v3_on_roof_eval.sh
```



### 3. Predict

Use trained NN to predict building properties.




### 4. Enhance

Use [*SURF*](https://github.com/charlesxwang/SURF) to predict missing building information:
###### Year built
```
cd src/training/yearBuilt
python yearBuiltNNSpatial.py
```
<img src="docs/images/yearBuilt-prediction-error.png" width="600">




### 5. BIM data release
SimCenter will post obtained data here.
###### Data 1. Atlantic coastal cities, NJ (coming soon)
<img src="docs/images/AtlanticCities.png" width="600">
