#### Roof

We have compiled the image data set for training, which can be found here:

Charles Wang. (2019). Random satellite images of buildings (Version v1.0) [Data set]. Zenodo. [http://doi.org/10.5281/zenodo.3521067](http://doi.org/10.5281/zenodo.3521067)

##### training
```
cd src/training/roof/2_train
# train (better to run on a GPU machine)
sh finetune_inception_v3_on_roof_train.sh
```

The training takes a long time on laptops. 
If you don't want to run the training process, we have a CNN trained on TACC and can be downloaded [here](https://berkeley.box.com/shared/static/awyyc22sjwknn9xg3p7wru4v5zwnlkjp.zip).
Put the downloaded file inside src/training/roof/tmp/roof-traindir/ and unzip it.


`If you have your own data, you can train your own neural nets following the above procedures.`
