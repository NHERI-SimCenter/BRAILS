#### Roof

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


