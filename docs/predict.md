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
