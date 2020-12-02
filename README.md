# <img src="https://raw.githubusercontent.com/NHERI-SimCenter/BRAILS/master/docs/images/logo/Logo.png" alt="logo" height="80"/> <span style="color:#FFFFFF;background-color: #000000;">B</span>uilding <span style="color:#FFFFFF;background-color: #000000;">R</span>ecognition using <span style="color:#FFFFFF;background-color: #000000;">AI</span> at <span style="color:#FFFFFF;background-color: #000000;">L</span>arge-<span style="color:#FFFFFF;background-color: #000000;">S</span>cale.

<img src="https://raw.githubusercontent.com/NHERI-SimCenter/BRAILS/master/docs/images/brails-demo.gif" alt="BRAILS" height="250"/>

#

## What is BRAILS

BRAILS is the acronym for Building Recognition using AI at Large-Scale, 
which is an AI-Based pipeline for city-scale building information modeling (BIM).

## How to install


```
pip install BRAILS
```

## How to use

The following example can be found in this [Google Colab Notebook](https://colab.research.google.com/drive/1zspDwK-rGA1gYcHZDnrQr_3Z27JL-ooS?usp=sharing).

Example images can be downloaded like this.

```
wget https://zenodo.org/record/4095668/files/image_examples.zip
```

```python
# import modules
from brails.RoofTypeClassifier import RoofClassifier
from brails.OccupancyClassClassifier import OccupancyClassifier
from brails.SoftstoryClassifier import SoftstoryClassifier

# initialize a roof classifier
roofModel = RoofClassifier()

# initialize an occupancy classifier
occupancyModel = OccupancyClassifier()

# initialize a soft-story classifier
ssModel = SoftstoryClassifier()

# use the roof classifier 

imgs = ['image_examples/Roof/gabled/76.png',
        'image_examples/Roof/hipped/54.png',
        'image_examples/Roof/flat/94.png']

predictions = roofModel.predict(imgs)

# use the occupancy classifier 

imgs = ['image_examples/Occupancy/RES1/51563.png',
        'image_examples/Occupancy/RES3/65883.png']

predictions = occupancyModel.predict(imgs)

# use the softstory classifier 

imgs = ['image_examples/Softstory/Others/3110.jpg',
        'image_examples/Softstory/Softstory/901.jpg']

predictions = ssModel.predict(imgs)

```

The predictions look like this:
```
Image :  image_examples/Roof/gabled/76.png     Class : gabled (83.21%)
Image :  image_examples/Roof/hipped/54.png     Class : hipped (100.0%)
Image :  image_examples/Roof/flat/94.png     Class : flat (97.68%)
Results written in file roofType_preds.csv

Image :  image_examples/Occupancy/RES1/51563.png     Class : RES1 (99.99%)
Image :  image_examples/Occupancy/RES3/65883.png     Class : RES3 (98.67%)
Results written in file occupancy_preds.csv

Image :  image_examples/Softstory/Others/3110.jpg     Class : others (96.13%)
Image :  image_examples/Softstory/Softstory/901.jpg     Class : softstory (96.31%)
Results written in file softstory_preds.csv
```

## Documents

Read the document <a href="https://nheri-simcenter.github.io/BRAILS-Documentation/index.html">here</a>.

More details in paper: <a href="https://arxiv.org/abs/1910.06391">here</a>.


## How to cite

Charles Wang, Qian Yu, Frank McKenna, Barbaros Cetiner, Stella X. Yu, Ertugrul Taciroglu & Kincho H. Law. (2019, October 11). NHERI-SimCenter/BRAILS: v1.0.1 (Version v1.0.1). Zenodo. http://doi.org/10.5281/zenodo.3483208


## Acknowledgement
This material is based upon work supported by the National Science Foundation under Grant No. 1612843.

## Contact
Charles Wang, NHERI SimCenter, UC Berkeley, c_w@berkeley.edu


