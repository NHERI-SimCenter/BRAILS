# <img src="https://raw.githubusercontent.com/NHERI-SimCenter/BRAILS/master/docs/images/logo/Logo.png" alt="logo" height="80"/> <span style="color:#FFFFFF;background-color: #000000;">B</span>uilding <span style="color:#FFFFFF;background-color: #000000;">R</span>ecognition using <span style="color:#FFFFFF;background-color: #000000;">AI</span> at <span style="color:#FFFFFF;background-color: #000000;">L</span>arge-<span style="color:#FFFFFF;background-color: #000000;">S</span>cale.

<img src="https://raw.githubusercontent.com/NHERI-SimCenter/BRAILS/master/docs/images/brails-demo.gif" alt="BRAILS" height="250"/>

#

## What is BRAILS

BRAILS is the acronym for Building Recognition using AI at Large-Scale, 
which is an AI-based pipeline for city-scale building information modeling (BIM).

## How to install

```
pip install BRAILS
```

If you have difficulties installing BRAILS, please check the [troubleshooting page](https://nheri-simcenter.github.io/BRAILS-Documentation/common/user_manual/troubleshooting.html).


## Documents

Read the online document <a href="https://nheri-simcenter.github.io/BRAILS-Documentation/index.html">here</a>.


## Quickstart


### Example 1: Modules


The following example shows how to use BRAILS modules. 

This example can also be found in the document [here](https://nheri-simcenter.github.io/BRAILS-Documentation/common/user_manual/examples.html), 
you can run it on you local computer or you can test it in this [notebook](https://colab.research.google.com/drive/1zspDwK-rGA1gYcHZDnrQr_3Z27JL-ooS?usp=sharing) on Google Colab.

Images used in examples can be downloaded by clicking [here](https://zenodo.org/record/4562949/files/image_examples.zip) or 
using the following the command:

```
wget https://zenodo.org/record/4562949/files/image_examples.zip
```

```python
# import modules
from brails.modules import RoofClassifier, OccupancyClassifier, SoftstoryClassifier


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

Image :  image_examples/Occupancy/RES1/51563.png     Class : RES1 (66.41%)
Image :  image_examples/Occupancy/RES3/65883.png     Class : RES3 (49.51%)
Results written in file occupancy_preds.csv

Image :  image_examples/Softstory/Others/3110.jpg     Class : others (96.13%)
Image :  image_examples/Softstory/Softstory/901.jpg     Class : softstory (96.31%)
Results written in file softstory_preds.csv
```


### Example 2: Workflow

This example shows how to create a building inventory by specifying the name of a city. 

You can also specify a bounding box. 

Check details of this example and more examples [here](https://nheri-simcenter.github.io/BRAILS-Documentation/common/user_manual/examples.html), or test them in this [notebook](https://colab.research.google.com/drive/1tG6xVRCmDyi6K8TWgoNd_31vV034VcSO?usp=sharing) on Google Colab.


```python
# Import the module from BRAILS
from brails.CityBuilder import CityBuilder

# Initialize the CityBuilder
cityBuilder = CityBuilder(attributes=['softstory','occupancy','roofshape'], 
                   numBldg=10,random=True, place='Lake Charles, Louisiana', 
                   GoogleMapAPIKey='put-your-key-here')

# create the city-scale BIM file
BIM = cityBuilder.build()

```

The definitions of the parameters in this example can be found [here](https://nheri-simcenter.github.io/BRAILS-Documentation/common/user_manual/examples.html). 

The result BIM is a geopandas dataframe:

index geometry	                                        |   ID	|   roofShape	|   roofShapeProb	|   softStory	|softStoryProb	|occupancy	|occupancyProb
--------------------------------------------------------|-------|---------------|-------------------|---------------|---------------|-----------|------------
0	POLYGON ((-93.21912 30.22786, -93.21892 30.227...	|   0	|   softstory	|   0.761644	    |   COM	    |    0.878260	|    flat	|    0.999769 
1	POLYGON ((-93.21517 30.22412, -93.21491 30.224...	|   1	|   softstory	|   0.500260	    |   RES1	    |    0.549517	|    hipped	|    0.961910 
2	POLYGON ((-93.21411 30.22617, -93.21427 30.226...	|   2	|   softstory	|   0.994021	    |   COM	    |    0.594705	|    hipped	|    0.999313 
3	POLYGON ((-93.16719 30.23006, -93.16715 30.230...	|   3	|   softstory	|   0.969902	    |   COM	    |    0.372285	|    flat	|    0.735594 
4	POLYGON ((-93.25565 30.21074, -93.25550 30.211...	|   4	|   softstory	|   0.000000	    |   COM	    |    0.000000	|    flat	|    0.998508 
5	POLYGON ((-93.20388 30.22758, -93.20388 30.227...	|   5	|   others   	|   0.971890	    |   COM	    |    0.913591	|    flat	|    0.996405 
6	POLYGON ((-93.21610 30.22505, -93.21613 30.224...	|   6	|   softstory	|   0.000000	    |   COM	    |    0.000000	|    flat	|    0.963075 
7	POLYGON ((-93.16549 30.22877, -93.16546 30.228...	|   7	|   others   	|   0.841312	    |   RES3	    |    0.441689	|    hipped	|    0.534562 
8	POLYGON ((-93.21525 30.22513, -93.21523 30.225...	|   8	|   others   	|   0.830462	    |   RES1	    |    0.383844	|    flat	|    0.786514 
9	POLYGON ((-93.21924 30.23054, -93.21949 30.230...	|   9	|   softstory	|   0.000000	    |   COM	    |    0.000000	|    flat	|    0.986143 




## How to cite

```
@article{wang2021machine,
  title={Machine learning-based regional scale intelligent modeling of building information for natural hazard risk management},
  author={Wang, Chaofeng and Yu, Qian and Law, Kincho H and McKenna, Frank and Stella, X Yu and Taciroglu, Ertugrul and Zsarn{\'o}czay, Adam and Elhaddad, Wael and Cetiner, Barbaros},
  journal={Automation in Construction},
  volume={122},
  pages={103474},
  year={2021},
  publisher={Elsevier},
  doi="https://doi.org/10.1016/j.autcon.2020.103474"
}
```
The pdf is <a href="https://www.researchgate.net/publication/346957248_Machine_Learning-based_Regional_Scale_Intelligent_Modeling_of_Building_Information_for_Natural_Hazard_Risk_Management">here</a>.

## Acknowledgement
This material is based upon work supported by the National Science Foundation under Grant No. 1612843.

## Contact
Charles Wang, NHERI SimCenter, UC Berkeley, c_w@berkeley.edu


