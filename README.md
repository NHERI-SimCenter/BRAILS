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


### Example 1


The following example can be found in the [here](https://nheri-simcenter.github.io/BRAILS-Documentation/common/user_manual/examples.html)

Example images can be downloaded like this.

```
wget https://zenodo.org/record/4095668/files/image_examples.zip
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

Image :  image_examples/Occupancy/RES1/51563.png     Class : RES1 (99.99%)
Image :  image_examples/Occupancy/RES3/65883.png     Class : RES3 (98.67%)
Results written in file occupancy_preds.csv

Image :  image_examples/Softstory/Others/3110.jpg     Class : others (96.13%)
Image :  image_examples/Softstory/Softstory/901.jpg     Class : softstory (96.31%)
Results written in file softstory_preds.csv
```


### Example 2

This example shows how to create a building inventory by specifying the name of a city. 

You can also specify a bounding box. 

Check details of the examples [here](https://nheri-simcenter.github.io/BRAILS-Documentation/common/user_manual/examples.html).


```python
# Import the module from BRAILS
from brails.CityBuilder import CityBuilder

# Initialize the CityBuilder
cityBuilder = CityBuilder(attributes=['story','occupancy','roofshape'], 
                   numBldg=10,random=False, place='lake charles',state='la', 
                   GoogleMapAPIKey='put-your-key-here')

# create the city-scale BIM file
BIM = cityBuilder.build()

```

The definitions of the parameters in this example can be found [here](https://nheri-simcenter.github.io/BRAILS-Documentation/common/user_manual/examples.html). 

The result BIM is a geopandas dataframe:
```
index geometry	                                       ID	roofShape	roofShapeProb	softStory	softStoryProb	occupancy	occupancyProb
0	POLYGON ((-93.18634 30.26957, -93.18626 30.269...	0	gabled	       0.985102	    softstory	0.985102	    RES1	    0.996449
1	POLYGON ((-93.18812 30.25996, -93.18812 30.260...	1	gabled	       0.903468	    others      0.903468	    RES1	    0.999988
2	POLYGON ((-93.18746 30.26043, -93.18746 30.260...	2	hipped	       0.790183	    others      0.790183	    RES1	    1.000000
3	POLYGON ((-93.18283 30.26018, -93.18294 30.260...	3	flat	       0.414026	    softstory	0.414026	    RES1	    0.999875
4	POLYGON ((-93.18224 30.26446, -93.18240 30.264...	4	flat	       0.956571	    softstory	0.956571	    RES1	    0.999984
5	POLYGON ((-93.17564 30.26633, -93.17564 30.266...	5	flat	       0.982985	    others      0.982985	    RES1	    0.999994
6	POLYGON ((-93.21555 30.23522, -93.21555 30.235...	6	flat	       0.992871	    softstory	0.992871	    RES3	    0.971049
7	POLYGON ((-93.21243 30.22394, -93.21243 30.224...	7	flat	       0.490653	    softstory	0.490653	    RES1	    0.894999
8	POLYGON ((-93.21002 30.22489, -93.21002 30.224...	8	hipped	       0.769291	    others      0.769291	    RES1	    0.904881
9	POLYGON ((-93.21001 30.22770, -93.20999 30.227...	9	flat	       0.991286	    others      0.991286	    RES1	    0.688759

```





## Documents

Read the online document <a href="https://nheri-simcenter.github.io/BRAILS-Documentation/index.html">here</a>.




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


