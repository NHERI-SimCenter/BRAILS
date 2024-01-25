

## What is BRAILS?

BRAILS (Building and Infrastructure Recognition using AI at Large-Scale) provides a set of Python modules that utilize deep learning (DL), and computer vision (CV) techniques to extract information from satellite and street level images. The BRAILS framework also provides turn-key applications allowing users to put individual modules together to determine multiple attributes in a single pass or train general-purpose image classification, object detection, or semantic segmentation models.

## Documentation

Online documentation is available at <a href="https://nheri-simcenter.github.io/BRAILS-Documentation/index.html">https://nheri-simcenter.github.io/BRAILS-Documentation</a>.

## Quickstart

### Installation


The easiest way to install the latest version of BRAILS is using ``pip``:
```
pip install BRAILS
```

### Example: InventoryGenerator Workflow

This example demonstrates how to use the ``InventoryGenerator`` method embedded in BRAILS to generate regional-level inventories. 

The primary input to ``InventoryGenerator`` is location. ``InventoryGenerator`` accepts four different location input: 1) region name, 2) list of region names, 3) bounding box of a region, 4) A GeoJSON file containing building footprints.

Please note that you will need a Google API Key to run ``InventoryGenerator``.

```python
#import InventoryGenerator:
from brails.InventoryGenerator import InventoryGenerator

# Initialize InventoryGenerator:
invGenerator = InventoryGenerator(location='Berkeley, CA',
                                  nbldgs=100, randomSelection=True,
                                  GoogleAPIKey="")

# Run InventoryGenerator to generate an inventory for the entered location:
# To run InventoryGenerator for all enabled attributes set attributes='all':
invGenerator.generate(attributes=['numstories','roofshape','buildingheight'])

# View generated inventory:
invGenerator.inventory

```

## Acknowledgements

This work is based on material supported by the National Science Foundation under grants CMMI 1612843 and CMMI 2131111.


## Contact

NHERI-SimCenter nheri-simcenter@berkeley.edu

## How to cite

```
@software{cetiner_2024_10448047,
  author       = {Barbaros Cetiner and
                  Charles Wang and
                  Frank McKenna and
                  Sascha Hornauer and
                  Jinyan Zhao and
                  Yunhui Guo and
                  Stella X. Yu and
                  Ertugrul Taciroglu and
                  Kincho H. Law},
  title        = {BRAILS Release v3.1.0},
  month        = jan,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v3.1.0},
  doi          = {10.5281/zenodo.10448047},
  url          = {https://doi.org/10.5281/zenodo.10448047}
}
```
