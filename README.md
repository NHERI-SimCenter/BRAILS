[![DOI](https://zenodo.org/badge/184673734.svg)](https://zenodo.org/badge/latestdoi/184673734)
[![PyPi version](https://badgen.net/pypi/v/BRAILS/)](https://pypi.org/project/BRAILS/)
[![PyPI download month](https://img.shields.io/pypi/dm/BRAILS.svg)](https://pypi.python.org/pypi/BRAILS/)

<p align="center">
:stop_sign: This repository is DEPRECATED and is no longer supported. For the latest supported version of BRAILS, please visit the <a href="https://github.com/NHERI-SimCenter/BrailsPlusPlus">BRAILS++ repository</a>.
</p>

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

The primary input to ``InventoryGenerator`` is location. ``InventoryGenerator`` accepts four different ``location`` input types: 1) region name, 2) list of region names, 3) a tuple containing the coordinates for two opposite vertices of a bounding box for a region (e.g., ``(vert1lon,vert1lat,vert2lon,vert2lat)``), and a 4) GeoJSON file containing building footprints or location points.

InventoryGenerator automatically detects building locations in a region by downloading footprint data for the ``location`` input. The three footprint data sources, ``fpSource``, included in BRAILS are i) OpenStreetMaps, ii) Microsoft Global Building Footprints dataset, and iii) FEMA USA Structures. The keywords for these sources are ``osm``, ``ms``, and ``usastr``, respectively.

``InventoryGenerator`` also allows inventory data to be imported from the National Structure Inventory or another user-specified file to create a baseline building inventory.

Please note that to run the ``generate`` method of ``InventoryGenerator``, you will need a Google API Key.

```python
#import InventoryGenerator:
from brails.InventoryGenerator import InventoryGenerator

# Initialize InventoryGenerator:
invGenerator = InventoryGenerator(location='Berkeley, CA',
                                  fpSource='usastr', 
                                  baselineInv='nsi',
                                  lengthUnit='m',
                                  outputFile='BaselineInvBerkeleyCA.geojson')

# View a list of building attributes that can be obtained using BRAILS:
invGenerator.enabled_attributes()

# Run InventoryGenerator to generate an inventory for the entered location:
# To run InventoryGenerator for all enabled attributes set attributes='all':
invGenerator.generate(attributes=['numstories','roofshape','buildingheight'],
                      GoogleAPIKey='ENTER-YOUR-API-KEY-HERE',
                      nbldgs=100,
                      outputFile='BldgInvBerkeleyCA.geojson')

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
  title        = {BRAILS Release v3.1.1},
  month        = feb,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v3.1.1},
  doi          = {10.5281/zenodo.10606032},
  url          = {https://doi.org/10.5281/zenodo.10606032}
}
```
