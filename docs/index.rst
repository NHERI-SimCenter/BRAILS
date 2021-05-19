.. s3hark Application documentation master file, created by
   sphinx-quickstart on Mon Dec  9 12:35:25 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Building Recognition using AI at Large-Scale
=================================================


.. image:: images/brails-demo.gif
  :width: 700
  :alt: BRAILS

What is |brailsName|   
---------------------------

The SimCenter tool - Building Recognition using AI at Large-Scale (BRAILS) is an AI-enabled software to assist regional-scale simulations. BRAILS is a prototype development that utilizes machine learning (ML), deep learning (DL), and computer vision (CV) to extract information from satellite and street view images for being used in computational modeling and risk assessment of the built environment. It also provides the architecture, engineering, and construction professionals the insight and tools to more efficiently plan, design, construct, and manage buildings and infrastructure systems. 

The released v2.0 is re-structured with modules for performing specific analyses of images. 
The expanded module library enables BRAILS’ capability of predicting a broader spectrum of building attributes including occupancy class, roof type, foundation elevation, year built, soft-story. 

The new release also features a streamlined workflow, CityBuilder, for automatic creation of regional-scale building inventories by fusing multiple sources of data, including OpenStreetMap, Microsoft Footprint Data, Google Maps, and extracting information from them using the modules.

Examples of BRAILS' application in natural hazard engineering include: 
The identification of roof shapes, occupancy type, number of stories, construction year, and foundation elevation to improve the damage and loss calculations for the hurricane workflow; The identification of soft-story buildings to improve models in earthquake workflows. 



.. toctree::
   :caption: User Manual
   :maxdepth: 1
   :numbered: 2

   common/user_manual/acknowledgments
   common/user_manual/about
   common/user_manual//installation
   common/user_manual/userGuide
   common/user_manual/troubleshooting
   common/user_manual/examples
   common/user_manual/bugs
   common/license
   


.. _lbl-technical-manual:

.. toctree::
   :caption: Technical Manual
   :maxdepth: 1
   :numbered: 2

   common/technical_manual/theory
   common/technical_manual/vnv


.. _lbl-developer-manual:

.. toctree::
   :caption: Developer Manual
   :maxdepth: 1
   :numbered: 2

   common/developer_manual/how_to_extend/how_to_build
   common/developer_manual/how_to_extend/architecture
   common/developer_manual/how_to_extend/how_to_extend
   common/developer_manual/coding_style/coding_style
 

How to cite
---------------------------

Charles Wang, Sascha Hornauer, Barbaros Cetiner, Yunhui Guo, Frank McKenna, Qian Yu, Stella X. Yu, Ertugrul Taciroglu, & Kincho H. Law. (2021, March 1). NHERI-SimCenter/BRAILS: Release v2.0.0 (Version v2.0.0). Zenodo. http://doi.org/10.5281/zenodo.4570554

Contact
---------------------------
Charles Wang, NHERI SimCenter, University of California, Berkeley, c_w@berkeley.edu 

References
---------------------------

.. bibliography:: common/technical_manual/references.bib 