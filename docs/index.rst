Building Recognition using AI at Large-Scale
============================================


.. image:: images/brails-demo.gif
  :width: 700
  :alt: BRAILS

What is |brailsName|   
--------------------

The software framework Building Recognition using AI at Large-Scale, |app|, is  provided by |SimCenter| to assist in creating the assest inventories needed to perform regional scale simulations to study the effects of a hazard on a region. The |app| framework provides a set of modules that utilize machine learning (ML), deep learning (DL), and computer vision (CV) techniques to extract information from satellite and street view images. Each of the modules performs a specific task. For example, for building assets inventories, seperate modules provide information on occupancy class, roof type, foundation elevation, roof elevation, and year built among others. The |app| framework will also provide workflows that allow the users to put the individual modules together to determine multiple attributes in a single pass. One such workflow application, CityBuilder, is an application released with |app| to  create regional-scale building inventories by utilizing multiple sources of data, including OpenStreetMap, Microsoft Footprint Data, Google Maps, and extracting information from them using the |app| modules.

|SimCenter|, which provides applications that are used in the assessment of the impact of natural hazard at the regional scale, is currently developing |app| in order to generate inventories for use in its `testbeds <https://nheri-simcenter.github.io/R2D-Documentation/>`_ and will in the future include |app| applications in it's `R2D <https://simcenter.designsafe-ci.org/research-tools/r2dtool/>`_ application. |app|, as a consequence, is under constant development with new modules and workflows being constantly added.

|SimCenter| also provides trained models so that users can immediately use of many of the modules within |app|. These trained models, which are released on Zenodo, are the product of |SimCenter| ML development/validation efforts and are being constantly updated as the |SimCenter| training sets are expanded upon. While use of these training sets are encouraged, it should be noted that the models are trained using data obtained and labelled for |SimCenter| testbeds and using the sparse data available from `OpenStreetMap <https://www.openstreetmap.org/#map=5/38.007/-95.844>`_. How these modules generalize to new and unseen data from different geographical locations depends on how similar they are to the training data. Generalization of machine learning models remains an active research area. Users should exercise cautions when the modules are used beyond their intended purposes and trained model ability.

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
NHERI-SimCenter nheri-simcenter@berkeley.edu

References
---------------------------

.. bibliography:: common/technical_manual/references.bib 
