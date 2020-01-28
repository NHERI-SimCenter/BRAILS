.. s3hark Application documentation master file, created by
   sphinx-quickstart on Mon Dec  9 12:35:25 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Building Recognition using AI at Large-Scale
=====================================================================


.. image:: images/brails-demo.gif
  :width: 700
  :alt: BRAILS

What is |brailsName|
---------------------------
|brailsName| is the acronym for Building Recognition using AI at Large-Scale, which is an AI-Based pipeline for city-scale building information modeling (BIM).

BRAILS helps users create regional-scale database of building inventory, where buildings properties are extracted from satellite or street view images using deep learning.

BIM, as an efficient way for describing buildings, gives architecture, engineering, and construction (AEC) professionals the insight and tools to more efficiently plan, design, construct, and manage buildings and infrastructure.

Natural disasters cause losses to human society by damaging or destroying buildings, which consequently endangers lives and goods. Buildings are the major components of a human built environment, hence are of the major considerations in planning for, responding to and recovering from disasters. BIM contains information showing building geometry, components, material, usage, etc, which makes it an ideal source for generation and running of simulations of building behavior under both normal and emergency scenarios. For example, based on BIM, structure engineers can create numerical models for dynamic simulations of seismic loading conditions.

.. toctree::
   :caption: User Manual
   :maxdepth: 1
   :numbered: 2

   
   common/user_manual/installation/installation
   common/user_manual/usage/preparedata 
   common/user_manual/usage/train 
   common/user_manual/usage/enhance 
   common/license


.. _lbl-technical-manual:

.. toctree::
   :caption: Technical Manual
   :maxdepth: 2
   :numbered: 2

   common/technical_manual/theory


.. _lbl-developer-manual:

.. toctree::
   :caption: Developer Manual
   :maxdepth: 2
   :numbered: 2

   common/developer_manual/dev
 

How to cite
---------------------------

Charles Wang. (2019, September 28). NHERI-SimCenter/SURF: v0.2.0 (Version v0.2.0). Zenodo. http://doi.org/10.5281/zenodo.3463676

License
---------------------------

The |brailsName| application is distributed under the BSD 3-Clause license, see :ref:`lbl-license` for details.

Acknowledgement
---------------------------

This material is based upon work supported by the National Science Foundation under Grant No. 1612843. 
Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do 
not necessarily reflect the views of the National Science Foundation.

Contact
---------------------------
Charles Wang, NHERI SimCenter, University of California, Berkeley, c_w@berkeley.edu 