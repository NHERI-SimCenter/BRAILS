.. _occupancyTheory:

Occupancy classifier
==========================

Occupancy class is an important parameter for natural hazard risk analysis of a building.
The occupancy model in the current version of BRAILS is trained on a dataset consists of residential buildings 
that are classified into three categories: single-family buildings (RES1), 
multi-family buildings (RES3) and commercial buildings (COM). Data of other occupancy classes are being collected and will be added to the new versions of BRAILS. 


Examples of different occupancy type is shown in :numref:`occupancyexample`.

.. _occupancyexample:
.. list-table:: Examples of different occupancy types

    * - .. figure:: ../../images/technical/RES1.jpg

           Single-Family Building

      - .. figure:: ../../images/technical/RES3.jpg

           Multi-Family Building

      - .. figure:: ../../images/image_examples/Occupancy/COM/42915.jpg

           Commercial Building

The classifier is trained using a 50-layer ResNet :cite:`he2016deep`, a widely used ConvNet architecture for images feature recognition. 

Its architecture is shown in :numref:`fig_resnet`.

.. _fig_resnet:
.. figure:: ../../images/technical/ResNet.png
  :width: 70%
  :alt: ResNet

  ResNet





   
