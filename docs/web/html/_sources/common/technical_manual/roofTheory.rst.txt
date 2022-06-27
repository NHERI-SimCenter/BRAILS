.. _roofTheory:

Roof type classifier
==========================

Roof type is a crucial information for wind hazard analyses of buildings because it is 
a key attribute needed for consideration of wind effects on structures. 

There are three major roof types, as shown in :numref:`roof_types`, that are widely used in the world: flat, gabled, hipped. 

.. _roof_types:
.. list-table:: Roof prototypes

    * - .. figure:: ../../images/technical/flat.jpg

           Flat

      - .. figure:: ../../images/technical/gable.jpg

           Gabled

      - .. figure:: ../../images/technical/hip.jpg

           Hipped

Correspondingly, a typical satellite image of each roof type is shown in :numref:`roof_images`.

.. _roof_images:
.. list-table:: Example satellite images of different roof types

    * - .. figure:: ../../images/image_examples/Roof/flat/94.png 

           Flat

      - .. figure:: ../../images/image_examples/Roof/gabled/76.png

           Gabled

      - .. figure:: ../../images/image_examples/Roof/hipped/54.png 

           Hipped




Satellite images are a scalable source for inferring roof type information.
In the attempt to determine roof type for every building in a region, 
a ConvNet classifier 
is trained to take a satellite image of a building and predicts its roof type.  
A training data set of 6,000 satellite images (2,000 for each roof type: flat, gabled, hipped) is collected.  
Specifically,  ResNet :cite:`he2016deep`, which is a widely-used ConvNet architecture for image feature recognition, 
is employed. 

The architecture of the model is shown in :numref:`fig_resnet`.
In this module, we used a 50-layer ResNet.

.. _fig_resnet:
.. figure:: ../../images/technical/ResNet.png
  :width: 70%
  :alt: ResNet

  ResNet




