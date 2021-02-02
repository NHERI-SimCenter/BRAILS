.. _lbl-nfloorDetector-vnv:

Number of Floor Detector
==============================

On a randomly selected set of in-the-wild building images from New Jersey's Bergen, Middlesex, and Moris Counties, the model attains an F1-score of 86%. Here, in-the-wild building images are defined as street-level photos that may contain multiple buildings and are captured with random camera properties. :numref:`confusion_nFloorWild` is the confusion matrix of the model inferences on the aforementioned in-the-wild test set.

.. confusion_nFloorWild:
.. figure:: ../../images/technical/confusion_nFloorWild.png
   :width: 40 %
   :alt: Confusion matrix (in-the-wild dataset)

   Confusion matrix of the pretrained model on the in-the-wild test set


If the test images are constrained such that a single building exists in each image and the images are captured such that the image plane is nearly parallel to the frontal plane of the building facade, the F1-score of the model is determined as 94.7%. :numref:`confusion_nFloorClean` shows the confusion matrix for the pretrained model on a test set generated according to these constraints.

.. _confusion_nFloorClean:
.. figure:: ../../images/technical/confusion_nFloorClean.png
   :width: 40 %
   :alt: Confusion matrix (clean dataset)

   Confusion matrix of the pretrained model on the dataset containing lightly distorted/obstructed images of individual buildings