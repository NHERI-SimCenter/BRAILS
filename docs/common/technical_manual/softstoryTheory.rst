.. _softstoryTheory:

Soft-story
==========================

Structural deficiencies identification is a crucial task in seismic risk evaluation of buildings. 
In case of multi-story structures, abrupt vertical variations of story stiffness are known to significantly increase the likelihood of collapse during earthquakes. 
These buildings are called soft story buildings.
Identifying these buildings is vital in earthquake loss estimation and mitigation. 


One example of soft-story failure is shown in :numref:`ssexample`.

.. _ssexample:
.. list-table::

    * - .. figure:: ../../images/technical/ss.png
           :width: 60%

           Soft-story collapse

      - .. figure:: ../../images/technical/ss-frame.png
           :width: 60%

           Failure mechanism



Street view images are a scalable source for inferring soft-story information.
In the attempt to screen soft-story buildings in a region, 
a ConvNet classifier 
is trained to take a street-view image of a building and classify it as soft-story or non-soft-story.  
The training data set contains 1,302 satellite images (566 soft-story and 736 non-soft-story).  
Specifically,  InceptionV3 :cite:`szegedy2016rethinking`, which is a widely-used ConvNet architecture for image feature recognition that has been shown to attain good results with an accuracy greater than 78.1\% on the ImageNet dataset, is employed. 

The InceptionV3 is made up of a series of symmetric and asymmetric building blocks, 
including convolutions, average pooling, max pooling, concats, dropouts, and fully connected layers. 
Batchnorm is used extensively throughout the model and applied to activation inputs. 
Loss is computed via Softmax. 
The architecture of the model is shown in :numref:`fig_InceptionV3`.

.. _fig_InceptionV3:
.. figure:: ../../images/technical/inceptionv3.png
  :width: 100%
  :alt: InceptionV3

  InceptionV3





   
