.. _lbl-dataPrep:

Data Preperation for Training and Inference
================================================

Training Data
-----------------

The accuracy of a deep learning model is highly dependent on the dataset used to train it. If the training data contains minimal noise and includes a sufficient level of features necessary for model development, high model prediction accuracies are attainable. In image-based applications, data noise can be in many forms, such as incorrect image labels and imagery that lack the features sought by the model (e.g., occluded features, etc.). 

In developing the pre-trained models in BRAILS, training images are subjected to rigorous prescreening to ensure sufficient visibility of target buildings in each image in the training set. Without this screening step, creating models with reasonable confidence levels becomes difficult, since the training accuracies being inversely proportional to the extent of noise. :numref:`noiseEffect` shows an example of one of many of our observations of this condition throughout SimCenter's model development efforts. The confusion matrices in both figures are for models generated on identical model architecture and hyperparameters. The first confusion matrix in :numref:`noiseEffect` is for the model trained on a dataset of images that contained 20% noisy data, while the second matrix is for the model trained on a  dataset of images that were fully prescreened before model training. After 100 epochs, the former model attains an F1-score of 47.43%; the latter achieves an F1-score of 79.15% for the same validation set. 

.. _noiseEffect:
.. list-table:: Effect of high noise levels on training performance

    * - .. figure:: figures/allDataNotCleaned.png

      - .. figure:: figures/allDataCleaned.png

:numref:`noisyImages` shows a sample set of images removed after prescreening.

.. _noisyImages:
.. list-table:: Sample noisy images removed by the prescreening algorithm

    * - .. figure:: figures/badImage1.jpg

      - .. figure:: figures/badImage2.jpg

      - .. figure:: figures/badImage3.jpg

      - .. figure:: figures/badImage4.jpg

    * - .. figure:: figures/badImage5.jpg

      - .. figure:: figures/badImage6.jpg

      - .. figure:: figures/badImage7.jpg

      - .. figure:: figures/badImage8.jpg

:numref:`cleanImages` shows a sample set of images that were deemed suitable for model consumption by the prescreening algorithm.

.. _noisyImages:
.. list-table:: Sample clean images retained by prescreening

    * - .. figure:: figures/cleanImage1.jpg

      - .. figure:: figures/cleanImage2.jpg

      - .. figure:: figures/cleanImage3.jpg

      - .. figure:: figures/cleanImage4.jpg

    * - .. figure:: figures/cleanImage5.jpg

      - .. figure:: figures/cleanImage6.jpg

      - .. figure:: figures/cleanImage7.jpg

      - .. figure:: figures/cleanImage8.jpg


Data Suitable for Inference
--------------------------------
For obtaining meaningful results from the models bundled with BRAILS, use of images that satisfy the following criteria is essential.

1. Images should contain buildings with little to no obstructions. 
2. If possible, images should contain a single building only.
3. The types of buildings that the predictions are performed on should not be substantially different in appearance from the building inventories used to establish the pretrained models.

If the images used to predict building attributes meet all three criteria, attribute predictions at the accuracy levels comparable to what is reported for each module will be more achievable.