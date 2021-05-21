.. _lbl-dataPrep:

Data Preperation for Training and Inference
================================================

The accuracy of a deep learning model is highly dependent on the dataset used to train it. If the training data contains minimal noise and includes a sufficient level of features necessary for model development, high model prediction accuracies are attainable. In image-based applications, data noise can be in many forms, such as incorrect image labels and imagery that lack the features sought by the model (e.g., occluded features, etc.). 

In developing the pre-trained models in BRAILS, training images are subjected to rigorous prescreening to ensure sufficient visibility of target buildings in each image in the training set. Without this screening step, creating models with reasonable confidence levels becomes difficult, with the training accuracies being inversely proportional to the extent of noise. Figure shows an example of one of our observations of this condition throughout SimCenter's model development efforts. The confusion matrices in both figures are for models generated using identical model architecture and hyperparameters. The model in Figure was trained on a dataset of images that contained 20% noisy data, while the model in Figure was trained on a  dataset of images that were fully prescreened before model training. After 100 epochs, the former model attained an F1-score of 47.43%; the latter achieved an F1-score of 79.15% for the same validation set. 

.. list-table::

    * - .. figure:: /figures/allDataNotCleaned.png

      - .. figure:: /figures/allDataCleaned.png

