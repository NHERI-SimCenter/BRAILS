.. _lbl-occupancyClassifier-vnv:

Occupancy Classifier
========================

The trained classifier is tested on a ground truth dataset that can be downloaded from `here <https://zenodo.org/record/4553803/files/occupancy_validation_images.zip>`_.
We firstly obtained a set of randomly selected buildings in the United States with occupancy tags found on OpenStreetMap.
We then downloaded the street view images from Google Street View for each building. 
The dataset contains 98 single family buildings (RES1), 97 multi-family buildings (RES3) and 98 commercial buildings (COM). 
Examples of these street view images can be found in :ref:`lbl-occupancyClassifier`. 

The accuracy, precision, recall and F1 are all found to be 100% for this dataset.


Run the following python script to test on this dataset.

.. code-block:: python 

    # download the testing dataset

    import wget
    import zipfile
    wget.download('https://zenodo.org/record/4553803/files/occupancy_validation_images.zip')
    with zipfile.ZipFile('occupancy_validation_images.zip', 'r') as zip_ref:
        zip_ref.extractall('.')

    # prepare the image lists

    import shutil
    import os
    import pandas as pd
    from glob import glob

    class_names = ['RES3', 'COM' ,'RES1']

    labels = []
    images = []
    for clas in class_names:
        imgs = glob(f'occupancy_validation_images/{clas}/*.jpg')
        for img in imgs:
            labels.append(clas)
            images.append(img)

    # import the module
    from brails.modules import OccupancyClassifier

    # initialize the classifier
    occupancyModel = OccupancyClassifier()

    # use the model to predict
    pred = occupancyModel.predict(images)
    predictions = pred['prediction'].tolist()

    # Plot results
    from brails.utils.plotUtils import plot_confusion_matrix
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import f1_score,accuracy_score

    print(' Accuracy is   : {}, Random guess is 0.33'.format(accuracy_score(predictions,labels)))
    cnf_matrix = confusion_matrix(predictions,labels)
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix',normalize=False,xlabel='Labels',ylabel='Predictions')



The confusion matrix tested on this dataset is shown in :numref:`fig_confusion_occupancy2`.


.. _fig_confusion_occupancy2:
.. figure:: ../../images/technical/confusion_occupancy_v2.png
  :width: 40%
  :alt: Confusion matrix occupancy class

  Confusion matrix - Occupancy Class classifier
