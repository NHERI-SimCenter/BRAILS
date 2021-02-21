.. _lbl-occupancyClassifier-vnv:

Occupancy Classifier
========================

An overall accuracy of 67.92% is obtained when the trained classifier is tested on a ground truth dataset `dataset <https://zenodo.org/record/4553803/files/occupancy_validation_images.zip>`_.

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



The confusion matrix tested on this dataset is shown in :numref:`fig_confusion_occupancy`.


.. _fig_confusion_occupancy:
.. figure:: ../../images/technical/confusion_occupancy_v2.png
  :width: 40%
  :alt: Confusion matrix occupancy class

  Confusion matrix - Occupancy Class classifier
