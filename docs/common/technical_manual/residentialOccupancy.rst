.. _lbl-occupancyClassifier-vnv:

Occupancy Classifier
========================

An overall accuracy of 85.43% is obtained when the trained classifier is tested on a ground truth dataset :cite:`chaofeng_wang_2020_4386991`.

Run the following python script to test on this dataset.

.. code-block:: python 

    # download the testing dataset

    import wget
    import zipfile
    wget.download('https://zenodo.org/record/4386991/files/occupancy_test.zip')
    with zipfile.ZipFile('occupancy_test.zip', 'r') as zip_ref:
        zip_ref.extractall('.')

    # prepare the image lists

    import shutil
    import os
    import pandas as pd
    from glob import glob

    mfList = glob('occupancy_test/multi-family/*.png')
    sfList = glob('occupancy_test/single-family/*.png')

    # import the module
    from brails.modules import OccupancyClassifier

    # initialize a roof classifier
    occupancyModel = OccupancyClassifier()

    # define the paths of images in a list
    imgs=mfList+sfList

    # use the model to predict
    predictions = occupancyModel.predict(imgs)

    predictions['label']=predictions['image'].apply(lambda x: 'RES1' if 'single' in x else 'RES3')
    prediction = predictions['prediction'].values.tolist()
    label = predictions['label'].values.tolist()

    # Plot results

    class_names = ['single-family','multi-family']
    from brails.utils.plotUtils import plot_confusion_matrix
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import f1_score,accuracy_score

    print(' Accuracy is   : {}, Random guess is 0.5'.format(accuracy_score(prediction,label)))
    cnf_matrix = confusion_matrix(prediction,label)
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix',normalize=False,xlabel='Labels',ylabel='Predictions')



The confusion matrix tested on this dataset is shown in :numref:`fig_confusion_occupancy`.


.. _fig_confusion_occupancy:
.. figure:: ../../images/technical/confusion_occupancy.png
  :width: 40%
  :alt: Confusion matrix occupancy class

  Confusion matrix - Occupancy Class classifier
