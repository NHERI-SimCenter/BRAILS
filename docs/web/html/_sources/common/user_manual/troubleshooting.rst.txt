.. _lblTroubleshooting:

Troubleshooting
===============

Installation
^^^^^^^^^^^^^^^^^^^

Windows users may experience difficulties because of two dependencies required: GDAL and Fiona.  

If you are a Conda user and you are installing BRAILS in the Conda environment, this is unlikely to happen. 

If you are not a Conda user, the solution is to download Fiona and GDAL's wheel 
files (https://www.lfd.uci.edu/~gohlke/pythonlibs/) and manually install them.

Make sure you choose the correct files based on your Python version.

For example, if you are using Python3.8:

1. Go to https://www.lfd.uci.edu/~gohlke/pythonlibs/ and download these files: 

    GDAL-3.1.4-cp38-cp38-win_amd64.whl 

    Fiona-1.8.18-cp38-cp38-win_amd64.whl

  here cp38 means python3.8.

2. pip install GDAL-3.1.4-cp38-cp38-win_amd64.whl

3. pip install Fiona-1.8.18-cp38-cp38-win_amd64.whl


Then you'll be able to:

pip3 install BRAILS


Internet connection
^^^^^^^^^^^^^^^^^^^^^

The trained models and accompanying datasets, when called the first time, need to be downloaded from the internet. 

Images also need to be downloaded during the running.

Therefore, please make sure you are connected to the internet.
