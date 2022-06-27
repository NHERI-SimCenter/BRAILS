.. _lbl-install:

Installation
================

BRAILS is developed in Python3. 
You can install BRAILS using `pip <https://pip.pypa.io/en/stable/installation/>`_ package management system by issuing the command:

.. code-block:: none 

    pip install git+https://github.com/NHERI-SimCenter/BRAILS

To install an earlier release of BRAILS using `pip <https://pip.pypa.io/en/stable/installation/>`_:

.. code-block:: none 

    pip install https://github.com/NHERI-SimCenter/BRAILS/releases/download/{Release Tag}/BRAILS-{Release Tag}.zip

For example, to install v1.9.0 of BRAILS:

.. code-block:: none 

    pip install https://github.com/NHERI-SimCenter/BRAILS/releases/download/v1.9.0/BRAILS-v1.9.0.zip

Windows users may experience difficulty installing BRAILS because of two required dependencies: GDAL and Fiona. 
Please see the :ref:`lblTroubleshooting` section for further assistance with installation issues.