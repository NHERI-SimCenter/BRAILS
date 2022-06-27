.. _lbl-vnv:

Validation 
=============================

This section provides the validation test results for the following ML modules:

.. toctree::
    :maxdepth: 0

    roof
    occupancy
    softstory
    nfloor
   

.. note:: 

   1. Year Built Classifier is currently under active development and testing. More details about the training, modification, improvement, and validation of this module can be found `here <https://github.com/NHERI-SimCenter/BRAILS/tree/master/brails/modules/Year_Built_Classifier>`_.
   2. Raised Foundation Classifier is currently under active development and testing. More details about the training, modification, improvement, and validation of this module can be found `here <https://github.com/NHERI-SimCenter/BRAILS/tree/master/brails/modules/Foundation_Classification>`_.

.. note:: 

   DISCLAIMER: 
   The modules are implemented to demonstrate the potentials of ML methods to help establish building attributes and inventories for regional scale simulation. 
   The modules are tested extensively using the data sets as reported herein for validations. 
   How these modules generalize to new and unseen data from different geographical locations depends on how similar they are to the training data. 
   User are referred to :ref:`lbl-understand` for examples on the data-generalization topic.
   Generalization of machine learning models remains an active research area.  
   Users should exercise cautions when the modules are used beyond their intended purposes and trained model ability. 
