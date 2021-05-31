.. _lblAbout:

*****
About
*****

BRAILS is an open-source software framework maintained at the |tool github link| and 
released under a **BSD clause 2** license, :numref:`lblLicense`. The function of  
|app| is to generate regional asset information, presently information on buildings, utilizing data from multiple sources. The gathering such information for a region, or even for city, is a laborious and expensive undertaking. The purpose of |app| is to assist researchers and decision makers in various fields, such as urban planning, risk management, etc. in this assest building task.

|SimCenter|, which provides applications that are used in the assessment of the impact of natural hazard at the regional scale, is currently using |app| to generate inventories for use in it's `testbeds <https://nheri-simcenter.github.io/R2D-Documentation/>`_ and will in the future include |app| applications in it's `R2D <https://simcenter.designsafe-ci.org/research-tools/r2dtool/>`_ application.


The |app| software framework provides applications that will first gather and then process the data to generate these asset inventories. The framework consists of individual modules that can be stiched together to form applications that will gather data from online resources, e.g. Google Street View, on-line databases, tax assessors offices and will then process the data to generate assest inventories, e.g. building inventory for a county or region. Many of the modules use machine learning to determine individual asseet attributes, e.g. roof shape, number of stores, occupancy. For machine learning algorithm modules, SimCenter releases python scripts for both training, teasting and processing. For processing, the SimCenter releases on Zenodo a number of previosuly trained models that allows users to use the models without the need to train the models. The trained models are continuously being updated as SimCenter expands the training sets it uses.

..  note::

    DISCLAIMER: The modules are implemented to demonstrate the potentials of ML methods to help establish building attributes and inventories for regional scale simulation. The modules are tested extensively using the data sets as reported herein for validations. How these modules generalize to new and unseen data from different geographical locations depends on how similar they are to the training data. User are referred to :ref:`lbl-understand` for examples on the data-generalization topic. Generalization of machine learning models remains an active research area. Users should exercise cautions when the modules are used beyond their intended purposes and trained model ability.



