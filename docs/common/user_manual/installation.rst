.. _lbl-install:

Installation
================

..  note::

    The following instructions assume that `Python <https://www.python.org/>`_ is already installed on your system. If you do not have Python installed, please follow the steps listed `here <https://nheri-simcenter.github.io/SimCenterBootcamp2022/source/setupInstructions.html#python-days>`_ to set up Python. Another good alternative to installing Python is `Anaconda <https://www.anaconda.com/>`_, which offers a relatively more straightforward interface for package management and working with different versions of Python. To set up Python through an Anaconda Distribution, please follow the instructions provided `here <https://docs.anaconda.com/anaconda/install/>`_.

.. tip::
    An effective way to eliminate conflicts between dependencies of individual Python projects is to isolate them from each other by installing them in different virtual environments. Please install |app| in a virtual environment to eliminate such potential conflicts. Depending on your Python installation, please follow the `instructions for vanilla Python <https://virtualenv.pypa.io/en/latest/user_guide.html#introduction>`_ or `instructions for Anaconda <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ to create and use a virtual environment.
	
|app| is a `Python <https://www.python.org/>`_ package for regional-level inventory generation and training/testing custom deep learning models for image recognition.  
You can install the latest version of |app| using `pip <https://pip.pypa.io/en/stable/installation/>`_ package management system by issuing the command:

.. code-block::

    pip install BRAILS
    
Once this command is fully executed, you can verify that |app| is successfully installed by importing |app|:

.. code-block::

    import brails

If you receive an error message after running the command above, please see the :ref:`lblTroubleshooting` section for assistance.
    
.. 
	To install an earlier release of BRAILS using `pip <https://pip.pypa.io/en/stable/installation/>`_ use the syntax:

	.. code-block::

	    pip install BRAILS=={release version}

	For example, to install v2.0.5 of BRAILS:

	.. code-block::

	    pip install BRAILS==2.0.5
