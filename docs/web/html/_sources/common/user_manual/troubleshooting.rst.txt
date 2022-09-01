.. _lblTroubleshooting:

Troubleshooting
===============

CUDA Errors
^^^^^^^^^^^^^^^^^^^
Installation instructions, by default install the latest version of PyTorch and the compatible CUDA version. If your computed is equipped with an older graphics card (GPU) that does not support this default CUDA installation, you will need to 

1. Uninstall Pytorch

2. Identify the CUDA version compatible with your GPU

3. Reinstall PyTorch for this version of CUDA

Internet connection
^^^^^^^^^^^^^^^^^^^^^
The trained models and accompanying datasets, when called the first time, need to be downloaded from the internet. Images also need to be downloaded during the running. Therefore, please make sure you are connected to the internet.
