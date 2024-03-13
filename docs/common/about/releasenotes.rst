.. _lbl-release:
.. role:: blue

*************
Release Notes
*************
Major Version 3
=================
   .. warning::

      The major version number was increased from 2 to 3 as changes were made to input and output formats of |app|. This means old examples will not be loaded in this version of the tool.

   .. dropdown::    **Version 3.1.1** (:blue:`Current`)
      :open:

      **Release date:** February 2024

      **Highlights**

        #. Enabled rapidtools feature to extract building-level imagery from NHERI RAPID aerial and street-level image data
        #. Added the capability to pull National Structure Inventory as baseline building inventory
        #. Updated TranspInventoryGenerator for more stable API queries
        #. Updated FacadeParser for more effective memory utilization

   .. dropdown::    **Version 3.1.0**

      **Release date:** December 2023

      **Highlights**

        #. Enabled Transportation inventory generation for roadways, bridges, tunnels, and railroads,
        #. Revised FacadeParser to calculate dimensions from depth maps (as opposed to ray tracing from camera coordinates) for more robust dimension predictions,
        #. Added FEMA USA Structures as an additional building footprint source.


   .. dropdown::    **Version 3.0.0**

      **Release date:** September 2022

      **Highlights**

        #. Added new modules for predicting building height, building window area, first-floor height, existence of chimneys, and existence of garages, roof cover type, roof eave height, roof pitch,
        #. Streamlined location inputs by resolving the input through Nominatim API,
        #. Enabled on-the-fly footprint parsing from Microsoft Footprint Database and OpenStreetMaps,
        #. Added the capability to remove neighboring buildings from street-level and satellite imagery,
        #. Enabled inventory output creation in a CSV format compatible with R2D,
        #. Added pipelines for generalized image classification and semantic segmentation models.
