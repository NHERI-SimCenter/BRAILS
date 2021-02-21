# -*- coding: utf-8 -*-
"""
/*------------------------------------------------------*
|                         BRAILS                        |
|                                                       |
| Author: Charles Wang,  UC Berkeley, c_w@berkeley.edu  |
|                                                       |
| Date:    10/15/2020                                   |
*------------------------------------------------------*/
"""


zoo = {
    'roofType':{
        'fileURL' : 'https://zenodo.org/record/4059084/files/roof_classifier_v0.1.h5',
        'classNames' : ['flat', 'gabled', 'hipped']
    } ,
    'residentialOccupancyClass':{
        'fileURL' : 'https://zenodo.org/record/4091547/files/occupancy-78-78-79.h5',
        'classNames' : ['RES1', 'RES3']
    } ,
    'occupancyClass':{
        'fileURL' : 'https://zenodo.org/record/4553798/files/occupancy_InceptionV3_V0.2.h5',
        'classNames' : ['RES3', 'COM' ,'RES1']
    } ,
    'softstory':{
        'fileURL' : 'https://zenodo.org/record/4094334/files/softstory-80-81-87.5-v0.1.h5',
        'classNames' : ['others', 'softstory']
    } 
}