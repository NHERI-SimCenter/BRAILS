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
        'fileURL' : 'https://zenodo.org/record/4567613/files/rooftype_ResNet50_V0.2.h5',
        'classNames' : ['flat', 'gabled', 'hipped']
    } ,
    'residentialOccupancyClass':{
        'fileURL' : 'https://zenodo.org/record/4091547/files/occupancy-78-78-79.h5',
        'classNames' : ['RES1', 'RES3']
    } ,
    'occupancyClass':{
        'fileURL' : 'https://zenodo.org/record/4795548/files/occupancy_ResNet50_V0.2.h5',
        'classNames' : ['COM' ,'RES1','RES3']
    } ,
    'softstory':{
        'fileURL' : 'https://zenodo.org/record/4565554/files/softstory_ResNet50_V0.1.h5',
        'classNames' : ['others', 'softstory']
    } 
}
