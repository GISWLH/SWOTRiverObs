'''
Copyright (c) 2018-, California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.
All rights reserved.

Author (s): Alex Fore
'''
import os
import numpy as np
from collections import OrderedDict as odict

from SWOTRiver.products.product import Product, FILL_VALUES, textjoin

class L2PIXCVector(Product):
    UID = "l2_hr_pixcvector"

    # copied from L2HRPIXC
    ATTRIBUTES = [
        'wavelength',
        'near_range',
        'nominal_slant_range_spacing',
        'start_time',
        'stop_time',
        'cycle_number',
        'pass_number',
        'tile_number',
        'swath_side',
        'tile_name',
        'ephemeris',
        'yaw_flip',
        'hpa_mode',
        'processing_beamwidth',
        'inner_first_latitude',
        'inner_first_longitude',
        'inner_last_latitude',
        'inner_last_longitude',
        'outer_first_latitude',
        'outer_first_longitude',
        'outer_last_latitude',
        'outer_last_longitude',
        'slc_first_line_index_in_tvp',
        'slc_end_line_index_in_tvp',
        'l1b_hr_slc_input_version',
        'static_karin_cal_input_version',
        'ref_dem_input_version',
        'water_mask_input_version',
        'static_geophys_input_version',
        'dynamic_geohys_input_version',
        'int_lr_xover_cal_input_version',
        'l2_hr_pixc_processor_nonbaseline_config_parameters',
        'ellipsoid_semi_major_axis',
        'ellipsoid_flattening',
        'mission_name',
        'institution',
        'source',
        'history',
        'conventions',
        'title',
        'contacts',
        'references',
        ]

    DIMENSIONS = odict([['points', 0]])
    VARIABLES = odict([
        ['azimuth_index',
         odict([['dtype', 'i4'],
                ['long_name', 'rare interferogram azimuth index'],
                ['units', '1'],
                ['valid_min', 0],
                ['valid_max', 999999],
                ['comment', 'rare interferogram azimuth index'],
                ])],
        ['range_index',
         odict([['dtype', 'i4'],
                ['long_name', 'rare interferogram range index'],
                ['units', '1'],
                ['valid_min', 0],
                ['valid_max', 999999],
                ['comment', 'rare interferogram range index'],
                ])],
        ['latitude_vectorproc',
         odict([['dtype', 'f8'],
                ['long_name', 'latitude'],
                ['standard_name', 'latitude'],
                ['units', 'degrees_north'],
                ['valid_min', -90],
                ['valid_max', 90],
                ['comment', 'geodetic latitude (degrees north of equator)'],
                ])],
        ['longitude_vectorproc',
         odict([['dtype', 'f8'],
                ['long_name', 'longitude'],
                ['standard_name', 'longitude'],
                ['units', 'degrees_east'],
                ['valid_min', 0],
                ['valid_max', 360],
                ['comment', 'longitude (east of the prime meridian)'],
                ])],
        ['height_vectorproc',
         odict([['dtype', 'f4'],
                ['long_name', 'height above reference ellipsoid'],
                ['units', 'm'],
                ['valid_min', -999999],
                ['valid_max', 999999],
                ['comment', 'height above reference ellipsoid'],
                ])],
        ['node_index',
         odict([['dtype', 'i4'],
                ['long_name', 'node index'],
                ['units', '1'],
                ['valid_min', 0],
                ['valid_max', 2147483647],
                ['comment', 'index of node this pixel was assigned to'],
                ])],
        ['reach_index',
         odict([['dtype', 'i4'],
                ['long_name', 'reach index'],
                ['units', '1'],
                ['valid_min', 0],
                ['valid_max', 2147483647],
                ['comment', 'index of reach this pixel was assigned to'],
                ])],
        ['river_tag',
         odict([['dtype', 'i4'],
                ['long_name', 'river tag'],
                ['units', '1'],
                ['valid_min', 0],
                ['valid_max', 2147483647],
                ['comment', 'river tag'],
                ])],
        ['segmentation_label',
         odict([['dtype', 'i4'],
                ['long_name', 'segmentation label'],
                ['units', '1'],
                ['valid_min', 0],
                ['valid_max', 2147483647],
                ['comment', 'segmentation label'],
                ])],
        ['good_height_flag',
         odict([['dtype', 'u1'],
                ['long_name', 'good height flag'],
                ['valid_min', 0],
                ['valid_max', 1],
                ['comment', 'good height flag'],
                ])],
        ['distance_to_node',
         odict([['dtype', 'f4'],
                ['long_name', 'distance to node'],
                ['units', 'm'],
                ['valid_min', 0],
                ['valid_max', 9999],
                ['comment', 'distance to node'],
                ])],
        ['along_reach',
         odict([['dtype', 'f4'],
                ['long_name', 'along reach distance'],
                ['units', 'm'],
                ['valid_min', -999999],
                ['valid_max', 999999],
                ['comment', 'along reach distance'],
                ])],
        ['cross_reach',
         odict([['dtype', 'f4'],
                ['long_name', 'across reach distance'],
                ['units', 'm'],
                ['valid_min', -999999],
                ['valid_max', 999999],
                ['comment', 'across reach distance'],
                ])],
        ['pixc_index',
         odict([['dtype', 'i4'],
                ['long_name', 'pixel cloud index'],
                ['units', '1'],
                ['valid_min', 0],
                ['valid_max', 2147483647],
                ['comment', 'index in pixel cloud product'],
                ])],
        ['lake_flag',
         odict([['dtype', 'u1'],
                ['long_name', 'lake flag'],
                ['valid_min', 0],
                ['valid_max', 1],
                ['comment', 'lake flag'],
                ])],
        ])

    for name, reference in VARIABLES.items():
        reference['dimensions'] = DIMENSIONS
