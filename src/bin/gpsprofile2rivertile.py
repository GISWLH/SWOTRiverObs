#!/usr/bin/env python
'''
Copyright (c) 2022-, California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.
All rights reserved.

Author (s): Alex Fore
'''
import os
import ast
import argparse
import logging

import RDF
import SWOTRiver.Estimate
import SWOTRiver.products.calval
from SWOTRiver.errors import RiverObsException

LOGGER = logging.getLogger('gps_prof2river')

def main():
    """Sample script for running calval data through RiverObs"""
    parser = argparse.ArgumentParser()
    parser.add_argument('gps_profile', help='GPS profile file')
    parser.add_argument('out_riverobs_file', help='Output NetCDF file')
    parser.add_argument('rdf_file', help='Static config params')
    parser.add_argument(
        '-l', '--log-level', type=str, default="info",
        help="logging level, one of: debug info warning error")
    args = parser.parse_args()

    level = {'debug': logging.DEBUG, 'info': logging.INFO,
             'warning': logging.WARNING, 'error': logging.ERROR}[args.log_level]
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=level, format=format)

    config = RDF.RDF()
    config.rdfParse(args.rdf_file)
    config = dict(config)

    LOGGER.debug('Converting GPS profile to netCDF4 file')
    
    gpsnc = SWOTRiver.products.calval.GPSProfile.from_ncfile(args.gps_profile)
    #breakpoint()
    #gpsnc = SWOTRiver.products.calval.GPSProfile.from_native(args.gps_profile)
    gps_profile_basename = os.path.basename(args.gps_profile)
    fake_pixc_fname = 'pixc_{}'.format(gps_profile_basename)
    gpsnc.to_ncfile(fake_pixc_fname)

    # typecast most config values with eval since RDF won't do it for me
    # (excluding strings)
    for key in config.keys():
        if key in ['geolocation_method', 'reach_db_path', 'height_agg_method',
                   'area_agg_method', 'slope_method', 'outlier_method']:
            if config[key].lower() != 'none':
                continue
        print(key)
        config[key] = ast.literal_eval(config[key])

    LOGGER.debug('Computing bounding box')
    bbox = gpsnc.compute_bounding_box()

    estimator = SWOTRiver.Estimate.CalValToRiverTile(fake_pixc_fname)
    estimator.load_config(config)

    # generate empty output file on errors
    try:
        estimator.do_river_processing()
    except RiverObsException as exception:
        LOGGER.error(
            'Unable to continue river processing: {}'.format(exception))

    # Build and write a rivertile-style output file
    estimator.build_products()
    estimator.rivertile_product.to_ncfile(args.out_riverobs_file)

if __name__ == "__main__":
    main()
