#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu April 09 09:49:53 2020

___e-mail__ = cesar_husillos@tutanota.com
__author__ = 'Cesar Husillos'

VERSION:
    1.0 Initial version
"""


# ---------------------- IMPORT SECTION ----------------------
import os
import argparse
import glob
import re

import pandas as pd
import jinja2 # HTML templates packages for documentation and logging

from mcFits import *

# ------------------------ MAIN FUNCTION SECTION -----------------------------
def main():
    parser = argparse.ArgumentParser(prog='iop3_pipeline.py', \
    conflict_handler='resolve',
    description='''Main program that reads, classify, reduces and calibrate 
    FITS located at input directory. ''',
    epilog='''''')
    parser.add_argument("config_dir", help="Configuration parameter files directory") # mandatory argument
    parser.add_argument("input_dir", help="Input directory")  # mandatory argument
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    parser.add_argument("--border_image",
       action="store",
       dest="border_image",
       type=int,
       default=15,
       help="Discarded FITS border pixels [default: %(default)s].")
    parser.add_argument('-v', '--verbose', action='count', default=0,
        help="Show running and progress information [default: %(default)s].")
    args = parser.parse_args()

    # Checking input parameters
    if not os.path.exists(args.config_dir):
        str_err = 'ERROR: Config dir "{}" not available'
        print(str_err.format(args.config_dir))
        return 1


    # Pattern for directories structure in IOP3 pipeline
    # Directory structure...
    # INPUT
    # [root_data]/data/raw/MAPCAT_yyyy-mm-dd
    # OUTPUT reduction
    # [root_data]/data/reduction/MAPCAT_yyyy-mm-dd
    # OUTPUT calibration
    # [root_data]/data/calibration/MAPCAT_yyyy-mm-dd
    # OUTPUT polarization
    # [root_data]/data/final/MAPCAT_yyyy-mm-dd

    input_dir = os.path.abspath(args.input_dir) # absolute path
    config_dir = os.path.abspath(args.config_dir) # absolute path

    # Does input verify pattern given above?
    pattern = re.findall('(/data/raw/MAPCAT_\d{4}-\d{2}-\d{2})', input_dir)
    if not len(pattern):  # pattern not found
        print('ERROR: Input directory pattern "[root_data]/data/raw/MAPCAT_yyyy-mm-dd" not verified.')
        return 2
    if not os.path.isdir(input_dir):
        str_err = 'ERROR: Input directory "{}" not available'
        print(str_err.format(input_dir))
        return 3

    # Getting run date (input directory must have pattern like *MAPCAT_YYYY-MM-DD)
    date_run = re.findall('MAPCAT_(\d{4}-\d{2}-\d{2})', input_dir)[0]

    path_line = input_dir.split('/')
    reduction_dir = '/'.join(path_line[:-2] + ['reduction', path_line[-1]])
    calibration_dir = '/'.join(path_line[:-2] + ['calibration', path_line[-1]])
    polarization_dir = '/'.join(path_line[:-2] + ['final', path_line[-1]])
    print(f"reduction_dir = {reduction_dir}")
    print(f"calibration_dir = {calibration_dir}")
    print(f"final_dir (polarization calculations) = {polarization_dir}")
    # root_dir, input_dirname = os.path.split(input_dir)
    # reduction_dir = os.path.join(root_dir, os.path.join(input_dirname, "_reduction"))
    if not os.path.exists(reduction_dir):
        try:
            os.makedirs(reduction_dir)
        except IOError:
            str_err = 'ERROR: Output reduction directory "{}" couldn\'t be created'
            print(str_err.format(reduction_dir))
            return 4
        else:
            message = "Output reduction directory set to '{}'"
            print(message.format(reduction_dir))

    border_image = args.border_image
    if args.border_image < 0:
        border_image = 0
    elif args.border_image >= 300:
        str_err = "ERROR: Border size too big for realistic analysis ({} given)."
        print(str_err.format(args.border_image))
        return 5

    # 1st STEP: Input raw image reduction
    com_reduction = f"python iop3_reduction.py --border_image={border_image} {config_dir} {reduction_dir} {input_dir}"
    print(com_reduction)
    subprocess.Popen(com_reduction, shell=True).wait()

    return -1

    # 2nd STEP: Input reduced images calibration
    reduced_fits = glob.glob(os.path.join(reduction_dir, 'caf-*-sci-agui.fits'))
    reduced_fits.sort()
    for rf in reduced_fits:
        im_time = re.findall('caf-(\d{8}-\d{2}:\d{2}:\d{2})-sci-agui.fits', rf)[0]
        im_time = im_time.replace(':', '')
        cal_dir = os.path.join(calibration_dir, im_time)
        com_calibration = f"python iop3_calibration.py {args.config_dir} {cal_dir} {rf}"
        # print('+' * 100)
        # print(com_calibration)
        # print('+' * 100)
        # subprocess.Popen(com_calibration, shell=True).wait()
        # Alternative command
        """"
        if not os.path.isdir(cal_dir):
            try:
                os.makedirs(cal_dir)
            except IOError:
                print(f"ERROR: Calibration directory '{cal_dir}' could no be generated.")
                raise
        with open(os.path.join(cal_dir, im_time + '.log'), 'w') as log_file:
            subprocess.Popen(com_calibration, shell=True, stdout=log_file).wait()
        """

    # return -1

    #  3rd STEP: Computing polarimetric parameters
    print("COMPUTING POLARIMETRY. Please wait...")
    com_polarimetry = f"python iop3_polarimetry.py {calibration_dir} {polarization_dir}"
    print(com_polarimetry)
    subprocess.Popen(com_polarimetry, shell=True).wait()

    return -1

    # 4th STEP: Inserting results in database
    # raw_csv = os.path.join(reduction_dir, 'input_data_raw.csv')
    # red_csv = os.path.join(reduction_dir, 'output_data_red.csv')
    # bias_csv = os.path.join(reduction_dir, 'masterbias_data.csv')
    # flats_csv = os.path.join(reduction_dir, 'masterflats_data.csv')
    
    # info_files = glob.glob(os.path.join(calibration_dir, '*/*final_info.csv'))
    # cal_df = pd.concat([pd.read_csv(p) for p in info_files], ignore_index=True)
    # cal_csv = os.path.join(calibration_dir, 'output_data_cal.csv')
    # cal_df.to_csv(cal_csv, index=False)
    
    # Inserting information on Database...
    # com_insertdb = f"python iop3_add_db_info.py {raw_csv} {bias_csv} {flats_csv} {red_csv} {cal_csv}"
    data_dir = input_dir.split('data')[0] + 'data'
    com_insertdb = f"python iop3_add_db_info.py {data_dir} {date_run}"
    print(com_insertdb)
    with open(os.path.join(polarization_dir, 'db.log'), 'w') as log_file:
        subprocess.Popen(com_insertdb, shell=True, stdout=log_file).wait()


    return 0

    #

if __name__ == '__main__':
    if not main():
        print("Done!!")
