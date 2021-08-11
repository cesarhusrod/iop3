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
# import collections
import os
# import sys
import argparse
# import glob
# import subprocess
# import math
# import pickle
# import pprint
# import datetime
# import re
# from io import StringIO
# import time
from collections import defaultdict
from pprint import pprint

import numpy as np
import pandas as pd

# from astropy.coordinates import SkyCoord
# import astropy.units as u

# HTML templates packages for documentation and logging
import jinja2 # templates module

# from mcReduction import *
import mcReduction
from mcFits import *

# ------------------------ MAIN FUNCTION SECTION -----------------------------
def main():
    parser = argparse.ArgumentParser(prog='mcPipe_OK.py', \
    conflict_handler='resolve',
    description='''Main program that reads, classify and reduces data from 
    FITS located at input directory. ''',
    epilog='''''')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')
    parser.add_argument("config_dir", help="Configuration parameter files directory")
    parser.add_argument("reduction_dir", help="Reduction output directory")
    parser.add_argument("input_dir", help="Input directory")
    parser.add_argument("--border_image",
       action="store",
       dest="border_image",
       type=int,
       default=0,
       help="True is input file is for clusters [default: %(default)s].")
    # parser.add_argument("--calibration_dir",
    #     action="store",
    #     dest="calibration_dir",
    #     default="./",
    #     help="Output directory [default: %(default)s]")
    #parser.add_argument("--bool_main_file",
    #    action="store",
    #    dest="bool_main_file",
    #    type=bool,
    #    default=False,
    #    help="True is input file is for clusters [default: %(default)s].")
    parser.add_argument('-v', '--verbose', action='count', default=0,
        help="Show running and progress information [default: %(default)s].")
    args = parser.parse_args()

    # Checking input parameters
    if not os.path.exists(args.config_dir):
        str_err = 'ERROR: Config dir "{}" not available'
        print(str_err.format(args.config_dir))
        return 1
    
    if not os.path.isdir(args.reduction_dir):
        try:
            os.makedirs(args.reduction_dir)
        except IOError:
            str_err = 'ERROR: Could not create output reduction directory "{}"'
            print(str_err.format(args.reduction_dir))
            return 2

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        str_err = 'ERROR: Input directory "{}" not available'
        print(str_err.format(input_dir))
        return 3
    
    border_image = args.border_image
    if args.border_image < 0:
        border_image = 0
    elif args.border_image > 300:
        str_err = 'ERROR: Border image "{}" too high for getting realistic results.'
        print(str_err.format(args.border_image))
        return 4
    
    # Getting run date (input directory must have pattern like *MAPCAT_yyyy-mm-dd)
    date_run = re.findall('MAPCAT_(\d{4}-\d{2}-\d{2})', input_dir)[0]

    if not os.path.exists(args.reduction_dir):
        try:
            os.makedirs(args.reduction_dir)
        except IOError:
            str_err = 'ERROR: Output reduction directory "{}" couldn\'t be created'
            print(str_err.format(args.reduction_dir))
            return 4
        else:
            message = "Output reduction directory set to '{}'"
            print(message.format(args.reduction_dir))

    # Raw input data
    data_raw_input = defaultdict(list)

    # Create generic object of class mcReduction
    oReduction = mcReduction.mcReduction(input_dir, args.reduction_dir,
                                         border=border_image)

    # Input FITS classification printing
    # print('Bias = {}'.format(oReduction.bias))
    # print('Flats = {}'.format(oReduction.flats))
    # print('Science = {}'.format(oReduction.science))
    # print('Description = {}'.format(oReduction.science.describe))

    # Getting input FITS info
    some_raw_keys = ['INSFLNAM', 'NAXIS1', 'NAXIS2', 'RA', 'DEC', 'OBJECT', \
        'EXPTIME', 'DATE-OBS', 'EQUINOX', 'MJD-OBS', 'INSTRSCL', 'IMAGETYP']

    input_fits = oReduction.bias['FILENAME'].tolist() + oReduction.flats['FILENAME'].tolist() + \
        oReduction.science['FILENAME'].tolist()

    for fi in input_fits:
        ff = mcFits(fi, border=args.border_image)
        if ff.header['NAXIS1'] < 1024:
            continue
        data_raw_input['PATH'].append(fi)
        data_raw_input['RUN_DATE'].append(date_run)
        
        pol_ang = ''
        if 'INSPOROT' in ff.header:
            pol_ang = round(float(ff.header['INSPOROT']), 1)
        data_raw_input['POLANGLE'].append(pol_ang)
        
        for some_key in some_raw_keys:
            data_raw_input[some_key].append(ff.header.get(some_key, ''))
        
        info = ff.stats()
        data_raw_input['MAX'] = info['MAX']
        data_raw_input['MIN'] = info['MIN']
        data_raw_input['MEAN'] = info['MEAN']
        data_raw_input['STD'] = info['STD']
        data_raw_input['MED'] = info['MEDIAN']
        data_raw_input['TYPE'] = info['TYPE']
        
    
    # Saving input raw data
    # Ouput CSV data file info
    csv_data_raw = os.path.join(args.reduction_dir, 'input_data_raw.csv')
    df_raw_out = pd.DataFrame(data_raw_input)
    # print('df_raw_out.info()')
    # print(df_raw_out.info())
    df_raw_out.to_csv(csv_data_raw, index=False)


    # ------------------- MasterBIAS generation -------------------
    if oReduction.createMasterBIAS() != 0:
        print("MasterBIAS generation failed.")
        return 6

    # MASTERBIAS plotting
    mcBIAS = mcFits(oReduction.masterBIAS, border=oReduction.border)
    info = mcBIAS.stats()
    # print('MasterBIAS statistics \n\n{}'.format(info))
    plotBIAS = oReduction.masterBIAS.replace('.fits', '.png')
    title = '{} masterBIAS'.format(os.path.split(args.input_dir)[1])
    mcBIAS.plot(title)

    plotBIASHist = oReduction.masterBIAS.replace('.fits', '_histogram.png')
    histo_par = {'xlabelsize':8,
                 'ylabelsize': 8,
                 'bins': 100,
                 'color':'#6022aa',
                 'alpha':0.6}
    mcBIAS.plot_histogram(plotBIASHist, title=title, histogram_params=histo_par)


    # Collecting masterBIAS info (there is only one by night)
    data_masterbias_info = {}

    # Collecting FLATS info
    some_bias_keys = ['NAXIS1', 'NAXIS2', 'SOFT', 'PROCDATE', 'BIASOP', 'PXBORDER', \
        'OBJECT', 'MAX', 'MIN', 'MEAN', 'STD', 'MED']

    data_masterbias_info['RUN_DATE'] = [date_run]
    data_masterbias_info['PATH'] = oReduction.masterBIAS
    data_masterbias_info['PLOT'] = [plotBIAS]
    data_masterbias_info['HISTOGRAM'] = [plotBIASHist]

    for some_key in some_bias_keys:
        data_masterbias_info[some_key] = [mcBIAS.header.get(some_key, '')]
    
    for j in range(20):
        if f'BIAS{j}' in mcBIAS.header:
            data_masterbias_info[f'BIAS{j}'] = [mcBIAS.header.get(f'BIAS{j}', '')]
        else:
            break

    # Saving masterbias data
    # Ouput CSV data file info
    # pprint.pprint(data_masterbias_info)
    csv_masterbias_data = os.path.join(args.reduction_dir, 'masterbias_data.csv')
    df_masterbias_out = pd.DataFrame(data_masterbias_info)
    df_masterbias_out.to_csv(csv_masterbias_data, index=False)
    

    # --------------- MasterFLATs generation ---------------------
    data_masterflats_info = defaultdict(list)

    if oReduction.createMasterFLAT():
        print("MasterFLATs generation failed.")
        return 7
    for k, v in oReduction.masterFLAT.items():
        print('- ' * 5 + v)
        mcFLAT = mcFits(v, border=oReduction.border)

        mf_info = mcFLAT.stats()
        # print('FLATS statistics = {}'.format(mf_info))
        plotFLAT = oReduction.masterFLAT[k].replace('.fits', '.png')
        title = f"MF (date, pol.angle)=({oReduction.date}, {k})"
        mcFLAT.plot(title)
        plotFLATHist = oReduction.masterFLAT[k].replace('.fits', '_histogram.png')

        mcFLAT.plot_histogram(plotFLATHist, title=title, \
            histogram_params=histo_par)
        
        # Collecting FLATS info
        some_flats_keys = ['NAXIS1', 'NAXIS2', 'SOFT', 'PROCDATE', 'FLATOP', \
            'PXBORDER', 'OBJECT', 'MBIAS', 'MAX', 'MIN', 'MEAN', 'STD', 'MED']

        data_masterflats_info['PATH'].append(v)
        data_masterflats_info['RUN_DATE'].append(date_run)
        pol_ang = ''
        if 'INSPOROT' in mcFLAT.header:
            pol_ang = round(float(mcFLAT.header['INSPOROT']), 1)
        data_masterflats_info['POLANGLE'].append(pol_ang)
        data_masterflats_info['PLOT'].append(plotFLAT)
        data_masterflats_info['HISTOGRAM'].append(plotFLATHist)

        for some_key in some_flats_keys:
            data_masterflats_info[some_key].append(mcFLAT.header.get(some_key, ''))

        for j in range(20): # adding raw FLAT used for generating masterFLAT
            if "FLAT{j}" in mcFLAT.header:
                data_masterflats_info["FLAT{j}"].append(mcFLAT.header.get("FLAT{j}", ''))
            else:
                break
        
    # Saving masterbias data
    # Ouput CSV data file info
    csv_masterflats_data = os.path.join(args.reduction_dir, 'masterflats_data.csv')
    df_masterflats_out = pd.DataFrame(data_masterflats_info)
    df_masterflats_out.to_csv(csv_masterflats_data, index=False)

    # --------------- Making FITS set reduction operations --------------
    
    # info container
    data_red_out = defaultdict(list)

    # Science FITS
    oReduction.reduce()

    #print('Plotting figures. Please wait...')
    for index in range(oReduction.science.shape[0]):
        sci = oReduction.science.iloc[index]
        path_red = os.path.join(oReduction.out_dir, os.path.split(sci['FILENAME'])[1])
        if not os.path.exists(path_red):
            str_warn = "WARNING: '{}' not available."
            print(str_warn.format(path_red))
            continue
        mcRED = mcFits(path_red)

        # FWHM
        dictSEx = dict()
        dictSEx['CATALOG_NAME'] = mcRED.path.replace('.fits', '.cat')
        dictSEx['CONFIG_FILE'] = os.path.join(args.config_dir, 'daofind.sex')
        mcRED.compute_fwhm(dictSEx)
        mcRED = 0 # Forcing to write FWHM dataSEX
        mcRED = mcFits(path_red, border=border_image)

        # print(f"{path_red} statistics...")
        # print(mcRED.stats())
        # print('')

        plotSCI = mcRED.path.replace('.fits', '.png')
        title_pattern = "{} - {} - {:.2f} deg - {:.1f} s"
        title = title_pattern.format(os.path.split(mcRED.path)[1][:-5], \
            sci['procOBJ'], float(sci['INSPOROT']), float(sci['EXPTIME']))
        mcRED.plot(title)
        plotSCIHist = mcRED.path.replace('.fits', '_histogram.png')
        mcRED.plot_histogram(plotSCIHist, title=title, histogram_params=histo_par)

        # Collecting REDUCED FITS info
        some_reduction_keys = ['NAXIS1', 'NAXIS2', 'SOFT', 'PROCDATE', 'PXBORDER', \
            'INSFLNAM',  \
            'RA', 'DEC', 'OBJECT', 'EXPTIME', 'DATE-OBS', 'EQUINOX', 'MJD-OBS', \
            'BIAS', 'FLAT', 'FWHM', 'FWHMSTD', 'FWNSOURC', 'FWHMFLAG', 'FWHMELLI', \
            'MAX', 'MIN', 'MEAN', 'STD', 'MED']
        data_red_out['PATH'].append(mcRED.path)
        data_red_out['RUN_DATE'].append(date_run)
        pol_ang = ''
        if 'INSPOROT' in mcRED.header:
            pol_ang = round(float(mcRED.header['INSPOROT']), 1)
        data_red_out['POLANGLE'].append(pol_ang)
        data_red_out['PLOT'].append(plotSCI)
        data_red_out['HISTOGRAM'].append(plotSCIHist)

        for some_key in some_reduction_keys:
            data_red_out[some_key].append(mcRED.header.get(some_key, ''))

    # Saving output reduced data
    # Output CSV data file
    csv_data_red = os.path.join(args.reduction_dir, 'output_data_red.csv')
    df_red_out = pd.DataFrame(data_red_out)
    df_red_out.to_csv(csv_data_red, index=False)

    return 0

if __name__ == '__main__':
    if not main():
        print("*" * 80)
        print("Reduction done successfully!!")
        print("*" * 80)
