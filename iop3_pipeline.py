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
from datetime import datetime
import os
import argparse
import glob
import re
from collections import defaultdict
from typing import DefaultDict
import math
from venv import create
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

# Coordinate system transformation package and modules
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import FK5
import astropy.units as u
from iop3_astrometric_calibration import execute_command

from mcFits import *
from mcReduction import mcReduction

# ----------------------- FUNCTIONS SECTION ------------------------------

def read_blazar_file(blazar_csv):
    """
    """
    df_mapcat = pd.read_csv(blazar_csv, comment='#')
    # print(df_mapcat.info())
    # getting coordinates in degrees unit
    c  = []
    for ra, dec in zip(df_mapcat['ra2000_mc'], df_mapcat['dec2000_mc']):
        c.append("{} {}".format(ra, dec))

    mapcat_coords = SkyCoord(c, frame=FK5, unit=(u.hourangle, u.deg), \
    obstime="J2000")
    df_mapcat['ra2000_mc_deg'] = mapcat_coords.ra.deg
    df_mapcat['dec2000_mc_deg'] = mapcat_coords.dec.deg

    return df_mapcat

def create_dataframe(fits_paths, keywords=[]):
    """
        Search for keywords in fit_path headers given and return it as 
        pandas.DataFrame.

    Args:
        keywords (list): List of keywords.

    Returns:
        pandas.DataFrame: With 'PATH' as additional column.
    """
    info = DefaultDict(list)
    
    for fp in fits_paths:
        inf = mcFits(fp).get_data(keywords=keywords)
        info['PATH'].append(fp)
        for k in keywords:
            info[k].append(inf[k])
    
    return pd.DataFrame(info)

def closest_blazar(blazar_data, path_fits):
    # Getting header informacion
    i_fits = mcFits(path_fits)
    input_head = i_fits.header
    # Central FITS coordinates
    if 'MAPCAT' in path_fits:
        icoords = "{} {}".format(input_head['RA'], input_head['DEC'])
        input_coords = SkyCoord(icoords, frame=FK5, unit=(u.deg, u.deg), \
                                    obstime="J2000")
    else:
        if 'OBJCTRA' in input_head:
            icoords = "{} {}".format(input_head['OBJCTRA'], input_head['OBJCTDEC'])
        else:
            print('Object coordinates are missing from header of {}'.format(path_fits) )
            icoords = "0 0"
        input_coords = SkyCoord(icoords, frame=FK5, unit=(u.hourangle, u.deg), \
                    obstime="J2000")
    # Blazars subset...
    df_blazars = blazar_data[blazar_data['IAU_name_mc'].notna()]
    c  = []
    if input_coords.ra.value==0.0 and input_coords.dec.value==0.0:
        for name,ra,dec in zip(df_blazars['IAU_name_mc'],df_blazars['ra2000_mc'], df_blazars['dec2000_mc']):
            s = SequenceMatcher(None, name, path_fits.split('/')[-1].split('-')[0])
            if s.ratio() > 0.6:
                input_coords=SkyCoord("{} {}".format(ra,dec), frame=FK5, unit=(u.hourangle, u.deg), \
                                          obstime="J2000")
                print('Found this object in blazar list: %s' % name)
                print('with name similar to (from fits file): %s' % path_fits)
                print('Using its coordinares instead, take with caution!')
                break

    for ra, dec in zip(df_blazars['ra2000_mc'], df_blazars['dec2000_mc']):
        c.append("{} {}".format(ra, dec))
    blazar_coords = SkyCoord(c, frame=FK5, unit=(u.hourangle, u.deg), \
    obstime="J2000")
    # Closest MAPCAT source to FITS central coordinates
    # Distance between this center FITS and MAPCAT targets (in degrees)
    distances = input_coords.separation(blazar_coords)
    # Closest source in complete set...
    i_min = distances.deg.argmin()
    
    return df_blazars.iloc[i_min], distances.deg[i_min]

def subsets(data):
    """It analyzes data passed and make subsets of observations according to date-obs.
    Args:
        data (pandas.DataFrame): Data from objects taken from 4 polarization angles grism.
        name (str): object name for subsetting

    Returns:
        list: of valid subsets of observation for object called 'name'.
    """
    sub_s = []
   
    # search for duplicates
    while len(data.index) > 0: # iterate until there is no more observations
        print(f'data elements = {len(data.index)}')
        if 'INSPOROT' in data:
            index_dup = data.duplicated(['INSPOROT'], keep='last') # false for last duplicated (angle, type) item
        else:
            index_dup = data.duplicated(['FILTER'], keep='last') # false for last duplicated (angle, type) item
        sub_s.append(data[~index_dup])  # append last set of duplicated items to list
        data = data[index_dup] # delete previous last repeated set of observations
    
    return sub_s

def object_groups(data_object):
    """It returns a list of data subsets.
    
    The number of subsets depends on the number of series of observations taken 
    night for the same object. The usual number of elements for each subset
    is equal to number of angles set for the instrument (usually 4).
    
    Args:
        data (pandas.DataFrame): Data from objects taken from 4 polarization angles grism.
        name (str): object name for subsetting

    Returns:
        list: of valid subsets of observation for object called 'name'.
    """
    data_sets = []
    
    # checking EXPTIME keyword: every set of measurements in different angles must have same EXPTIME
    exptimes = sorted(data_object['EXPTIME'].unique().tolist())
    print(f"EXPTIMES = {exptimes}")
    
    # If several EXPTIMES where taken, then several groups must be processed
    groups = []
    for et in exptimes:
        groups.append(data_object[data_object['EXPTIME'] == et])
        
    for g in groups:
        data_sets = data_sets + subsets(g)
    
    return data_sets        



def group_calibration(data, calibration_dir, config_dir, tol_pixs=10, overwrite=False, crotation=3):
    # Processing each group
    calibration = {'CAL_IMWCS': [], 'CAL_NO-IMWCS': [], 'NO-CAL': []}
    non_calibrated_group_commands =[]
    non_calibrated_group_datetimes =[]
    if data is None:
        print('WARNING: Group has no information')
        return calibration
    
    # print(type(data)) 
    for index, row in data.iterrows():
        print(row)
        # print(row['dateobs'])
        if 'MAPCAT' in calibration_dir:
            dt_obj = datetime.fromisoformat(row['DATE-OBS'])
        else:
            dt_obj = datetime.fromisoformat(row['DATE-OBS'][:-3])
        im_time = dt_obj.strftime('%Y%m%d-%H%M%S')
        cal_dir = os.path.join(calibration_dir, im_time)
        
        # generating calibration output directory
        if not os.path.isdir(cal_dir):
            try:
                os.makedirs(cal_dir)
            except IOError:
                print(f"ERROR: Calibration directory '{cal_dir}' could no be generated.")
                raise

        # calibration command
        reduced = row['PATH'].replace('raw', 'reduction')
        
        com_calibration = "python iop3_astrometric_calibration.py --crotation={} --tol_pixs={} {} {} {}"
        if overwrite:
            com_calibration = "python iop3_astrometric_calibration.py --crotation={} --overwrite --tol_pixs={} {} {} {}"

        com_calibration = com_calibration.format(crotation, tol_pixs, config_dir, cal_dir, reduced)
        print('+' * 100)
        print(com_calibration)
        print('+' * 100)
        with open(os.path.join(cal_dir, im_time + '.log'), 'w') as log_file:
            res = subprocess.run(com_calibration, stdout=log_file, \
                stderr=subprocess.PIPE, shell=True)
            if res.returncode:
                print(f'ASTROCALIBRATION,ERROR,"Failed for calibrating {reduced} file."')
        # Checking for succesful calibration
        calibrated = glob.glob(os.path.join(cal_dir, '*final.fit*'))
        if calibrated:
            calibration['CAL_IMWCS'].append(calibrated[0])
            # Photometric calibration
            if overwrite:
                com_photocal = f"python iop3_photometric_calibration.py --overwrite {config_dir} {cal_dir} {calibrated[0]}"
            else:
                com_photocal = f"python iop3_photometric_calibration.py {config_dir} {cal_dir} {calibrated[0]}"
            print('+' * 100)
            print(com_photocal)
            print('+' * 100)
            
            res = subprocess.run(com_photocal, stdout=subprocess.PIPE, \
                stderr=subprocess.PIPE, shell=True)
            if res.returncode:
                print(f'PHOTOCALIBRATION,ERROR,"Failed for calibrating {calibrated[0]} file."')
                print(f'(Error code = {res.returncode})')
        else:
            non_calibrated_group_commands.append(row['PATH'])
            non_calibrated_group_datetimes.append(row['DATE-OBS'])

    return calibration

def get_best_rotangle(path, config_dir, cal_dir, tol_pixs=5):
    """Try serveral rotation angles and gets the one who maximize the number of matches.

    Args:
        path (str): Input FITS path.
        config_dir (str): Configuration directory path.
        cal_dir (str): Calibration output directory.
        tol_pixs (int, optional): Pixel tolerance between matches. Defaults to 5.

    Returns:
        tuple: (float, int)
                (Best rotation angle for astrometric calibration, number of matched sources)
    """
    crotation = 0
    wcs_best_match = 0
    wcsmatch = None
    if not os.path.exists(cal_dir):
        try:
            os.makedirs(cal_dir)
        except IOError:
            print(f'ASTROCALIBRATION,ERROR,"Could not create output directory \'{cal_dir}\'."')
            return -99, -99
    com_cal = "python iop3_astrometric_calibration.py --crotation={} --overwrite --tol_pixs={} {} {} {}"
    # getting best rotation angle for astrometrical calibration
    for crot in np.arange(0, 5.5, 0.5):
        cal_dir_angle = os.path.join(cal_dir, f'rotangle_{crot}')
        com_calibration = com_cal.format(crot, tol_pixs, config_dir, \
            cal_dir_angle, path)
        print(com_calibration)
        res = execute_command(com_calibration)
        if res.returncode:
            print(res)
            return res.returncode
        # read astrocalibration file
        astro_csv = glob.glob(os.path.join(cal_dir_angle, '*astrocal_process_info.csv'))
        
        if astro_csv:
            print(f"Astrometric file = '{astro_csv[0]}'")
            data_cal = pd.read_csv(astro_csv[0])
            wcsmatch = data_cal['WCSMATCH'][0]
            if  wcsmatch > wcs_best_match:
                crotation = crot
                wcs_best_match = wcsmatch
    
        print(f"(wcs_best_match, WCSMATCH) = ({wcs_best_match}, {wcsmatch})")
    
    return crotation, wcs_best_match
            
def check_directories(dirs):
    """[summary]

    Args:
        dirs (list): Directory paths for ckecking.
        
    Returns:
        int: First list (index+1) of failed input directory.
            0 means that every directories exist.
    """
    ret_val = 0
    # Checking input parameters
    for ind, dire in enumerate(dirs):
        if not os.path.exists(dire):
            ret_val = ind + 1
            break
            
    return ret_val

def create_directories(raw_dir, path_subdir='raw'):
    """Create directories for saving reduction, calibration and final phoyometrical results.

    Args:
        path_subdir (str): Raw FITS images to process.
        keyword (str): Part of input directory to replace for generate new output paths.

    Returns:
        dict: Output directory dictionary. If everything was well, it 
            must contain 3 pairs keyword:values, 
            'reduction_dir', 'calibration_dir' and 'polarization_dir'.º
    """
    dirs = {}
    dirs['reduction_dir'] = raw_dir.replace(path_subdir, 'reduction')
    dirs['calibration_dir'] = raw_dir.replace(path_subdir, 'calibration')
    dirs['polarization_dir'] = raw_dir.replace(path_subdir, 'final')
    
    out_dirs = {}
    for k, v in dirs.items():
        if not os.path.exists(v):
            try:
                os.makedirs(v)
            except IOError:
                print(f'ERROR: Output reduction directory "{v}" couldn\'t be created')
                break
        out_dirs[k] = v
    
    return out_dirs


def contains_valid_coords(fits_path, keywordRA='RA', keywordDEC='DEC'):
    # Getting header informacion
    i_fits = mcFits(fits_path)
    input_head = i_fits.header
    if keywordRA not in input_head:
        print(f'PIPELINE,ERROR,Input coordinates (RA DEC) not in HEADER')
        return False
    else:
        # Central FITS coordinates
        icoords = "{} {}".format(input_head[keywordRA], input_head[keywordDEC])
    try:
        input_coords = SkyCoord(icoords, frame=FK5, unit=(u.deg, u.deg), \
        obstime="J2000")
    except ValueError:
        print(f'PIPELINE,ERROR,Input coordinates (RA DEC) = ({icoords}) for "{fits_path}" are not valid.')
        return False
    
    return True

def recover_fits_coords(fits_path, blazar_data, verbose=True):
    """_summary_

    Args:
        fits_path (_type_): _description_
        blazar_data (_type_): _description_
        verbose (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    recover = False
    if 'MAPCAT' in fits_path:
        keywordRA = 'RA'
        keywordDEC = 'DEC'
    else:
        keywordRA = 'OBJCTRA'
        keywordDEC = 'OBJCTDEC'
    if not contains_valid_coords(fits_path, keywordRA, keywordDEC):
        recover = True
        # Changing coordinates to IOP3 source
        i_fits = mcFits(fits_path)
        objname = i_fits.header['OBJECT'].split()[0]
        blazar = blazar_data[blazar_data['IAU_name_mc'] == objname]
        if len(blazar.index) > 0:
            # ra2000_mc,dec2000_mc
            newRA = blazar['ra2000_mc_deg'].values[0]
            newDEC = blazar['dec2000_mc_deg'].values[0]
            cards = [('RA', newRA, ''), \
                ('DEC', newDEC, '')]
            i_fits.update_header(cards)
            if verbose:
                print(f'FITS = {fits_path}')
                print(f"(oldRA, oldDEC) = ({i_fits.header['RA']}, {i_fits.header['DEC']})")
                print(f'(OBJECT, newRA, newDEC) = ({objname}, {newRA}, {newDEC})')
    return recover

def contains_valid_dateobs(fits_path, keyword='DATE-OBS'):
    """Check the existence of 'DATE-OBS' FITS keyword.

    Args:
        fits_path (str): Inputs FITS path.
    
    Returns:
        bool:
            True, means fits_path has valid observation date keyword.
            False, in other case.
    """
    return (keyword in mcFits(fits_path).header)


def has_near_calibrators(fits_path, blazars_data, max_deg_dist=0.5):
    """Check if a blazar closer than 'max_deg_dist' to FITS center is available in 'blazars_data'.

    Args:
        fits_path (str): Inputs FITS path.
        blazars_data (pandas.DataFrame): IOP3 calibrators info.
        max_deg_dist (float): max distance (in degrees) allowed between 
            'fits_path' image center And blazars/star in 'blazars_data'.
    
    Returns:
        True, means fits_path is closer than 'max_deg_dist' degrees to an IOP3 calibrator.
        False, in other case.
    """
    calibrator, dist_arcs = closest_blazar(blazars_data, fits_path)
    
    return (dist_arcs <= max_deg_dist)

def is_blazar(fits_path, blazars_data, max_deg_dist=0.5):
    """Check if FITS target is a blazar.

    Args:
        fits_path (str): Inputs FITS path.
        blazars_data (pandas.DataFrame): IOP3 calibrators info.
        max_deg_dist (float): max distance (in degrees) allowed between 
            'fits_path' image center And blazars/star in 'blazars_data'.
    
    Returns:
        bool: True, fits contains a blazar
             False, fits does not contain a blazar
    """
    calibrator, dist_arcs = closest_blazar(blazars_data, fits_path)
    
    if (dist_arcs > max_deg_dist) or (calibrator['Rmag_mc'] > 0) :
        return False
    return True

def is_star(fits_path, blazars_data, max_deg_dist=0.5):
    """Check if FITS center is close to IOP3 calibration star.

    Args:
        fits_path (str): Inputs FITS path.
        blazars_data (pandas.DataFrame): IOP3 calibrators info.
        max_deg_dist (float): max distance (in degrees) allowed between 
            fits_path image center And blazars/star in 'blazars_data'.
    
    Returns:
        bool: True, fits contains a star.
             False, fits does not contain a star.
    """
    calibrator, dist_arcs = closest_blazar(blazars_data, fits_path)
    
    if (dist_arcs > max_deg_dist) or (calibrator['Rmag_mc'] < 0) :
        return False
    
    return True

def is_saturated(fits_path, max_median=40000):
    """Check if FITS data is not valid because of sobre-exposed data.

    Args:
        fits_path (str): Inputs FITS path.
        max_median (float): Max allowed value for median data.
    
    Returns:
        bool: True if saturated.
    """
    return np.median(mcFits(fits_path).data) > max_median
    
# ------------------------ MAIN FUNCTION SECTION -----------------------------
def main():
    parser = argparse.ArgumentParser(prog='iop3_pipeline.py', \
    conflict_handler='resolve',
    description='''Main program that reads, classify, reduces and calibrates 
    FITS located at input directory. ''',
    epilog="")

    parser.add_argument("-b", "--border_image",
       action="store",
       dest="border_image",
       type=int,
       default=15,
       help="Discarded FITS border pixels. Values between 0 and 150 are accepted. Else, it is set to 15 [default: %(default)s].")
    parser.add_argument("--tol_pixs",
       action="store",
       dest="tol_pixs",
       type=int,
       default=15,
       help="Tolerance for distance in pixels for matching between objects in external catalog and FITS detections. [default: %(default)s].")
    parser.add_argument('--ignore_farcalib', dest='ignore_farcalib', action='store_true', \
        help='If False, pipeline stops execution if some science FITS has no close enough IOP3 calibrator.')
    parser.add_argument("--overwrite", dest='overwrite', action='store_true', \
        help='Pipeline overwrite previous calibrations.')
    parser.add_argument("--skip_reduction", dest='skip_reduction', action='store_true', \
        help='Skip reduction process in pipeline steps.')    
    parser.add_argument("--skip_astrocal", dest='skip_astrocal', action='store_true', \
        help='Skip astrometric calibration process in pipeline steps.')
    parser.add_argument("--skip_photocal", dest='skip_photocal', action='store_true', \
        help='Skip photometric calibration process in pipeline steps.')
    parser.add_argument("--skip_photometry", dest='skip_photometry', action='store_true', \
        help='Skip photometry process in pipeline steps.')
    parser.add_argument("--skip_polarimetry", dest='skip_polarimetry', action='store_true', \
        help='Skip polarimetry computation in pipeline steps.')
    parser.add_argument("--skip_db_registration", dest='skip_db_registration', action='store_true', \
        help='Skip registering/updating run information in database as last pipeline step.')
    parser.add_argument('-v', '--verbose', action='count', default=0,
        help="Show running and progress information [default: %(default)s].")
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    parser.add_argument("config_dir", help="Configuration parameter files directory") # mandatory argument
    parser.add_argument("input_dir", help="Input directory")  # mandatory argument
    
    args = parser.parse_args()

    # Absolute input/output paths
    input_dir = os.path.abspath(args.input_dir) # absolute path
    config_dir = os.path.abspath(args.config_dir) # absolute path
    
    dires = [input_dir, config_dir]
    index = check_directories(dires)
    if index:
        print(f'CHECKING,ERROR,"Input directory {dires[index - 1]} does not exists."')
        return 1
    
    proc_dirs = create_directories(input_dir)
    if len(proc_dirs.keys()) != 3:
        for k in ['reduction_dir', 'calibration_dir', 'final_dir']:
            if k not in proc_dirs:
                print(f'CHECKING,ERROR,"\'{k}\' output directory could not be created."')
        return 2

    # Checking image border        
    border_image = args.border_image
    if border_image < 0 or border_image > 150:
        print(f'Given border param ({border_image}) can not be accepted. Setting value to 15.')
        border_image = 15
    
    # Reading blazar file
    blazar_path = os.path.join(args.config_dir, 'blazar_photo_calib_last.csv')
    
    blazar_data = read_blazar_file(blazar_path)
    
    # Does input verify pattern given above?
    pattern = re.findall('(/\w+/\w+/\w+/\d{6})', input_dir)
    if not len(pattern):  # pattern not found
        print('ERROR: Input directory pattern "[root_data]/*/raw/*/yymmdd" not verified.')
        return 3

    # Getting run date (input directory must have pattern like *YYMMDD)
    dt_run = re.findall('(\d{6})', input_dir)[0]
    # date_run = f'20{dt_run[:2]}-{dt_run[2:4]}-{dt_run[-2:]}'
    
    # --------------- Classifying input FITS -----------------------
    oRed = mcReduction(input_dir, proc_dirs['reduction_dir'], \
        border=args.border_image)
    # print(f'oRed.science = {oRed.science}')
    # print(oRed.science.iloc[0])

    # ------------ Checking input FITS content --------------- #
    input_paths = oRed.science['FILENAME'].values
    input_bias = oRed.bias['FILENAME'].values
    input_flats = oRed.flats['FILENAME'].values

    # Trying to recover bad coordinates in FITS using OBJECT
    for p in input_paths:
        recover_fits_coords(p, blazar_data)
    
    # Rejected because of non valid RA,DEC coordinates
    if 'MAPCAT' in input_dir:
        keywordRA = 'RA'
        keywordDEC = 'DEC'
    else:
        keywordRA = 'OBJCTRA'
        keywordDEC = 'OBJCTDEC'    
    non_valid_coords = [p for p in input_paths if not contains_valid_coords(p, keywordRA, keywordDEC)]
    input_paths = [p for p in input_paths if contains_valid_coords(p, keywordRA, keywordDEC)]

    # Rejected because of non-valid observation DATE keyword found in FITS header
    non_valid_dateobs = [p for p in input_paths if not contains_valid_dateobs(p)]
    input_paths = [p for p in input_paths if contains_valid_dateobs(p)]

    # Rejected FITS because image center are far from IOP3 calibrators. FLATS and BIAS are excluded.
    far_calibrators = [p for p in input_paths if (not has_near_calibrators(p, blazar_data))]
    input_paths = [p for p in input_paths if has_near_calibrators(p, blazar_data)]

    # FITS that contain Blazars
    blazar_paths = [p for p in input_paths if (is_blazar(p, blazar_data) and not is_saturated(p))]
    # FITS that contain stars
    star_paths = [p for p in input_paths if (is_star(p, blazar_data) and not is_saturated(p))]
    
    print('------------------------- INPUT FITS STATISTICS ---------------------')
    print(f'Total science paths = {len(input_paths)}')
    print(f'Non valid DATE-OBS paths = {len(non_valid_dateobs)}')
    print(f'Non close enough calibrator paths = {len(far_calibrators)}')
    print(f'Blazar paths = {len(blazar_paths)}')
    print(f'Star paths = {len(star_paths)}')
    print('----------------------------------------------------------------------')
    
    # return -1

    # Printing info in detail
    if non_valid_dateobs:
        message = 'CHECKING,WARNING,"Non valid DATE-OBS keyword found in {}"'
        for path in non_valid_dateobs:
            print(message.foramt(path))

    if far_calibrators:
        message = 'CHECKING,ERROR,"Close calibrator was not found for \'{}\' (OBJECT = \'{}\')."'
        message += '(RA, DEC) = ({}, {})'
        for path in far_calibrators:
            if 'RA' in mcFits(path).header:
                print(message.format(path, mcFits(path).header['OBJECT'], \
                                         mcFits(path).header['RA'], mcFits(path).header['DEC']))
            elif 'OBJCTRA' in mcFits(path).header:
                print(message.format(path, mcFits(path).header['OBJECT'], \
                                         mcFits(path).header['OBJCTRA'], mcFits(path).header['OBJCTDEC']))
            else:
                print("No RA/DEC info for object "+path)

            blazar, distance = closest_blazar(blazar_data, path)
            print(f"\tClosest one = {blazar['IAU_name_mc']}")
            
        
        if not args.ignore_farcalib:
            return 4

    # **************  1st STEP: Input raw image reduction  ******************** #
    if not args.skip_reduction:
        com_reduction = "python iop3_reduction.py --border_image={} {} {} {}"
        com_reduction = com_reduction.format(border_image, config_dir, \
            proc_dirs['reduction_dir'], input_dir)
        print(com_reduction)
        # Command execution
        res_reduction = subprocess.run(com_reduction,stdout=subprocess.PIPE, \
            stderr=subprocess.PIPE, shell=True)
        if res_reduction.returncode:
            message = f'REDUCTION,ERROR,"Could not reduce {dt_run} night run."'
            print(message)
            print(f'STDOUT = {res_reduction.stdout.decode("UTF-8")}')
            print(f'STDERR = {res_reduction.stderr.decode("UTF-8")}')
            return 1

    # ****************** 2nd STEP: Input reduced images calibration  ************* #
    pol_sources = False # It checks that more than one blazar was calibrated successfully
    if 'MAPCAT' in input_dir:
        df_blazars = create_dataframe(blazar_paths, keywords=['DATE-OBS', 'OBJECT', 'EXPTIME', 'INSPOROT'])
    else:
        df_blazars = create_dataframe(blazar_paths, keywords=['DATE-OBS', 'OBJECT', 'EXPTIME', 'FILTER'])
    if len(df_blazars.index) > 0:
        pol_sources = True
  
    # ----------------- BLAZARS PROCESSING -----------------
    
    # Creating Blazars DataFrame
    if len(df_blazars.index) > 0:
        df_blazars['CLOSE_IOP3'] = [closest_blazar(blazar_data, bp)[0]['IAU_name_mc'] for bp in df_blazars['PATH'].values]
        df_blazars['CLOSE_IOP3_RA'] = [closest_blazar(blazar_data, bp)[0]['ra2000_mc_deg'] for bp in df_blazars['PATH'].values]
        df_blazars['CLOSE_IOP3_DEC'] = [closest_blazar(blazar_data, bp)[0]['dec2000_mc_deg'] for bp in df_blazars['PATH'].values]
        
        # Getting best FITS rotation for astrocalibration
        rot_dir = os.path.join(proc_dirs['calibration_dir'], 'rotation_angle')
        crot = None
        wcsmatch_best = None
        rot_path = os.path.join(rot_dir, 'result.txt')

        if  os.path.exists(rot_path):
            data_res = pd.read_csv(rot_path)
            crot = data_res['ROT_ANGLE'].values[0]
            wcsmatch_best = data_res['WCSMATH_SOURCES'].values[0]
            path_fits = data_res['PATH_FITS'].values[0]
        else:
            # The algorithm will try to compute RUN rotating angle 
            # until a minimum number of sources be detected
            n_min = 10
            
            # splitting OBJECT column and adding columns to dataframe
            try:
                df_blazars_obj = pd.DataFrame(df_blazars.OBJECT.str.split().str[0].tolist(), \
                    columns=['OBJ'])
            except ValueError:
                print(df_blazars.OBJECT.str.split().tolist())
                raise
            
            df_blazars = pd.concat([df_blazars, df_blazars_obj], axis=1)
            df_blazars.sort_values(by='EXPTIME', inplace=True, ascending=False)
            # print(df_blazars['EXPTIME'])
            
            # return -99
            # grouping by object and getting first fit for each group
            candidate_paths = df_blazars.groupby('OBJ')['PATH'].last()
            
            # candidate_paths.sort_values(by='EXPTIME', inplace=True, ascending=False)
            print('------- Rotation candidates ----------')
            # sort by descending EXPTIME
            exptimes = []
            paths = []
            objects = []
            for cp in candidate_paths:
                i_fits = mcFits(cp)
                paths.append(cp)
                exptimes.append(i_fits.header['EXPTIME'])
                objects.append(i_fits.header['OBJECT'])
            # change to Numpy arrays
            a_exptimes = np.array(exptimes)
            a_objects = np.array(objects)
            a_paths = np.array(paths)
            
            # sorting by descending exptimes
            index_sort = np.argsort(a_exptimes)
            sorted_paths = a_paths[index_sort[::-1]]
            for t, o, p in zip(a_exptimes[index_sort[::-1]], a_objects[index_sort[::-1]], a_paths[index_sort[::-1]]):
                print(f'{t} seconds (object = {o}) -> {p}')
        
            for cp in a_paths[index_sort[::-1]]:
                cp_reduct = cp.replace('raw', 'reduction')
                i_fits = mcFits(cp_reduct)
                etime = i_fits.header["EXPTIME"]
                obj = i_fits.header["OBJECT"]
                print(f'ASTROCALIBRATION,INFO,"Computing best rotation angle on \'{cp_reduct}\' (OBJECT={obj}, EXPTIME={etime})"')
                crot, wcsmatch_best  = get_best_rotangle(cp_reduct, args.config_dir, rot_dir, tol_pixs=2)
                print(f'ASTROCALIBRATION,INFO,"Rotated angle computed and number of mathed sources = ({crot}, {wcsmatch_best})"')
                if (wcsmatch_best is None) or (wcsmatch_best < n_min):
                    continue

                # If rotation angle computation was satifactory, then it saves results in file
                with open(rot_path, 'w') as fout:
                    fout.write('ROT_ANGLE,WCSMATH_SOURCES,PATH_FITS\n')
                    fout.write(f'{crot},{wcsmatch_best},{cp_reduct}')
                break

        if wcsmatch_best < 10:
            print(f"ASTROCALIBRATION,ERROR,'Not enough matches for confident rotation angle computation. Please, look at \'{rot_dir}\' for more info.'")
            return 2
        
        print(f' ---------- Best rotation angle for astrometric calibration = {crot} ({wcsmatch_best} matches) -------')
        print(df_blazars)
        pol_sources = True
        # sorting by DATE-OBS
        df_blazars = df_blazars.sort_values('DATE-OBS', ascending=True)
        # Processing each astro-calibrated FITS
        for index, row in df_blazars.iterrows():
            reduced = row['PATH'].replace('raw', 'reduction')
            print(f"DATE-OBS = {row['DATE-OBS']}")
            if 'MAPCAT' in input_dir:
                dt_obj = datetime.fromisoformat(row['DATE-OBS'])
            else:
                dt_obj = datetime.fromisoformat(row['DATE-OBS'][:-3])
            if len(row['DATE-OBS']) <= 10: # it only contains date in format YYY-mm-dd
                i_fits = mcFits(row['PATH'])
                if 'DATE' in i_fits.header and len(i_fits.header['DATE']) > 10:
                    dt_obj = datetime.fromisoformat(i_fits.header['DATE'])        

            im_time = dt_obj.strftime('%Y%m%d-%H%M%S')
            print(f'FITS DateTime = {im_time}')
            cal_dir = os.path.join(proc_dirs['calibration_dir'], im_time)

            if not args.skip_astrocal:
                com_calibration = "python iop3_astrometric_calibration.py --crotation={} --tol_pixs={} {} {} {}"
                if args.overwrite:
                    com_calibration = "python iop3_astrometric_calibration.py --crotation={} --overwrite --tol_pixs={} {} {} {}"

                com_calibration = com_calibration.format(crot, args.tol_pixs, config_dir, cal_dir, reduced)
                print('+' * 100)
                print(com_calibration)
                print('+' * 100)
                if not os.path.exists(cal_dir):
                    try:
                        os.makedirs(cal_dir)
                    except IOError:
                        str_err = 'ASTROCALIBRATION,ERROR,"Could not create output directory {}"'
                        print(str_err.format(cal_dir))
                        return 3
                with open(os.path.join(cal_dir, im_time + '.log'), 'w') as log_file:
                    res = subprocess.run(com_calibration, stdout=log_file, \
                        stderr=subprocess.PIPE, shell=True)
                    if res.returncode:
                        print(f'ASTROCALIBRATION,ERROR,"Failed for calibrating {reduced} file."')
                
    # Photometric calibration 
    calibrated = sorted(glob.glob(os.path.join(proc_dirs['calibration_dir'], '*-*/*final.fit*')))
    # print(calibrated)
    df_astrocal_blazars = create_dataframe(calibrated, keywords=['DATE-OBS', 'OBJECT', 'EXPTIME', 'INSPOROT', 'BLZRNAME', 'FWHM'])
    # print(df_astrocal_blazars[df_astrocal_blazars['BLZRNAME'].isnull()]['PATH'].values)

    ###################################################################################

    print(df_astrocal_blazars.info())
    print(df_astrocal_blazars.head())

    fwhm_mean_csv = os.path.join(proc_dirs['calibration_dir'], 'mean_fwhm.csv')
    blazar_names = df_astrocal_blazars['BLZRNAME'].unique().tolist()
    df_fwhm_mean = None

    if not os.path.exists(fwhm_mean_csv):
        fwhm_mean = DefaultDict(list)
        # computing mean FWHM
        print('Computing median FWHM/object...')
        for name in blazar_names:
            # print(f'blazar name = {name}')
            fwhm_object = df_astrocal_blazars[df_astrocal_blazars['BLZRNAME'] == name]['FWHM'].values
            # print(fwhm_object)
            fwhm_mean['BLZRNAME'].append(name)
            fwhm_mean['MEAN_FWHM'].append(round(np.nanmean(fwhm_object), 2))

        print('Final FWHM / object')
        print('-' * 40)
        for n, m in zip(fwhm_mean['BLZRNAME'], fwhm_mean['MEAN_FWHM']):
            print(f'{n} = {m}')
        
        df_fwhm_mean = pd.DataFrame(fwhm_mean)
        df_fwhm_mean.to_csv(fwhm_mean_csv, index=False)
        print(f'Writing blazars mean FWHM file in \'{fwhm_mean_csv}\'.')
    else:
        df_fwhm_mean = pd.read_csv(fwhm_mean_csv)
    
    print(df_fwhm_mean)

    if not args.skip_photocal:
        for acalib in calibrated:
            # get astrocalibrated blazar names ("BLZRNAME") from FITS
            calib_header = mcFits(acalib).header
            blzrname = calib_header.get('BLZRNAME', None)

            # Querying blazar data info file.
            # If aperture is given, the script takes it.
            try:
                aperas = blazar_data[blazar_data["IAU_name_mc"] == blzrname]["aper_mc"].values[0]
                aper_pix = aperas / calib_header['PIXSCALE']
                print(f'aper_pix from blazars file = "{aper_pix}"')
            except:
                aper_pix=float("nan")
            # If there is no aperture asigned for this object, 2 times mean FWHM for night object is taken
            if math.isnan(aper_pix):
                print(df_fwhm_mean.info())
                try:
                    fwhm = df_fwhm_mean[df_fwhm_mean["BLZRNAME"] == blzrname]["MEAN_FWHM"].values[0]
                except:
                    print("Couldn't measure FWHM, setting it to 0 and taking 12 pix as aperture")
                    fwhm = 0
                print(f'FWHM = {fwhm}')
                # Never (for ordinary apertures) a value lower than 12 pixels has been taken. So if
                # FWHM is so low than (2 * FWHM) < 12, then 12 is selected.
                aper_pix = max([2 * fwhm, 12])
            
            # Rounding pixel aperture...
            aper_pix = round(aper_pix, 1)

            cal_dir, fits_name = os.path.split(acalib)

            print(f'(BLZRNAME, FWHM) = ({blzrname}, {aper_pix})')

            # Photocalibration command
            # ---------------- REALMENTE, AQUI NO HACE FALTA APERPIX ---------------------
            if args.overwrite:
                com_photocal = f"python iop3_photometric_calibration.py --overwrite --aper_pix={aper_pix} {config_dir} {cal_dir} {acalib}"
            else:
                com_photocal = f"python iop3_photometric_calibration.py --aper_pix={aper_pix} {config_dir} {cal_dir} {acalib}"
            
            print('+' * 100)
            print(com_photocal)
            print('+' * 100)
            res = subprocess.run(com_photocal, stdout=subprocess.PIPE, \
                stderr=subprocess.PIPE, shell=True)
            if res.returncode:
                print(f'PHOTOCALIBRATION,ERROR,"Failed for calibrating {acalib} file."')
            
    # ------------- STARS PROCESSING ------------------
    # Creating Stars DataFrame
    if 'MAPCAT' in input_dir:
        df_stars =  create_dataframe(star_paths, keywords=['DATE-OBS', 'OBJECT', 'EXPTIME', 'INSPOROT'])
    else:
        df_stars =  create_dataframe(star_paths, keywords=['DATE-OBS', 'OBJECT', 'EXPTIME', 'FILTER']) 
    if len(df_stars.index) > 0:

        #df_stars['CLOSE_IOP3'] = [closest_blazar(blazar_data, bp)[0]['IAU_name_mc'] for bp in df_stars['PATH'].values]
        # sorting by DATE-OBS
        df_stars = df_stars.sort_values('DATE-OBS', ascending=True)
    
        # processing stars...
        for index, row in df_stars.iterrows():
            if 'MAPCAT' in input_dir:
                dt_obj = datetime.fromisoformat(row['DATE-OBS'])
            else:
                dt_obj = datetime.fromisoformat(row['DATE-OBS'][:-3])
            i_fits = mcFits(row['PATH'])
            if len(row['DATE-OBS']) <= 10: # it only contains date in format YYY-mm-dd
                if 'DATE' in i_fits.header and len(i_fits.header['DATE']) > 10:
                    dt_obj = datetime.fromisoformat(i_fits.header['DATE'])

            im_time = dt_obj.strftime('%Y%m%d-%H%M%S')
            print(f'im_time = {im_time}')
            cal_dir = os.path.join(proc_dirs['calibration_dir'], im_time)
            
            if not args.skip_astrocal:
                print('Calibrating star: ')
                print(f'{row}')
                cmd = 'python iop3_astrometric_calibration.py --is_star --tol_pixs={}  {} {} {}'
                if args.overwrite:
                    cmd = 'python iop3_astrometric_calibration.py --is_star --tol_pixs={} --overwrite {} {} {}'
                cmd = cmd.format(args.tol_pixs, config_dir, cal_dir, row['PATH'].replace('raw', 'reduction'))
                print('+' * 100)
                print(cmd)
                print('+' * 100)
        
                res = subprocess.run(cmd, stdout=subprocess.PIPE, \
                    stderr=subprocess.PIPE, shell=True)
                if res.returncode:
                    message = 'ASTROCALIBRATION,ERROR,"Failed processing star: DATE-OBS={}, OBJECT={}, EXPTIME={}"'
                    print(message.format(row['DATE-OBS'], row['OBJECT'], row['EXPTIME']))

            if not args.skip_photocal:    
                # Photometric calibration: stars have fixed aperture

                if 'fits' in os.path.split(row['PATH'])[1]:
                    calibrated = os.path.join(cal_dir, os.path.split(row['PATH'])[1].replace('.fits', '_final.fits'))
                else:
                    calibrated = os.path.join(cal_dir, os.path.split(row['PATH'])[1].replace('.fit', '_final.fit'))

                if not os.path.exists(calibrated):
                    print(f'PHOTOCALIBRATION,WARNING,Could not calibrate "{calibrated}"')
                    continue
                i_fits = mcFits(calibrated)

                cmd_photocal = ""
                i_fits = mcFits(calibrated)
                # Querying blazar name for this calibrated FITS
                blzr_name = i_fits.header.get('BLZRNAME', None)
                if blzr_name is None:
                    fits_path = row['PATH']
                    print(f'PHOTOMETRY,ERROR,"No blazar asigned to FITS \'{fits_path}\'"')
                    continue
                else:
                    blzr_name = blzr_name.strip()
                # Querying blazar data info file.
                # If aperture is given, the script takes it.
                aperas = blazar_data[blazar_data["IAU_name_mc"] == blzr_name]["aper_mc"].values[0]
                aper = aperas / i_fits.header['PIXSCALE']
                print(f'aper = {aper}')
                cmd_photocal = "python iop3_photometric_calibration.py --aper_pix={} {} {} {}"
                if args.overwrite:
                    cmd_photocal = "python iop3_photometric_calibration.py --overwrite --aper_pix={} {} {} {}"    
                cmd_photocal = cmd_photocal.format(aper, config_dir, cal_dir, calibrated)
                print('+' * 100)
                print(cmd_photocal)
                print('+' * 100)
                res = subprocess.run(cmd_photocal, stdout=subprocess.PIPE, \
                    stderr=subprocess.PIPE, shell=True)
                if res.returncode:
                    message = 'PHOTOCALIBRATION,ERROR,"Failed processing star: DATE-OBS={}, OBJECT={}, EXPTIME={}"'
                    print(message.format(row['DATE-OBS'], row['OBJECT'], row['EXPTIME']))

    #  3rd STEP: Getting aperture photometry
    if not args.skip_photometry:
        astro_photo_calibrated = sorted(glob.glob(os.path.join(proc_dirs['calibration_dir'], '*-*/*final.fit*')))
        print(f'PHOTOMETRY,INFO,"{len(astro_photo_calibrated)} files for getting photometry."')
        for ap_calib in astro_photo_calibrated:
            i_fits = mcFits(ap_calib)
            # Querying blazar name for this calibrated FITS
            blzr_name = i_fits.header.get('BLZRNAME', None)
            if blzr_name is None:
                print(f'PHOTOMETRY,ERROR,"No blazar asigned to FITS \'{ap_calib}\'"')
                continue
            # Querying blazar data info file.
            # If aperture is given, the script takes it.
            try:
                aperas = blazar_data[blazar_data["IAU_name_mc"] == blzr_name]["aper_mc"].values[0]
                aper = aperas / i_fits.header['PIXSCALE']
            except:
                aper=float("nan")
            print(f'aper = {aper}')
            # If there is no aperture asigned for this object, 2 times mean FWHM for night object is taken
            if math.isnan(aper):
                try:
                    fwhm = df_fwhm_mean[df_fwhm_mean["BLZRNAME"] == blzr_name]["MEAN_FWHM"].values[0]
                except:
                    print("Couldn't calculate FWHM, setting it to 0 and taking 12 px as aperture")
                    fwhm = 0
                print(f'FWHM = {fwhm}')
                # Never (for ordinary apertures) a value lower than 12 pixels has been taken. So if
                # FWHM is so low than (2 * FWHM) < 12, then 12 is selected.
                aper = max([2 * fwhm, 12])
            
            base_dir, name = os.path.split(ap_calib)
            if 'MAGZPT' not in i_fits.header: # not photometrically calibrated
                print(f'PHOTOMETRY,ERROR,"No MAGZPT in \'{ap_calib}\'"')
                continue
            cmd = 'python iop3_photometry.py --aper_pix={} {} {} {}'
            cmd = cmd.format(round(aper, 1), args.config_dir, base_dir, ap_calib)
            print(cmd)

            try:
                res = subprocess.run(cmd, stdout=subprocess.PIPE, \
                                         stderr=subprocess.PIPE, shell=True, check=True)
                if res.returncode:
                    message = 'PHOTOMETRY,ERROR,"Failed processing star: DATE-OBS={}, OBJECT={}, EXPTIME={}"'
                    print(message.format(i_fits.header['DATE-OBS'], i_fits.header['OBJECT'], i_fits.header['EXPTIME']))
            except:
                message = 'PHOTOMETRY,ERROR,"Failed processing star: DATE-OBS={}, OBJECT={}, EXPTIME={}"'
                print(message.format(i_fits.header['DATE-OBS'], i_fits.header['OBJECT'], i_fits.header['EXPTIME']))
                continue

    #  4th STEP: Computing polarimetric parameters
    if not args.skip_polarimetry:
        if pol_sources:
            print("COMPUTING POLARIMETRY. Please wait...")
            com_polarimetry = f"python iop3_polarimetry.py {proc_dirs['calibration_dir']} {proc_dirs['polarization_dir']}"
            print('+' * 100)
            print(com_polarimetry)
            print('+' * 100)
            subprocess.Popen(com_polarimetry, shell=True).wait()

    # 5th STEP: Inserting results in database
    telescope=input_dir.split('/')[-2]
    if not args.skip_db_registration:
        data_dir = input_dir.split('data')[0] + 'data'
        com_insertdb = f"python iop3_add_db_info.py {data_dir} {dt_run} {telescope}"
        print(com_insertdb)
        with open(os.path.join(proc_dirs['polarization_dir'], 'db.log'), 'w') as log_file:
            subprocess.Popen(com_insertdb, shell=True, stdout=log_file).wait()

    

    # 6th: Generate plot for each observed blazar

    date=df_blazars['DATE-OBS'].values[0].split('T')[0]
    for name in blazar_names:
        com_plot_query = f'python generate_and_save_plots_from_iop3db.py --out_dir={proc_dirs["polarization_dir"]} "{name}" --full_range=True'
        com_plot_query_tonight = f'python generate_and_save_plots_from_iop3db.py --out_dir={proc_dirs["polarization_dir"]} "{name}" --date_start={date} --date_end={date}'
        print(com_plot_query)    
        subprocess.Popen(com_plot_query, shell=True).wait()
        print(com_plot_query_tonight)
        subprocess.Popen(com_plot_query_tonight, shell=True).wait()

    return 0
if __name__ == '__main__':
    if not main():
        print("IOP3 process successfully completed!!")
