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
from venv import create

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
    """"""
    # Getting header informacion
    i_fits = mcFits(path_fits)
    input_head = i_fits.header
    
    # Central FITS coordinates
    icoords = "{} {}".format(input_head['RA'], input_head['DEC'])
    input_coords = SkyCoord(icoords, frame=FK5, unit=(u.deg, u.deg), \
    obstime="J2000")

    # Blazars subset...
    df_blazars = blazar_data[blazar_data['IAU_name_mc'].notna()]
    c  = []
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
        index_dup = data.duplicated(['INSPOROT'], keep='last') # false for last duplicated (angle, type) item
        sub_s.append(data[~index_dup])  # append last set of duplicated items to list
        data = data[index_dup] # delete previous last repeated set of observations
    
    return sub_s

def object_groups(data_object):
    """"It returns a list of data subsets.
    
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
        dt_obj = datetime.fromisoformat(row['DATE-OBS'])
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
            stderr=subprocess.PIPE, shell=True, check=True)
            if res.returncode:
                print(f'ASTROCALIBRATION,ERROR,"Failed for calibrating {reduced} file."')
        
        # Checking for succesful calibration
        calibrated = glob.glob(os.path.join(cal_dir, '*final.fits'))
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
                stderr=subprocess.PIPE, shell=True, check=True)
            if res.returncode:
                print(f'PHOTOCALIBRATION,ERROR,"Failed for calibrating {calibrated[0]} file."')
        else:
            non_calibrated_group_commands.append(row['PATH'])
            non_calibrated_group_datetimes.append(row['DATE-OBS'])

    # # After al process, check for non-successful calibrated FITS in group
    # if calibration['CAL_IMWCS']: # if, at least one calibration was successful
    #     for ncfits, nc_dt in zip(non_calibrated_group_commands, non_calibrated_group_datetimes):
    #         dt_obj = datetime.fromisoformat(nc_dt)
    #         im_time = dt_obj.strftime('%Y%m%d-%H%M%S')
    #         cal_dir = os.path.join(calibration_dir, im_time)
    #         # rebuilding calibration command
    #         # calibrated model group FITS
    #         model_cal = calibration['CAL_IMWCS'][0] # TODO: choose lightly rotated better
    #         try:
    #             if overwrite:
    #                 new_calibration_com = f"python iop3_astrometric_calibration.py --overwrite --tol_pixs={tol_pixs} --fits_astrocal={model_cal} {config_dir} {cal_dir} {ncfits}"
    #             else:
    #                 new_calibration_com = f"python iop3_astrometric_calibration.py --tol_pixs={tol_pixs} --fits_astrocal={model_cal} {config_dir} {cal_dir} {ncfits}"
    #         except:
    #             print(f"calibration['CAL_IMWCS'] = {calibration['CAL_IMWCS']}")
    #             print(f'ncfits = {ncfits}')
    #             # print(f'new_calibration_com = {new_calibration_com}')
    #             raise
    #         print('+' * 100)
    #         print(new_calibration_com)
    #         print('+' * 100)
    #         with open(os.path.join(cal_dir, im_time + '.log'), 'w') as log_file:
    #             res = subprocess.run(new_calibration_com, stdout=log_file, \
    #                 stderr=subprocess.PIPE, shell=True, check=True)
    #             if res.returncode:
    #                 print(f'ASTROCALIBRATION,ERROR,"Failed for calibrating {calibrated[0]} file."')
        
    #         # Checking for successful calibration result
    #         calibrated = glob.glob(os.path.join(cal_dir, '*final.fits'))
    #         if calibrated:
    #             calibration['CAL_NO-IMWCS'].append(calibrated[0])
    #             # Photometric calibration
    #             if overwrite:
    #                 com_photocal = f"python iop3_photometric_calibration.py --overwrite {config_dir} {cal_dir} {calibrated[0]}"
    #             else:
    #                 com_photocal = f"python iop3_photometric_calibration.py {config_dir} {cal_dir} {calibrated[0]}"
    #             print('+' * 100)
    #             print(com_photocal)
    #             print('+' * 100)
    #             res = subprocess.run(com_photocal, stdout=subprocess.PIPE, \
    #                 stderr=subprocess.PIPE, shell=True, check=True)
    #             if res.returncode:
    #                 print(f'PHOTOCALIBRATION,ERROR,"Failed for calibrating {calibrated[0]} file."')
    #         else:
    #             calibration['NO-CAL'].append(ncfits)

    return calibration

# def object_calibration(data, calibration_dir, config_dir, tol_pixs=10, crotation=3):
#     """
#     """
#     subsets = object_groups(data)
    
#     res = []
#     for sset in subsets:
#         res.append(group_calibration(sset, calibration_dir, config_dir, \
#             tol_pixs=tol_pixs, crotation=crotation))
    
#     return res

def get_best_rotangle(path, config_dir, cal_dir, tol_pixs=5):
    """Try serveral rotation angles and gets the one who maximize the number of matches.

    Args:
        path (str): Input FITS path.
        config_dir (str): Configuration directory path.
        cal_dir (str): Calibration output directory.
        tol_pixs (int, optional): Pixel tolerance between matches. Defaults to 5.

    Returns:
        float: Best rotation angle for astrometric calibration.
    """
    crotation = 0
    wcs_best_match = 0
    wcsmatch = None
    com_cal = "python iop3_astrometric_calibration.py --crotation={} --overwrite --tol_pixs={} {} {} {}"
    # getting best rotation angle for astrometrical calibration
    for crot in np.arange(0, 5.5, 0.5):
        com_calibration = com_cal.format(crot, tol_pixs, config_dir, \
            cal_dir, path)
        print(com_calibration)
        res = execute_command(com_calibration)
        if res.returncode:
            print(res)
            return res.returncode
        # read astrocalibration file
        astro_csv = glob.glob(os.path.join(cal_dir, '*astrocal_process_info.csv'))
        
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
    """[summary]

    Args:
        path_subdir (str): Run of raw FITS images to process.
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

def contains_valid_dateobs(fits_path, keyword='DATE-OBS'):
    """[summary]

    Args:
        fits_path (str): Inputs FITS path.
    
    Returns:
        bool:
            True, means fits_path has valid observation date keyword.
            False, in other case.
    """
    return (keyword in mcFits(fits_path).header)


def has_near_calibrators(fits_path, blazars_data, max_deg_dist=0.5):
    """[summary]

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
    """[summary]

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
    description='''Main program that reads, classify, reduces and calibrate 
    FITS located at input directory. ''',
    epilog='''''')

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
       default=10,
       help="Tolerance for distance in pixels for matching between objects in external catalog and FITS detections. [default: %(default)s].")
    # parser.add_argument("--crotation",
    #    action="store",
    #    dest="crotation",
    #    type=float,
    #    default=3,
    #    help="Rotation angle in degrees to align FITS image North (up). [default: %(default)s].")
    parser.add_argument('--ignore_farcalib', dest='ignore_farcalib', action='store_true', \
        help='If False, pipeline stops execution if some science FITS has no close enough IOP3 calibrator.')
    parser.add_argument("--overwrite", dest='overwrite', action='store_true', \
        help='Pipeline overwrite previous calibrations.')
    parser.add_argument("--skip_reduction", dest='skip_reduction', action='store_true', \
        help='Skip reduction process in pipeline steps.')
    parser.add_argument("--skip_calibration", dest='skip_calibration', action='store_true', \
        help='Skip astrometric calibration process in pipeline steps.')
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
    pattern = re.findall('(/data/raw/\w+/\d{6})', input_dir)
    if not len(pattern):  # pattern not found
        print('ERROR: Input directory pattern "[root_data]/data/raw/*/yymmdd" not verified.')
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

    # Rejected because of non-valid observation DATE keyword found in FITS header
    non_valid_dateobs = [p for p in input_paths if not contains_valid_dateobs(p)]
    # Rejected FITS because image center are far from IOP3 calibrators. FLATS and BIAS are excluded.
    far_calibrators = [p for p in input_paths if (not has_near_calibrators(p, blazar_data))]
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
            print(message.format(path, mcFits(path).header['OBJECT'], \
                mcFits(path).header['RA'], mcFits(path).header['DEC']))
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
        stderr=subprocess.PIPE, shell=True, check=True)
        if res_reduction.returncode:
            message = f'REDUCTION,ERROR,"Could not reduce {dt_run} night run."'
            print(message)
            print(f'STDOUT = {res_reduction.stdout.decode("UTF-8")}')
            print(f'STDERR = {res_reduction.stderr.decode("UTF-8")}')
            return 1

    # return -1

    # ****************** 2nd STEP: Input reduced images calibration  ************* #

    # The idea is to do calibration grouping images close in time and
    # referred to same object
    
    pol_sources = False # If there is no images from starts or blazars, no calibration routines are needed
    df_blazars = create_dataframe(blazar_paths, keywords=['DATE-OBS', 'OBJECT', 'EXPTIME', 'INSPOROT'])
    if len(df_blazars.index) > 0:
        pol_sources = True
    df_stars =  create_dataframe(star_paths, keywords=['DATE-OBS', 'OBJECT', 'EXPTIME', 'INSPOROT'])
    if len(df_stars.index) > 0:
        pol_sources = True
            
    if not args.skip_calibration:
        # Creating Blazars DataFrame
        df_blazars = create_dataframe(blazar_paths, keywords=['DATE-OBS', 'OBJECT', 'EXPTIME', 'INSPOROT'])
        if len(df_blazars.index) > 0:
            df_blazars['CLOSE_IOP3'] = [closest_blazar(blazar_data, bp)[0]['IAU_name_mc'] for bp in df_blazars['PATH'].values]
            df_blazars['CLOSE_IOP3_RA'] = [closest_blazar(blazar_data, bp)[0]['ra2000_mc_deg'] for bp in df_blazars['PATH'].values]
            df_blazars['CLOSE_IOP3_DEC'] = [closest_blazar(blazar_data, bp)[0]['dec2000_mc_deg'] for bp in df_blazars['PATH'].values]
            
            
            cal_dir = os.path.join(proc_dirs['calibration_dir'], 'rotation_angle')
            crot = None
            wcsmatch_best = None
            res_path = os.path.join(cal_dir, 'result.txt')
    
            if not os.path.exists(res_path):
                # getting LONG EXPTIME
                df_blazars = df_blazars.sort_values('EXPTIME', ascending=False)
                path_fits = df_blazars['PATH'].iloc[0].replace('raw', 'reduction')
                crot, wcsmatch_best  = get_best_rotangle(path_fits, args.config_dir, cal_dir, tol_pixs=2)
                # saving result in file
                with open(res_path, 'w') as fout:
                    fout.write(f'{crot},{wcsmatch_best}')
            else:        
                data_res = open(res_path).read().split(',') 
                crot = float(data_res[0]) 
                wcsmatch_best = int(data_res[1])   
                
            if wcsmatch_best < 10:
                print("ASTROCALIBRATION,ERROR,'Not enought matches for confident rotation angle computation. Please, look at \'{cal_dir}\' for more info.'")
                return -99
            
            print(f' ---------- Best rotation angle for astrometric calibration = {crot} ({wcsmatch_best} matches) -------')
            
            pol_sources = True
            # sorting by DATE-OBS
            df_blazars = df_blazars.sort_values('DATE-OBS', ascending=True)

            # print(f'Blazar objects = {df_blazars}')

            # Grouping by closest IOP3 blazar
            obj_blazar = sorted(df_blazars['CLOSE_IOP3'].unique().tolist())

            # processing blazars...
            for obj in obj_blazar:
                df_object = df_blazars[df_blazars['CLOSE_IOP3'] == obj]
                print('********* Processing group:')
                print(df_object)
                
                # calibration setting 
                subsets = object_groups(df_object)
                res_group_calibration = []
                for sset in subsets:
                    res_group_calibration.append(group_calibration(sset, proc_dirs['calibration_dir'], \
                        config_dir, overwrite=args.overwrite, tol_pixs=args.tol_pixs, crotation=crot))
                print("----------- Calibration results: ")
                print(res_group_calibration)
                # break


        # Creating Stars DataFrame
        df_stars =  create_dataframe(star_paths, keywords=['DATE-OBS', 'OBJECT', 'EXPTIME', 'INSPOROT'])
        if len(df_stars.index) > 0:
            pol_sources = True
            df_stars['CLOSE_IOP3'] = [closest_blazar(blazar_data, bp)[0]['IAU_name_mc'] for bp in df_stars['PATH'].values]
            # sorting by DATE-OBS
            df_stars = df_stars.sort_values('DATE-OBS', ascending=True)
        
            # processing stars...
            for index, row in df_stars.iterrows():
                dt_obj = datetime.fromisoformat(row['DATE-OBS'])
                im_time = dt_obj.strftime('%Y%m%d-%H%M%S')
                cal_dir = os.path.join(proc_dirs['calibration_dir'], im_time)
                
                print('Calibrating star: ')
                print(f'{row}')
                cmd = ''
                if args.overwrite:
                    cmd = 'python iop3_astrometric_calibration.py --is_star --tol_pixs={} --overwrite {} {} {}'
                else:
                    cmd = 'python iop3_astrometric_calibration.py --is_star --tol_pixs={}  {} {} {}'
                cmd = cmd.format(args.tol_pixs, config_dir, cal_dir, row['PATH'].replace('raw', 'reduction'))
                print('+' * 100)
                print(cmd)
                print('+' * 100)
                res = subprocess.run(cmd, stdout=subprocess.PIPE, \
                    stderr=subprocess.PIPE, shell=True, check=True)
                if res.returncode:
                    message = 'ASTROCALIBRATION,ERROR,"Failed processing star: DATE-OBS={}, OBJECT={}, EXPTIME={}"'
                    print(message.format(row['DATE-OBS'], row['OBJECT'], row['EXPTIME']))
                
                # Photometric calibration
                calibrated = os.path.join(cal_dir, os.path.split(row['PATH'])[1].replace('.fits', '_final.fits'))
                cmd_photocal = ""
                if args.overwrite:
                    cmd_photocal = "python iop3_photometric_calibration.py --overwrite {} {} {}"
                else:
                    cmd_photocal = "python iop3_photometric_calibration.py {} {} {}"
                cmd_photocal = cmd_photocal.format(config_dir, cal_dir, calibrated)
                print('+' * 100)
                print(cmd_photocal)
                print('+' * 100)
                res = subprocess.run(cmd_photocal, stdout=subprocess.PIPE, \
                    stderr=subprocess.PIPE, shell=True, check=True)
                if res.returncode:
                    message = 'PHOTOCALIBRATION,ERROR,"Failed processing star: DATE-OBS={}, OBJECT={}, EXPTIME={}"'
                    print(message.format(row['DATE-OBS'], row['OBJECT'], row['EXPTIME']))

    #  3rd STEP: Computing polarimetric parameters
    if not args.skip_polarimetry:
        if pol_sources:
            print("COMPUTING POLARIMETRY. Please wait...")
            com_polarimetry = f"python iop3_polarimetry.py {proc_dirs['calibration_dir']} {proc_dirs['polarization_dir']}"
            print('+' * 100)
            print(com_polarimetry)
            print('+' * 100)
            subprocess.Popen(com_polarimetry, shell=True).wait()

    # 4th STEP: Inserting results in database
    if not args.skip_db_registration:
        data_dir = input_dir.split('data')[0] + 'data'
        com_insertdb = f"python iop3_add_db_info.py {data_dir} {dt_run}"
        print(com_insertdb)
        with open(os.path.join(proc_dirs['polarization_dir'], 'db.log'), 'w') as log_file:
            subprocess.Popen(com_insertdb, shell=True, stdout=log_file).wait()

    return 0

if __name__ == '__main__':
    if not main():
        print("Done!!")
