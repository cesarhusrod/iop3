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
    #Getting telescope type
    tel_type=re.findall('(\w+/)', 
                        path_fits)[len(re.findall('(\w+/)', 
                                                  path_fits))-2][:-1]
    # Getting header informacion
    i_fits = mcFits(path_fits)
    input_head = i_fits.header
    # Central FITS coordinates
    if tel_type=='MAPCAT':
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

def group_calibration(data, calibration_dir, config_dir, overwrite=False):
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
        calibrated = glob.glob(os.path.join(cal_dir, '*final.fit*'))
        com_calibration = f"python iop3_astrometric_calibration.py --overwrite={overwrite} {config_dir} {cal_dir} {reduced}"
        print('+' * 100)
        print(com_calibration)
        print('+' * 100)
        with open(os.path.join(cal_dir, im_time + '.log'), 'w') as log_file:
            res = subprocess.run(com_calibration, stdout=log_file, \
                                     stderr=subprocess.PIPE, shell=True, check=True)
            if res.returncode:
                print(f'ASTROCALIBRATION,ERROR,"Failed for calibrating {reduced} file."')

        # Checking for succesful calibration
        calibrated = glob.glob(os.path.join(cal_dir, '*final.fit*'))
        if calibrated:
            calibration['CAL_IMWCS'].append(calibrated[0])
            # Photometric calibration
            com_photocal = f"python iop3_photometric_calibration.py --overwrite={overwrite} {config_dir} {cal_dir} {calibrated[0]}"
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
        #print("CUIDAO CON ESTE BREAK QUITALO")
        #break
    # After al process, check for non-successful calibrated FITS in group
    if calibration['CAL_IMWCS']: # if, at least one calibration was successful
        for ncfits, nc_dt in zip(non_calibrated_group_commands, non_calibrated_group_datetimes):
            dt_obj = datetime.fromisoformat(nc_dt)
            im_time = dt_obj.strftime('%Y%m%d-%H%M%S')
            cal_dir = os.path.join(calibration_dir, im_time)
            # rebuilding calibration command
            # calibrated model group FITS
            model_cal = calibration['CAL_IMWCS'][0] # TODO: choose lightly rotated better
            try:
                new_calibration_com = f"python iop3_astrometric_calibration.py --fits_astrocal={model_cal} {config_dir} {cal_dir} {ncfits}"
            except:
                print(f"calibration['CAL_IMWCS'] = {calibration['CAL_IMWCS']}")
                print(f'ncfits = {ncfits}')
                # print(f'new_calibration_com = {new_calibration_com}')
                raise
            print('+' * 100)
            print(new_calibration_com)
            print('+' * 100)
            with open(os.path.join(cal_dir, im_time + '.log'), 'w') as log_file:
                res = subprocess.run(new_calibration_com, stdout=log_file, \
                    stderr=subprocess.PIPE, shell=True, check=True)
                if res.returncode:
                    print(f'ASTROCALIBRATION,ERROR,"Failed for calibrating {calibrated[0]} file."')
        
            # Checking for successful calibration result
            calibrated = glob.glob(os.path.join(cal_dir, '*final.fit*'))
            if calibrated:
                calibration['CAL_NO-IMWCS'].append(calibrated[0])
                # Photometric calibration
                com_photocal = f"python iop3_photometric_calibration.py {config_dir} {cal_dir} {calibrated[0]}"
                print('+' * 100)
                print(com_photocal)
                print('+' * 100)
                res = subprocess.run(com_photocal, stdout=subprocess.PIPE, \
                    stderr=subprocess.PIPE, shell=True, check=True)
                if res.returncode:
                    print(f'PHOTOCALIBRATION,ERROR,"Failed for calibrating {calibrated[0]} file."')
            else:
                calibration['NO-CAL'].append(ncfits)

    return calibration

def object_calibration(data, calibration_dir, config_dir):
    """
    """
    subsets = object_groups(data)
    
    res = []
    for sset in subsets:
        res.append(group_calibration(sset, calibration_dir, config_dir))
    
    return res

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
        0, means fits_path is closer than 'max_deg_dist' degrees to an IOP3 calibrator.
        1, in other case.
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
    """[summary]

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
       help="Discarded FITS border pixels. Values between 0 and 150 are accepted. Else, it is set to 15 [default: %(default)s].")
    parser.add_argument("--overwrite",
       action="store",
       dest="overwrite",
       type=str,
       default=False,
       help="If True, previous calibration is ignored and done again. [default: %(default)s].")
    parser.add_argument('-v', '--verbose', action='count', default=0,
        help="Show running and progress information [default: %(default)s].")
    args = parser.parse_args()
    
    #Transform overwrite parameter properly to boolean
    if args.overwrite in ('true', 'True', '1', 'y', 'yes', 'Yes'):
        args.overwrite=True
    elif args.overwrite in ('false', 'False', '0', 'n', 'no', 'No'):
        args.overwrite=False
    else:
        print("Wrong or no value for --overwrite parameter. Setting it to default (False)")
        args.overwrite=False

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
    # Rejected because of non-valid observation keyword found in FITS header
    non_valid_dateobs = [p for p in input_paths if not contains_valid_dateobs(p)]
    # Rejected FITS because no DATE-OBS header keyword present
    far_calibrators = [p for p in input_paths if not has_near_calibrators(p, blazar_data)]
    # FITS that contain Blazars
    blazar_paths = [p for p in input_paths if is_blazar(p, blazar_data)]
    # FITS that contain stars
    star_paths = [p for p in input_paths if is_star(p, blazar_data)]
    
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
        far_data = {}
        message = 'CHECKING,ERROR,"Close calibrator was not found for \'{}\' (OBJECT = \'{}\')."'
        for path in far_calibrators:
            print(message.format(path, mcFits(path).header['OBJECT']))
        #return 4

    # **************  1st STEP: Input raw image reduction  ******************** #
    com_reduction = "python iop3_reduction.py --border_image={} {} {} {}"
    com_reduction = com_reduction.format(border_image, config_dir, \
        proc_dirs['reduction_dir'], input_dir)
    print(com_reduction)
    print("IM SKIPPING REDUCTION UNCOMMENT THIS SECTION!!!")
    # Command execution
    #res_reduction = subprocess.run(com_reduction,stdout=subprocess.PIPE, \
    #   stderr=subprocess.PIPE, shell=True, check=True)
    #if res_reduction.returncode:
    #    message = f'REDUCTION,ERROR,"Could not reduce {dt_run} night run."'
    #    print(message)
    #    print(f'STDOUT = {res_reduction.stdout.decode("UTF-8")}')
    #    print(f'STDERR = {res_reduction.stderr.decode("UTF-8")}')
    #    return 1

    #return -1

    # ****************** 2nd STEP: Input reduced images calibration  ************* #

    # The idea is to do calibration grouping images close in time and
    # referred to same object

    # Creating Blazars DataFrame 
    if 'MAPCAT' in input_dir:
        df_blazars = create_dataframe(blazar_paths, keywords=['DATE-OBS', 'OBJECT', 'EXPTIME', 'INSPOROT'])
    else:
        df_blazars = create_dataframe(blazar_paths, keywords=['DATE-OBS', 'OBJECT', 'EXPTIME', 'FILTER'])

    if not len(blazar_paths)==0:
        df_blazars['CLOSE_IOP3'] = [closest_blazar(blazar_data, bp)[0]['IAU_name_mc'] for bp in df_blazars['PATH'].values]
    # sorting by DATE-OBS
    df_blazars = df_blazars.sort_values('DATE-OBS', ascending=True)

    # Creating Stars DataFrame
    if 'MAPCAT' in input_dir:
        df_stars =  create_dataframe(star_paths, keywords=['DATE-OBS', 'OBJECT', 'EXPTIME', 'INSPOROT'])
    else:
        df_stars =  create_dataframe(star_paths, keywords=['DATE-OBS', 'OBJECT', 'EXPTIME', 'FILTER'])

    print(star_paths)
    if not len(star_paths)==0:
        df_stars['CLOSE_IOP3'] = [closest_blazar(blazar_data, bp)[0]['IAU_name_mc'] for bp in df_stars['PATH'].values]
        # sorting by DATE-OBS
        df_stars = df_stars.sort_values('DATE-OBS', ascending=True)

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
                config_dir, overwrite=args.overwrite))
            #print("BELLA WHERE HAVE YOU BEEN LOCA QUITA ESTOS BREAKS")
            #break
        #break
        print("----------- Calibration results: ")
        print(res_group_calibration)
        

    # processing stars...
    for index, row in df_stars.iterrows():
        if 'MAPCAT' in input_dir:
            dt_obj = datetime.fromisoformat(row['DATE-OBS'])
        else:
            dt_obj = datetime.fromisoformat(row['DATE-OBS'][:-3])
        im_time = dt_obj.strftime('%Y%m%d-%H%M%S')
        cal_dir = os.path.join(proc_dirs['calibration_dir'], im_time)
        
        print('Calibrating star: ')
        print(f'{row}')
        cmd = 'python iop3_astrometric_calibration.py --is_star=True --overwrite={} {} {} {}'
        cmd = cmd.format(args.overwrite, config_dir, cal_dir, row['PATH'].replace('raw', 'reduction'))
        print('+' * 100)
        print(cmd)
        print('+' * 100)
        res = subprocess.run(cmd, stdout=subprocess.PIPE, \
            stderr=subprocess.PIPE, shell=True, check=True)
        if res.returncode:
            message = 'ASTROCALIBRATION,ERROR,"Failed processing star: DATE-OBS={}, OBJECT={}, EXPTIME={}"'
            print(message.format(row['DATE-OBS'], row['OBJECT'], row['EXPTIME']))
        # Photometric calibration
        cmd_photocal = "python iop3_photometric_calibration.py --overwrite={} {} {} {}"
        if 'fits' in os.path.split(row['PATH'])[1]:
            calibrated = os.path.join(cal_dir, os.path.split(row['PATH'])[1].replace('.fits', '_final.fits'))
        else:
            calibrated = os.path.join(cal_dir, os.path.split(row['PATH'])[1].replace('.fit', '_final.fit'))
        cmd_photocal = cmd_photocal.format(args.overwrite, config_dir, cal_dir, calibrated)
        print('+' * 100)
        print(cmd_photocal)
        print('+' * 100)
        res = subprocess.run(cmd_photocal, stdout=subprocess.PIPE, \
            stderr=subprocess.PIPE, shell=True, check=True)
        if res.returncode:
            message = 'PHOTOCALIBRATION,ERROR,"Failed processing star: DATE-OBS={}, OBJECT={}, EXPTIME={}"'
            print(message.format(row['DATE-OBS'], row['OBJECT'], row['EXPTIME']))
    #print("OYE QUITA ESTE BREAK")
    #break
    # return -1

    #  3rd STEP: Computing polarimetric parameters
    print("COMPUTING POLARIMETRY. Please wait...")
    com_polarimetry = f"python iop3_polarimetry.py {proc_dirs['calibration_dir']} {proc_dirs['polarization_dir']}"
    print(com_polarimetry)
    subprocess.Popen(com_polarimetry, shell=True).wait()

    return -1

    # 4th STEP: Inserting results in database
    data_dir = input_dir.split('data')[0] + 'data'
    com_insertdb = f"python iop3_add_db_info.py {data_dir} {dt_run}"
    print(com_insertdb)
    with open(os.path.join(proc_dirs['polarization_dir'], 'db.log'), 'w') as log_file:
        subprocess.Popen(com_insertdb, shell=True, stdout=log_file).wait()

    return 0

if __name__ == '__main__':
    if not main():
        print("Done!!")
