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

from mcFits import *


# ----------------------- FUNCTIONS SECTION ------------------------------

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
        index_dup = data.duplicated(['angle'], keep='last') # false for last duplicated (angle, type) item
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
    exptimes = sorted(data_object['exptime'].unique().tolist())
    print(f"EXPTIMES = {exptimes}")
    
    # If several EXPTIMES where taken, then several groups must be processed
    groups = []
    for et in exptimes:
        groups.append(data_object[data_object['exptime'] == et])
        
    for g in groups:
        data_sets = data_sets + subsets(g)
    
    return data_sets        
        

def object_calibration(data, calibration_dir, config_dir):
    
    subsets = object_groups(data)
    
    res = []
    for sset in subsets:
        res.append(group_calibration(sset, calibration_dir, config_dir))
    
    return res


# def object_calibration(data, calibration_dir, config_dir, is_star=False):
#     print(f'Number of FITS for object {data["object"].iloc[0]} = {len(data.index)}')    
#     df_object_valid_first = None
#     df_object_valid_last = None
#     if len(data.index) >= 8: 
#         # selecting first angles ocurrences
#         df_object_valid_first = data.drop_duplicates(subset=['angle'], keep='first')
#         df_object_valid_last = data.drop_duplicates(subset=['angle'], keep='last')
#     else:
#         df_object_valid_last = data.drop_duplicates(subset=['angle'], keep='last')
    
#     calibration_first = group_calibration(df_object_valid_first, calibration_dir, config_dir)
#     calibration_last = group_calibration(df_object_valid_last, calibration_dir, config_dir)

#     return calibration_first, calibration_last


def group_calibration(data, calibration_dir, config_dir):
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
        dt_obj = datetime.fromisoformat(row['dateobs'])
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
        com_calibration = f"python iop3_astrometric_calibration.py {config_dir} {cal_dir} {row['path']}"
        print('+' * 100)
        print(com_calibration)
        print('+' * 100)
        with open(os.path.join(cal_dir, im_time + '.log'), 'w') as log_file:
            subprocess.Popen(com_calibration, shell=True, stdout=log_file).wait()
        
        # Checking for succesful calibration
        calibrated = glob.glob(os.path.join(cal_dir, '*final.fits'))
        if calibrated:
            calibration['CAL_IMWCS'].append(calibrated[0])
            # Photometric calibration
            com_photocal = f"python iop3_photometric_calibration.py {config_dir} {cal_dir} {calibrated[0]}"
            subprocess.Popen(com_photocal, shell=True).wait()
        else:
            non_calibrated_group_commands.append(row['path'])
            non_calibrated_group_datetimes.append(row['dateobs'])

    # After al process, check for non-sucessful calibrated FITS in group
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
                subprocess.Popen(new_calibration_com, shell=True, stdout=log_file).wait()
        
            # Checking for succesful calibration result
            calibrated = glob.glob(os.path.join(cal_dir, '*final.fits'))
            if calibrated:
                calibration['CAL_NO-IMWCS'].append(calibrated[0])
                # Photometric calibration
                com_photocal = f"python iop3_photometric_calibration.py {config_dir} {cal_dir} {calibrated[0]}"
                print('+' * 100)
                print(com_photocal)
                print('+' * 100)
                subprocess.Popen(com_photocal, shell=True).wait()
            else:
                calibration['NO-CAL'].append(ncfits)

    return calibration


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
    parser.add_argument("--thresh_exptime",
       action="store",
       dest="thresh_exptime",
       type=float,
       default=2.0,
       help="For FITS that have EXPTIME lower or equal than 'thresh_exptime', star calibration process is applied. [default: %(default)s].")
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
    # [root_data]/data/raw/[project]/yymmdd
    # OUTPUT reduction
    # [root_data]/data/reduction/[project]/yymmdd
    # OUTPUT calibration
    # [root_data]/data/calibration/[project]/yymmdd
    # OUTPUT polarization
    # [root_data]/data/final/[project]/yymmdd
    #
    # where [project] could be MAPCAT

    input_dir = os.path.abspath(args.input_dir) # absolute path
    config_dir = os.path.abspath(args.config_dir) # absolute path

    # Does input verify pattern given above?
    pattern = re.findall('(/data/raw/\w+/\d{6})', input_dir)
    if not len(pattern):  # pattern not found
        print('ERROR: Input directory pattern "[root_data]/data/raw/*/yymmdd" not verified.')
        return 2
    if not os.path.isdir(input_dir):
        str_err = 'ERROR: Input directory "{}" not available'
        print(str_err.format(input_dir))
        return 3

    # Getting run date (input directory must have pattern like *YYMMDD)
    dt_run = re.findall('(\d{6})', input_dir)[0]
    # date_run = f'20{dt_run[:2]}-{dt_run[2:4]}-{dt_run[-2:]}'

    path_line = input_dir.split('/')
    reduction_dir = input_dir.replace('raw', 'reduction')
    calibration_dir = input_dir.replace('raw', 'calibration')
    polarization_dir = input_dir.replace('raw', 'final')
    print(f"reduction_dir = {reduction_dir}")
    print(f"calibration_dir = {calibration_dir}")
    print(f"final_dir (polarization) = {polarization_dir}")
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
    # Command execution
    # subprocess.Popen(com_reduction, shell=True).wait()

    # return -1

    # 2nd STEP: Input reduced images calibration

    # The idea is to do calibration grouping images close in time and
    # referred to same object
    reduced_fits = glob.glob(os.path.join(reduction_dir, '*.fits'))
    
    d_red = defaultdict(list)
    for rf in reduced_fits:
        r_fits = mcFits(rf)
        obj = r_fits.header['OBJECT']
        if obj == 'Master BIAS' or obj == 'Master FLAT':
            continue 
        d_red['object'].append(obj.split(' ')[0])
        d_red['path'].append(rf)
        d_red['dateobs'].append(r_fits.header['DATE-OBS'])
        d_red['angle'].append(r_fits.header['INSPOROT'])
        d_red['exptime'].append(r_fits.header['EXPTIME'])

    # transformation to numpy array for sorting files by DATE-OBS and OBJECT header keywords
    # Target is creating groups of observations for the same OBJECT and four different 
    #     polarization angles
    df_calib = pd.DataFrame(d_red)
    
    # classify using EXPTIME
    df_blazar = df_calib[df_calib['exptime'] > args.thresh_exptime]
    df_star = df_calib[df_calib['exptime'] <= args.thresh_exptime]
    
    # sort by ascending dateobs, object, and angle
    df_blazar = df_blazar.sort_values(['dateobs', 'object', 'angle'], ascending=(True, True, True))
    
    # now, select valid observations for each group
    obj_blazar = sorted(np.unique(df_blazar['object'].values).tolist())
    
    # print(f'Blazar objects = {obj_blazar}')
    
    for obj in obj_blazar:
        df_object = df_calib[df_calib['object'] == obj]
        print('********* Processing group:')
        print(df_object)
        
        # calibration setting 
        subsets = object_groups(df_object)
        res_group_calibration = []
        for sset in subsets:
            res_group_calibration.append(group_calibration(sset, calibration_dir, config_dir))
        print("----------- Calibration results: ")
        print(res_group_calibration)
        # break

    # processing stars...
    for index, row in df_star.iterrows():
        dt_obj = datetime.fromisoformat(row['dateobs'])
        im_time = dt_obj.strftime('%Y%m%d-%H%M%S')
        cal_dir = os.path.join(calibration_dir, im_time)
        
        print(f'Calibrating star: {row}')
        cmd = 'python iop3_astrometric_calibration.py --is_star=True {} {} {}'
        cmd = cmd.format(config_dir, cal_dir, row['path'])
        print('+' * 100)
        print(cmd)
        print('+' * 100)
        res = subprocess.run(cmd, stdout=subprocess.PIPE, \
            stderr=subprocess.PIPE, shell=True, check=True)
        if res.returncode:
            message = 'ASTROCALIBRATION,ERROR,"Failed processing star: DATE-OBS={}, OBJECT={}, EXPTIME={}"'
            print(message.format(row['dateobs'], row['object'], row['exptime']))
        # Photometric calibration
        cmd_photocal = "python iop3_photometric_calibration.py {} {} {}"
        calibrated = os.path.join(cal_dir, os.path.split(row['path'])[1].replace('.fits', '_final.fits'))
        cmd_photocal = cmd_photocal.format(config_dir, cal_dir, calibrated)
        print('+' * 100)
        print(cmd_photocal)
        print('+' * 100)
        res = subprocess.run(cmd_photocal, stdout=subprocess.PIPE, \
            stderr=subprocess.PIPE, shell=True, check=True)
        if res.returncode:
            message = 'PHOTOCALIBRATION,ERROR,"Failed processing star: DATE-OBS={}, OBJECT={}, EXPTIME={}"'
            print(message.format(row['dateobs'], row['object'], row['exptime']))
    
    # return -1

    #  3rd STEP: Computing polarimetric parameters
    print("COMPUTING POLARIMETRY. Please wait...")
    com_polarimetry = f"python iop3_polarimetry.py {calibration_dir} {polarization_dir}"
    print(com_polarimetry)
    subprocess.Popen(com_polarimetry, shell=True).wait()

    return -1

    # 4th STEP: Inserting results in database
    data_dir = input_dir.split('data')[0] + 'data'
    com_insertdb = f"python iop3_add_db_info.py {data_dir} {dt_run}"
    print(com_insertdb)
    with open(os.path.join(polarization_dir, 'db.log'), 'w') as log_file:
        subprocess.Popen(com_insertdb, shell=True, stdout=log_file).wait()

    return 0

if __name__ == '__main__':
    if not main():
        print("Done!!")
