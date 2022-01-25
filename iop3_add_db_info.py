#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon June 25 17:50:04 2021

___e-mail__ = cesar_husillos@tutanota.com
__author__ = 'Cesar Husillos'

VERSION:
    1.0 Initial version
"""

# ---------------------- IMPORT SECTION ----------------------
from hashlib import new
import os
import argparse
import glob
from getpass import getpass
from datetime import datetime, date

import pandas as pd

import mysql.connector

from astropy.io import fits
from mcFits import mcFits 

def res2df(db_cursor):
    return 0


def register_raw(data_dir, run_date, db_object):
    """
    Register new RAW fits in data_dir from nightly run in database 'image_raw' table using
    connection given by db_object.

    Args:
        data_dir (string): base directory. FITS are stored in data_dir/raw directory.
        run_date (string): format given by pattern YYMMDD where YY is year, MM is month and DD y day.
        db_object (mysql.connector.db object): it manages database connection.

    Returns:
        int:
            if int >= 0, it represents number of new raw FITS registered in database.
    
    Exception:
        IOError: if raw FITS catalog file was not found.
    """
    new_registrations = 0
    raw_dir = os.path.join(data_dir, f'raw/MAPCAT/{run_date}/')
    try:
        catalog = os.path.join(raw_dir, f"{run_date}.cat")
        lines = [l for l in open(catalog).read().split('\n') if len(l) > 0]
    except IOError:
        print(f"ERROR: catalog '{catalog}' not available.")
        raise

    # Create cursor for database operation
    db_cursor = db_object.cursor()

    # Checking each raw FITS 
    for l in lines[2:]:
        raw_name = l.split('\t')[0]
        
        sql_search = f"SELECT id FROM image_raw WHERE path='{raw_name}'"
        db_cursor.execute(sql_search)
        res_search = db_cursor.fetchall()
        if db_cursor.rowcount == 0:
            # Insert new register
            raw_path = os.path.join(raw_dir, raw_name)
            try:
                raw_fits = mcFits(raw_path)
            except:
                print(f"ERROR: processing {raw_path}")
                raise
            raw_header = raw_fits.header
            raw_stats = raw_fits.stats()
            
            params = ['date_run', 'path', 'date_obs', 'naxis1', \
                'naxis2', 'object', 'type', 'ra2000', 'dec2000', \
                'exptime', 'equinox', 'mjd_obs', 'pixscale', \
                'filtname', 'telescope', 'instrument', 'pol_angle', \
                'min', 'max', 'mean', 'std', 'median']

            str_params = ['date_run', 'path', 'date_obs', 'object', \
                'type', 'filtname', 'telescope', 'instrument']
            
            fits_keywords = ['NAXIS1', 'NAXIS2', 'OBJECT', 'IMAGETYP', 'RA', 'DEC', 'EXPTIME', \
                'EQUINOX', 'MJD-OBS', 'INSTRSCL', 'INSFLNAM', 'TELESCOP', \
                'INSTRUME', 'INSPOROT']
            # date_obs = datetime.strptime(raw_header['DATE'], '%Y-%d-%mT%H:%M:%S')
            # r_date = datetime.strptime(run_date, '%Y-%d-%m')
            date_obs = raw_header['DATE'].replace('T', ' ')
            r_date = run_date
            values = [r_date, raw_name, date_obs] + \
                [raw_header[k] for k in fits_keywords] + \
                [raw_stats[k] for k in ['MIN', 'MAX', 'MEAN', 'STD', 'MEDIAN']]
            
            # Inserting new register on database.image_raw
            v = ','.join([f"'{val}'" if par in str_params else f"{val}" for par, val in zip(params, values)])
            sql_insert = f"INSERT INTO image_raw ({','.join(params)}) VALUES ({v})"
            print(f"SQL command (image_raw) = '{sql_insert}'")

            try:
                db_cursor.execute(sql_insert)
            except:
                print(f"SQL ERROR: {sql_insert}")
                raise

            # Commiting insertion
            db_object.commit()

            if db_cursor.rowcount > 0:
                print(f"INFO: {db_cursor.rowcount} register/s inserted successfully.")
            new_registrations += db_cursor.rowcount

        # if new_registrations > 1:
        #     break
    
    return new_registrations



def register_masterbias(data_dir, run_date, db_object):
    """
    Register new masterBIAS fits in data_dir from nightly run in database 'image_raw' table using
    connection given by db_object.

    Args:
        data_dir (string): base directory. FITS are stored in data_dir/raw directory.
        run_date (string): format given by pattern YYMMDD where YY is year, MM is month and DD y day.
        db_object (mysql.connector.db object): it manages database connection.

    Returns:
        int:
            if int >= 0, it represents number of new masterBIAS FITS registered in database.
    
    Exception:
        IOError: if masterBIAS catalog file was not found.
    """
    new_registrations = 0
    reduction_dir = os.path.join(data_dir, f'reduction/MAPCAT/{run_date}/')
    try:
        catalog = os.path.join(reduction_dir, "masterbias_data.csv")
        data = pd.read_csv(catalog)
    except IOError:
        print(f"ERROR: catalog '{catalog}' not available.")
        raise

    # Create cursor for database operation
    db_cursor = db_object.cursor()

    # Checking each masterBIAS FITS 
    for index, row in data.iterrows():
        dire, mb_name = os.path.split(row['PATH'])
        
        sql_search = f"SELECT id FROM master_bias WHERE path='{mb_name}'"
        db_cursor.execute(sql_search)
        res_search = db_cursor.fetchall()
        if db_cursor.rowcount == 0:
            # Insert new register
            try:
                red_fits = mcFits(row['PATH'])
            except:
                print(f"ERROR: processing {row['PATH']}")
                raise

            red_header = red_fits.header
            red_stats = red_fits.stats()
            
            params = ['type', 'date_run', 'path', 'proc_date', \
                'naxis1', 'naxis2', 'soft', 'pix_border', \
                'bias_operation', \
                'min', 'max', 'mean', 'std', 'median']

            str_params = ['date_run', 'path', 'proc_date', 'soft', \
                'bias_operation', 'type']
            
            # Keywords are the same for CSV and fits files.
            keywords = ['NAXIS1', 'NAXIS2', 'SOFT', 'PXBORDER', 'BIASOP']
            
            date_proc = red_header['PROCDATE'].replace('T', ' ')
            mb_type = red_header['OBJECT']
            values = [mb_type, run_date, mb_name, date_proc] + \
                [red_header[k] for k in keywords] + \
                [red_stats[k] for k in ['MIN', 'MAX', 'MEAN', 'STD', 'MEDIAN']]
            
            # Inserting new register on database.image_raw
            v = ','.join([f"'{val}'" if par in str_params else f"{val}" for par, val in zip(params, values)])
            sql_insert = f"INSERT INTO master_bias ({','.join(params)}) VALUES ({v})"
            print(f"SQL command (image_raw) = '{sql_insert}'")

            try:
                db_cursor.execute(sql_insert)
            except:
                print(f"SQL ERROR: {sql_insert}")
                raise

            # Commiting insertion
            db_object.commit()

            if db_cursor.rowcount > 0:
                print(f"INFO: {db_cursor.rowcount} register/s inserted successfully.")
            new_registrations += db_cursor.rowcount

        # if new_registrations > 1:
        #     break
    
    return new_registrations

def register_rawbias(data_dir, run_date, db_object):
    """
    Register for each run_date masterBIAS every raw fits used for its generation.

    Args:
        data_dir (string): base directory. FITS are stored in data_dir/raw directory.
        run_date (string): format given by pattern YYMMDD where YY is year, 
            MM is month and DD y day.
        db_object (mysql.connector.db object): it manages database connection.

    Returns:
        int:
            if int >= 0, it represents number of new relationship between 
                masterBIAS and raw fits registered in database.
    
    Exception:
        IOError: 
            if masterBIAS catalog file was not found.
    """
    new_registrations = 0
    reduction_dir = os.path.join(data_dir, f'reduction/MAPCAT/{run_date}/')
    try:
        catalog = os.path.join(reduction_dir, "masterbias_data.csv")
        data = pd.read_csv(catalog)
    except IOError:
        print(f"ERROR: catalog '{catalog}' not available.")
        raise

    # Create cursor for database operation
    db_cursor = db_object.cursor()

    # Checking each masterBIAS FITS (usually only one for each night) 
    for index, row in data.iterrows():
        dire, mb_name = os.path.split(row['PATH'])
        
        sql_search = f"SELECT id FROM master_bias WHERE path='{mb_name}'"
        db_cursor.execute(sql_search)
        res_search = db_cursor.fetchall()
        if db_cursor.rowcount > 0:
            id_mb = None
            for rs in res_search:
                id_mb = rs[0]
                break
            # now, I get names from raw images used for masterBIAS generation
            try:
                bias = mcFits(row['PATH'])
            except:
                print(f"ERROR: processing {row['PATH']}")
                raise

            bias_header = bias.header

            keywords = [f'BIAS{n}' for n in range(20)]
            names = [bias_header[k] for k in keywords if k in bias_header]
            
            print(f"names = {names}")

            raw_ids = []
            for n in names:
                sql_search = f"SELECT id FROM `image_raw` WHERE `path`='{n}'"
                print(f"sql_search = {sql_search}")
                db_cursor.execute(sql_search)
                res_search = db_cursor.fetchall()
                if db_cursor.rowcount == 1:
                    raw_ids.append(res_search[0][0])
                else:
                    print("ERROR")
                    return -1
            
            values = ','.join([f"({id_mb}, {rid})" for rid in raw_ids])
            sql_insert = f"INSERT INTO `raw_bias` (`master_bias_id`, `raw_id`) VALUES {values}"
            
            # Inserting new registers on database.raw_bias
            print(f"SQL command (raw_bias) = '{sql_insert}'")

            try:
                db_cursor.execute(sql_insert)
            except:
                print(f"SQL ERROR: {sql_insert}")
                raise
            
            # Commiting insertion
            db_object.commit()

            if db_cursor.rowcount > 0:
                print(f"INFO: {db_cursor.rowcount} register/s inserted successfully.")
            new_registrations += db_cursor.rowcount

        # if new_registrations > 1:
        #     break
    
    return new_registrations


def register_masterflats(data_dir, run_date, db_object):
    """
    Register new masterFLATS fits in data_dir from nightly run in database 'image_raw' table using
    connection given by db_object.

    Args:
        data_dir (string): base directory. FITS are stored in data_dir/raw directory.
        run_date (string): format given by pattern YYMMDD where YY is year, MM is month and DD y day.
        db_object (mysql.connector.db object): it manages database connection.

    Returns:
        int:
            if int >= 0, it represents number of new masterFLAT FITS registered in database.
    
    Exception:
        IOError: if masterFLAT catalog file was not found.
    """
    new_registrations = 0
    reduction_dir = os.path.join(data_dir, f'reduction/MAPCAT/{run_date}/')
    try:
        catalog = os.path.join(reduction_dir, "masterflats_data.csv")
        data = pd.read_csv(catalog)
    except IOError:
        print(f"ERROR: catalog '{catalog}' not available.")
        raise

    # Create cursor for database operation
    db_cursor = db_object.cursor()

    # Checking each masterFLAT FITS (4 per night, one for each grism angle) 
    for index, row in data.iterrows():
        dire, mf_name = os.path.split(row['PATH'])
        
        sql_search = f"SELECT id FROM master_flat WHERE path='{mf_name}'"
        db_cursor.execute(sql_search)
        res_search = db_cursor.fetchall()
        if db_cursor.rowcount == 0:
            # Insert new register
            try:
                mf_fits = mcFits(row['PATH'])
            except:
                print(f"ERROR: processing {row['PATH']}")
                raise

            mf_header = mf_fits.header
            mf_stats = mf_fits.stats()

            # getting masterBIAS id
            r_date = datetime.strptime(run_date, '%y%m%d')
            sql_search_mb = f"SELECT `id` FROM `master_bias` WHERE date_run = '{r_date}'"
            print(f"sql_search_mb = {sql_search_mb}")
            db_cursor.execute(sql_search_mb)
            res_search_mb = db_cursor.fetchall()
            mb_id = None
            if db_cursor.rowcount == 1:
                mb_id = res_search_mb[0][0]
            else:
                print("ERROR: No masterBIAS found for this masterFLAT")
                return -2
            # return -3
            params = ['master_bias_id', 'type', 'date_run', 'path', 'proc_date', \
                'naxis1', 'naxis2', 'pol_angle', 'soft', 'pix_border', \
                'flat_operation', \
                'min', 'max', 'mean', 'std', 'median']

            str_params = ['date_run', 'path', 'proc_date', 'soft', \
                'flat_operation', 'type']
            
            # Keywords are the same for CSV and fits files.
            keywords = ['NAXIS1', 'NAXIS2', 'INSPOROT', 'SOFT', 'PXBORDER', 'FLATOP']
            
            date_proc = mf_header['PROCDATE'].replace('T', ' ')
            mf_type = mf_header['OBJECT']
            values = [mb_id, mf_type, run_date, mf_name, date_proc] + \
                [mf_header[k] for k in keywords] + \
                [mf_stats[k] for k in ['MIN', 'MAX', 'MEAN', 'STD', 'MEDIAN']]
            
            # Inserting new register on database.image_raw
            v = ','.join([f"'{val}'" if par in str_params else f"{val}" for par, val in zip(params, values)])
            sql_insert = f"INSERT INTO master_flat ({','.join(params)}) VALUES ({v})"
            print(f"SQL command (master_flat) = '{sql_insert}'")

            try:
                db_cursor.execute(sql_insert)
            except:
                print(f"SQL ERROR: {sql_insert}")
                raise

            # Commiting insertion
            db_object.commit()

            if db_cursor.rowcount > 0:
                print(f"INFO: {db_cursor.rowcount} register/s inserted successfully.")
            new_registrations += db_cursor.rowcount

        # if new_registrations > 1:
        #     break
    
    return new_registrations

def register_rawflats(data_dir, run_date, db_object):
    """
    Register for each run_date masterFLATS every raw fits used for its generation.

    Args:
        data_dir (string): base directory. FITS are stored in data_dir/raw directory.
        run_date (string): format given by pattern YYMMDD where YY is year, 
            MM is month and DD y day.
        db_object (mysql.connector.db object): it manages database connection.

    Returns:
        int:
            if int >= 0, it represents number of new relationship between 
                masterFLATs and raw fits registered in database.
    
    Exception:
        IOError: 
            if masterFLATs catalog file was not found.
    """
    new_registrations = 0
    reduction_dir = os.path.join(data_dir, f'reduction/MAPCAT/{run_date}/')
    try:
        catalog = os.path.join(reduction_dir, "masterflats_data.csv")
        data = pd.read_csv(catalog)
    except IOError:
        print(f"ERROR: catalog '{catalog}' not available.")
        raise

    # Create cursor for database operation
    db_cursor = db_object.cursor()

    # Checking each masterFLATs FITS (4 per night, one for each considered grism angle) 
    for index, row in data.iterrows():
        dire, mf_name = os.path.split(row['PATH'])
        
        sql_search = f"SELECT id FROM master_flat WHERE path='{mf_name}'"
        db_cursor.execute(sql_search)
        res_search = db_cursor.fetchall()
        if db_cursor.rowcount > 0:
            id_mf = res_search[0][0]
            # now, I get names from raw images used for masterBIAS generation
            try:
                flat = mcFits(row['PATH'])
            except:
                print(f"ERROR: processing {row['PATH']}")
                raise

            flat_header = flat.header

            keywords = [f'FLAT{n}' for n in range(20)]
            names = [flat_header[k] for k in keywords if k in flat_header]
            
            print(f"names = {names}")

            raw_ids = []
            for n in names:
                sql_search = f"SELECT id FROM `image_raw` WHERE `path`='{n}'"
                print(f"sql_search = {sql_search}")
                db_cursor.execute(sql_search)
                res_search = db_cursor.fetchall()
                if db_cursor.rowcount == 1:
                    raw_ids.append(res_search[0][0])
                else:
                    print("ERROR")
                    return -1
            
            values = ','.join([f"({id_mf}, {rid})" for rid in raw_ids])
            sql_insert = f"INSERT INTO `raw_flat` (`master_flat_id`, `raw_id`) VALUES {values}"
            
            # Inserting new registers on database.raw_bias
            print(f"SQL command (raw_bias) = '{sql_insert}'")

            try:
                db_cursor.execute(sql_insert)
            except:
                print(f"SQL ERROR: {sql_insert}")
                raise
            
            # Commiting insertion
            db_object.commit()

            if db_cursor.rowcount > 0:
                print(f"INFO: {db_cursor.rowcount} register/s inserted successfully.")
            new_registrations += db_cursor.rowcount

        # if new_registrations > 1:
        #     break
    
    return new_registrations

def register_reduced(data_dir, run_date, db_object):
    """
    Register new REDUCED fits located in data_dir from nightly run 
    in database 'image_reduced' table using connection given by db_object.

    Args:
        data_dir (string): base directory. FITS are stored in data_dir/raw directory.
        run_date (string): format given by pattern YYMMDD where YY is year, MM is month and DD y day.
        db_object (mysql.connector.db object): it manages database connection.

    Returns:
        int:
            if int >= 0, it represents number of new reduced FITS registered in database.
    
    Exception:
        IOError: if reduced FITS catalog file was not found.
    """
    new_registrations = 0
    reduced_dir = os.path.join(data_dir, f'reduction/MAPCAT/{run_date}/')
    try:
        catalog = os.path.join(reduced_dir, "output_data_red.csv")
        data = pd.read_csv(catalog)
    except IOError:
        print(f"ERROR: catalog '{catalog}' not available.")
        raise

    # Create cursor for database operation
    db_cursor = db_object.cursor()

    # Checking each raw FITS 
    for index, row in data.iterrows():
        dire, red_name = os.path.split(row['PATH'])
        
        sql_search = f"SELECT `id` FROM `image_reduced` WHERE path='{red_name}'"
        db_cursor.execute(sql_search)
        res_search = db_cursor.fetchall()
        if db_cursor.rowcount == 0:
            # Insert new register
            red_path = os.path.join(reduced_dir, red_name)
            try:
                red_fits = mcFits(red_path)
            except:
                print(f"ERROR: processing {red_path}")
                continue
            red_header = red_fits.header
            red_stats = red_fits.stats()

            # getting raw id
            sql_search_raw_id = f"SELECT `id` FROM `image_raw` WHERE `path` = '{red_name}'"
            print(f"sql_search = {sql_search_raw_id}")
            db_cursor.execute(sql_search_raw_id)
            res_search_raw_id = db_cursor.fetchall()
            if db_cursor.rowcount == 1:
                raw_id = res_search_raw_id[0][0]
            else:
                print(f"ERROR: No found raw_id for reduced fits '{red_name}'")
                return -1
            
            # getting master_bias_id
            sql_search_mb_id = f"SELECT `id` FROM `master_bias` WHERE `path` = '{red_header['BIAS']}'"
            print(f"sql_search = {sql_search_mb_id}")
            db_cursor.execute(sql_search_mb_id)
            res_search_mb_id = db_cursor.fetchall()
            if db_cursor.rowcount == 1:
                mb_id = res_search_mb_id[0][0]
            else:
                print(f"ERROR: No found master_bias_id for reduced fits '{red_name}'")
                return -1
            

            # getting master_flat_id
            sql_search_mf_id = f"SELECT `id` FROM `master_flat` WHERE `path` = '{red_header['FLAT']}'"
            print(f"sql_search = {sql_search_mf_id}")
            db_cursor.execute(sql_search_mf_id)
            res_search_mf_id = db_cursor.fetchall()
            if db_cursor.rowcount == 1:
                mf_id = res_search_mf_id[0][0]
            else:
                print(f"ERROR: No found master_flat_id for reduced fits '{red_name}'")
                return -1
            
            
            params = ['raw_id', 'master_bias_id', 'master_flat_id', \
                'date_run', 'date_obs', 'path', 'proc_date', 'soft', \
                'pix_border', 'naxis1', 'naxis2', 'type', 'object', \
                'ra2000', 'dec2000', 'exptime', 'equinox', 'mjd_obs', \
                'pixscale', 'filtname', 'telescope', 'instrument', \
                'pol_angle', 'fwhm', 'fwhm_std', 'fwhm_nsources', \
                'fwhm_flag', 'fwhm_ellip', 'min', 'max', 'mean', 'std', 'median']

            str_params = ['date_run', 'path', 'date_obs', 'object', \
                'soft', 'proc_date', 'type', 'object', 'filtname', \
                'telescope', 'instrument']
            
            fits_keywords = ['SOFT', 'PXBORDER', 'NAXIS1', 'NAXIS2', \
                'IMAGETYP', 'OBJECT', 'RA', 'DEC', 'EXPTIME', \
                'EQUINOX', 'MJD-OBS', 'INSTRSCL', 'INSFLNAM', 'TELESCOP', \
                'INSTRUME', 'INSPOROT', 'FWHM', 'FWHMSTD', 'FWNSOURC', \
                'FWHMFLAG', 'FWHMELLI'] # , 'MIN', 'MAX', 'MEAN', 'STD', 'MED']
            
            # Preprocessing some fields...
            date_obs = red_header['DATE'].replace('T', ' ')
            proc_date = red_header['PROCDATE'].replace('T', ' ')

            values = [raw_id, mb_id, mf_id, run_date, date_obs, red_name, proc_date] + \
                [red_header[k] for k in fits_keywords] + \
                [red_stats[k] for k in ['MIN', 'MAX', 'MEAN', 'STD', 'MEDIAN']]
            v = ','.join([f"'{val}'" if par in str_params else f"{val}" for par, val in zip(params, values)])

            # Inserting new register on database.image_raw
            sql_insert = f"INSERT INTO image_reduced ({','.join(params)}) VALUES ({v})"
            print(f"SQL command (image_reduced) = '{sql_insert}'")
            
            try:
                db_cursor.execute(sql_insert)
            except:
                print(f"SQL ERROR: {sql_insert}")
                raise
            
            # Commiting insertion
            db_object.commit()

            if db_cursor.rowcount > 0:
                print(f"INFO: {db_cursor.rowcount} register/s inserted successfully.")
            new_registrations += db_cursor.rowcount

        # if new_registrations > 1:
        #     break
    
    return new_registrations

def register_calibrated(data_dir, run_date, db_object):
    """
    Register new CALIBRATED fits located in data_dir from nightly run 
    in database 'image_calibrated' table using connection given by db_object.

    Args:
        data_dir (string): base directory. FITS are stored in data_dir/raw directory.
        run_date (string): format given by pattern YYMMDD where YY is year, MM is month and DD y day.
        db_object (mysql.connector.db object): it manages database connection.

    Returns:
        int:
            if int >= 0, it represents number of new reduced FITS registered in database.
    
    Exception:
        IOError: if reduced FITS catalog file was not found.
    """
    new_registrations = 0
    calibration_dir = os.path.join(data_dir, f'calibration/MAPCAT/{run_date}/')

    cal_directories = glob.glob(os.path.join(calibration_dir, '*-*'))
    cal_directories.sort()

    # Create cursor for database operation
    db_cursor = db_object.cursor()

    # working on each subdirectory
    for cal_dir in cal_directories:
        if not os.path.isdir(cal_dir):
            continue
        print(f"INFO: working in directory '{cal_dir}'\n")
        try:
            searching = os.path.join(cal_dir, '*final_photocal_process_info.csv')
            cal_info_path = glob.glob(searching)[0]
            data = pd.read_csv(cal_info_path)
        except:
            print(f"--------------ERROR: catalog '{searching}' not available.")
            continue

        # Checking each calibrated FITS 
        for index, row in data.iterrows():
            dire, cal_name = os.path.split(row['PATH'])
            
            sql_search = f"SELECT `id` FROM `image_calibrated` WHERE path='{cal_name}'"
            db_cursor.execute(sql_search)
            res_search = db_cursor.fetchall()
            if db_cursor.rowcount == 0:
                # Insert new register
                cal_path = os.path.join(cal_dir, cal_name)
                try:
                    cal_fits = mcFits(cal_path)
                except:
                    print(f"ERROR: processing {cal_path}")
                    raise
                cal_header = cal_fits.header
                cal_stats = cal_fits.stats()

                # getting reduced id
                red_name = cal_name.replace('_final', '')
                sql_search_red_id = f"SELECT `id` FROM `image_reduced` WHERE `path` = '{red_name}'"
                print(f"sql_search = {sql_search_red_id}")
                db_cursor.execute(sql_search_red_id)
                res_search_red_id = db_cursor.fetchall()
                if db_cursor.rowcount == 1:
                    red_id = res_search_red_id[0][0]
                else:
                    print(f"ERROR: No found red_id for calibrated fits '{cal_name}'")
                    return -1
                
                # getting blazar_id
                blazar_name = None
                if 'BLZRNAME' in cal_header:
                    blazar_name = cal_header['BLZRNAME']
                elif 'BLZRNAME' in row:
                    blazar_name = row['BLZRNAME']
                else: # blazar name empty
                    try:
                        blazar_info_path = glob.glob(os.path.join(cal_dir, '_photocal_res.csv'))[0]
                        bz_data = pd.read_csv(blazar_info_path)
                        blazar_name = data['MC-IAU-NAME'].values[0]
                    except:
                        print("ERROR: no blazar info file")
                
                blazar_id = None
                if len(blazar_name):
                    sql_search_bl_id = f"SELECT `id` FROM `blazar_source` WHERE `name_IAU` = '{blazar_name}'"
                    print(f"sql_search = {sql_search_bl_id}")
                    db_cursor.execute(sql_search_bl_id)
                    res_search_bl_id = db_cursor.fetchall()
                    if db_cursor.rowcount == 1:
                        blazar_id = res_search_bl_id[0][0]
                    else:
                        print(f"ERROR: No found blazar_id for calibrated fits '{cal_name}'")
                        # return -1                
                
                params = ['reduced_id', 'blazar_id', \
                    'date_run', 'date_obs', 'path', 'proc_date', \
                    'secpix1', 'secpix2', 'soft', \
                    'naxis1', 'naxis2', 'pix_border', 'type', 'object', \
                    'ra2000', 'dec2000', 'exptime', 'equinox', 'mjd_obs', \
                    'pixscale', 'filtname', 'telescope', 'instrument', \
                    'pol_angle', 'fwhm', 'fwhm_std', 'fwhm_nsources', \
                    'fwhm_flag', 'fwhm_ellip', 'softdet', 'crval1', 'crval2', \
                    'epoch', 'crpix1', 'crpix2', 'cdelt1', 'cdelt2', \
                    'ctype1','ctype2', 'cd1_1', 'cd1_2', 'cd2_1' , 'cd2_2', \
                    'wcsrefcat', 'wcsmatch', 'wcsnref', 'wcstol', 'crota1' , \
                    'crota2', 'wcssep', 'imwcs', \
                    'mag_zpt', 'ns_zpt', 'mag_zpt_std', \
                    'min', 'max', 'mean', 'std', 'median']

                str_params = ['date_run', 'path', 'date_obs', 'object', \
                    'soft', 'proc_date', 'type', 'object', 'filtname', \
                    'telescope', 'instrument', 'softdet', 'ctype1', 'ctype2', \
                    'wcsrefcat', 'imwcs', 'ra2000', 'dec2000']
                
                fits_keywords = ['SOFT', 'NAXIS1', 'NAXIS2', 'PXBORDER', \
                    'IMAGETYP', 'OBJECT', 'RA', 'DEC', 'EXPTIME', \
                    'EQUINOX', 'MJD-OBS', 'INSTRSCL', 'INSFLNAM', 'TELESCOP', \
                    'INSTRUME', 'INSPOROT', 'FWHM', 'FWHMSTD', 'FWNSOURC', \
                    'FWHMFLAG', 'FWHMELLI', 'SOFTDET', 'CRVAL1', 'CRVAL2', \
                    'EPOCH', 'CRPIX1', 'CRPIX2', 'CDELT1', 'CDELT2', 'CTYPE1', \
                    'CTYPE2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'WCSRFCAT', \
                    'WCSMATCH', 'WCSNREF', 'WCSTOL', 'CROTA1', 'CROTA2', \
                    'WCSSEP', 'IMWCS', 'MAGZPT', 'NSZPT', 'STDMAGZP'] 
                    
                    # , 'MIN', 'MAX', 'MEAN', 'STD', 'MED']
                
                # Preprocessing some fields...
                date_obs = cal_header['DATE'].replace('T', ' ')
                proc_date = cal_header['PROCDATE'].replace('T', ' ')
                secpix1 = None
                secpix2 = None
                if 'SECPIX' in cal_header:
                    secpix1 = cal_header['SECPIX']
                    secpix2 = cal_header['SECPIX']
                if 'SECPIX1' in cal_header:
                    secpix1 = cal_header['SECPIX1']
                if 'SECPIX2' in cal_header:
                    secpix2 = cal_header['SECPIX2']
                values = [red_id, blazar_id, run_date, date_obs, cal_name, \
                    proc_date, secpix1, secpix2] + \
                    [cal_header[k] for k in fits_keywords] + \
                    [cal_stats[k] for k in ['MIN', 'MAX', 'MEAN', 'STD', 'MEDIAN']]
                v = ','.join([f"'{val}'" if par in str_params else f"{val}" for par, val in zip(params, values)])

                # Inserting new register on database.image_raw
                sql_insert = f"INSERT INTO image_calibrated ({','.join(params)}) VALUES ({v})"
                print(f"SQL command (image_calibrated) = '{sql_insert}'")

                try:
                    db_cursor.execute(sql_insert)
                except:
                    print(f"SQL ERROR: {sql_insert}")
                    raise
                
                # Commiting insertion
                db_object.commit()

                if db_cursor.rowcount > 0:
                    print(f"INFO: {db_cursor.rowcount} register/s inserted successfully.")
                new_registrations += db_cursor.rowcount

        # if new_registrations > 1:
        #     break
    
    return new_registrations

def register_blazar_measure(data_dir, run_date, db_object):
    """
    Register for each run_date and calibrated fits data associated to blazar.

    Args:
        data_dir (string): base directory. FITS are stored in data_dir/raw directory.
        run_date (string): format given by pattern YYMMDD where YY is year, 
            MM is month and DD y day.
        db_object (mysql.connector.db object): it manages database connection.

    Returns:
        int:
            if int >= 0, it represents number of blazars processed and registered in database.
    
    Exception:
        IOError, IndexError: 
            if blazar catalog file was not found.
    """
    new_registrations = 0
    calibration_dir = os.path.join(data_dir, f'calibration/MAPCAT/{run_date}/')

    cal_directories = glob.glob(os.path.join(calibration_dir, '*-*'))
    cal_directories.sort()

    # Create cursor for database operation
    db_cursor = db_object.cursor()

    # working on each subdirectory
    for cal_dir in cal_directories:
        if not os.path.isdir(cal_dir):
            continue
        print(f"INFO: working in directory '{cal_dir}'\n")
        try:
            searching = os.path.join(cal_dir, '*_final_photocal_res.csv')
            blazar_info_path = glob.glob(searching)[0]
            blazar_data = pd.read_csv(blazar_info_path)
        except:
            print(f"--------------ERROR: catalog '{searching}' not available.")
            continue

        # getting calibrated FITS
        dire, name = os.path.split(blazar_info_path)
        cal_name = name.replace('_final_photocal_res.csv', '_final.fits')

        sql_search = f"SELECT `id` FROM `image_calibrated` WHERE path='{cal_name}'"
        db_cursor.execute(sql_search)
        res_search = db_cursor.fetchall()
        if db_cursor.rowcount == 1:
            cal_id = res_search[0][0]
        else:
            print(f'sql_search = {sql_search}')
            print(f'query result = {res_search}')
            print('ERROR: No associated calibrated FITS for this catalog')
            return -2
        # Checking previous information on blazar in blazar_measure table...
        sql_search_cal_id = f"SELECT `cal_id` FROM `blazar_measure` WHERE `cal_id`={cal_id}"
        db_cursor.execute(sql_search_cal_id)
        res_search_cal_id = db_cursor.fetchall()

        if db_cursor.rowcount == 0:
            # Insert new register

            # getting calibrated FITS
            cal_path = blazar_info_path.replace('_final_photocal_res.csv', '_final.fits')
            try:
                cal_fits = mcFits(cal_path)
            except:
                print(f"ERROR: processing {cal_path}")
                raise
            cal_header = cal_fits.header
            cal_stats = cal_fits.stats()

            # getting blazar_id
            blazar_name = blazar_data['MC-IAU-NAME'].values[0]
            sql_search_bl_id = f"SELECT `id` FROM `blazar_source` WHERE `name_IAU` = '{blazar_name}'"
            print(f"sql_search = {sql_search_bl_id}")
            db_cursor.execute(sql_search_bl_id)
            res_search_bl_id = db_cursor.fetchall()
            if db_cursor.rowcount == 1:
                blazar_id = res_search_bl_id[0][0]
            else:
                print(f"ERROR: No found blazar_id for calibrated fits '{cal_name}'")
                return -3 
    
            # Taking info about ordinary and extraordinary sources...
            params = ['cal_id', 'blazar_id', 'date_obs', \
                'date_run', 'mjd_obs', 'pol_angle', \
                'source_type', 'object', 'fwhm', \
                'ra2000', 'dec2000', 'flux_max', 'flux_aper', \
                'fluxerr_aper', 'mag_aper', 'magerr_aper', \
                'class_star', 'ellipticity', 'flags']

            str_params = ['date_run', 'date_obs', 'object', 'source_type']
            
            blazar_keywords = ['RUN_DATE', 'MJD-OBS', 'ANGLE', 'TYPE', 'OBJECT', \
                'FWHM_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', 'FLUX_MAX', \
                'FLUX_APER', 'FLUXERR_APER', 'MAG_APER', 'MAGERR_APER', \
                'CLASS_STAR', 'ELLIPTICITY', 'FLAGS']

            for index, row in blazar_data.iterrows():            
                # Preprocessing some fields...
                date_obs = row['DATE-OBS'].replace('T', ' ')
                values = [cal_id, blazar_id, date_obs] + \
                    [row[k] for k in blazar_keywords]

                v = ','.join([f"'{val}'" if par in str_params else f"{val}" for par, val in zip(params, values)])

                # Inserting new register on database.image_raw
                sql_insert = f"INSERT INTO `blazar_measure` ({','.join(params)}) VALUES ({v})"
                print(f"SQL command (blazar_measure) = '{sql_insert}'")

                try:
                    db_cursor.execute(sql_insert)
                except:
                    print(f"SQL ERROR: {sql_insert}")
                    raise
                
                # Commiting insertion
                db_object.commit()

                if db_cursor.rowcount > 0:
                    print(f"INFO: {db_cursor.rowcount} register/s inserted successfully.")
                new_registrations += db_cursor.rowcount

        # if new_registrations > 1:
        #     break
    
    return new_registrations

def register_polarimetry_data(data_dir, run_date, db_object):
    """
    Register for each value of polarization and angle of polarization for
    each blazar observed at run_date.

    Args:
        data_dir (string): base directory. FITS are stored in data_dir/raw directory.
        run_date (string): format given by pattern YYMMDD where YY is year, 
            MM is month and DD y day.
        db_object (mysql.connector.db object): it manages database connection.

    Returns:
        int:
            if int >= 0, it represents number of polarimetrical computations registered in database.
    
    Exception:
        IOError, IndexError: 
            if polarimetry catalog file was not found.
    """
    new_registrations = 0

    # Create cursor for database operation
    db_cursor = db_object.cursor()
    
    dt = datetime.strptime(run_date, '%y%m%d')

    catalog = os.path.join(data_dir, f'final/MAPCAT/{run_date}/MAPCAT_polR_{dt.strftime("%Y-%m-%d")}.csv')
    try:
        pol_data = pd.read_csv(catalog)
        print(f'Number of polarimetry data = {len(pol_data.index)}')
        print('*' * 60)
    except IOError:
        print(f"ERROR: polarimetry catalog '{catalog}' not found.")
        raise
    
    params = ['blazar_id', 'date_run', \
        'rjd-50000', 'name', 'P', 'dP', 'Theta', 'dTheta', 'R', 'dR']

    str_params = ['name', 'date_run']

    pol_keywords = ['DATE_RUN', 'RJD-50000', 'MC-IAU-NAME', \
        'P', 'dP', 'Theta', 'dTheta', 'R', 'Sigma']
        
    # Checking previous information on blazar in blazar_measure table...
    for index, row in pol_data.iterrows():
        print(f'index = {index}')
        # print(f'row ------------\n {row}')
        # getting blazar_id
        blazar_name = row['MC-IAU-NAME']
        sql_search_bl_id = f"SELECT `id` FROM `blazar_source` WHERE `name_IAU` = '{blazar_name}'"
        print(f"sql_search = {sql_search_bl_id}")
        db_cursor.execute(sql_search_bl_id)
        res_search_bl_id = db_cursor.fetchall()
        
        blazar_id = None
        if len(res_search_bl_id) == 1:
            blazar_id = res_search_bl_id[0][0]
        else:
            print(f"ERROR: No found blazar_id for blazar called '{blazar_name}' in polarimetry info row.")
            return -3 
    
        r_date = datetime.strptime(run_date, '%y%m%d').strftime("%Y-%m-%d")
        sql_search_pol_data = f"SELECT * FROM `polarimetry_data` WHERE `blazar_id` = {blazar_id} AND `date_run` = '{r_date}' AND (ABS(`rjd-50000` - {row['RJD-50000']}) < 0.0001)"
        print(f'sql_search = {sql_search_pol_data}')
        db_cursor.execute(sql_search_pol_data)
        res_search_pol_data = db_cursor.fetchall()
        print(f'res_search_pol_data = {res_search_pol_data}')

        if len(res_search_pol_data) == 0:
            # Insert new register
            values = [blazar_id] + \
                [row[k] for k in pol_keywords]

            v = ','.join([f"'{val}'" if par in str_params else f"{val}" for par, val in zip(params, values)])

            # Inserting new register on database.image_raw
            sql_insert = f"INSERT INTO `polarimetry_data` (`{'`,`'.join(params)}`) VALUES ({v})"
            print(f"SQL command (blazar_measure) = '{sql_insert}'")

            try:
                db_cursor.execute(sql_insert)
            except:
                print(f"SQL ERROR: {sql_insert}")
                raise
            
            # Commiting insertion
            db_object.commit()
            new_registrations += 1

        # if new_registrations > 1:
        #     break
    
    return new_registrations



def main():
    parser = argparse.ArgumentParser(prog='iop3_add_db_info.py', \
    conflict_handler='resolve',
    description='''Read CSV files from run_date and insert new info in database.''',
    epilog='''''')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    # parser.add_argument("dbengine", help="Configuration parameter files directory")
    parser.add_argument("input_data_dir", help="Input data base directory")
    parser.add_argument("run_date", help="Run date in format YYMMDD")
    parser.add_argument("--db_server",
        action="store",
        dest="db_server",
        default="localhost",
        help="Database server name or IP [default: %(default)s]")
    parser.add_argument("--db_name",
        action="store",
        dest="db_name",
        default="iop3db",
        help="Database name [default: %(default)s]")
    parser.add_argument("--db_user",
        action="store",
        dest="db_user",
        default="iop3admin",
        help="Database admin name [default: %(default)s]")
    parser.add_argument("--db_password",
        action="store",
        dest="db_password",
        default="IOP3_db_admin",
        help="Database admin user password [default: %(default)s]")
    parser.add_argument('-v', '--verbose', action='count', default=0,
        help="Show running and progress information [default: %(default)s].")
    
    args = parser.parse_args()

    # Checking mandatory parameters
    if not os.path.isdir(args.input_data_dir):
        print(f"ERROR: Not valid input_data_dir '{args.input_data_dir}'")
        return 1
    
    dirs = ['raw', 'reduction', 'calibration', 'final']
    directories = {k: os.path.join(args.input_data_dir, f'{k}/MAPCAT/{args.run_date}/') for k in dirs}
    
    print(f'Directories = {directories}')
    
   
    for k, v in directories.items():
        if not os.path.isdir(v):
            print(f"ERROR: Directory '{v}' not available.")
            return 2

    # Checking optional parameters
    db_user = args.db_user
    if not len(db_user):
        prompt =  f"Write login for allowed user to insert/update info on database '{args.db_name}': "
        db_user = input(prompt)
    
    db_password = args.db_password
    if not len(db_password):
        db_password = getpass(f"Password of {db_user}@{args.db_server}: ")

    # Database connection
    my_db = mysql.connector.connect(host=args.db_server, \
        user=db_user, password=db_password, database=args.db_name)

    my_cursor = my_db.cursor()

    # Selecting database for table creation
    my_cursor.execute(f"USE {args.db_name};")
    
    # return 1

    # Processing RAW images (image_raw table) from run date
    # res = register_raw(args.input_data_dir, args.run_date, my_db)
    # print(f"Raw image registration result -> {res}")
    
    # return 1

    # Processing MASTERBIAS (master_bias table) from run date 
    # res = register_masterbias(args.input_data_dir, args.run_date, my_db)
    # print(f"MasterBIAS registration result -> {res}")

    # return 1

    # Processing RAW-BIAS (raw_bias table) from run date
    # res = register_rawbias(args.input_data_dir, args.run_date, my_db)
    # print(f"Relation between raw and masterBIAS registration result -> {res}")

    # return 1

    # Processing MASTERFLATS (master_flat table) from run date
    # res = register_masterflats(args.input_data_dir, args.run_date, my_db)
    # print(f"MasterFLATS registration result -> {res}")
    
    # return 1

    # Processing RAW-FLATS (raw_flat table) from run date
    # res = register_rawflats(args.input_data_dir, args.run_date, my_db)
    # print(f"Relation between raw and masterFLATS registration result -> {res}")
    
    # return 1

    # Processing REDUCED images (image_reduced table) from run date
    # res = register_reduced(args.input_data_dir, args.run_date, my_db)
    # print(f"Reduced image registration result -> {res}")
    
    # return 1

    # Processing CALIBRATED images (image_calibrated table) from run date
    # res = register_calibrated(args.input_data_dir, args.run_date, my_db)
    # print(f"Calibrated image registration result -> {res}")
    
    # return 1

    # Processing BLAZAR MEASURE (blazar_measure table) from run date
    # res = register_blazar_measure(args.input_data_dir, args.run_date, my_db)
    # print(f"blazar data registration result -> {res}")
    
    # return 1

    # Processing POLARIMETRY DATA (polarimetry_data table) from run date
    res = register_polarimetry_data(args.input_data_dir, args.run_date, my_db)
    print(f"Polarimetry data registration result -> {res}")
    
    return 1

    # Unuseful table. Info can be get by querying...
    """SELECT 
        bm.cal_id, bm.date_run, bm.class_star, bm.source_type, 
        bm.ellipticity, bm.flags, bm.date_obs, bm.pol_angle, 
        pd.name, pd.P, pd.dP, pd.Theta, pd.dTheta, pd.R, pd.dR 
    FROM 
        polarimetry_data as pd 
        INNER JOIN 
            blazar_measure as bm 
        ON 
            pd.blazar_id=bm.blazar_id 
    ORDER BY pd.name, bm.date_obs;"""
    # Processing BLAZAR_POLARIMETRY (blazar_polarimetry table) from run date
    # res = register_polarimetry_blazar(args.input_data_dir, args.run_date, my_db)
    # print(f"Relations between polarimetry measures and blazars result -> {res}")



    return 0

if __name__ == '__main__':
    print(main())