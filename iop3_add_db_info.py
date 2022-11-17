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
from datetime import datetime, date, timedelta

import pandas as pd

import mysql.connector

from astropy.io import fits
from astropy.time import Time

from mcFits import mcFits 
import astropy.units as u

def get_run_date(mjd):
    """Returns night run as Python datetime object for 
    Modified Julian Date given by 'mjd' input parameter.

    Returns:
        datetime.datetime: Run night observation date.
    """
    t = Time(mjd, format="mjd", scale="utc")
    t_dt = t.to_datetime()
    
    if t_dt.hour < 12: # fits taken in previous day
        t_dt = t_dt - timedelta(days=1)

    return t_dt


def res2df(db_cursor):
    return 0


def register_raw(data_dir, run_date, db_object, telescope):
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
    #raw_dir = os.path.join(data_dir, f'{run_date}/')
    raw_dir = data_dir
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
        sql_search = f"SELECT id FROM image_raw WHERE path='{raw_name}' AND date_run='{run_date}'"
        db_cursor.execute(sql_search)
        res_search = db_cursor.fetchall()
        action="INSERT"
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
            try:
                if 'RA' not in raw_header:
                    ra = raw_header['OBJCTRA'].split(' ')
                    dec = raw_header['OBJCTDEC'].split(' ')
                raw_header['RA'] = ((((int(ra[0]) * 3600 + int(ra[1]) * 60 + int(float(ra[2])))/3600) * u.hourangle).to(u.deg)).to_value()
                raw_header['DEC'] = ((((int(dec[0]) * 3600 + int(dec[1]) * 60 + int(float(dec[2])))/3600) * u.hourangle).to(u.deg)).to_value()
            except:
                print("This is probably not a science file, no coordinates found")
                continue
            if 'INSPOROT' not in raw_header and 'FILTER' in raw_header:
                if raw_header['FILTER'] in ['R', 'U', 'V', 'B', 'Clear', 'I']:
                    raw_header['INSPOROT'] = -999
                else:
                    try:
                        if "_45" in raw_header['FILTER']:
                            raw_header['INSPOROT'] = -45
                        else:
                            raw_header['INSPOROT'] = int(raw_header['FILTER'][1:]) 
                    except:
                        raw_header['INSPOROT'] = -999
            if 'MJD-OBS' not in raw_header and 'JD' in raw_header:
                raw_header['MJD-OBS'] = raw_header['JD'] - 2400000.5
            fits_keywords = ['NAXIS1', 'NAXIS2', 'OBJECT', 'IMAGETYP', 'RA', 'DEC', 'EXPTIME', \
                                 'EQUINOX', 'MJD-OBS', 'INSTRSCL', 'INSFLNAM', 'TELESCOP', \
                                 'INSTRUME', 'INSPOROT']
            # date_obs = datetime.strptime(raw_header['DATE'], '%Y-%d-%mT%H:%M:%S')
            # r_date = datetime.strptime(run_date, '%Y-%d-%m')
            if 'DATE' in raw_header:
                date_obs = raw_header['DATE'].replace('T', ' ')
            else:
                date_obs = raw_header['DATE-OBS'].replace('T', ' ')
            r_date = run_date
            values = [r_date, raw_name, date_obs] + \
                [raw_header.get(k, 'NULL') for k in fits_keywords] + \
                [raw_stats.get(k, 'NULL') for k in ['MIN', 'MAX', 'MEAN', 'STD', 'MEDIAN']]
            
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



def register_masterbias(data_dir, run_date, db_object, telescope):
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
    #reduction_dir = data_dir.replace('raw', 'reduction') 
    #reduction_dir = os.path.join(reduction_dir, f'{run_date}/')
    reduction_dir=data_dir

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
        
        sql_search = f"SELECT id FROM master_bias WHERE path='{mb_name}' AND date_run='{run_date}'"
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
            
            params = ['type', 'date_run', 'path', 'proc_date', 'Telescope', \
                'naxis1', 'naxis2', 'soft', 'pix_border', \
                'bias_operation', \
                'min', 'max', 'mean', 'std', 'median']

            str_params = ['date_run', 'path', 'proc_date', 'soft', \
                'bias_operation', 'type']
            
            # Keywords are the same for CSV and fits files.
            keywords = ['NAXIS1', 'NAXIS2', 'SOFT', 'PXBORDER', 'BIASOP']
            
            date_proc = red_header['PROCDATE'].replace('T', ' ')
            mb_type = red_header['OBJECT']
            values = [mb_type, run_date, mb_name, date_proc, f'"{telescope}"'] + \
                [red_header.get(k, 'NULL') for k in keywords] + \
                [red_stats.get(k, 'NULL')for k in ['MIN', 'MAX', 'MEAN', 'STD', 'MEDIAN']]
            
            #query="ALTER TABLE master_bias ADD Telescope VARCHAR(100)"
            #sql_command = "ALTER TABLE master_bias"
            #db_cursor.execute(query)
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

def register_rawbias(data_dir, run_date, db_object, telescope):
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
    #reduction_dir = data_dir.replace('raw', 'reduction') 
    #reduction_dir = os.path.join(reduction_dir, f'{run_date}/')
    reduction_dir=data_dir

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
            names = [bias_header.get(k, 'NULL') for k in keywords if k in bias_header]
            
            print(f"names = {names}")

            raw_ids = []
            for n in names:
                sql_search = f"SELECT id FROM `image_raw` WHERE `path`='{n}' and `date_run`='{run_date}'"
                print(f"sql_search = {sql_search}")
                db_cursor.execute(sql_search)
                res_search = db_cursor.fetchall()
                print(n)
                print(db_cursor.rowcount)
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


def register_masterflats(data_dir, run_date, db_object, telescope):
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
    #reduction_dir = data_dir.replace('raw', 'reduction') 
    #reduction_dir = os.path.join(reduction_dir, f'{run_date}/')
    reduction_dir=data_dir

    try:
        catalog = os.path.join(reduction_dir, "masterflats_data.csv")
        data = pd.read_csv(catalog)
    except IOError:
        print(f"ERROR: catalog '{catalog}' not available.")
        raise

    # Create cursor for database operation
    db_cursor = db_object.cursor()
    #query="ALTER TABLE master_flat ADD Telescope VARCHAR(100)"
    #db_cursor.execute(query)
    # Checking each masterFLAT FITS (4 per night, one for each grism angle) 
    for index, row in data.iterrows():
        dire, mf_name = os.path.split(row['PATH'])
        
        sql_search = f"SELECT id FROM master_flat WHERE path='{mf_name}' AND date_run='{run_date}'"
        db_cursor.execute(sql_search)
        res_search = db_cursor.fetchall()
        if db_cursor.rowcount == 0:
        # Insert new register<
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
            params = ['master_bias_id', 'type', 'date_run', 'path', 'proc_date', 'Telescope', \
                'naxis1', 'naxis2', 'pol_angle', 'soft', 'pix_border', \
                'flat_operation', \
                'min', 'max', 'mean', 'std', 'median']

            str_params = ['date_run', 'path', 'proc_date', 'soft', \
                'flat_operation', 'type']
            
            # Keywords are the same for CSV and fits files.
            if ('INSPOROT' not in mf_header and 'FILTER' in mf_header):
                if mf_header['FILTER'] in ['R', 'U', 'V', 'B', 'Clear','I']:
                    mf_header['INSPOROT'] = -999
                else:
                    mf_header['INSPOROT'] = int(mf_header['FILTER'][1:]) 
            if 'R' in mf_header['INSPOROT']:
                if mf_header['INSPOROT']=='R':
                    mf_header['INSPOROT'] = -999
                elif "_45" in mf_header['INSPOROT']:
                    mf_header['INSPOROT'] = -45
                else:
                    mf_header['INSPOROT'] = int(mf_header['INSPOROT'][1:])
            if mf_header['INSPOROT'] in ['I','R', 'U', 'V', 'B', 'Clear']:
                mf_header['INSPOROT'] = -999
                
            keywords = ['NAXIS1', 'NAXIS2', 'INSPOROT', 'SOFT', 'PXBORDER', 'FLATOP']
            
            date_proc = mf_header['PROCDATE'].replace('T', ' ')
            mf_type = mf_header['OBJECT']
            values = [mb_id, mf_type, run_date, mf_name, date_proc, f'"{telescope}"'] + \
                [mf_header.get(k, 'NULL') for k in keywords] + \
                [mf_stats.get(k, 'NULL') for k in ['MIN', 'MAX', 'MEAN', 'STD', 'MEDIAN']]
            
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

def register_rawflats(data_dir, run_date, db_object, telescope):
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
    #reduction_dir = data_dir.replace('raw', 'reduction') 
    #reduction_dir = os.path.join(reduction_dir, f'{run_date}/')
    reduction_dir=data_dir

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
            names = [flat_header.get(k, 'NULL') for k in keywords if k in flat_header]
            
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

def register_reduced(data_dir, run_date, db_object, telescope):
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
    #reduction_dir = data_dir.replace('raw', 'reduction') 
    #reduction_dir = os.path.join(reduction_dir, f'{run_date}/')
    reduction_dir=data_dir

    try:
        catalog = os.path.join(reduction_dir, "output_data_red.csv")
        data = pd.read_csv(catalog)
    except IOError:
        print(f"ERROR: catalog '{catalog}' not available.")
        raise

    # Create cursor for database operation
    db_cursor = db_object.cursor()

    # Checking each raw FITS 
    for index, row in data.iterrows():
        dire, red_name = os.path.split(row['PATH'])
        sql_search = f"SELECT `id` FROM `image_reduced` WHERE path='{red_name}' AND date_run='{run_date}'"
        db_cursor.execute(sql_search)
        res_search = db_cursor.fetchall()

        if db_cursor.rowcount == 0:
            # Insert new register
            red_path = os.path.join(reduction_dir, red_name)
            try:
                red_fits = mcFits(red_path)
            except:
                print(f"ERROR: processing {red_path}")
                continue
            red_header = red_fits.header
            red_stats = red_fits.stats()

            # getting raw id
            sql_search_raw_id = f"SELECT `id` FROM `image_raw` WHERE `path` = '{red_name}' AND date_run='{run_date}'"
            print(f"sql_search = {sql_search_raw_id}")
            db_cursor.execute(sql_search_raw_id)
            res_search_raw_id = db_cursor.fetchall()
            if db_cursor.rowcount == 1:
                raw_id = res_search_raw_id[0][0]
            else:
                print(f"ERROR: No found raw_id for reduced fits '{red_name}'")
                
            
            # getting master_bias_id
            sql_search_mb_id = f"SELECT `id` FROM `master_bias` WHERE `path` = '{red_header['BIAS']}'"
            print(f"sql_search = {sql_search_mb_id}")
            db_cursor.execute(sql_search_mb_id)
            res_search_mb_id = db_cursor.fetchall()
            
            if db_cursor.rowcount == 1:
                mb_id = res_search_mb_id[0][0]
            else:
                print(f"ERROR: No found master_bias_id for reduced fits '{red_name}'")
                
            

            # getting master_flat_id
            sql_search_mf_id = f"SELECT `id` FROM `master_flat` WHERE `path` = '{red_header['FLAT']}'"
            print(f"sql_search = {sql_search_mf_id}")
            db_cursor.execute(sql_search_mf_id)
            res_search_mf_id = db_cursor.fetchall()
            if db_cursor.rowcount == 1:
                mf_id = res_search_mf_id[0][0]
            else:
                print(f"ERROR: No found master_flat_id for reduced fits '{red_name}'")
                
            
            
            params = ['raw_id', 'master_bias_id', 'master_flat_id', \
                'date_run', 'date_obs', 'path', 'proc_date', 'soft', \
                'pix_border', 'naxis1', 'naxis2', 'type', 'object', \
                'ra2000', 'dec2000', 'exptime', 'equinox', 'mjd_obs', \
                'pixscale', 'filtname', 'telescope', 'instrument', \
                'pol_angle', 'min', 'max', 'mean', 'std', 'median']
                # 'fwhm', 'fwhm_std', 'fwhm_nsources', 'fwhm_flag', 'fwhm_ellip'

            str_params = ['date_run', 'path', 'date_obs', 'object', \
                'soft', 'proc_date', 'type', 'object', 'filtname', \
                'telescope', 'instrument']
            
            if 'RA' not in red_header:
                ra = red_header['OBJCTRA'].split(' ')
                dec = red_header['OBJCTDEC'].split(' ')
                red_header['RA'] = ((((int(ra[0]) * 3600 + int(ra[1]) * 60 + int(float(ra[2])))/3600) * u.hourangle).to(u.deg)).to_value()
                red_header['DEC'] = ((((int(dec[0]) * 3600 + int(dec[1]) * 60 + int(float(dec[2])))/3600) * u.hourangle).to(u.deg)).to_value()
            if 'INSPOROT' not in red_header and 'FILTER' in red_header:
                if red_header['FILTER'] in ['I','R', 'U', 'V', 'B', 'Clear']:
                    red_header['INSPOROT'] = -999
                elif '_45' in red_header['FILTER']:
                    red_header['INSPOROT'] = -45 
                else:
                    red_header['INSPOROT'] = int(red_header['FILTER'][1:]) 
            if 'MJD-OBS' not in red_header and 'JD' in red_header:
                red_header['MJD-OBS'] = red_header['JD'] - 2400000.5


            fits_keywords = ['SOFT', 'PXBORDER', 'NAXIS1', 'NAXIS2', \
                                 'IMAGETYP', 'OBJECT', 'RA', 'DEC', 'EXPTIME', \
                                 'EQUINOX', 'MJD-OBS', 'INSTRSCL', 'INSFLNAM', 'TELESCOP', \
                                 'INSTRUME', 'INSPOROT']
                # , 'FWHM', 'FWHMSTD', 'FWNSOURC', 'FWHMFLAG', 'FWHMELLI'] 
                # , 'MIN', 'MAX', 'MEAN', 'STD', 'MED']
            
            # Preprocessing some fields...
            if 'DATE' in red_header:
                date_obs = red_header['DATE'].replace('T', ' ')
            else:
                date_obs = red_header['DATE-OBS'].replace('T', ' ')
            proc_date = red_header['PROCDATE'].replace('T', ' ')

            values = [raw_id, mb_id, mf_id, run_date, date_obs, red_name, proc_date] + \
                [red_header.get(k, 'NULL') for k in fits_keywords] + \
                [red_stats.get(k, 'NULL') for k in ['MIN', 'MAX', 'MEAN', 'STD', 'MEDIAN']]
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

def register_calibrated(data_dir, run_date, db_object, telescope):
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
    #calibration_dir = data_dir.replace('raw', 'calibration') 
    #calibration_dir = os.path.join(calibration_dir, f'{run_date}/')
    calibration_dir=data_dir
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
            sql_search = f"SELECT `id` FROM `image_calibrated` WHERE path='{cal_name}' AND date_run='{run_date}'"
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
                sql_search_red_id = f"SELECT `id` FROM `image_reduced` WHERE `path` = '{red_name}' AND date_run='{run_date}'"
                print(f"sql_search = {sql_search_red_id}")
                db_cursor.execute(sql_search_red_id)
                res_search_red_id = db_cursor.fetchall()
                if db_cursor.rowcount == 1:
                    red_id = res_search_red_id[0][0]
                else:
                    print(f"ERROR: No found red_id for calibrated fits '{cal_name}'")
                    continue
                
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
                    'crotation', \
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
                if 'RA' not in cal_header:
                    ra = cal_header['OBJCTRA'].split(' ')
                    dec = cal_header['OBJCTDEC'].split(' ')
                    cal_header['RA'] = ((((int(ra[0]) * 3600 + int(ra[1]) * 60 + int(float(ra[2])))/3600) * u.hourangle).to(u.deg)).to_value()
                    cal_header['DEC'] = ((((int(dec[0]) * 3600 + int(dec[1]) * 60 + int(float(dec[2])))/3600) * u.hourangle).to(u.deg)).to_value()
                if 'INSPOROT' not in cal_header and 'FILTER' in cal_header:
                    if cal_header['FILTER'] in ['R', 'I','U', 'V', 'B', 'Clear']:
                        cal_header['INSPOROT'] = -999
                    elif "_45" in cal_header['FILTER']:
                        cal_header['INSPOROT'] = -45 
                    else:
                        cal_header['INSPOROT'] = int(cal_header['FILTER'][1:]) 
                if 'MJD-OBS' not in cal_header and 'JD' in cal_header:
                    cal_header['MJD-OBS'] = cal_header['JD'] - 2400000.5
                
                fits_keywords = ['SOFT', 'CROTATION', \
                    'NAXIS1', 'NAXIS2', 'PXBORDER', \
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
                if 'DATE' in cal_header:
                    date_obs = cal_header['DATE'].replace('T', ' ')
                else:
                    date_obs = cal_header['DATE-OBS'].replace('T', ' ')
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
                    [cal_header.get(k, 'NULL') for k in fits_keywords] + \
                    [cal_stats.get(k, 'NULL') for k in ['MIN', 'MAX', 'MEAN', 'STD', 'MEDIAN']]
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

def register_photometry(data_dir, run_date, db_object, telescope):
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
    #calibration_dir = data_dir.replace('raw', 'calibration') 
    #calibration_dir = os.path.join(calibration_dir, f'{run_date}/')
    calibration_dir=data_dir
    dt = datetime.strptime(run_date, '%y%m%d')

    cal_directories = glob.glob(os.path.join(calibration_dir, '*-*'))
    cal_directories.sort()

    # Create cursor for database operation
    db_cursor = db_object.cursor()
    #query="ALTER TABLE polarimetry ADD Telescope VARCHAR(100)"
    #db_cursor.execute(query)

    # working on each subdirectory
    for cal_dir in cal_directories:
        if not os.path.isdir(cal_dir):
            continue
        print(f"INFO: working in directory '{cal_dir}'\n")
        try:
            searching = os.path.join(cal_dir, '*_final_photometry.csv')
            photometry_info_path = glob.glob(searching)[0]
            print(f'Reading file: {photometry_info_path}')
            photometry_data = pd.read_csv(photometry_info_path)
            print(photometry_data.info())
        except:
            print(f"--------------ERROR: catalog '{searching}' not available.")
            continue
        
        # getting calibrated FITS
        dire, name = os.path.split(photometry_info_path)
        if 'MAPCAT' in photometry_info_path:
            cal_name = name.replace('_final_photometry.csv', '_final.fits')
        else:
            cal_name = name.replace('_final_photometry.csv', '_final.fit')
        sql_search = f"SELECT `id` FROM `image_calibrated` WHERE path='{cal_name}' AND date_run='{run_date}'"
        db_cursor.execute(sql_search)
        res_search = db_cursor.fetchall()
        if db_cursor.rowcount == 1:
            cal_id = res_search[0][0]
        else:
            print(f'sql_search = {sql_search}')
            print(f'query result = {res_search}')
            print('ERROR: No associated calibrated FITS for this catalog')
            continue


        # getting calibrated FITS
        if 'MAPCAT' in photometry_info_path:
            cal_path = photometry_info_path.replace('_final_photometry.csv', '_final.fits')
        else:
            cal_path = photometry_info_path.replace('_final_photometry.csv', '_final.fit')
        try:
            cal_fits = mcFits(cal_path)
        except:
            print(f"ERROR: processing {cal_path}")
            raise
        cal_header = cal_fits.header
        cal_stats = cal_fits.stats()
        # Checking that Blazars are identical for ORDINARY/EXTRAORDINARY counterparts
        name_blazar_o = photometry_data['name_mc_O'].values[0]
        name_blazar_e = photometry_data['name_mc_E'].values[0]
        print(name_blazar_e)
        if name_blazar_o != name_blazar_e:
            print(f'ORD ({name_blazar_o}) & EXTRAORD ({name_blazar_e}) blazars are differents!')
            return 2

        # getting blazar_id
        blazar_name = name_blazar_o
        blazar_name_IAU = photometry_data['IAU_name_mc_O'].values[0]
        sql_search_bl_id = f"SELECT `id` FROM `blazar_source` WHERE `name` = '{blazar_name}'"
        print(f"sql_search = {sql_search_bl_id}")
        db_cursor.execute(sql_search_bl_id)
        res_search_bl_id = db_cursor.fetchall()
        if db_cursor.rowcount == 1:
            blazar_id = res_search_bl_id[0][0]
        else:
            print(f"ERROR: No found blazar_id for calibrated fits '{cal_name}'")
            return 3 
    
        # Parameters from input photometry file
        #  index_O,id_mc_O,id_blazar_mc_O,aper_mc_O,IAU_name_mc_O,ra2000_mc_O,dec2000_mc_O,name_mc_O,Rmag_mc_O,Rmagerr_mc_O,PolDeg_mc_O,ErrPolDeg_mc_O,PolAngle_mc_O,ErrPolAngle_mc_O,ra2000_mc_deg_O,dec2000_mc_deg_O,index_O,NUMBER_O,MAG_AUTO_O,MAGERR_AUTO_O,FLUX_AUTO_O,FLUXERR_AUTO_O,FLUX_APER_O,FLUXERR_APER_O,MAG_APER_O,MAGERR_APER_O,X_IMAGE_O,Y_IMAGE_O,ALPHA_J2000_O,DELTA_J2000_O,FLAGS_O,CLASS_STAR_O,FWHM_IMAGE_O,FWHM_WORLD_O,ELONGATION_O,ELLIPTICITY_O,DISTANCE_DEG_O,
        # index_E,id_mc_E,id_blazar_mc_E,aper_mc_E,IAU_name_mc_E,ra2000_mc_E,dec2000_mc_E,name_mc_E,Rmag_mc_E,Rmagerr_mc_E,PolDeg_mc_E,ErrPolDeg_mc_E,PolAngle_mc_E,ErrPolAngle_mc_E,ra2000_mc_deg_E,dec2000_mc_deg_E,index_E,NUMBER_E,MAG_AUTO_E,MAGERR_AUTO_E,FLUX_AUTO_E,FLUXERR_AUTO_E,FLUX_APER_E,FLUXERR_APER_E,MAG_APER_E,MAGERR_APER_E,X_IMAGE_E,Y_IMAGE_E,ALPHA_J2000_E,DELTA_J2000_E,FLAGS_E,CLASS_STAR_E,FWHM_IMAGE_E,FWHM_WORLD_E,ELONGATION_E,ELLIPTICITY_E,DISTANCE_DEG_E,RA_J2000_O,DEC_J2000_O,RA_J2000_E,DEC_J2000_E,
        # APERPIX,FWHM,SECPIX,DATE-OBS,MJD-OBS,RJD-50000,EXPTIME,ANGLE,MAGZPT

        # Taking info about ordinary and extraordinary sources...
        params = ['cal_id', 'blazar_id','name','name_IAU','Telescope'] # Parameters got from database registers
        params += ['date_run', 'mjd_obs', 'rjd-50000', 'pol_angle', \
                       'aperpix', 'fwhm', 'secpix', 'exptime', 'magzpt']
        
        photo_params = ['mag_auto', 'magerr_auto', 'flux_auto', 'fluxerr_auto', \
                            'flux_aper', 'fluxerr_aper', 'mag_aper', 'magerr_aper', \
                            'x_image', 'y_image', 'alpha_j2000', 'delta_j2000', 'flags', \
                            'class_star', 'fwhm_image', 'fwhm_world', 'elongation', 'ellipticity', \
                            'distance_deg']
            
        # String parameters
        str_params = ['date_run', 'source_type', 'name', 'name_IAU', 'Telescope']
        
        blazar_keywords = ['MJD-OBS', 'RJD-50000', 'ANGLE', \
                               'APERPIX', 'FWHM', 'SECPIX', 'EXPTIME', 'MAGZPT']
        blazar_photometry_keywords = ['MAG_AUTO', 'MAGERR_AUTO', 'FLUX_AUTO', \
                                          'FLUXERR_AUTO', 'FLUX_APER', 'FLUXERR_APER', 'MAG_APER', \
                                          'MAGERR_APER', 'X_IMAGE', 'Y_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', \
                                          'FLAGS', 'CLASS_STAR', 'FWHM_IMAGE', 'FWHM_WORLD', 'ELONGATION', \
                                          'ELLIPTICITY', 'DISTANCE_DEG']
            
        # There is only one line for each photometry file
        # mjd_obs = photometry_data['MJD-OBS']
        # try:
        #     r_date = get_run_date(photometry_data['MJD-OBS'].values[0])
        # except:
        #     print(photometry_data['MJD-OBS'])
        #     raise
        date_iso=photometry_data['DATE-OBS'].values[0]
        cal_id=cal_id
        values = [cal_id, blazar_id,blazar_name, blazar_name_IAU,telescope, dt.strftime("%Y-%m-%d")]
        values += [photometry_data[k].values[0] for k in blazar_keywords]

        # Checking previous information on blazar in photometry table...
        sql_search_cal_id = f"SELECT `cal_id` FROM `photometry` WHERE `cal_id`={cal_id} AND date_run='{run_date}'"
        db_cursor.execute(sql_search_cal_id)
        res_search_cal_id = db_cursor.fetchall()
        
        if db_cursor.rowcount == 0:
            # Insert new register
            for t in ['O', 'E']:
                par = params + photo_params + ['source_type']
                vals = values + [photometry_data[k + f'_{t}'].values[0] for k in blazar_photometry_keywords] + [t]
                # formatting values
                v = ','.join([f"'{va}'" if p in str_params else f"{va}" for p, va in zip(par, vals)])
                print(v)
                # Inserting new register on database.photometry
                #sql_insert = f"INSERT INTO `photometry` (`{'`,`'.join(par)}`) VALUES ({v})"
                sql = f"INSERT INTO `photometry` (`{'`,`'.join(par)}`) VALUES ({v})"
                new_registrations += db_cursor.rowcount
                print(f"SQL command (photometry) = {sql}")
                try:
                    db_cursor.execute(sql)
                except:
                    print(f"SQL ERROR: {sql}")
                    raise
                # Commiting insertion
                db_object.commit()
                if db_cursor.rowcount > 0:
                    print(f"INFO: {db_cursor.rowcount} register/s inserted successfully.")
        else:
            # Update register
            for t in ['O', 'E']:
                par = params + photo_params + ['source_type']
                vals = values + [photometry_data[k + f'_{t}'].values[0] for k in blazar_photometry_keywords] + [t]
                # formatting values
                val_fmt = [f"'{val}'" if par in str_params else f"{val}" for par, val in zip(par, vals)]
                v = ','.join([f"'{va}'" if p in str_params else f"{va}" for p, va in zip(par, vals)])
                pairs = []
                for k, j in zip(par, val_fmt):
                    pairs.append(f'`{k}` = {j}')
                sql = f"INSERT INTO `photometry` (`{'`,`'.join(par)}`) VALUES ({v}) ON DUPLICATE KEY UPDATE {','.join(pairs)}"
        
                print(f"SQL command (photometry) = {sql}")
                try:
                    db_cursor.execute(sql)
                except:
                    print(f"SQL ERROR: {sql}")
                    raise
                # Commiting insertion
                db_object.commit()
                if db_cursor.rowcount > 0:
                    print(f"INFO: {db_cursor.rowcount} register/s inserted successfully.")
            
    return new_registrations

def register_photometry_refstars(data_dir, run_date, db_object, telescope):
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
    #calibration_dir = data_dir.replace('raw', 'calibration') 
    #calibration_dir = os.path.join(calibration_dir, f'{run_date}/')
    calibration_dir=data_dir
    dt = datetime.strptime(run_date, '%y%m%d')

    cal_directories = glob.glob(os.path.join(calibration_dir, '*-*'))
    cal_directories.sort()

    # Create cursor for database operation
    db_cursor = db_object.cursor()
    #query="ALTER TABLE polarimetry ADD Telescope VARCHAR(100)"
    #db_cursor.execute(query)

    # working on each subdirectory
    for cal_dir in cal_directories:
        print(cal_dir)
        if not os.path.isdir(cal_dir):
            continue
        print(f"INFO: working in directory '{cal_dir}'\n")
        try:
            searching = os.path.join(cal_dir, '*_final_photometry.csv')
            photometry_info_path = glob.glob(searching)[0]
            print(f'Reading file: {photometry_info_path}')
            photometry_data = pd.read_csv(photometry_info_path)
            print(photometry_data.info())
        except:
            print(f"--------------ERROR: catalog '{searching}' not available.")
            continue
        if photometry_data['name_mc_O'].values.shape[0]>=2:
            # getting calibrated FITS
            dire, name = os.path.split(photometry_info_path)
            if 'MAPCAT' in photometry_info_path:
                cal_name = name.replace('_final_photometry.csv', '_final.fits')
            else:
                cal_name = name.replace('_final_photometry.csv', '_final.fit')
            sql_search = f"SELECT `id` FROM `image_calibrated` WHERE path='{cal_name}' AND date_run='{run_date}'"
            db_cursor.execute(sql_search)
            res_search = db_cursor.fetchall()
            if db_cursor.rowcount == 1:
                cal_id = res_search[0][0]
            else:
                print(f'sql_search = {sql_search}')
                print(f'query result = {res_search}')
                print('ERROR: No associated calibrated FITS for this catalog')
                return 1


            # getting calibrated FITS
            if 'MAPCAT' in photometry_info_path:
                cal_path = photometry_info_path.replace('_final_photometry.csv', '_final.fits')
            else:
                cal_path = photometry_info_path.replace('_final_photometry.csv', '_final.fit')
            try:
                cal_fits = mcFits(cal_path)
            except:
                print(f"ERROR: processing {cal_path}")
                raise
            cal_header = cal_fits.header
            cal_stats = cal_fits.stats()
            
            # Checking that Blazars are identical for ORDINARY/EXTRAORDINARY counterparts
            name_blazar_o = photometry_data['name_mc_O'].values[1]
            name_blazar_e = photometry_data['name_mc_E'].values[1]
            print(name_blazar_e)
            if name_blazar_o != name_blazar_e:
                print(f'ORD ({name_blazar_o}) & EXTRAORD ({name_blazar_e}) blazars are differents!')
                return 2

            # getting blazar_id
            blazar_name = name_blazar_o
            blazar_name_IAU = photometry_data['IAU_name_mc_O'].values[1]
            sql_search_bl_id = f"SELECT `id` FROM `blazar_source` WHERE `name` = '{blazar_name}'"
            print(f"sql_search = {sql_search_bl_id}")
            db_cursor.execute(sql_search_bl_id)
            res_search_bl_id = db_cursor.fetchall()
            if db_cursor.rowcount == 1:
                blazar_id = res_search_bl_id[0][0]
            else:
                print(f"ERROR: No found blazar_id for calibrated fits '{cal_name}'")
                return 3 
    
            # Parameters from input photometry file
            #  index_O,id_mc_O,id_blazar_mc_O,aper_mc_O,IAU_name_mc_O,ra2000_mc_O,dec2000_mc_O,name_mc_O,Rmag_mc_O,Rmagerr_mc_O,PolDeg_mc_O,ErrPolDeg_mc_O,PolAngle_mc_O,ErrPolAngle_mc_O,ra2000_mc_deg_O,dec2000_mc_deg_O,index_O,NUMBER_O,MAG_AUTO_O,MAGERR_AUTO_O,FLUX_AUTO_O,FLUXERR_AUTO_O,FLUX_APER_O,FLUXERR_APER_O,MAG_APER_O,MAGERR_APER_O,X_IMAGE_O,Y_IMAGE_O,ALPHA_J2000_O,DELTA_J2000_O,FLAGS_O,CLASS_STAR_O,FWHM_IMAGE_O,FWHM_WORLD_O,ELONGATION_O,ELLIPTICITY_O,DISTANCE_DEG_O,
            # index_E,id_mc_E,id_blazar_mc_E,aper_mc_E,IAU_name_mc_E,ra2000_mc_E,dec2000_mc_E,name_mc_E,Rmag_mc_E,Rmagerr_mc_E,PolDeg_mc_E,ErrPolDeg_mc_E,PolAngle_mc_E,ErrPolAngle_mc_E,ra2000_mc_deg_E,dec2000_mc_deg_E,index_E,NUMBER_E,MAG_AUTO_E,MAGERR_AUTO_E,FLUX_AUTO_E,FLUXERR_AUTO_E,FLUX_APER_E,FLUXERR_APER_E,MAG_APER_E,MAGERR_APER_E,X_IMAGE_E,Y_IMAGE_E,ALPHA_J2000_E,DELTA_J2000_E,FLAGS_E,CLASS_STAR_E,FWHM_IMAGE_E,FWHM_WORLD_E,ELONGATION_E,ELLIPTICITY_E,DISTANCE_DEG_E,RA_J2000_O,DEC_J2000_O,RA_J2000_E,DEC_J2000_E,
            # APERPIX,FWHM,SECPIX,DATE-OBS,MJD-OBS,RJD-50000,EXPTIME,ANGLE,MAGZPT

            # Taking info about ordinary and extraordinary sources...
            params = ['cal_id', 'blazar_id','name','name_IAU','Telescope'] # Parameters got from database registers
            params += ['date_run', 'mjd_obs', 'rjd-50000', 'pol_angle', \
                           'aperpix', 'fwhm', 'secpix', 'exptime', 'magzpt']
            
            photo_params = ['mag_auto', 'magerr_auto', 'flux_auto', 'fluxerr_auto', \
                                'flux_aper', 'fluxerr_aper', 'mag_aper', 'magerr_aper', \
                                'x_image', 'y_image', 'alpha_j2000', 'delta_j2000', 'flags', \
                                'class_star', 'fwhm_image', 'fwhm_world', 'elongation', 'ellipticity', \
                                'distance_deg']
            
            # String parameters
            str_params = ['date_run','name', 'name_IAU','source_type', 'Telescope']
            
            blazar_keywords = ['MJD-OBS', 'RJD-50000', 'ANGLE', \
                                   'APERPIX', 'FWHM', 'SECPIX', 'EXPTIME', 'MAGZPT']
            blazar_photometry_keywords = ['MAG_AUTO', 'MAGERR_AUTO', 'FLUX_AUTO', \
                                              'FLUXERR_AUTO', 'FLUX_APER', 'FLUXERR_APER', 'MAG_APER', \
                                              'MAGERR_APER', 'X_IMAGE', 'Y_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', \
                                              'FLAGS', 'CLASS_STAR', 'FWHM_IMAGE', 'FWHM_WORLD', 'ELONGATION', \
                                              'ELLIPTICITY', 'DISTANCE_DEG']

            # There is only one line for each photometry file
            # mjd_obs = photometry_data['MJD-OBS']
            # try:
            #     r_date = get_run_date(photometry_data['MJD-OBS'].values[0])
            # except:
            #     print(photometry_data['MJD-OBS'])
            #     raise

                
            cal_id=cal_id
            values = [cal_id, blazar_id, blazar_name, blazar_name_IAU,telescope, dt.strftime("%Y-%m-%d")]
            values += [photometry_data[k].values[1] for k in blazar_keywords]

            # Checking previous information on blazar in photometry table...
            sql_search_cal_id = f"SELECT `cal_id` FROM `photometry_reference_stars` WHERE `cal_id`={cal_id} AND date_run='{run_date}'"
            db_cursor.execute(sql_search_cal_id)
            res_search_cal_id = db_cursor.fetchall()
        
            if db_cursor.rowcount == 0:
                # Insert new register
                for t in ['O', 'E']:
                    par = params + photo_params + ['source_type']
                    vals = values + [photometry_data[k + f'_{t}'].values[1] for k in blazar_photometry_keywords] + [t]
                    # formatting values
                    v = ','.join([f"'{va}'" if p in str_params else f"{va}" for p, va in zip(par, vals)])
                    print(v)
                    # Inserting new register on database.photometry
                
                    sql_insert = f"INSERT INTO `photometry_reference_stars` (`{'`,`'.join(par)}`) VALUES ({v})"
                    print(f"SQL command (photometry) = '{sql_insert}'")

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
            else:
                # Update register
                for t in ['O', 'E']:
                    par = params + photo_params + ['source_type']
                    vals = values + [photometry_data[k + f'_{t}'].values[0] for k in blazar_photometry_keywords] + [t]
                # formatting values
                    val_fmt = [f"'{val}'" if par in str_params else f"{val}" for par, val in zip(par, vals)]
                    v = ','.join([f"'{va}'" if p in str_params else f"{va}" for p, va in zip(par, vals)])
                    pairs = []
                    for k, j in zip(par, val_fmt):
                        pairs.append(f'`{k}` = {j}')
                    sql = f"INSERT INTO `photometry_reference_stars` (`{'`,`'.join(par)}`) VALUES ({v}) ON DUPLICATE KEY UPDATE {','.join(pairs)}"
        
                    print(f"SQL command (photometry) = {sql}")
                    try:
                        db_cursor.execute(sql)
                    except:
                        print(f"SQL ERROR: {sql}")
                        raise
                    # Commiting insertion
                    db_object.commit()
                    if db_cursor.rowcount > 0:
                        print(f"INFO: {db_cursor.rowcount} register/s inserted successfully.")    
    return new_registrations


def register_polarimetry_data(data_dir, run_date, db_object, telescope):
    """
    Register for each value of polarization and angle of polarization for
    each blazar observed at run_date.

    Args:
        data_dir (string): base directory. FITS are stored in data_dir/raw directory.
        run_date (string): format given by pattern YYMMDD where YY is year, 
            MM is month and DD y day.
        db_object (mysql.connector.db object): it manages database connection.

    Returns:
        tuple: (num_insertions, num_updates)
            where 
                - 'num_insertions' represents number of polarimetrical computations 
            registered in database.
                - 'num_updates', is the number of updates registed in database.
    
    Exception:
        IOError, IndexError: 
            if polarimetry catalog file was not found.
    """
    new_registrations = 0
    updates = 0

    # Create cursor for database operation
    db_cursor = db_object.cursor()
    
    dt = datetime.strptime(run_date, '%y%m%d')
    r_date = dt.strftime("%Y-%m-%d")
    #catalog_dir = data_dir.replace('raw', 'final') 
    #catalog_dir = os.path.join(catalog_dir, f'{run_date}/')
    catalog_dir = data_dir
    catalog = os.path.join(catalog_dir, f'{telescope}_polR_{r_date}.csv')
    
    try:
        pol_data = pd.read_csv(catalog)
        print(f'Number of polarimetry data = {len(pol_data.index)}')
        print('*' * 60)
    except IOError:
        print(f"ERROR: polarimetry catalog '{catalog}' not found.")
        raise
    
    params = ['blazar_id', 'Telescope','date_run', \
        'rjd-50000','mjd_obs', 'name','alternative_name', 'P', 'dP', 'Theta', 'dTheta', 'R', 'dR', \
        'Q', 'dQ', 'U', 'dU', 'exptime', 'aperpix', 'aperas', 'num_angles', 'flux_std_mean_ratio', 'flag']

    str_params = ['name', 'alternative_name','date_run', 'Telescope']

    pol_keywords = ['DATE_RUN', 'RJD-50000','MJD-OBS', 'MC-IAU-NAME','MC-NAME', \
        'P', 'dP', 'Theta', 'dTheta', 'R', 'Sigma', \
        'Q', 'dQ', 'U', 'dU', 'EXPTIME', 'APERPIX', 'APERAS', 'NUM_ROTATION', 'flux_std_mean_ratio', 'flag']
        
    flag_manual=False #Set the manual flagginf to false by default

    # Checking previous information on blazar in photometry table...
    for index, row in pol_data.iterrows():
        print(f'index = {index}')
        # print(f'row ------------\n {row}')
        # getting blazar_id
        blazar_name = row['MC-IAU-NAME'].strip()
        blazar_mc_name = row['MC-NAME']
        sql_search_bl_id = f"SELECT `id` FROM `blazar_source` WHERE `name_IAU` like '{blazar_name}'"
        print(f"sql_search = {sql_search_bl_id}")
        db_cursor.execute(sql_search_bl_id)
        res_search_bl_id = db_cursor.fetchall()
        
        blazar_id = None
        if len(res_search_bl_id) == 1:
            blazar_id = res_search_bl_id[0][0]
        else:
            print(f"ERROR: No found blazar_id for blazar called '{blazar_name}' in polarimetry info row.")
            return -3 
        
        sql_search_pol_data = f"SELECT * FROM `polarimetry` WHERE `blazar_id` = {blazar_id} AND `date_run` = '{r_date}' AND (ABS(`rjd-50000` - {row['RJD-50000']}) < 0.0001)"
        print(f'sql_search = {sql_search_pol_data}')
        db_cursor.execute(sql_search_pol_data)
        res_search_pol_data = db_cursor.fetchall()
        print(f'res_search_pol_data = {res_search_pol_data}')

        sql = ''
        if len(res_search_pol_data) == 0:
            # Insert new register
            values = [blazar_id, telescope]  + [row[k] for k in pol_keywords]

            v = ','.join([f"'{val}'" if par in str_params else f"{val}" for par, val in zip(params, values)])

            # Inserting new register on database.image_raw
            sql = f"INSERT INTO `polarimetry` (`{'`,`'.join(params)}`) VALUES ({v})"
            new_registrations += 1
        else:
            values = [blazar_id, telescope] + [row[k] for k in pol_keywords]
            val_fmt = [f"'{val}'" if par in str_params else f"{val}" for par, val in zip(params, values)]
            
            pairs = []
            for k, v in zip(params, val_fmt):
                pairs.append(f'`{k}` = {v}')
            sql = f"UPDATE `polarimetry` SET {','.join(pairs)} WHERE (ABS(`rjd-50000` - {row['RJD-50000']}) < 0.0001)"
            updates += 1
        
        print(f'sql insert/update = {sql}')
        try:
            db_cursor.execute(sql)
        except:
            print(f"SQL WARNING: {sql}")
            # raise
        
        # Commiting insertion
        db_object.commit()
        

        # if new_registrations > 1:
        #     break
    
    return new_registrations, updates

def register_polarimetry_refstars_data(data_dir, run_date, db_object, telescope):
    """
    Register for each value of polarization and angle of polarization for
    each blazar observed at run_date.

    Args:
        data_dir (string): base directory. FITS are stored in data_dir/raw directory.
        run_date (string): format given by pattern YYMMDD where YY is year, 
            MM is month and DD y day.
        db_object (mysql.connector.db object): it manages database connection.

    Returns:
        tuple: (num_insertions, num_updates)
            where 
                - 'num_insertions' represents number of polarimetrical computations 
            registered in database.
                - 'num_updates', is the number of updates registed in database.
    
    Exception:
        IOError, IndexError: 
            if polarimetry catalog file was not found.
    """
    new_registrations = 0
    updates = 0

    # Create cursor for database operation
    db_cursor = db_object.cursor()
    
    dt = datetime.strptime(run_date, '%y%m%d')
    r_date = dt.strftime("%Y-%m-%d")
    #catalog_dir = data_dir.replace('raw', 'final') 
    #catalog_dir = os.path.join(catalog_dir, f'{run_date}/')
    catalog_dir = data_dir
    catalog = os.path.join(catalog_dir, f'{telescope}_polR_{r_date}_reference_stars.csv')
    
    try:
        pol_data = pd.read_csv(catalog)
        print(f'Number of polarimetry data = {len(pol_data.index)}')
        print('*' * 60)
    except IOError:
        print(f"ERROR: polarimetry catalog '{catalog}' not found.")
        raise
    
    params = ['blazar_id', 'Telescope','date_run', \
        'rjd-50000', 'mjd_obs', 'name', 'alternative_name', 'Rmag_lit','P', 'dP', 'Theta', 'dTheta', 'R', 'dR', \
        'Q', 'dQ', 'U', 'dU', 'exptime', 'aperpix', 'aperas', 'num_angles', 'flux_std_mean_ratio', 'flag']

    str_params = ['name', 'alternative_name', 'date_run']

    pol_keywords = ['DATE_RUN', 'RJD-50000','MJD-OBS', 'MC-IAU-NAME','MC-NAME','RMAG-LIT', \
        'P', 'dP', 'Theta', 'dTheta', 'R', 'Sigma', \
        'Q', 'dQ', 'U', 'dU', 'EXPTIME', 'APERPIX', 'APERAS', 'NUM_ROTATION', 'flux_std_mean_ratio', 'flag']
        
    # Checking previous information on blazar in photometry table...
    for index, row in pol_data.iterrows():
        print(f'index = {index}')
        # print(f'row ------------\n {row}')
        # getting blazar_id
        blazar_name = row['MC-NAME']
        blazar_mc_name = row['MC-NAME']
        sql_search_bl_id = f"SELECT `id` FROM `blazar_source` WHERE `name` like '{blazar_name}'"
        print(f"sql_search = {sql_search_bl_id}")
        db_cursor.execute(sql_search_bl_id)
        res_search_bl_id = db_cursor.fetchall()
        
        blazar_id = None
        if len(res_search_bl_id) == 1:
            blazar_id = res_search_bl_id[0][0]
        else:
            print(f"ERROR: No found blazar_id for blazar called '{blazar_name}' in polarimetry info row.")
            return -3 
        
        sql_search_pol_data = f"SELECT * FROM `polarimetry_reference_stars` WHERE `blazar_id` = {blazar_id} AND `date_run` = '{r_date}' AND (ABS(`rjd-50000` - {row['RJD-50000']}) < 0.0001)"
        print(f'sql_search = {sql_search_pol_data}')
        db_cursor.execute(sql_search_pol_data)
        res_search_pol_data = db_cursor.fetchall()
        print(f'res_search_pol_data = {res_search_pol_data}')

        sql = ''
        if len(res_search_pol_data) == 0:
            # Insert new register
            values = [blazar_id, f'"{telescope}"'] + [row[k] for k in pol_keywords]

            v = ','.join([f"'{val}'" if par in str_params else f"{val}" for par, val in zip(params, values)])

            # Inserting new register on database.image_raw
            sql = f"INSERT INTO `polarimetry_reference_stars` (`{'`,`'.join(params)}`) VALUES ({v})"
            new_registrations += 1
        else:
            values = [blazar_id, f'"{telescope}"'] + [row[k] for k in pol_keywords]
            val_fmt = [f"'{val}'" if par in str_params else f"{val}" for par, val in zip(params, values)]
            
            pairs = []
            for k, v in zip(params, val_fmt):
                pairs.append(f'`{k}` = {v}')
            
            sql = f"UPDATE `polarimetry_reference_stars` SET {','.join(pairs)} WHERE (ABS(`rjd-50000` - {row['RJD-50000']}) < 0.0001)"
            updates += 1
        
        print(f'sql insert/update = {sql}')
        
        try:
            db_cursor.execute(sql)
        except:
            print(f"SQL WARNING, REGISTRATION FAILED: {sql}")
            new_registrations = 0
            # raise
        
        # Commiting insertion
        db_object.commit()
        

        # if new_registrations > 1:
        #     break
    
    return new_registrations, updates



def main():
    parser = argparse.ArgumentParser(prog='iop3_add_db_info.py', \
    conflict_handler='resolve',
    description='''Read CSV files from run_date and insert new info in database.''',
    epilog='')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    # parser.add_argument("dbengine", help="Configuration parameter files directory")
    parser.add_argument("input_data_dir", help="Input data base directory")
    parser.add_argument("run_date", help="Run date in format YYMMDD")
    parser.add_argument("telescope", help="Input telescope name")
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
    telescope = args.telescope
        
    directories = {k: os.path.join(args.input_data_dir, f'{k}/{telescope}/{args.run_date}/') for k in dirs}
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
        user=db_user, password=db_password, database=args.db_name, ssl_disabled=True)

    my_cursor = my_db.cursor()

    # Selecting database for table creation
    my_cursor.execute(f"USE {args.db_name};")
    
    # return 1

    # Processing RAW images (image_raw table) from run date
    res = register_raw(directories['raw'], args.run_date, my_db, telescope)
    print(f"Raw image registration result -> {res}")
    
    # return 1

    # Processing MASTERBIAS (master_bias table) from run date 
    res = register_masterbias(directories['reduction'], args.run_date, my_db, telescope)
    print(f"MasterBIAS registration result -> {res}")

    # return 1

    # Processing RAW-BIAS (raw_bias table) from run date
    res = register_rawbias(directories['reduction'], args.run_date, my_db, telescope)
    print(f"Relation between raw and masterBIAS registration result -> {res}")

    # return 1

    # Processing MASTERFLATS (master_flat table) from run date
    res = register_masterflats(directories['reduction'], args.run_date, my_db, telescope)
    print(f"MasterFLATS registration result -> {res}")
    
    # return 1

    # Processing RAW-FLATS (raw_flat table) from run date
    res = register_rawflats(directories['reduction'], args.run_date, my_db, telescope)
    print(f"Relation between raw and masterFLATS registration result -> {res}")
    
    # return 1

    # Processing REDUCED images (image_reduced table) from run date
    res = register_reduced(directories['reduction'], args.run_date, my_db, telescope)
    print(f"Reduced image registration result -> {res}")
    
    # return 1

    # Processing CALIBRATED images (image_calibrated table) from run date
    res = register_calibrated(directories['calibration'], args.run_date, my_db, telescope)
    print(f"Calibrated image registration result -> {res}")
    
    # return 1

    # Processing BLAZAR MEASURE (photometry table) from run date
    res = register_photometry(directories['calibration'], args.run_date, my_db, telescope)
    print(f"blazar data registration result -> {res}")

    res = register_photometry_refstars(directories['calibration'], args.run_date, my_db, telescope)
    print(f"blazar data registration result -> {res}")
    # return 1

    # Processing POLARIMETRY DATA (polarimetry_data table) from run date
    res = register_polarimetry_data(directories['final'], args.run_date, my_db, telescope)
    print(f"Polarimetry data registration result -> {res}")

    res = register_polarimetry_refstars_data(directories['final'], args.run_date, my_db, telescope)
    print(f"Polarimetry data registration result -> {res}")

    
    # return 1

    # # Unuseful table. Info can be get by querying...
    # """SELECT 
    #     bm.cal_id, bm.date_run, bm.class_star, bm.source_type, 
    #     bm.ellipticity, bm.flags, bm.date_obs, bm.pol_angle, 
    #     pd.name, pd.P, pd.dP, pd.Theta, pd.dTheta, pd.R, pd.dR 
    # FROM 
    #     polarimetry_data as pd 
    #     INNER JOIN 
    #         photometry as p 
    #     ON 
    #         pd.blazar_id=p.blazar_id 
    # ORDER BY pd.name, p.date_obs;"""
    # # Processing BLAZAR_POLARIMETRY (blazar_polarimetry table) from run date
    # # res = register_polarimetry_blazar(args.input_data_dir, args.run_date, my_db)
    # # print(f"Relations between polarimetry measures and blazars result -> {res}")



    return 0

if __name__ == '__main__':
    print(main())
