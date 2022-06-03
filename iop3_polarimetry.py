#!/usr/bin/env python3

import sys
from typing import DefaultDict

from astropy.utils import data
sys.path.append('/home/cesar/entornos/py38/lib/python3.8/site-packages')
import argparse
import os
import re
import math
import glob
import pprint

import numpy as np
import pandas as pd
from collections import defaultdict

from astropy.io import fits

# Coordinate system transformation package and modules
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import match_coordinates_sky  # Used for searching sources in catalog
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
import astropy.coordinates as coord
import astropy.units as u
# Astrotime
from astropy.time import Time


def subsets(data, num_angles=4):
    """It analyzes data passed and maje subsets of observations according to date-obs
    and rotator angle.
    Args:
        data (pandas.DataFrame): Data from objects taken from 4 polarization angles grism.
        name (str): object name for subsetting

    Returns:
        list: of valid subsets of observation for object called 'name'.
    """
    sub_s = []
    # taking angles
    angles_rot = sorted(data['ANGLE'].unique().tolist())
    print(f"ROTATED ANGLES = {angles_rot}")
    if len(angles_rot) < 4:
        message = 'POLARIMETRY,WARNING,"Not complete rotation angles (only {}) measurements for object {}"'
        print(message.format(angles_rot, data['MC-IAU-NAME'].iloc[0]))
    
    if len(data.index) > 4:
        message = 'POLARIMETRY,INFO,"More than 4 observations taken this run for object {}"'
        print(message.format(data['MC-IAU-NAME'].iloc[0]))
    
    # sort by datetime
    data.sort_values(by=['DATE-OBS', 'ANGLE', 'TYPE'], \
        inplace=True, ascending=False)
    
    # search for duplicates
    while len(data.index) > 0: # iterate until there is no more observations
        print(f'data elements = {len(data.index)}')
        index_dup = data.duplicated(['ANGLE', 'TYPE'], keep='last') # false for last duplicated (angle, type) item
        sub_s.append(data[~index_dup])  # append last set of duplicated items to list
        data = data[index_dup] # delete previous last repeated set of observations
    
    return sub_s
        
        
        

def object_measures(data, name):
    """"It returns a list of data subsets, according to object 'name'.
    
    The number of subsets depends on the number of series observation taken this 
    night for this object called 'name'. The usual number of elements for each subset
    is equal to number of angles set for the instrument (usually 4).
    
    Args:
        data (pandas.DataFrame): Data from objects taken from 4 polarization angles grism.
        name (str): object name for subsetting

    Returns:
        list: of valid subsets of observation for object called 'name'.
    """
    data_sets = []
    print(f"***** Processing object called '{name}' ********")
    
    # Filtering to 'name' object measurements
    # data_object = data[data['NAME'] == name]
    data_object = data[data['MC-IAU-NAME'] == name]

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

def polarimetry(df):
    """Compute polarimetric parameters.
    Args:
        df (pandas.DataFrame): Data from object taken from 4 polarization angles.

    Returns:
        tuple: (P, dP, Theta, dTheta)

    Formula taken from "Zapatero_Osorio_2005_ApJ_621_445.pdf"
    """
    
    # print("Ordinary data")
    # print(df_o[['MJD-OBS', 'DATE-OBS', 'MC-IAU-NAME', 'FLUX_APER', 'FLUXERR_APER']])

    # print("Extraordinary data")
    # print(df_e[['MJD-OBS', 'DATE-OBS', 'MC-IAU-NAME', 'FLUX_APER', 'FLUXERR_APER']])

    #print(df_o['FLUX_AUTO'][df_o[key_polangle] == '0.0'].values[0])
    # print('Ordinary =')
    # print(df_o)
    # print(df_o.info())

    all_angles = df['ANGLE'].unique().tolist()
    
    if len(all_angles) == 1:
        return [None] * 8 
    elif len(all_angles) == 2:
        # valid combinations are: 0 and 22.5 or 45 and 67.5
        if (0 in all_angles and 45 in all_angles) or (22.5 in all_angles and 67.5 in all_angles):
            return [None] * 8
    
    # aliases
    fo = df['FLUX_APER_O']
    fe = df['FLUX_APER_E']
    ferr_o = df['FLUXERR_APER_O']
    ferr_e = df['FLUXERR_APER_E']
    
    a0 = df['ANGLE'] == 0
    a22 = df['ANGLE'] == 22.5
    a45 = df['ANGLE'] == 45
    a67 = df['ANGLE'] == 67.5
    
    # ORDINARY measurements
    if a0.sum() == 0: # No measures in rotator angle equals to 0
        # taking data from 45
        o_0 = fe[a45].iloc[0]
        oe_0 = ferr_e[a45].iloc[0]
        e_0 = fo[a45].iloc[0]
        ee_0 = ferr_o[a45].iloc[0]
    else:
        o_0 = fo[a0].iloc[0]
        oe_0 = ferr_o[a0].iloc[0]
        e_0 = fe[a0].iloc[0]
        ee_0 = ferr_e[a0].iloc[0]

    if a22.sum() == 0: # No measures in rotator angle equals to 22.5
        # taking data from 67.5
        o_22 = fe[a67].iloc[0]
        oe_22 = ferr_e[a67].iloc[0]
        e_22 = fo[a67].iloc[0]
        ee_22 = ferr_o[a67].iloc[0]
    else:
        o_22 = fo[a22].iloc[0]
        oe_22 = ferr_o[a22].iloc[0]
        e_22 = fe[a22].iloc[0]
        ee_22 = ferr_e[a22].iloc[0]
        
    if a45.sum() == 0: # No measures in rotator angle equals to 45
        # taking data from 0
        o_45 = fe[a0].iloc[0]
        oe_45 = ferr_e[a0].iloc[0]
        e_45 = fo[a0].iloc[0]
        ee_45 = ferr_o[a0].iloc[0]
    else:
        o_45 = fo[a45].iloc[0]
        oe_45 = ferr_o[a45].iloc[0]
        e_45 = fe[a45].iloc[0]
        ee_45 = ferr_e[a45].iloc[0]

    if a67.sum() == 0: # No measures in rotator angle equals to 67.5
        # taking data from 22.5
        o_67 = fe[a22].iloc[0]
        oe_67 = ferr_e[a22].iloc[0]
        e_67 = fo[a22].iloc[0]
        ee_67 = ferr_o[a22].iloc[0]
    else:
        o_67 = fo[a67].iloc[0]
        oe_67 = ferr_o[a67].iloc[0]
        e_67 = fe[a67].iloc[0]
        ee_67 = ferr_e[a67].iloc[0]

    # str_out = '{} (0, 22, 45, 67) = ({}, {}, {}, {})'
    # print(str_out.format('Ord', o_0, o_22, o_45, o_67))
    # print(str_out.format('Ext', e_0, e_22, e_45, e_67))

    try:
        RQ = math.sqrt((o_0 / e_0) / (o_45 / e_45))
    except:
        print(f"ERROR: computing RQ = math.sqrt((o_0 / e_0) / (o_45 / e_45))")
        print(f"(o_0, e_0, o_45, e_45) = ({o_0}, {e_0}, {o_45}, {e_45})")
        raise
    
    try:
        dRQ = RQ * math.sqrt((oe_0 / o_0) ** 2 + (ee_0 / e_0) ** 2 + \
            (oe_45 / o_45) ** 2 + (ee_45 / e_45) ** 2)
    except:
        print(f"ERROR: computing dRQ = RQ * math.sqrt((oe_0/o_0) ** 2 + (ee_0 / e_0) ** 2 + \
            (oe_45 / o_45) ** 2 + (ee_45 / e_45) ** 2)")
        print(f"(oe_0, o_0, ee_0, e_0) = ({oe_0}, {o_0}, {ee_0}, {e_0})")
        print(f"(oe_45, o_45, ee_45, e_45) = ({oe_45}, {o_45}, {ee_45}, {e_45})")
        raise
    
    try:
        RU = math.sqrt((o_22 / e_22) / (o_67 / e_67))
    except:
        print(f"ERROR:computing RU = math.sqrt((o_22 / e_22) / (o_67 / e_67))")
        print(f"(o_22, e_22, o_67, e_67) = ({o_22}, {e_22}, {o_67}, {e_67})")
        raise
    
    try:
        dRU = RU * math.sqrt((oe_22 / o_22) ** 2 + (ee_22 / e_22) ** 2 + \
            (oe_67 / o_67) ** 2 + (ee_67 / e_67) ** 2)
    except:
        print(f"ERROR: computing dRU = RU * math.sqrt((oe_22 / o_22) ** 2 + (ee_22 / e_22) ** 2 + \
            (oe_67 / o_67) ** 2 + (ee_67 / e_67) ** 2)")
        print(f"(oe_22, o_22, ee_22, e_22) = ({oe_22}, {o_22}, {ee_22}, {e_22})")
        print(f"(oe_67, o_67, ee_67, e_67) = ({oe_67}, {o_67}, {ee_67}, {e_67})")
        raise

    val_str = '(RQ+-dRQ, RU+-dRU) = ({}+-{}, {}+-{})'
    # print(val_str.format(RQ, dRQ, RU, dRU))

    Q_I = (RQ - 1) / (RQ + 1)
    dQ_I = Q_I * math.sqrt(2 * (dRQ / RQ) ** 2)
    U_I = (RU - 1) / (RU + 1)
    dU_I = U_I * math.sqrt(2 * (dRU / RU) ** 2)

    # val_str = '(Q_I +- (dQ_I), U_I +- (dU_I)) = ({} +- ({}), {} +- ({}))'
    # print(val_str.format(Q_I, dQ_I, U_I, dU_I))

    try:
        P = math.sqrt(Q_I ** 2 + U_I ** 2)
        dP = P * math.sqrt((dRQ / RQ) ** 2 + (dRU / RU) ** 2) / 2
    except ZeroDivisionError:
        # print(f"(Q_I, U_I, RQ, dRQ, RU, dRU) = ({Q_I}, {U_I}, {RQ}, {dRQ}, {RU}, {dRU})")
        raise

    try:
        Theta_0 = 0
        
        if Q_I >= 0:
            Theta_0 = math.pi 
            if U_I > 0:
                Theta_0 = -1 * math.pi
            # if Q_I < 0:
            #     Theta_0 = math.pi / 2
        print(f'Theta_0 = {Theta_0}')
        Theta = 0.5 * math.degrees(math.atan(U_I / Q_I) + Theta_0)
        dTheta = dP / P * 28.6
    except ZeroDivisionError:
        # print(f"(U_I, Q_I, P, dP) = ({U_I}, {Q_I}, {P}, {dP})")
        raise

    # pol_vals = 'P = {}, dP = {} \nTheta = {}, dTheta = {}'
    # print(pol_vals.format(P * 100, dP * 100, Theta, dTheta))

    return P, dP, Theta, dTheta, RQ, dRQ, RU, dRU

def compute_polarimetry(data_object):
    """Given input data, it calls polarimetry function to get
    polarimetric magnitudes.

    Args:
        data_object (pandas.DataFrame): Data from object taken from 4 polarization angles.

    Returns:
        dict: Keywords are [P, dP, Theta, dTheta, R, Sigma, RJD-5000, ID-MC, ID-BLAZAR-MC, MC-NAME, MC-IAU-NAME, NAME]
    
    """
    result = DefaultDict()
    
    name = data_object['IAU_name_mc_O'].values[0]
    dates = sorted(data_object['RJD-50000'].unique().tolist())
    # print(len(data_object.index))
    # print(f'dates = {dates}')
    index_date = 2 # 3rd observation () for group of 3 o 4 observations
    if len(dates) == 1:
        index_date = 0
    elif len(dates) == 2:
        index_date = 1
    obs_date = dates[index_date] 
    
    result['RJD-5000'] = round(obs_date, 4)
    result['ID-MC'] = data_object['id_mc_O'].values[0]
    result['ID-BLAZAR-MC'] = data_object['IAU_name_mc_O'].values[0]
    result['MC-NAME'] = data_object['name_mc_O'].values[0]
    result['EXPTIME'] = data_object['EXPTIME'].values[0] # .split()[0]      
    result['NUM_ROTATION'] = len(data_object.index)
    
    # Computing polarimetry parameters
    # print(data_object)
    # print(data_object.info())
    # print(data_object)
    try:
        P, dP, Theta, dTheta, RQ, dRQ, RU, dRU = polarimetry(data_object)
    except ZeroDivisionError:
        print(f'\tZeroDivisionError while processing object called "{name}"')
        raise
    
    if P is None:
        result['P'] = P
        result['dP'] = dP
        result['Theta'] = Theta
        result['dTheta'] = dTheta
        result['R'] = None
        result['Sigma'] = None
        result['Q'] = RQ
        result['dQ'] = dRQ
        result['U'] = RU
        result['dU'] = dRU
        
        return result
    
    
    # HOW DO I COMPUTE MAGNITUDE AND ITS ERROR?
    # m = mean(C_i - 2.5 * log10(FLUX_ISO_O + FLUX_ISO_E))
    # e_m = std(C_i - 2.5 * log10(FLUX_ISO_O + FLUX_ISO_E))
    # print(data_object.info())
    fluxes = data_object['FLUX_APER_O'] + data_object['FLUX_APER_E']
    try:
        mags = data_object['MAGZPT'].values - 2.5 * np.log10(fluxes.values)
    except:
        obs_datetimes = sorted(data_object['DATE-OBS'].unique().tolist())
        print(f'POLARIMETRY,ERROR,"Processing data from {name}, date-obs ={obs_datetimes}')
        print(f'\tzps = {data_object["MAGZPT"].values}')
        print(f'\tFlux_apers\n{data_object[["FLUX_APER_O", "FLUX_APER_E"]]}')
        raise

    result['P'] = round(P * 100, 3)
    result['dP'] = round(dP * 100, 3)
    result['Theta'] = round(Theta, 2)
    result['dTheta'] = round(dTheta, 2)
    result['R'] =  round(mags.mean(), 2)
    result['Sigma'] = round(max([data_object['MAGERR_APER_O'].values.max(), \
        data_object['MAGERR_APER_E'].values.max()]), 4)
    result['Q'] = round(RQ, 4)
    result['dQ'] = round(dRQ, 4)
    result['U'] = round(RU, 4)
    result['dU'] = round(dRU, 4)

    return result


def make_groups(data_res):
    """Make groups of eight (or less) observations based on DATE-OBS and blazar/star name.
    Each date (4 different or less) have measures for Ordinary & Extraodinary sources.

    Args:
        data_res (pd.DataFrame): Astro-photometric calibration and target measurements results.

    Returns:
        list: Each item is a pandas Dataframe with 
    """
    groups = []
    # sort by MJD-OBS ascending
    df = data_res.sort_values(by=['MJD-OBS'], ascending=True)
    df = df.reset_index()

    df_o = df[df['TYPE'] == 'O']
    df_o = df_o.reset_index()
    df_e = df[df['TYPE'] == 'E']
    df_e = df_e.reset_index()

    # print('Ordinary measurements...')
    # print(df_o.loc[:,['ANGLE', 'TYPE', 'EXPTIME', 'MC-IAU-NAME', 'DATE-OBS', 'FLUX_APER', 'FLUXERR_APER']])
    # print("")
    # Working on ordinary subset. Indexes are common by Ordinary/Extraordinary sets.
    prev_blazar_name = None
    prev_angle = None
    angles = DefaultDict(int)
    indexes = []
    # print(df)
    for j, row in df_o.iterrows():
        print('')
        indexes = list(angles.values())

        if prev_blazar_name is None:
            # adding new observation
            # prev_blazar_name = row['MC-IAU-NAME']
            # prev_angle = row['ANGLE']
            # angles[row['ANGLE']] = j 
            pass
        # print(f'row = {row}')
        elif (row['ANGLE'] == prev_angle) and (row['MC-IAU-NAME'] == prev_blazar_name):
            # repeated observation. Overwrite previous one
            # angles[row['ANGLE']] == j
            pass
        elif (row['MC-IAU-NAME'] != prev_blazar_name) or row['ANGLE'] < prev_angle: # (len(indexes) > 3):
            # saving last group
            indexes = list(angles.values())
            # print(f'indexes = {indexes}')
            df_group = pd.concat([df_o.iloc[indexes], df_e.iloc[indexes]])
            df_group.sort_values(by=['DATE-OBS'], ascending=True)
            # print(df_group.loc[:,['ANGLE', 'TYPE', 'EXPTIME', 'MC-IAU-NAME', 'DATE-OBS', 'FLUX_APER', 'FLUXERR_APER']])
            groups.append(df_group)
            angles.clear()
            
        # In any case:
        # 1. Updating previous row data
        prev_blazar_name = row['MC-IAU-NAME']
        prev_angle = row['ANGLE']
        #  2. adding or updating indexes
        angles[row['ANGLE']] = j 

    if angles:
        # print('')
        indexes = list(angles.values())
        df_group = pd.concat([df_o.iloc[indexes], df_e.iloc[indexes]])
        # print(df_group.loc[:,['ANGLE', 'TYPE', 'EXPTIME', 'MC-IAU-NAME', 'DATE-OBS', 'FLUX_APER', 'FLUXERR_APER']])
        groups.append(df_group)

    return groups

def make_groups2(data_res):
    """Make groups of eight (or less) observations based on DATE-OBS and blazar/star name.
    Each date (4 different or less) have measures for Ordinary & Extraodinary sources.

    Args:
        data_res (pd.DataFrame): Astro-photometric calibration and target measurements results.

    Returns:
        list: Each item is a pandas Dataframe with 
    """
    groups = []

    # sort by MJD-OBS ascending
    df = data_res.sort_values(by=['RJD-50000'], ascending=True)
    df = df.reset_index()

    # print('Ordinary measurements...')
    # print(df_o.loc[:,['ANGLE', 'TYPE', 'EXPTIME', 'MC-IAU-NAME', 'DATE-OBS', 'FLUX_APER', 'FLUXERR_APER']])
    # print("")
    # Working on ordinary subset. Indexes are common by Ordinary/Extraordinary sets.
    prev_blazar_name = None
    prev_angle = None
    prev_exptime = None
    # angles = DefaultDict(int)
    indexes = []
    # print(df)
    for j, row in df.iterrows():
        print('')
        # indexes = list(angles.values())
        # print(f'row = {row}')
        if prev_blazar_name is None:
            # adding new observation
            # prev_blazar_name = row['MC-IAU-NAME']
            # prev_angle = row['ANGLE']
            # angles[row['ANGLE']] = j 
            pass
        elif (row['ANGLE'] == prev_angle) and (row['IAU_name_mc_O'] == prev_blazar_name) and \
            (row['EXPTIME'] == prev_exptime):
            # repeated observation. Overwrite previous one
            # angles[row['ANGLE']] == j
            pass
        elif (row['IAU_name_mc_O'] != prev_blazar_name) or row['ANGLE'] < prev_angle or \
            row['EXPTIME'] != prev_exptime: # (len(indexes) > 3):
            # saving last group
            # indexes = list(angles.values())
            # print(f'indexes = {indexes}')
            # df_group = pd.concat([df.iloc[indexes], df.iloc[indexes]])
            df_group = df.iloc[indexes]
            df_group.sort_values(by=['DATE-OBS'], ascending=True)
            # print(df_group.loc[:,['ANGLE', 'TYPE', 'EXPTIME', 'MC-IAU-NAME', 'DATE-OBS', 'FLUX_APER', 'FLUXERR_APER']])
            groups.append(df_group)
            # angles.clear()
            indexes.clear()
            
        # In any case:
        # 1. Updating previous row data
        prev_blazar_name = row['IAU_name_mc_O']
        prev_angle = row['ANGLE']
        prev_exptime = row['EXPTIME']
        #  2. adding or updating indexes
        # angles[row['ANGLE']] = j 
        indexes.append(j)

    #  if angles:
    if indexes:
        # print('')
        # indexes = list(angles.values())
        #df_group = pd.concat([df.iloc[indexes], df.iloc[indexes]])
        # print(df_group.loc[:,['ANGLE', 'TYPE', 'EXPTIME', 'MC-IAU-NAME', 'DATE-OBS', 'FLUX_APER', 'FLUXERR_APER']])
        #groups.append(df_group)
        df_group = df.iloc[indexes]
        df_group.sort_values(by=['DATE-OBS'], ascending=True)
        groups.append(df_group)

    return groups


def closest_object(ra_ref, dec_ref, ra_others, dec_others):
    """Check for objects close than min_dist_arcs.
    
    Args:
        ra_ref (_type_): _description_
        dec_ref (_type_): _description_
        ra_others (_type_): _description_
        dec_others (_type_): _description_
        
    Returns:
        tuple: (index_closest_object, distance_closest_object_arcs)
    """
    ref_coords = SkyCoord(ra_ref, dec_ref, frame=FK5, unit=(u.deg, u.deg), obstime="J2000")
    others_coords = SkyCoord(ra_others, dec_others, frame=FK5, unit=(u.deg, u.deg), obstime="J2000")
    
    # computing distances
    index, d2d, d3d  = match_coordinates_sky(ref_coords, others_coords, nthneighbor=1)
    
    d2d_arcsecs = d2d.deg * 3600.
    
    return index, d2d_arcsecs

def same_object(ras, decs, tol_arcs=5):
    """Check if input coordinates list corresponds to same object, with
    distance tolerance less or equal to 'tol_arcs' arcseconds.

    Args:
        ras (np.array): RA sources coordinates in degrees.
        decs (np.array): DEC sources coordinates in degrees.
        tol_arcs (float, optional): Threshold distance in arcsecs between coordinates. Defaults to 5.

    Returns:'ANGLE', 
        bool: True, if coordinates point same object. False in other case.
    """
       
    for j in range(int(len(ras) / 2)):
        ra_ref = ras[j]
        dec_ref = decs[j]
        other_ras = np.delete(ras, j)
        other_decs = np.delete(decs, j)
        
        ref_coords = SkyCoord(ra_ref, dec_ref, frame=FK5, unit=(u.deg, u.deg), obstime="J2000")
        others_coords = SkyCoord(other_ras, other_decs, frame=FK5, unit=(u.deg, u.deg), obstime="J2000")
    
        # computing distances
        index, d2d, d3d  = match_coordinates_sky(ref_coords, others_coords, nthneighbor=1)
        d2d_arcsecs = d2d.deg * 3600.

        if d2d_arcsecs > tol_arcs:
            return False
    
    return True

def check_group_coordinates(group, tol_arcs=5):
    """_summary_

    Args:
        group (_type_): _description_
        max_dist_arcs (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    
    same_o = same_object(group['ALPHA_J2000_O'].values, group['DELTA_J2000_O'].values, tol_arcs=tol_arcs)  
    same_e = same_object(group['ALPHA_J2000_E'].values, group['DELTA_J2000_E'].values, tol_arcs=tol_arcs)  
    
    if not same_o:
        print('Ordinary coordinates have too much variation. Not the same source.')
        return 1
    if not same_e:
        print('Extraordinary coordinates have too much variation. Not the same source.')
        return 2
    
    return 0

def main():
    parser = argparse.ArgumentParser(prog='iop3_polarimetry.py', \
    conflict_handler='resolve',
    description='''Main program. It searchs for every .res and final.fits files.
    Compute polarimetry parameters for each source (.res) and write
    output GLOBAL results file. ''',
    epilog='''''')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    parser.add_argument("calib_base_dir", help="Base directory for searching files.")
    parser.add_argument("output_dir", help="Output directory for storing results.")
    parser.add_argument('-v',
                        '--verbose',
                        action='count',
                        default=0,
                        help="Show running and progress information [default: %(default)s].")
    args = parser.parse_args()

    # processing .res files
    # results = glob.glob(os.path.join(args.calib_base_dir, '*-*/*_photocal_res.csv'))
    results = glob.glob(os.path.join(args.calib_base_dir, '*-*/*_photometry.csv'))
    # sort by name == sort by date (ASC)
    
    if not results:
        str_err = 'ERROR: No *_photometry.csv files found.'
        print(str_err)
        return 1

    results.sort()
    pprint.pprint(results)
    print(f"\nFound {len(results)} '*_photometry.csv' files.\n")

    # getting photocalibration process info, in order to get useful FITS parameters
    # cal_process = [r.replace('_photocal_res.csv', '_photocal_process_info.csv') for r in results]

    if not os.path.isdir(args.output_dir):
        try:
            os.makedirs(args.output_dir)
        except IOError:
            print(f'POLARIMETRY,ERROR,Could not create output directory {args.output_dir}"')
            return 2
    

    # Getting run date (input directory must have pattern like *yyyymmdd)
    dt_run = re.findall('(\d{6})', args.calib_base_dir)[-1]
    date_run = f'20{dt_run[:2]}-{dt_run[2:4]}-{dt_run[-2:]}'
    
    # Getting data from every VALID *_final.csv file into a new dataframe
    # data_proc = pd.concat([pd.concat([pd.read_csv(cp), pd.read_csv(cp)])  for cp in cal_process])
    data_res = pd.concat([pd.read_csv(r) for r in results])

    # Store photometry in CSV file, for further analysis
    data_res_csv = os.path.join(args.output_dir, f'photometry_{date_run}.csv')
    data_res.to_csv(data_res_csv, index=False)
    

    # print(data_res.info())
    # print(data_proc.info())

    # return -99

    # # append arcseconds per pixel info
    # data_res['SECPIX1'] = data_proc['SECPIX1'].values
    # data_res['SECPIX2'] = data_proc['SECPIX2'].values
    # print(data_res.info())
    # return -9

    # dictionary for storing results...
    pol_rows = []
    pol_data = DefaultDict(list)
    
    groups = make_groups2(data_res)
    for group in groups:
        name = group['IAU_name_mc_O'].values[0]
        print(f'GROUP {name}')
        print('-' * 30)
        print(group[['IAU_name_mc_O', 'DATE-OBS', 'ANGLE', 'EXPTIME', 'FLUX_APER_O', 'FLUX_APER_E', 'FLUXERR_APER_O', 'FLUXERR_APER_E']])
        print('')

        res = check_group_coordinates(group, tol_arcs=3)
        if res:
            print(res)
            c_o = SkyCoord(group['ALPHA_J2000_O'], group['DELTA_J2000_O'], \
                frame=FK5, unit=(u.deg, u.deg), obstime="J2000")
            c_e = SkyCoord(group['ALPHA_J2000_E'], group['DELTA_J2000_E'], \
                frame=FK5, unit=(u.deg, u.deg), obstime="J2000")
            
            print('Ordinary = ', c_o.ra.dms, c_o.dec.dms)
            print('Extraordinary = ', c_e.ra.dms, c_e.dec.dms)
            continue
        
        try:
            res_pol = compute_polarimetry(group)
            if res_pol['P'] is None:
                print('POLARIMETRY,WARNING,"Could not compute Polarization for this set of measurements"')
                continue
        except ZeroDivisionError:
            print('POLARIMETRY,ERROR,"EXCEPTION ZeroDivisionError: Found Zero Division Error in group processing.')
            raise
        except ValueError:
            print('POLARIMETRY,ERROR,"EXCEPTION ValueError: Found Value Error in group processing.')
            raise

        res_pol['DATE_RUN'] = date_run
        res_pol['EXPTIME'] = group['EXPTIME'].values[0]
        res_pol['APERPIX'] = group['APERPIX'].values[0]
        # arcsec per pixel as mean values of astrometric calibration computaion in RA and DEC.
        mean_secpix = round(np.nanmean(group['SECPIX'].values), 2)
        if np.isnan(mean_secpix):
            print(f'mean_secpix = {mean_secpix}')
            return -199
        
        res_pol['APERAS'] = res_pol['APERPIX'] * mean_secpix
        rp_sigma = res_pol['Sigma']
        # if rp_sigma < 0.01:
        #     rp_sigma = 0.01
        
        for k, v in res_pol.items():
            if k == 'Sigma':
                pol_data[k].append(rp_sigma)
                continue
            pol_data[k].append(v)

        index_obs = 2 # groups of three or four observations
        if len(group.index) <= 3:
            index_obs = 1
        obs_date = group['RJD-50000'].values[index_obs]
        pol_data['RJD-50000'].append(obs_date)
        row = [date_run, obs_date, group['IAU_name_mc_O'].values[0].strip()]
        
        row = row + [res_pol['P'], res_pol['dP'], \
            res_pol['Theta'], res_pol['dTheta'], \
            res_pol['Q'], res_pol['dQ'], \
            res_pol['U'], res_pol['dU'], \
            res_pol['R'], rp_sigma, \
            res_pol['APERPIX'], res_pol['APERAS'], \
            res_pol['NUM_ROTATION'], res_pol['EXPTIME']]
        pol_rows.append(row)

    # writing output night polarimetry file
    name_out_file = 'MAPCAT_polR_{}.res'.format(date_run)
    out_res = os.path.join(args.output_dir, name_out_file)
    print('out_res = ', out_res)
    with open(out_res, 'w') as fout:
        str_out = '\n{:12s} {:12.6f}  {:10s}{:>10}{:>10}   {:>8}{:>8}   {:>14}{:>7}   {:>8}{:>7}{:>7}{:>8}{:>6}{:>14.3f}{:>14}{:>10}'
        header = 'DATE_RUN        RJD-50000   Object            P+-dP(%)             Theta+-dTheta(deg.)      Q+-dQ             U+-dU          R      Sigma     APERPIX   APERAS   NUM_ROTATION  EXPTIME'
        fout.write(header)
        for lines in pol_rows:
            fout.write(str_out.format(*lines))

    # --------------------- CSV file
    name_out_csv = 'MAPCAT_polR_{}.csv'.format(date_run)
    out_csv = os.path.join(args.output_dir, name_out_csv)
    try:
        cols = ['P', 'dP', 'Theta', 'dTheta', 'Q', 'dQ', 'U', 'dU', \
            'R', 'Sigma', 'DATE_RUN', 'EXPTIME', 'RJD-50000', 'ID-MC', \
            'ID-BLAZAR-MC', 'MC-NAME', 'MC-IAU-NAME', 'OBJECT', 'APERPIX', 'APERAS', 'NUM_ROTATION', 'EXPTIME']
        df = pd.DataFrame(pol_data, columns=cols)
    except:
        print("pol_data")
        for k, v in pol_data.items():
            print(f"{k} -> {len(v)}")
        raise
    # Formatting
    df['RJD-50000'] = df['RJD-50000'].map(lambda x: '{0:.6f}'.format(x))
    df['P'] = df['P'].map(lambda x: '{0:.3f}'.format(x))
    df['dP'] = df['dP'].map(lambda x: '{0:.3f}'.format(x))
    df['Theta'] = df['Theta'].map(lambda x: '{0:.3f}'.format(x))
    df['dTheta'] = df['dTheta'].map(lambda x: '{0:.3f}'.format(x))
    df['Q'] = df['Q'].map(lambda x: '{0:.4f}'.format(x))
    df['dQ'] = df['dQ'].map(lambda x: '{0:.4f}'.format(x))
    df['U'] = df['U'].map(lambda x: '{0:.4f}'.format(x))
    df['dU'] = df['dU'].map(lambda x: '{0:.4f}'.format(x))
    df['R'] = df['R'].map(lambda x: '{0:.3f}'.format(x))
    df['Sigma'] = df['Sigma'].map(lambda x: '{0:.3f}'.format(x))
    df['APERAS'] = df['APERAS'].map(lambda x: '{0:.3f}'.format(x))
    df.to_csv(out_csv, index=False)
    
    return 0

    # sort by MJD
    data_res = data_res.sort_values(by=['MJD-OBS'])

    # Extract filter polarization angle and target name as new dataframe columns
    # data_res['ANGLE'] = data_res['OBJECT'].str.extract(r'\s([\d.]+)\s')
    data_res['OBJECT'] = np.array([' '.join(na.split(' ')[:-2]) for na in data_res['OBJECT'].values])
    # data_res['NAME'] = data_res['OBJECT'].str.extract(r'([a-zA-z0-9+-]+)\s')
    #print(data_res)
    #print(data_res.info())
    #return -99
    #Getting unique names
    object_names = data_res['MC-IAU-NAME'].unique()
    print('^' * 100)
    print('OBJECTS = ', object_names)
    print('^' * 100)

    pol_rows = []

    # dictionary for storing results...
    pol_data = DefaultDict(list)

    # Processing each target object
    for name in object_names:
        data_sets = object_measures(data_res, name)
            
        for data_object in data_sets:
            print('group')
            print('-' * 60)
            print(data_object)

            try:
                res_pol = compute_polarimetry(data_object)
                if res_pol['P'] is None:
                    print('POLARIMETRY,WARNING,"Could not compute Polarization for this set of measurements"')
                    continue
            except ZeroDivisionError:
                message = 'POLARIMETRY,ERROR,"EXCEPTION ZeroDivisionError: Found Zero Division Error in group processing; OBJECT={} DATE-OBS={}'
                message = message.format(name, data_object['DATE-OBS'])
                print(message)
                continue
            except ValueError:
                message = 'POLARIMETRY,ERROR,"EXCEPTION ValueError: Found Value Error in group processing; OBJECT={} DATE-OBS={}'
                message = message.format(name, data_object['DATE-OBS'])
                print(message)
                continue

            res_pol['DATE_RUN'] = date_run
            res_pol['EXPTIME'] = data_object['EXPTIME'].values[0]
            res_pol['APERPIX'] = data_object['APERPIX'].values[0]
            # arcsec per pixel as mean values of astrometric calibration computaion in RA and DEC.
            is_ord = data_object['TYPE'] == 'O'
            mean_secpix = (np.nanmean(data_object[is_ord]['SECPIX1'].values) + np.nanmean(data_object[is_ord]['SECPIX2'].values)) / 2
            if np.isnan(mean_secpix):
                print(f'mean_secpix = {mean_secpix}')
                print(data_object[is_ord]['SECPIX1'].values)
                print(data_object[is_ord]['SECPIX2'].values)
                return -199
            
            res_pol['APERAS'] = res_pol['APERPIX'] * mean_secpix
            rp_sigma = res_pol['Sigma']
            # if rp_sigma < 0.01:
            #     rp_sigma = 0.01
            
            for k, v in res_pol.items():
                if k == 'Sigma':
                    pol_data[k].append(rp_sigma)
                    continue
                pol_data[k].append(v)

            index_obs = 2 # groups of three or four observations
            if len(data_object['MJD-OBS'][data_object['TYPE'] == 'O']) < 3:
                index_obs = 1
            obs_date = data_object['MJD-OBS'][data_object['TYPE'] == 'O'].values[index_obs]
            pol_data['RJD-50000'].append(obs_date - 50000)
            row = [date_run, obs_date - 50000, name.strip()]
            
            row = row + [res_pol['P'], res_pol['dP'], \
                res_pol['Theta'], res_pol['dTheta'], \
                res_pol['Q'], res_pol['dQ'], \
                res_pol['U'], res_pol['dU'], \
                res_pol['R'], rp_sigma, \
                res_pol['APERPIX'], res_pol['APERAS'], res_pol['NUM_ROTATION']]
            pol_rows.append(row)
            # print('Lines to write down:')
            # pprint.pprint(pol_rows)

    # writing output night polarimetry file
    name_out_file = 'MAPCAT_polR_{}.res'.format(date_run)
    out_res = os.path.join(args.output_dir, name_out_file)
    print('out_res = ', out_res)
    with open(out_res, 'w') as fout:
        str_out = '\n{:12s} {:9.4f}   {:18s}{:>10}{:>7}   {:>7}{:>7}   {:>14}{:>7}   {:>8}{:>7}{:>7}{:>8}{:>6}{:>14.3f}{:>4}'
        header = 'DATE_RUN     RJD-50000   Object                 P+-dP(%)        Theta+-dTheta(deg.)     Q+-dQ             U+-dU          R      Sigma     APERPIX   APERAS   NUM_ROTATION'
        fout.write(header)
        for lines in pol_rows:
            fout.write(str_out.format(*lines))

    # --------------------- CSV file
    name_out_csv = 'MAPCAT_polR_{}.csv'.format(date_run)
    out_csv = os.path.join(args.output_dir, name_out_csv)
    try:
        cols = ['P', 'dP', 'Theta', 'dTheta', 'Q', 'dQ', 'U', 'dU', \
            'R', 'Sigma', 'DATE_RUN', 'EXPTIME', 'RJD-50000', 'ID-MC', \
            'ID-BLAZAR-MC', 'MC-NAME', 'MC-IAU-NAME', 'OBJECT', 'APERPIX', 'APERAS', 'NUM_ROTATION']
        df = pd.DataFrame(pol_data, columns=cols)
    except:
        print("pol_data")
        for k, v in pol_data.items():
            print(f"{k} -> {len(v)}")
        raise
    # Formatting
    df['RJD-50000'] = df['RJD-50000'].map(lambda x: '{0:.4f}'.format(x))
    df['P'] = df['P'].map(lambda x: '{0:.3f}'.format(x))
    df['dP'] = df['dP'].map(lambda x: '{0:.3f}'.format(x))
    df['Theta'] = df['Theta'].map(lambda x: '{0:.3f}'.format(x))
    df['dTheta'] = df['dTheta'].map(lambda x: '{0:.3f}'.format(x))
    df['Q'] = df['Q'].map(lambda x: '{0:.4f}'.format(x))
    df['dQ'] = df['dQ'].map(lambda x: '{0:.4f}'.format(x))
    df['U'] = df['U'].map(lambda x: '{0:.4f}'.format(x))
    df['dU'] = df['dU'].map(lambda x: '{0:.4f}'.format(x))
    df['R'] = df['R'].map(lambda x: '{0:.3f}'.format(x))
    df['Sigma'] = df['Sigma'].map(lambda x: '{0:.3f}'.format(x))
    df['APERAS'] = df['APERAS'].map(lambda x: '{0:.3f}'.format(x))
    df.to_csv(out_csv, index=False)

    return 0

if __name__ == '__main__':
    print(main())
