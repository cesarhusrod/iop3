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
import astropy.units as unit

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
    """It returns a list of data subsets, according to object 'name'.
    
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

def polarimetry_osn(df):
    """
    Compute polarimetric parameters for OSN data.
    Args:
        df (pandas.DataFrame): Data from object taken from 4 polarization angles.

    Returns:
        tuple: (P, dP, Theta, dTheta)

    Formula taken from "F. Moreno and O. Muñoz polarization.pdf"
    """
    df = df[df['TYPE'] == 'O'] #Only the ordinary source makes sense

    #values for T090
    qoff = 0.0645
    uoff = 0.0574
    Phi=math.radians(0)
    dPhi=math.radians(0)

    #values for 150
    #qoff = 0.0058
    dqoff = 0.001
    #uoff = 0.0334
    duoff = 0.003
 
    #Phi=math.radians(9.)
    #dPhi=math.radians(2.3)

    #values of T150 from polarization.pdf
    #qoff=0.031
    #uoff=0.024
            
    # For getting instrumental polarization
    #qoff=0
    #uoff=0
    #Phi=0
    #dPhi=0
    try:
        I_0 = (df['FLUX_APER'][df['ANGLE'] == 0]).values[0]
        dI_0 = (df['FLUXERR_APER'][df['ANGLE'] == 0]).values[0]
        I_45 = (df['FLUX_APER'][df['ANGLE'] == 45]).values[0]
        dI_45 = (df['FLUXERR_APER'][df['ANGLE'] == 45]).values[0]
        I_90 = (df['FLUX_APER'][df['ANGLE'] == 90]).values[0]
        dI_90 = (df['FLUXERR_APER'][df['ANGLE'] == 90]).values[0]
        I_135 = (df['FLUX_APER'][df['ANGLE'] == -45]).values[0]
        dI_135 = (df['FLUXERR_APER'][df['ANGLE'] == -45]).values[0]
    except:
        return -99, -99, -99, -99, -99, -99, -99, -99

    #Check if there is a value with big deviation from the rest:
    fluxes = np.array([I_0, I_45, I_90, I_135])
    flux_mean = fluxes.mean()
    flux_std = fluxes.std()
    dev_from_mean=(np.sqrt((fluxes-flux_mean)**2)/flux_mean)*100
    sig_from_mean=(np.sqrt((fluxes-flux_mean)**2)/flux_std)

    if (dev_from_mean>10).sum()>=1:
        flag="B"
    else:
        flag="G"
    
    qraw = (I_0 - I_90) / (I_0 + I_90)
    uraw = (I_45 - I_135) / (I_45 + I_135)
    
    #Applying error propagation...
    dqraw = qraw * math.sqrt(((dI_0**2+dI_90**2)/(I_0+I_90)**2)+(((dI_0**2+dI_90**2))/(I_0-I_90)**2))
    duraw = uraw * math.sqrt(((dI_45**2+dI_135**2)/(I_45+I_135)**2)+(((dI_45**2+dI_135**2))/(I_45-I_135)**2))

    qc = qraw - qoff
    uc = uraw - uoff

    dqc = math.sqrt(dqraw**2 + dqoff**2) 
    duc = math.sqrt(duraw**2 + duoff**2)

    q = qc*math.cos(2*Phi) - uc*math.sin(2*Phi)
    u = qc*math.sin(2*Phi) + uc*math.cos(2*Phi)
    
    dqa = qc*math.cos(2*Phi) * math.sqrt((dqc/qc)**2+((2*dPhi*math.sin(2*Phi))/(math.cos(2*Phi)))**2) 
    dqb = uc*math.sin(2*Phi) * math.sqrt((duc/uc)**2+((2*dPhi*math.cos(2*Phi))/(math.sin(2*Phi)))**2)
    dua = qc*math.sin(2*Phi) * math.sqrt((dqc/qc)**2+((2*dPhi*math.cos(2*Phi))/(math.sin(2*Phi)))**2) 
    dub = uc*math.cos(2*Phi) * math.sqrt((duc/uc)**2+((2*dPhi*math.sin(2*Phi))/(math.cos(2*Phi)))**2)
    
    #Just for tests
    #dqa=0
    #dua=0
    #dqb=0
    #dub=0
    
    dq = np.sqrt(dqa**2+dqb**2)
    du = np.sqrt(dua**2+dub**2)
    
    P = math.sqrt(q**2 + u**2)
    dP = P * (1/(q**2+u**2)) * math.sqrt((q*dq)**2+(u*du)**2)
    
    Theta_0 = 0
    if q >=0:
        Theta_0 = math.pi
        if u > 0:
            Theta_0 = -1 * math.pi

#    Theta_0 = 0
#    if q > 0:
#        if u >= 0:
#            Theta_0 = 0
#        if u < 0:
#            Theta_0 = math.pi / 2
#    elif q < 0:
#        Theta_0 = math.pi / 4
    Theta_0 = 0
    Theta = (1/2) * math.degrees(math.atan2(u,q) + Theta_0)
    dTheta = dP/P * 28.6
    
    
    return P, dP, Theta, dTheta, q, dq, u, du, flag, dev_from_mean.max()

def polarimetry(df):
    """Compute polarimetric parameters.
    Args:
        df (pandas.DataFrame): Data from object taken from 4 polarization angles.

    Returns:
        tuple: (P, dP, Theta, dTheta)

    Formula taken from "Zapatero_Osorio_2005_ApJ_621_445.pdf"
    """

    df_o = df[df['TYPE'] == 'O']
    df_e = df[df['TYPE'] == 'E']
    # print("Ordinary data")
    # print(df_o[['MJD-OBS', 'DATE-OBS', 'MC-IAU-NAME', 'FLUX_APER', 'FLUXERR_APER']])

    # print("Extraordinary data")
    # print(df_e[['MJD-OBS', 'DATE-OBS', 'MC-IAU-NAME', 'FLUX_APER', 'FLUXERR_APER']])

    #print(df_o['FLUX_AUTO'][df_o[key_polangle] == '0.0'].values[0])
    # print('Ordinary =')
    # print(df_o)
    # print(df_o.info())

    all_angles = df_o['ANGLE'].unique().tolist()
    
    if len(all_angles) == 1:
        return [None] * 8 
    elif len(all_angles) == 2:
        # valid combinations are: 0 and 22.5 or 45 and 67.5
        if (0 in all_angles and 45 in all_angles) or (22.5 in all_angles and 67.5 in all_angles):
            return [None] * 8
    
    # ORDINARY measurements
    if (df_o['ANGLE'] == 0).sum() == 0: # No measures in rotator angle equals to 0
        o_0 = (df_e['FLUX_APER'][df_e['ANGLE'] == 45]).values[-1]
        oe_0 = (df_e['FLUXERR_APER'][df_e['ANGLE'] == 45]).values[-1]
    else:
        o_0 = (df_o['FLUX_APER'][df_o['ANGLE'] == 0]).values[-1]
        oe_0 = (df_o['FLUXERR_APER'][df_o['ANGLE'] == 0]).values[-1]
    if (df_o['ANGLE'] == 22.5).sum() == 0: # No measures in rotator angle equals to 22.5
        o_22 = (df_e['FLUX_APER'][df_e['ANGLE'] == 67.5]).values[-1]
        oe_22 = (df_e['FLUXERR_APER'][df_e['ANGLE'] == 67.5]).values[-1]
    else:
        o_22 = (df_o['FLUX_APER'][df_o['ANGLE'] == 22.5]).values[-1]
        oe_22 = (df_o['FLUXERR_APER'][df_o['ANGLE'] == 22.5]).values[-1]
    if (df_o['ANGLE'] == 45).sum() == 0: # No measures in rotator angle equals to 45
        o_45 = (df_e['FLUX_APER'][df_e['ANGLE'] == 0]).values[-1]
        oe_45 = (df_e['FLUXERR_APER'][df_e['ANGLE'] == 0]).values[-1]
    else:
        o_45 = (df_o['FLUX_APER'][df_o['ANGLE'] == 45]).values[-1]
        oe_45 = (df_o['FLUXERR_APER'][df_o['ANGLE'] == 45]).values[-1]
    if (df_o['ANGLE'] == 67.5).sum() == 0: # No measures in rotator angle equals to 67.5
        o_67 = (df_e['FLUX_APER'][df_e['ANGLE'] == 22.5]).values[-1]
        oe_67 = (df_e['FLUXERR_APER'][df_e['ANGLE'] == 22.5]).values[-1]
    else:
        o_67 = (df_o['FLUX_APER'][df_o['ANGLE'] == 67.5]).values[-1]
        oe_67 = (df_o['FLUXERR_APER'][df_o['ANGLE'] == 67.5]).values[-1]

    # EXTRAORDINARY measurements
    if (df_o['ANGLE'] == 0).sum() == 0: # No measures in rotator angle equals to 0
        e_0 = (df_o['FLUX_APER'][df_o['ANGLE'] == 45]).values[-1]
        ee_0 = (df_o['FLUXERR_APER'][df_o['ANGLE'] == 45]).values[-1]
    else:
        e_0 = (df_e['FLUX_APER'][df_e['ANGLE'] == 0]).values[-1]
        ee_0 = (df_e['FLUXERR_APER'][df_e['ANGLE'] == 0]).values[-1]
    if (df_o['ANGLE'] == 22.5).sum() == 0: # No measures in rotator angle equals to 22.5
        e_22 = (df_o['FLUX_APER'][df_o['ANGLE'] == 67.5]).values[-1]
        ee_22 = (df_o['FLUXERR_APER'][df_o['ANGLE'] == 67.5]).values[-1]
    else:
        e_22 = (df_e['FLUX_APER'][df_e['ANGLE'] == 22.5]).values[-1]
        ee_22 = (df_e['FLUXERR_APER'][df_e['ANGLE'] == 22.5]).values[-1]
    if (df_o['ANGLE'] == 45).sum() == 0: # No measures in rotator angle equals to 45
        e_45 = (df_o['FLUX_APER'][df_o['ANGLE'] == 0]).values[-1]
        ee_45 = (df_o['FLUXERR_APER'][df_o['ANGLE'] == 0]).values[-1]
    else:
        e_45 = (df_e['FLUX_APER'][df_e['ANGLE'] == 45]).values[-1]
        ee_45 = (df_e['FLUXERR_APER'][df_e['ANGLE'] == 45]).values[-1]
    if (df_o['ANGLE'] == 67.5).sum() == 0: # No measures in rotator angle equals to 67.5
        e_67 = (df_o['FLUX_APER'][df_o['ANGLE'] == 22.5]).values[-1]
        ee_67 = (df_o['FLUXERR_APER'][df_o['ANGLE'] == 22.5]).values[-1]
    else:
        e_67 = (df_e['FLUX_APER'][df_e['ANGLE'] == 67.5]).values[-1]
        ee_67 = (df_e['FLUXERR_APER'][df_e['ANGLE'] == 67.5]).values[-1]

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
        dRQ = RQ * math.sqrt((oe_0/o_0) ** 2 + (ee_0 / e_0) ** 2 + \
            (oe_45 / o_45) ** 2 + (ee_45 / e_45) ** 2)

        #dRQ = RQ * (1/2) * ((o_45/e_45)/(o_0/e_0)) * math.sqrt((((oe_0/o_0)**2+(ee_0/e_0)**2)/(o_0/e_0)**2)+(((oe_45/o_45)**2+(ee_45/e_45)**2)/(o_45/e_45)**2))

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
        #dRU = RU * (1/2) * ((o_67/e_67)/(o_22/e_22)) * math.sqrt((((oe_22/o_22)**2+(ee_22/e_22)**2)/(o_22/e_22)**2)+(((oe_67/o_67)**2+(ee_67/e_67)**2)/(o_67/e_67)**2))

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

    val_str = '(Q_I +- (dQ_I), U_I +- (dU_I)) = ({} +- ({}), {} +- ({}))'
    # print(val_str.format(Q_I, dQ_I, U_I, dU_I))

    try:
        P = math.sqrt(Q_I ** 2 + U_I ** 2)
        dP = P * math.sqrt((dRQ / RQ) ** 2 + (dRU / RU) ** 2) / 2
        #dP = P * (1/(Q_I**2+U_I**2)) * math.sqrt((Q_I*dQ_I)**2+(U_I*dU_I)**2)
    except ZeroDivisionError:
        # print(f"(Q_I, U_I, RQ, dRQ, RU, dRU) = ({Q_I}, {U_I}, {RQ}, {dRQ}, {RU}, {dRU})")
        raise

    try:
        Theta_0 = 0
        
        if Q_I >= 0:
            Theta_0 = math.pi 
            if U_I > 0:
                Theta_0 = -1 * math.pi
        print(f'Theta_0 = {Theta_0}')
        Theta = 0.5 * math.degrees(math.atan(U_I/Q_I) + Theta_0)
        dTheta = dP / P * 28.6
    except ZeroDivisionError:
        # print(f"(U_I, Q_I, P, dP) = ({U_I}, {Q_I}, {P}, {dP})")
        raise

    pol_vals = 'P = {}, dP = {} \nTheta = {}, dTheta = {}'
    # print(pol_vals.format(P * 100, dP * 100, Theta, dTheta))

    return P, dP, Theta, dTheta, RQ, dRQ, RU, dRU


def compute_polarimetry(data_object):
    """Given input data, it calls polarimetry function to get
    polarimetric magnitudes.

    Args:
        data_object (pandas.DataFrame): Data from object taken from 4 polarization angles.

    Returns:
        dict: Keywords are [P, dP, Theta, dTheta, R, Sigma, MJD-5000, ID-MC, ID-BLAZAR-MC, MC-NAME, MC-IAU-NAME, NAME]
    
    """
    result = DefaultDict()
    
    name = data_object['OBJECT'].values[0]
    dates = sorted(data_object['MJD-OBS'].unique().tolist())
    # print(len(data_object.index))
    # print(f'dates = {dates}')
    index_date = 2 # 3rd observation () for group of 3 o 4 observations
    if len(dates) == 1:
        index_date = 0
    elif len(dates) == 2:
        index_date = 1

    obs_date = dates[index_date] 
    
        
    result['MJD-5000'] = round(obs_date - 50000, 4)
    result['ID-MC'] = data_object['ID-MC'].values[0]
    result['ID-BLAZAR-MC'] = data_object['ID-BLAZAR-MC'].values[0]
    result['MC-NAME'] = data_object['MC-NAME'].values[0]
    result['MC-IAU-NAME'] = data_object['MC-IAU-NAME'].values[0]
    result['OBJECT'] = data_object['OBJECT'].values[0] # .split()[0]      
    result['NUM_ROTATION'] = int(len(data_object.index) / 2)
    
    # Computing polarimetry parameters
    # print(data_object)
    # print(data_object.info())
    # print(data_object)
    try:
        if (data_object['ANGLE']==22.5).any()==True:
            P, dP, Theta, dTheta, RQ, dRQ, RU, dRU = polarimetry(data_object)
            flag="N/S"
            flux_std=-999
        else:
            P, dP, Theta, dTheta, RQ, dRQ, RU, dRU, flag, flux_std = polarimetry_osn(data_object)
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
        result['flag'] = flag
        result['flux_std'] = flux_std
        
        return result
    
    
    # HOW DO I COMPUTE MAGNITUDE AND ITS ERROR?
    # m = mean(C_i - 2.5 * log10(FLUX_ISO_O + FLUX_ISO_E))
    # e_m = std(C_i - 2.5 * log10(FLUX_ISO_O + FLUX_ISO_E))
    # print(data_object.info())
    is_ord = data_object['TYPE'] == 'O'
    zps = data_object['MAGZPT'][is_ord].values
    fluxes = data_object[['ANGLE', 'FLUX_APER']].groupby(['ANGLE']).sum()
    flux_errs = data_object[['ANGLE', 'FLUXERR_APER']].groupby(['ANGLE']).sum()
    if (data_object['ANGLE']==22.5).any()==True:
        try:
            mags = zps - 2.5 * np.log10(fluxes['FLUX_APER'].values)
        except ValueError:
            obs_datetimes = sorted(data_object['DATE-OBS'].unique().tolist())
            print(f'POLARIMETRY,ERROR,"Processing data from {name}, date-obs ={obs_datetimes}')
            print(f'\tzps = {zps}')
            print(f'\tFlux_apers = {fluxes["FLUX_APER"].values}')
            raise
    else:
        mags = data_object['MAG_APER']

    result['P'] = round(P * 100, 3)
    result['dP'] = round(dP * 100, 3)
    result['Theta'] = round(Theta, 2)
    result['dTheta'] = round(dTheta, 2)
    result['R'] =  round(mags.mean(), 2)
    result['Sigma'] = round(data_object['MAGERR_APER'].values.max(), 4)
    result['Q'] = round(RQ, 4)
    result['dQ'] = round(dRQ, 4)
    result['U'] = round(RU, 4)
    result['dU'] = round(dRU, 4)
    result['flag'] = flag
    result['flux_std'] = round(flux_std,4)
    return result

def main():
    parser = argparse.ArgumentParser(prog='iop3_polarimetry.py', \
    conflict_handler='resolve',
    description='''Main program. It searchs for every .res and final.fits files.
    Compute polarimetry parameters for each source (.res) and write
    output GLOBAL results file. ''',
    epilog='''''')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    parser.add_argument("calib_base_dir", help="Base directory for searching files.")
    parser.add_argument("output_dir", help="Output diretory for storing results.")
    parser.add_argument('-v',
                        '--verbose',
                        action='count',
                        default=0,
                        help="Show running and progress information [default: %(default)s].")
    args = parser.parse_args()

    # processing .res files
    results = glob.glob(os.path.join(args.calib_base_dir, '*-*/*_photocal_res.csv'))
    
    # sort by name == sort by date (ASC)
    
    if not results:
        str_err = 'ERROR: No *_photocal_res.csv files found.'
        print(str_err)
        return 1

    results.sort()
    pprint.pprint(results)
    print(f"\nFound {len(results)} '*_photocal_res.csv' files.\n")

    # getting photocalibration process info, in order to get useful FITS parameters
    cal_process = [r.replace('_photocal_res.csv', '_photocal_process_info.csv') for r in results]

    if not os.path.isdir(args.output_dir):
        try:
            os.makedirs(args.output_dir)
        except IOError:
            print(f'POLARIMETRY,ERROR,Could not create output directory {args.output_dir}"')
            return 2
    

    # Getting run date (input directory must have pattern like *MAPCAT_yyyy-mm-dd)
    dt_run = re.findall('(\d{6})', args.calib_base_dir)[-1]
    date_run = f'20{dt_run[:2]}-{dt_run[2:4]}-{dt_run[-2:]}'
    
    # Getting data from every VALID *_final.csv file into a new dataframe
    data_res = pd.concat([pd.read_csv(r) for r in results])
    data_proc = pd.concat([pd.concat([pd.read_csv(cp), pd.read_csv(cp)])  for cp in cal_process])

    print(data_res.info())
    print(data_proc.info())

    # return -99

    # append seconds per pixel info
    data_res['SECPIX1'] = data_proc['SECPIX1'].values
    data_res['SECPIX2'] = data_proc['SECPIX2'].values
    # print(data_res.info())
    # return -9

    # sort by MJD
    data_res = data_res.sort_values(by=['MJD-OBS'])

    # Extract filter polarization angle and target name as new dataframe columns
    # data_res['ANGLE'] = data_res['OBJECT'].str.extract(r'\s([\d.]+)\s')
    

    if 'deg' in data_res['OBJECT'].values[0]:
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
            print(data_object[['TYPE','OBJECT','ANGLE','MAG_APER', 'MAGERR_APER', 'FLUX_APER', 'FLUXERR_APER']])

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
            if len(data_object['MJD-OBS'][data_object['TYPE'] == 'O']) == 1:
                index_obs = 0
            obs_date = data_object['MJD-OBS'][data_object['TYPE'] == 'O'].values[index_obs]
            pol_data['MJD-50000'].append(obs_date)
            row = [date_run, obs_date - 50000, name.strip()]
            
            row = row + [res_pol['P'], res_pol['dP'], \
                res_pol['Theta'], res_pol['dTheta'], \
                res_pol['Q'], res_pol['dQ'], \
                res_pol['U'], res_pol['dU'], \
                res_pol['R'], rp_sigma, \
                res_pol['APERPIX'], res_pol['APERAS'], res_pol['NUM_ROTATION'], \
                res_pol['flag'], \
                res_pol['flux_std']             ]
            pol_rows.append(row)
             #print('Lines to write down:')
             #pprint.pprint(pol_rows)

    # writing output night polarimetry file
    
    if 'MAPCAT' in args.calib_base_dir:
        name_out_file = 'MAPCAT_polR_{}.res'.format(date_run)
    elif 'T090' in args.calib_base_dir:
        name_out_file = 'T090_polR_{}.res'.format(date_run)
    elif 'T150' in args.calib_base_dir:
        name_out_file = 'T150_polR_{}.res'.format(date_run)
    out_res = os.path.join(args.output_dir, name_out_file)
    print('out_res = ', out_res)
    with open(out_res, 'w') as fout:
        str_out = '\n{:12s} {:9.4f}   {:18s}{:>10}{:>7}   {:>7}{:>7}   {:>14}{:>7}   {:>8}{:>7}{:>7}{:>8}{:>6}{:>14.3f}{:>4}    {:s}   {:>14.3f}'
        header = 'DATE_RUN     MJD-50000   Object                 P+-dP(%)        Theta+-dTheta(deg.)     Q+-dQ             U+-dU          R      Sigma     APERPIX   APERAS   NUM_ROTATION   flag    flux_std'
        fout.write(header)
        for lines in pol_rows:
            fout.write(str_out.format(*lines))

    # --------------------- CSV file
    if 'MAPCAT' in args.calib_base_dir:
        name_out_csv = 'MAPCAT_polR_{}.csv'.format(date_run)
    elif 'T090' in args.calib_base_dir:
        name_out_csv = 'T090_polR_{}.csv'.format(date_run)
    elif 'T150' in args.calib_base_dir:
        name_out_csv = 'T150_polR_{}.csv'.format(date_run)
    out_csv = os.path.join(args.output_dir, name_out_csv)
    try:
        cols = ['P', 'dP', 'Theta', 'dTheta', 'Q', 'dQ', 'U', 'dU', \
            'R', 'Sigma', 'DATE_RUN', 'EXPTIME', 'MJD-50000', 'ID-MC', \
            'ID-BLAZAR-MC', 'MC-NAME', 'MC-IAU-NAME', 'OBJECT', 'APERPIX', 'APERAS', 'NUM_ROTATION', 'flag', 'flux_std']
        df = pd.DataFrame(pol_data, columns=cols)
    except:
        print("pol_data")
        for k, v in pol_data.items():
            print(f"{k} -> {len(v)}")
        raise
    # Formatting
    df['MJD-50000'] = df['MJD-50000'].map(lambda x: '{0:.4f}'.format(x))
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
    df['flag'] = df['flag'].map(lambda x: '{0:s}'.format(x))
    df['flux_std'] = df['flux_std'].map(lambda x: '{0:.3f}'.format(x))
    
    df.to_csv(out_csv, index=False)

    return 0

if __name__ == '__main__':
    print(main())
