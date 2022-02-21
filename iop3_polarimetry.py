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

def compute_polarimetry(data_object):
    """Given input data, it calls polarimetry function to get
    polarimetric magnitudes.

    Args:
        data_object (pandas.DataFrame): Data from object taken from 4 polarization angles.

    Returns:
        dict: Keywords are [P, dP, Theta, dTheta, R, Sigma, RJD-5000, ID-MC, ID-BLAZAR-MC, MC-NAME, MC-IAU-NAME, NAME]
    
    """
    result = DefaultDict()
    name = data_object['OBJECT'].values[0]
    
    dates = sorted(data_object['MJD-OBS'].unique().tolist())
    obs_date = dates[2] # 3rd observation
    
    
    # Computing polarimetry parameters
    # print(data_object)
    # print(data_object.info())
    # print(data_object)

    try:
        if (data_object['ANGLE']==22.5).any()==True:
            P, dP, Theta, dTheta, RQ, dRQ, RU, dRU = polarimetry(data_object)
        else:
            P, dP, Theta, dTheta, RQ, dRQ, RU, dRU = polarimetry_osn(data_object)
    except ZeroDivisionError:
        print(f'\tZeroDivisionError while processing object called "{name}"')
        raise
    
    # HOW DO I COMPUTE MAGNITUDE AND ITS ERROR?
    # m = mean(C_i - 2.5 * log10(FLUX_ISO_O + FLUX_ISO_E))
    # e_m = std(C_i - 2.5 * log10(FLUX_ISO_O + FLUX_ISO_E))
    # print(data_object.info())
    is_ord = data_object['TYPE'] == 'O'
    zps = data_object['MAGZPT'][is_ord].values
    fluxes = data_object[['ANGLE', 'FLUX_APER']].groupby(['ANGLE']).sum()
    flux_errs = data_object[['ANGLE', 'FLUXERR_APER']].groupby(['ANGLE']).sum()
    try:
        mags = zps - 2.5 * np.log10(fluxes['FLUX_APER'].values)
    except ValueError:
        obs_datetimes = sorted(data_object['DATE-OBS'].unique().tolist())
        print(f'POLARIMETRY,ERROR,"Processing data from {name}, date-obs ={obs_datetimes}')
        print(f'\tzps = {zps}')
        print(f'\tFlux_apers = {fluxes["FLUX_APER"].values}')
        raise

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
    result['RJD-5000'] = round(obs_date - 50000, 4)
    result['ID-MC'] = data_object['ID-MC'].values[0]
    result['ID-BLAZAR-MC'] = data_object['ID-BLAZAR-MC'].values[0]
    result['MC-NAME'] = data_object['MC-NAME'].values[0]
    result['MC-IAU-NAME'] = data_object['MC-IAU-NAME'].values[0]
    result['OBJECT'] = data_object['OBJECT']

    return result
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

    qoff=0.031
    uoff=0.024

    I_0 = (df['FLUX_APER'][df['ANGLE'] == 0]).values[0]
    dI_0 = (df['FLUXERR_APER'][df['ANGLE'] == 0]).values[0]
    I_45 = (df['FLUX_APER'][df['ANGLE'] == 45]).values[0]
    dI_45 = (df['FLUXERR_APER'][df['ANGLE'] == 45]).values[0]
    I_90 = (df['FLUX_APER'][df['ANGLE'] == 90]).values[0]
    dI_90 = (df['FLUXERR_APER'][df['ANGLE'] == 90]).values[0]
    I_135 = (df['FLUX_APER'][df['ANGLE'] == -45]).values[0]
    dI_135 = (df['FLUXERR_APER'][df['ANGLE'] == -45]).values[0]

    qraw = (I_0 - I_90) / (I_0 + I_90)
    uraw = (I_45 - I_135) / (I_45 + I_135)

    qc = qraw
    uc = uraw

    dqc = qc * math.sqrt(2*((dI_0/I_0)**2 + (dI_90/I_90)**2))
    duc = qc * math.sqrt(2*((dI_45/I_45)**2 + (dI_135/I_135)**2))
    
    P = math.sqrt(qc**2 + uc**2)
    dP = P * math.sqrt((dqc/qc)**2+(duc/uc)**2)
    
    Theta = (1/2) * math.degrees(math.atan(uc/qc))
    dTheta = dP/P
    
    return P, dP, Theta, dTheta, qc, dqc, uc, duc

def polarimetry(df):
    """
    Compute polarimetric parameters.
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
    o_0 = (df_o['FLUX_APER'][df_o['ANGLE'] == 0]).values[-1]
    oe_0 = (df_o['FLUXERR_APER'][df_o['ANGLE'] == 0]).values[-1]
    o_22 = (df_o['FLUX_APER'][df_o['ANGLE'] == 22.5]).values[-1]
    oe_22 = (df_o['FLUXERR_APER'][df_o['ANGLE'] == 22.5]).values[-1]
    o_45 = (df_o['FLUX_APER'][df_o['ANGLE'] == 45]).values[-1]
    oe_45 = (df_o['FLUXERR_APER'][df_o['ANGLE'] == 45]).values[-1]
    o_67 = (df_o['FLUX_APER'][df_o['ANGLE'] == 67.5]).values[-1]
    oe_67 = (df_o['FLUXERR_APER'][df_o['ANGLE'] == 67.5]).values[-1]

    e_0 = (df_e['FLUX_APER'][df_e['ANGLE'] == 0]).values[-1]
    ee_0 = (df_e['FLUXERR_APER'][df_e['ANGLE'] == 0]).values[-1]
    e_22 = (df_e['FLUX_APER'][df_e['ANGLE'] == 22.5]).values[-1]
    ee_22 = (df_e['FLUXERR_APER'][df_e['ANGLE'] == 22.5]).values[-1]
    e_45 = (df_e['FLUX_APER'][df_e['ANGLE'] == 45]).values[-1]
    ee_45 = (df_e['FLUXERR_APER'][df_e['ANGLE'] == 45]).values[-1]
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

    val_str = '(Q_I +- (dQ_I), U_I +- (dU_I)) = ({} +- ({}), {} +- ({}))'
    # print(val_str.format(Q_I, dQ_I, U_I, dU_I))

    try:
        P = math.sqrt(Q_I ** 2 + U_I ** 2)
        dP = P * math.sqrt((dRQ / RQ) ** 2 + (dRU / RU) ** 2) / 2
    except ZeroDivisionError:
        # print(f"(Q_I, U_I, RQ, dRQ, RU, dRU) = ({Q_I}, {U_I}, {RQ}, {dRQ}, {RU}, {dRU})")
        raise

    try:
        Theta = 0.5 * math.degrees(math.atan(U_I / Q_I))
        dTheta = dP / P * 28.6
    except ZeroDivisionError:
        # print(f"(U_I, Q_I, P, dP) = ({U_I}, {Q_I}, {P}, {dP})")
        raise

    pol_vals = 'P = {}, dP = {} \nTheta = {}, dTheta = {}'
    # print(pol_vals.format(P * 100, dP * 100, Theta, dTheta))

    return P, dP, Theta, dTheta, RQ, dRQ, RU, dRU

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

            if len(data_object.index) < 8:
                print(f'POLARIMETRY,WARNING,"Not enough observations for compute object \'{name}\' polarimetry"')
                continue

            try:
                res_pol = compute_polarimetry(data_object)
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
            mean_secpix = (data_object[is_ord]['SECPIX1'].mean() + data_object[is_ord]['SECPIX2'].mean()) / 2
            res_pol['APERAS'] = res_pol['APERPIX'] * mean_secpix
            rp_sigma = res_pol['Sigma']
            # if rp_sigma < 0.01:
            #     rp_sigma = 0.01
            
            for k, v in res_pol.items():
                if k == 'Sigma':
                    pol_data[k].append(rp_sigma)
                    continue
                pol_data[k].append(v)

            obs_date = data_object['MJD-OBS'][data_object['TYPE'] == 'O'].values[2]
            pol_data['RJD-50000'].append(obs_date - 50000)
            row = [date_run, obs_date - 50000, name.strip()]
            
            row = row + [res_pol['P'], res_pol['dP'], \
                res_pol['Theta'], res_pol['dTheta'], \
                res_pol['Q'], res_pol['dQ'], \
                res_pol['U'], res_pol['dU'], \
                res_pol['R'], rp_sigma, \
                res_pol['APERPIX'], res_pol['APERAS']]
            pol_rows.append(row)
            # print('Lines to write down:')
            # pprint.pprint(pol_rows)

    # writing output night polarimetry file
    name_out_file = 'MAPCAT_polR_{}.res'.format(date_run)
    out_res = os.path.join(args.output_dir, name_out_file)
    print('out_res = ', out_res)
    with open(out_res, 'w') as fout:
        str_out = '\n{:12s} {:9.4f}   {:18s}{:>10}{:>7}   {:>7}{:>7}   {:>14}{:>7}   {:>8}{:>7}{:>7}{:>8}{:>6}{:>14.3f}'
        header = 'DATE_RUN     RJD-50000   Object                 P+-dP(%)        Theta+-dTheta(deg.)     Q+-dQ             U+-dU          R      Sigma     APERPIX   APERAS'
        fout.write(header)
        for lines in pol_rows:
            fout.write(str_out.format(*lines))

    # --------------------- CSV file
    name_out_csv = 'MAPCAT_polR_{}.csv'.format(date_run)
    out_csv = os.path.join(args.output_dir, name_out_csv)
    try:
        cols = ['P', 'dP', 'Theta', 'dTheta', 'Q', 'dQ', 'U', 'dU', \
            'R', 'Sigma', 'DATE_RUN', 'EXPTIME', 'RJD-50000', 'ID-MC', \
            'ID-BLAZAR-MC', 'MC-NAME', 'MC-IAU-NAME', 'OBJECT', 'APERPIX', 'APERAS']
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
