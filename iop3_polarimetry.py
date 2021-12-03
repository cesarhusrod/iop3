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


def object_measures(data, name):
    """"Two params are keyword: grism angle and exptime. This function
    return valid subsets of input data.
    
    Args:
        data (pandas.DataFrame): Data from objects taken from 4 polarization angles grism.
        name (str): object name for subsetting

    Returns:
        list: of valid subsets of observation for object called 'name'."""
    data_sets = []
    print(f'***** Processing object called "{name}" ********')
    
    data_object = data[data['NAME'] == name]
    # checking EXPTIME keyword: every set of measurements in different angles must have same EXPTIME
    exptimes = data_object['EXPTIME'].unique()
    print(f"EXPTIMES = {exptimes}")
    for et in exptimes:
        print(f"\tAnalyzing EXPTIME = {et}")
        subdata_object = data_object[data_object['EXPTIME'] == et]
        angles_rot = subdata_object['ANGLE'].unique()
        print(f"\t\tangles_rot = {angles_rot}")
        if len(angles_rot) < 4:
            print("\t\tERROR: Not all rotation instrument were measured.")
            print(f"\t{angles_rot}")
            continue
        # Data taken in every filter, so now goes check number of measurements
        # print(subdata_object)
        if len(subdata_object.index) >= 16:
            # Maybe several exposures were taken
            subdata_object.sort_values(by=['ANGLE', 'TYPE', 'DATE-OBS'], \
                inplace=True, ascending=False)
            index = subdata_object[['ANGLE', 'TYPE']].duplicated()
            dt_object = subdata_object[index == False]
            if len(dt_object.index) == 8:
                data_sets.append(dt_object)            
            elif len(dt_object.index) > 8:
                # filtering by second time
                dt_object.sort_values(by=['ANGLE', 'TYPE', 'DATE-OBS'], \
                    inplace=True, ascending=False)
                index = dt_object[['ANGLE', 'TYPE']].duplicated()
                dt_object = dt_object[index == False]
                data_sets.append(dt_object)
            else: # low than 8 measurements (Ordinary/Extraordinary sources times 4 angles)
                str_err = '\tERROR: Not enough polarization angle measurements (only {})'
                str_err += ' 8 measurements are needed: 4 angles and Ordinary/Extraordinary sources.'
                print(str_err.format(dt_object['NAME'].size))
                print("\tDiscarded data\n\t----------------")
                print(dt_object[['DATE-OBS', 'MJD-OBS', 'TYPE', 'ANGLE', 'EXPTIME', 'OBJECT', 'MC-IAU-NAME', 'FLUX_APER', 'FLUXERR_APER']])
                continue
        elif len(subdata_object.index) > 8:
            subdata_object.sort_values(by=['ANGLE', 'TYPE', 'DATE-OBS'], \
                inplace=True, ascending=False)
            index = subdata_object[['ANGLE', 'TYPE']].duplicated()
            dt_object = subdata_object[index == False]
            if len(dt_object.index) == 8:
                data_sets.append(dt_object)
            else:
                str_err = '\tERROR: Not enough polarization angle measurements (only {})'
                str_err += ' 8 measurements are needed: 4 angles and Ordinary/Extraordinary sources.'
                print(str_err.format(dt_object['NAME'].size))
                print("\tDiscarded data\n\t----------------")
                print(dt_object[['DATE-OBS', 'MJD-OBS', 'TYPE', 'ANGLE', 'EXPTIME', 'OBJECT', 'MC-IAU-NAME', 'FLUX_APER', 'FLUXERR_APER']])
                continue
        elif len(subdata_object.index) == 8:
            data_sets.append(subdata_object)
        else:
            str_err = '\tERROR: Not enough polarization angle measurements (only {})'
            str_err += ' 8 measurements are needed: 4 angles and Ordinary/Extraordinary sources.'
            print(str_err.format(subdata_object['NAME'].size))
            print("\tDiscarded data\n\t----------------")
            print(subdata_object[['DATE-OBS', 'MJD-OBS', 'TYPE', 'ANGLE', 'EXPTIME', 'OBJECT', 'MC-IAU-NAME', 'FLUX_APER', 'FLUXERR_APER']])

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
    name = data_object['NAME'].values[0]
    obs_date = data_object['MJD-OBS'][data_object['TYPE'] == 'O'].values[2] # 3rd observation
    
    
    # Computing polarimetry parameters
    # print(data_object)
    # print(data_object.info())
    # print(data_object)
    try:
        P, dP, Theta, dTheta = polarimetry(data_object)
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
        print(f'\t---------ERROR processing data from {name}--------')
        print(f'\tzps = {zps}')
        print(f'\tFlux_apers = {fluxes["FLUX_APER"].values}')
        raise
    # mag_errs = zps - 2.5 * np.log10(flux_errs['FLUXERR_APER'].values)


    # flux_o = data_object['FLUX_APER'][is_ord].values
    # flux_e = data_object['FLUX_APER'][~is_ord].values
    # print(flux_o, flux_e)
    #
    # print('Numero de flujos ordinarios = {}'.format(flux_o.size))
    # print('Numero de flujos extraordinarios = {}'.format(flux_e.size))
    # objects = data_object['OBJECT'].unique()
    # # zps = np.array([zeropoints[o] for o in objects])
    # # print("Numero de zeropoints = {}".format(zps.size))
    # # print('Suma de flujos = {}'.format(flux_o + flux_e))
    # # mags = zps - 2.5 * np.log10(flux_o + flux_e)
    # row = row + [round(data_object['MAG_AUTO'][is_ord].values[0], 2), \
    # round(data_object['MAGERR_AUTO'][is_ord].values[0], 2)] # [round(mags.mean(), 3), round(mags.std(), 3)]

    # row += [round(mags.mean(), 2), round(mag_errs.max(), 2)]

    # Data process pass all possble tests
    # result['P'].append(round(P * 100, 3))
    # result['dP'].append(round(dP * 100, 3))
    # result['Theta'].append(round(Theta, 2))
    # result['dTheta'].append(round(dTheta, 2))
    # result['R'].append(round(mags.mean(), 2))
    # result['Sigma'].append(round(data_object['MAGERR_APER'].values.max(), 2))
    # result['RJD-5000'].append(round(obs_date - 50000, 4))
    # result['ID-MC'].append(data_object['ID-MC'].values[0])
    # result['ID-BLAZAR-MC'].append(data_object['ID-BLAZAR-MC'].values[0])
    # result['MC-NAME'].append(data_object['MC-NAME'].values[0])
    # result['MC-IAU-NAME'].append(data_object['MC-IAU-NAME'].values[0])
    # result['NAME'].append(name)

    result['P'] = round(P * 100, 3)
    result['dP'] = round(dP * 100, 3)
    result['Theta'] = round(Theta, 2)
    result['dTheta'] = round(dTheta, 2)
    result['R'] =  round(mags.mean(), 2)
    result['Sigma'] = round(data_object['MAGERR_APER'].values.max(), 2)
    result['RJD-5000'] = round(obs_date - 50000, 4)
    result['ID-MC'] = data_object['ID-MC'].values[0]
    result['ID-BLAZAR-MC'] = data_object['ID-BLAZAR-MC'].values[0]
    result['MC-NAME'] = data_object['MC-NAME'].values[0]
    result['MC-IAU-NAME'] = data_object['MC-IAU-NAME'].values[0]
    result['NAME'] = name
        

    return result

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

    return P, dP, Theta, dTheta


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
    results = glob.glob(os.path.join(args.calib_base_dir, '*-*/*_final.csv'))
    # sort by name == sort by date (ASC)
    
    if not results:
        str_err = 'ERROR: No *_final.csv files found.'
        print(str_err)
        return 1

    results.sort()
    pprint.pprint(results)
    print(f"Found {len(results)} '*_final.csv' files.")

    if not os.path.isdir(args.output_dir):
        try:
            os.makedirs(args.output_dir)
        except IOError:
            print(f"ERROR: Could not create output directory '{args.output_dir}'")
            return 2
    

    # Getting run date (input directory must have pattern like *MAPCAT_yyyy-mm-dd)
    dt_run = re.findall('(\d{6})', args.calib_base_dir)[-1]
    date_run = f'20{dt_run[:2]}-{dt_run[2:4]}-{dt_run[-2:]}'
    
    # Getting data from every VALID *_final.csv file into a new dataframe
    data_res = pd.concat([pd.read_csv(r) for r in results])
    # print(data_res.info())
    # return -9

    # sort by MJD
    data_res = data_res.sort_values(by=['MJD-OBS'])

    # Extract filter polarization angle and target name as new dataframe columns
    # data_res['ANGLE'] = data_res['OBJECT'].str.extract(r'\s([\d.]+)\s')
    data_res['NAME'] = data_res['OBJECT'].str.extract(r'([a-zA-z0-9+-]+)\s')
    #print(data_res)
    #print(data_res.info())
    #return -99
    #Getting unique names
    object_names = data_res['NAME'].unique()
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
            try:
                res_pol = compute_polarimetry(data_object)
            except ZeroDivisionError:
                print("EXCEPTON: Found Zero Division Error")
            except ValueError:
                print("EXCEPTON: Found Value Error")

            res_pol['DATE_RUN'] = date_run
            res_pol['EXPTIME'] = data_object['EXPTIME'].values[0]
            for k, v in res_pol.items():
                pol_data[k].append(v)

            obs_date = data_object['MJD-OBS'][data_object['TYPE'] == 'O'].values[2]
            row = [date_run, obs_date - 50000, name]
            row = row + [res_pol['P'], res_pol['dP'], \
                res_pol['Theta'], res_pol['dTheta'], \
                res_pol['R'], res_pol['Sigma']]
            pol_rows.append(row)
            # print('Lines to write down:')
            # pprint.pprint(pol_rows)

    # writing output night polarimetry file
    name_out_file = 'MAPCAT_polR_{}.res'.format(date_run)
    out_res = os.path.join(args.output_dir, name_out_file)
    print('out_res = ', out_res)
    with open(out_res, 'w') as fout:
        str_out = '\n{:12s} {:9.4f} {:8}{:>8}{:>7}{:>7}{:>7}{:>9}{:>6}'
        header = 'DATE_RUN   RJD-50000 Object     P+-dP(%)  Theta+-dTheta(deg.)  R     Sigma '
        fout.write(header)
        for lines in pol_rows:
            fout.write(str_out.format(*lines))

    # CSV file
    name_out_csv = 'MAPCAT_polR_{}.csv'.format(date_run)
    out_csv = os.path.join(args.output_dir, name_out_csv)
    try:
        cols = ['P', 'dP', 'Theta', 'dTheta', 'R', 'Sigma', 'DATE_RUN', 'EXPTIME', \
            'RJD-5000', 'ID-MC', 'ID-BLAZAR-MC', 'MC-NAME', 'MC-IAU-NAME', 'NAME']
        df = pd.DataFrame(pol_data, columns=cols)
    except:
        print("pol_data")
        for k, v in pol_data.items():
            print(f"{k} -> {len(v)}")
        raise
    df.to_csv(out_csv, index=False)

    return 0

if __name__ == '__main__':
    print(main())
