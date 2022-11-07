#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu April 15 17:38:23 2021

___e-mail__ = cesar_husillos@tutanota.com
__author__ = 'Cesar Husillos'

VERSION:
    0.1 Initial version, based on CAFOS_wcstools_perfect_pruebas_revision_HD.ipynb
    0.2 (Sat April 17, 2021) MAPCAT apertures are considered.
    0.9 (Sat March 12, 2022) Major refactoring.
"""


# ---------------------- IMPORT SECTION ----------------------
from ast import Try
from ssl import ALERT_DESCRIPTION_BAD_RECORD_MAC
# from cmath import isnan
from dataclasses import replace
import os
import glob
import argparse
import subprocess
import re
import math
from datetime import datetime,timedelta
from collections import defaultdict
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')

# Data structures libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


#import seaborn
import aplpy # FITS plotting library

from astropy.io import fits # FITS library
from astropy.time import Time
import astropy.wcs as wcs

# Coordinate system transformation package and modules
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import match_coordinates_sky  # Used for searching sources in catalog
from astropy.coordinates import FK5  # Low-level frames
import astropy.coordinates as coord
import astropy.units as u


from mcFits import mcFits

# HTML ouput template
import jinja2


# Photutils (aperture flux measurements)
# from photutils.aperture import aperture_photometry, CircularAperture, CircularAnnulus

# =================================
from astropy.utils.exceptions import AstropyWarning, AstropyDeprecationWarning
import warnings

# Ignore too many FITSFixedWarnings
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.simplefilter('ignore', category=AstropyDeprecationWarning)

# =================================

def read_sext_catalog(path, format='ASCII', verbose=False):
    """
    Read SExtractor output catalog given by 'path'.
    
    Args:
        path (str): SExtractor catalog path
        format (str): 'ASCII' or 'FTIS_LDAC' output SExtractor formats.
        verbose (bool): IT True, it prints process info.
        
    Returns:
        pd.DataFrame: Data from SExtractor output catalog.
    """
    if format == 'ASCII':
        cat = ''
        with open(path) as fin:
            cat = fin.read()
        campos = re.findall(r'#\s+\d+\s+([\w_]*)', cat)
        if verbose:
            print(f'Header catalog keywords = {campos}')

        data_sext = np.genfromtxt(path, names=campos)
        
        # Working with pandas DataFrame
        # data_sext = pd.DataFrame({k:np.atleast_1d(data_sext[k]) for k in campos})
    else:
        sext = fits.open(path)
        data_sext = sext[2].data
        campos = data_sext.columns.names
    
    data_sext = {k:list(data_sext[k]) for k in campos}
    data_sext = pd.DataFrame(data_sext)

    return data_sext

def execute_command(cmd, out=subprocess.PIPE, err=subprocess.PIPE, shell=True):
    """It executes command and wait for results.

    Args:
        cmd (str or list): Command to be executed.
        out (file object, optional): Standar output file for messages produced by 
            command execution. Defaults to subprocess.PIPE.
        err (file object, optional): Standar error file for messages produced by 
            command execution. Defaults to subprocess.PIPE.
        shell (bool, optional): If True, 'cmd' is a string. Else 'cmd' is a list. Defaults to True.

    Returns:
        CompletedProcess instance: It respresents a process that has finished.
            (https://docs.python.org/3/library/subprocess.html#subprocess.CompletedProcess)
    """
    
    result = subprocess.run(cmd, stdout=out, stderr=err, shell=shell)
    
    return result

def statistics(path_fits, border=15, sat_threshold=50000):
    input_fits = mcFits(path_fits, border=border)
    head = input_fits.header
    data = input_fits.data + 0.0

    new_data = data
    if border > 0:
        new_data = data[border:-border, border:-border]
    
    dictStats = {}
    
    dictStats['MINIMUM'] = new_data.min()
    dictStats['MAXIMUM'] = new_data.max()
    dictStats['MEAN'] = new_data.mean()
    dictStats['STD'] = new_data.std()
    dictStats['MEDIAN'] = np.median(new_data)

    dictStats['NPIX'] = head['NAXIS1'] * head['NAXIS2']
    dictStats['NSAT'] = (new_data >= sat_threshold).sum()
    dictStats['EXPTIME'] = head['EXPTIME']
    dictStats['FILENAME'] = os.path.split(path_fits)[1]
    dictStats['STD/MEAN'] = dictStats['STD'] / dictStats['MEAN']
    mean = dictStats['MEAN']
    median = dictStats['MEDIAN']
    dictStats['MEAN_MEDIAN'] = np.round((mean - median) / (mean + median) * 100, 3)

    return dictStats


def sext_params_detection(path_fits, border=15, sat_threshold=45000):
    """
    It analyzes FITS data and return best input parameters for maximize detection.
    
    Args:
        path_fits (str): Path to FITS file.
        border (int): Border size. It won't be used in statistics computation.
        sat_threshold (int): threshold pixel value considered as saturated.
        
    Returns:
        dict: Dictionary with best detection parameters for SExtractor.
    """    
    params = {}
    # default values
    params['FILTER'] = 'N'
    params['CLEAN'] = 'N'
    params['DETECT_MINAREA'] = 15
    params['ANALYSIS_THRESH'] = 1.0
    params['DETECT_THRESH'] = 1.0
    params['DEBLEND_MINCONT'] = 0.005

    fits = mcFits(path_fits)
    if 'CCDGAIN' in fits.header:
        params['GAIN'] = fits.header['CCDGAIN']

    # getting info about FITS
    dt = statistics(path_fits, border=border, sat_threshold=sat_threshold)

    if dt['EXPTIME'] > 1:
        params['FILTER'] = 'Y'
        params['CLEAN'] = 'Y'
        # params['FILTER_NAME'] = '/home/cesar/desarrollos/Ivan_Agudo/code/iop3/conf/filters_sext/mexhat_5.0_11x11.conv'
        # params['FILTER_NAME'] = '/home/cesar/desarrollos/Ivan_Agudo/code/iop3/conf/filters_sext/gauss_5.0_9x9.conv'
        params['FILTER_NAME'] = '/home/users/dreg/misabelber/GitHub/iop3/conf/filters_sext/tophat_5.0_5x5.conv'
        # params['FILTER_NAME'] = '/home/cesar/desarrollos/Ivan_Agudo/code/iop3/conf/filters_sext/tophat_5.0_5x5.conv'
    
    if dt['STD/MEAN'] > 2: # noisy
        params['ANALYSIS_THRESH'] = 1.5
        params['DETECT_THRESH'] = 1.5
    # elif dt['STD/MEAN'] > 5: # very noisy
    #     params['ANALYSIS_THRESH'] = 2.5
    #     params['DETECT_THRESH'] = 2.5


    return params

def plot_cat(path_fits, plot_out_path, cat, astro_coords=False, cat_format='ASCII', \
    color='red', title=None, border=15, sat_threshold=45000):
    """Plot data given by 'cat' over 'path_fits'

    Args:
        path_fits (str): _description_
        plot_out_path (str): _description_
        cat (str): _description_
        astro_coords (bool): If True, (ALPHA_J2000, DELTA_J2000) coordinates 
            are plotted. Else (X_IMAGE, Y_IMAGE). Defaults to False.
        cat_format (str, optional): Output SExtractor valid format ('ASCII' or 'FITS_LDAC'). Defaults to 'ASCII'.
    
    Returns:
        int: 0, if everything was fine.
    """
    # Read input FITS
    i_fits = mcFits(path_fits)
    if 'MAPCAT' in path_fits:
        fits_par = i_fits.get_data(keywords=['INSTRSCL', 'FWHM', 'EXPTIME', 'OBJECT', 'DATE-OBS'])
    else:
        fits_par = i_fits.get_data(keywords=['NAXIS1', 'FWHM', 'EXPTIME', 'OBJECT', 'DATE-OBS'])
    # Plotting detections
    data_cat = read_sext_catalog(cat, format=cat_format)
    
    coords = [data_cat['X_IMAGE'], data_cat['Y_IMAGE']] 
    ref_coords = 'pixel'
    astrocal = False
    if astro_coords:
        coords = [data_cat['ALPHA_J2000'], data_cat['DELTA_J2000']] 
        ref_coords = 'world'
        astrocal = True
    
    stats = statistics(path_fits, border=border, sat_threshold=sat_threshold)

    if not title:
        title = f"{fits_par['OBJECT']}, {fits_par['DATE-OBS']} ({fits_par['EXPTIME']} s, STD/MEAN = {round(stats['STD/MEAN'], 3)})"
    
    i_fits.plot(plot_out_path, title=title, coords=coords, \
        ref_coords=ref_coords, astroCal=astrocal, color=color, \
        dictParams={'aspect':'auto', 'vmin': 1, 'stretch': 'power', 'invert': True})    
        # dictParams={'aspect':'auto', 'vmin': 1, 'invert': True}, format='png')

    return 0

def get_radec_limits(path_fits):
    """It computes alpha_J2000, delta_J2000 limits for astrometric calibrated FITS. 
    
    Args:
        path_fits (str): FITS path.

    Returns:
        dict: dictionary with keywords 'ra_min', 'ra_max', 'dec_min', dec_max' in degrees.
    """
    fits = mcFits(path_fits)
    com1 = f'xy2sky -d {path_fits} 0 0'
    com2 = f'xy2sky -d {path_fits} {fits.header["NAXIS1"]} {fits.header["NAXIS2"]}'

    # print(f'{com1}')
    proc1 = subprocess.Popen(com1, shell=True, stdout=subprocess.PIPE)
    out1 = proc1.stdout.read().decode('utf-8')[:-2]
    data1 = [float(d) for d in out1.split()[:2]]
    # print(f'data1 = {data1}')
   
    # print(f'{com2}')
    proc2 = subprocess.Popen(com2, shell=True, stdout=subprocess.PIPE)
    out2 = proc2.stdout.read().decode('utf-8')[:-2]
    data2 = [float(d) for d in out2.split()[:2]]
    # print(f'data2 = {data2}')
   

    ras = [data1[0], data2[0]]
    decs = [data1[1], data2[1]]

    return {'ra_min': min(ras), 'ra_max': max(ras), 'dec_min': min(decs), 'dec_max': max(decs)}

def read_blazar_file(path, verbose=False):
    """It reads blazars info from file given by 'path'.
    
    It computes (ra, dec) blazar coordinates in degrees also.
    
    Args:
        path (str): path to blazars info CSV file.
        verbose (bool): If True, read info is plotted in stdout.
        
    Returns: 
        pandas.DataFrame: Information about IOP3 blazars.
    """
    df = pd.read_csv(path, comment='#')
    
    if verbose:
        print(df.info())
        
    # Getting sky coordinates in degrees
    c  = []
    for ra, dec in zip(df['ra2000_mc'], df['dec2000_mc']):
        c.append("{} {}".format(ra, dec))

    mapcat_coords = SkyCoord(c, frame=FK5, unit=(u.hourangle, u.deg), \
        obstime="J2000")
    df['ra2000_mc_deg'] = mapcat_coords.ra.deg
    df['dec2000_mc_deg'] = mapcat_coords.dec.deg
    
    return df

def closest_blazar(astronomical_deg_coords, blazar_path):
    """
    Get closest blazar to astronomical_deg_coords.
    
    Args:
        astronomical_deg_coords (tuple or list): Astronomical coordinates (ra, dec) in degrees.
        blazar_path (str): path to blazars info file.
        
    Returns:
        pandas.DataFrame: Info about closest blazar/s.
    """
    # read blazars file
    data = read_blazar_file(blazar_path)
    
    icoords = "{} {}".format(*(astronomical_deg_coords)) # tuple/list expansion
    input_coords = SkyCoord(icoords, frame=FK5, unit=(u.deg, u.deg), \
    obstime="J2000")
    
    # Blazars subset...
    df_blazars = data[data['IAU_name_mc'].notna()]  # take target sources from data
    c  = []
    for ra, dec in zip(df_blazars['ra2000_mc'], df_blazars['dec2000_mc']):
        c.append("{} {}".format(ra, dec))
    blazar_coords = SkyCoord(c, frame=FK5, unit=(u.hourangle, u.deg), \
    obstime="J2000")
    
    # Closest MAPCAT source to FITS central coordinates
    # Distance between this center FITS and MAPCAT targets
    distances = input_coords.separation(blazar_coords)
    
    # Closest source in complete set...
    i_min = distances.deg.argmin()
    
    blz_name = df_blazars['IAU_name_mc'].values[i_min]
    # print(f"Blazar closest source name = {blz_name}")
    cond = data['IAU_name_mc'] == blz_name
    
    return data[cond], distances.deg.min()

def detections_inside(data, ra_limits, dec_limits, \
    keywords={'RA': 'ra2000_mc_deg', 'DEC': 'dec2000_mc_deg'}):
    """Filter sources given by 'data' inside rectangular limits given by
    ra_limits and dec_limits.

    Args:
        data (pd.DataFrame): source info DataFrame.
        ra_limits (tuple/list): (ra_min, ra_max) in degrees.
        dec_limits (tuple/list): (dec_min, dec_max) in degrees.
        keywords (dict): 'RA', 'DEC' keywords for 'data' sky coordinates.
    Returns:
        pd.DataFrame: DataFrame with sources inside sky limits.
    """
    ##### ------ Searching for MAPCAT sources inside limits FITS coordinates ---- #####
    ra_min, ra_max = ra_limits[0], ra_limits[1]
    dec_min, dec_max = dec_limits[0], dec_limits[1]

    ra = keywords['RA']
    dec = keywords['DEC']
    dt = data.loc[data[ra] > ra_min]
    dt = dt.loc[dt[ra] < ra_max]
    dt = dt.loc[dt[dec] > dec_min]
    dt = dt.loc[dt[dec] < dec_max]
    
    return dt

def match_sources(ra1, dec1, ra2, dec2, num_close=1):
    """It matches catalog1 and catalog2. Coordinates for astrometric sky coordinates are given 
    by key_coord dictionaries (keywords 'RA' and 'DEC'). It returns num_close closest sources 
    in cat2 for each source in cat1.
    
    Args:
        ra1 (np.array): First catalog 'RA' coordinates (in degrees).
        dec1 (np.array): First catalog 'DEC' coordinates (in degrees).
        ra2 (np.array): Second catalog 'RA' coordinates (in degrees).
        dec2 (np.array): Second catalog 'DEC' coordinates (in degrees).
        num_close (int, optional): Number of closest sources in second set of coordinates 
            for each source first set of coordinates. Default: 1.
        
    Returns:
        tuple of tuples: (index/es in cat2 closest to cat1 ones, 2D-distances, 3d-distances).
            Each element is a tuple. Their len is given by cat1 sources.
    """
    # First catalog source sky coordinates
    c1 = SkyCoord(ra = ra1 * u.degree, dec = dec1 * u.degree)
    # Second catalog source sky coordinates
    c2 = SkyCoord(ra = ra2 * u.degree, dec = dec2 * u.degree)

    return match_coordinates_sky(c1, c2, nthneighbor=num_close)

def check_saturation(sext_flags):
    """Check for saturated SExtractor FLAGS in 'sext_flags'.
    
    As SExtractor manual says, if some source pixel si saturated then FLAGS take
    3th bit of FLAG to 1. That is, value 4 = 2^2 (3th bit) is activated
    
    Args:
        sext_flags (np.array): SExtractor FLAGS array.
        
    Returns:
        np.array of booleans: True means saturated FLAG.
    """
    # Binary codification and bit check
    bin_code = np.array([f"{format(int(flag), 'b') :0>8}"[-3] == '1' for flag in sext_flags], dtype=bool)
    
    return bin_code

def get_mapcat_sources(input_fits, blazar_path):
    """Search for MAPCAT sources included in area covered by FITS.

    Args:
        input_fits (str): FITS fille path.
        blazar_path (str): Path of file that contains MAPCAT sources.

    Returns:
        pd.DataFrame: It contains MAPCAT info provided by 'balazar_path'
            about sources in sky 'input_fits' area.
    """
    # RA,DEC limits...
    sky_limits = get_radec_limits(input_fits)
    
    # getting blazars info inside FITS sky area
    blazars_data = read_blazar_file(blazar_path)
    df_mc = detections_inside(blazars_data, \
        (sky_limits['ra_min'], sky_limits['ra_max']), \
        (sky_limits['dec_min'], sky_limits['dec_max']))
    
    return df_mc

def assoc_sources(catalog1, catalog2, max_deg_dist=0.0006, suffix='O'):
    """Match each source from 'catalog1' to closest one of 'catalog2'.

    Max matched distance betwwen sources is given by 'max_deg_dist' value.

    Args:
        catalog1 (pd.DataFrame): First catalog.
        catalog2 (pd.DataFrame): Second catalog.
        max_deg_dist (float, optional): Max distance between match sources in degrees. Defaults to 0.006.
        suffix (str, optional): Suffix to apply to output merged catalog. Defaults to 'O'.

    Returns:
        pd.DataFrame: Merged info from associated sources.
    """
    catalog2 = catalog2.reset_index()
    
      # Matching SExtractor detections with closest ORDINARY MAPCAT sources: 
    # Values returned: matched source indexes, 2D-distances, 3D-distances
    ra1 = catalog2['ra2000_mc_deg'].values
    dec1 = catalog2['dec2000_mc_deg'].values
    ra2 = catalog1['ALPHA_J2000'].values
    dec2 = catalog1['DELTA_J2000'].values
    idx, d2d, d3d = match_sources(ra1, dec1, ra2, dec2, num_close=1)

    # Selecting SExtractor closest sources
    data = catalog1.iloc[idx]
    data = data.reset_index()

    # matching catalog sources
    data['DISTANCE_DEG'] = list(d2d.deg)

    # Concatenating
    df_merge = pd.concat([catalog2, data], axis=1)
    print(f'(suffix, distances) =({suffix}, {d2d.deg})')
    # getting close enough matches
    fboo = df_merge['DISTANCE_DEG'] < max_deg_dist
    if fboo.sum() == 0:  # no near sources
        print('Merged catalogs')
        print(df_merge[['IAU_name_mc', 'ra2000_mc_deg', 'dec2000_mc_deg', \
            'ALPHA_J2000', 'DELTA_J2000', 'DISTANCE_DEG']])
    df_merge = df_merge[fboo]


    # renaming columns
    cols = {k: f'{k}_{suffix}' for k in df_merge.columns.values}
    df_merge.rename(columns = cols, inplace = True)

    # print(f' --------- Columnas para el merge del tipo {suffix} = {df_merge.columns.values}')

    return df_merge

def compute_zeropoint(input_fits, merged_data, output_png=None):
    """Compute zeropoint and plots saturated and not-saturated calibrators location.

    Args:
        input_fits (str): FITS path.
        merged_data (pd.DataFrame): Blazar and SExtractor closest detection info.

    Returns:
        tuple: (mag_zeropoint, magerr_zeropoint, num_non-saturated_calibrators)
    """ 
    i_fits = mcFits(input_fits)
     
    # checking for non-saturated calibrators
    print("checking Saturation")
    print(merged_data[['IAU_name_mc_O', 'DISTANCE_DEG_O', 'FLUX_APER_O', 'FLAGS_O', 'DISTANCE_DEG_E', 'FLUX_APER_E', 'FLAGS_E']])

    # filtering nan values
    flags_nan = np.isnan(merged_data['FLAGS_O'].values) | np.isnan(merged_data['FLAGS_E'].values)
    merged_data = merged_data[~flags_nan]
    ord_sat = check_saturation(merged_data['FLAGS_O'].values)
    ext_sat = check_saturation(merged_data['FLAGS_E'].values)
    
    # If ordinary or extraordinary counterpart is saturated, source 
    # is considered as saturated.
    sat_calibrator = np.logical_or(ord_sat, ext_sat)
    num_sat = sat_calibrator.sum()
    
    print(f'********************** {num_sat} calibrators is/are saturated. *********')
    if num_sat == ord_sat.size:
        # all calibrators are saturated, so no filtering operation will be applied
        print("----------- All calibrators are saturated. Non-saturation filter will not be applied --------------------")
        return None, None, 0

    # Getting coordinates
    non_sat_o_coords = [merged_data['ALPHA_J2000_O'][~sat_calibrator], \
        merged_data['DELTA_J2000_O'][~sat_calibrator]]
    non_sat_e_coords = [merged_data['ALPHA_J2000_E'][~sat_calibrator], \
        merged_data['DELTA_J2000_E'][~sat_calibrator]]
    sat_o_coords = [merged_data['ALPHA_J2000_O'][sat_calibrator], \
        merged_data['DELTA_J2000_O'][sat_calibrator]]
    sat_e_coords = [merged_data['ALPHA_J2000_E'][sat_calibrator], \
        merged_data['DELTA_J2000_E'][sat_calibrator]]
    
    coords = [non_sat_o_coords, non_sat_e_coords, sat_o_coords, sat_e_coords]
    
    # plotting saturated and non-saturated calibrators
    if not output_png:
        root, ext = os.path.splitext(input_fits)
        output_png = root + '_photo-calibrators.png'
        
    title_temp = "Photo-calibrators in {}, {} : used (green) and rejected (red)"
    header_data = i_fits.get_data(keywords=['OBJECT', 'DATE-OBS'])
    title = title_temp.format(i_fits.header['OBJECT'], i_fits.header['DATE-OBS'])
    print(f'output_png = {output_png}')
    i_fits.plot(output_png, title=title, \
        colorBar=True, ref_coords='world', astroCal=True, \
        color=['green', 'green', 'red', 'red'], coords=coords, \
        dictParams={'invert':'True'})

    # non-saturated calibrators total flux (Ordinary + Extraordinary)
    # Using SExtractor APER measures
    if 'MAPCAT' in input_fits:
        calibrators_total_flux = (merged_data['FLUX_APER_O'] + \
                                      merged_data['FLUX_APER_E'])[~sat_calibrator]
    else:
        calibrators_total_flux = merged_data['FLUX_APER_O'][~sat_calibrator]
    
    # Computing ZEROPOINT using non-saturated calibrators (or all of them if 
    # they are all saturated)
    zps = merged_data['Rmag_mc_O'][~sat_calibrator].values + 2.5 * np.log10(calibrators_total_flux.values)
    
    return zps.mean(), zps.std(), len(zps)


def merge_mapcat_sextractor(df_sext, df_mc, input_fits, max_deg_dist=0.0006):
    """_summary_

    Args:
        def_sext (pd.DataFrame): SExtractor output catalog.
        df_mc (pd.DataFrame): Blazar IOP3 sources info.
        input_fits (str): FITS path.
        max_deg_dist (float, optional): MAx distance between matched sources. Defaults to 0.0006.

    Returns:
        pd.DataFrame: Output merged catalog.
    """
    root, ext = os.path.splitext(input_fits) 
    i_fits = mcFits(input_fits)
    header = i_fits.header

    data_match_o = assoc_sources(df_sext, df_mc, max_deg_dist=max_deg_dist, suffix='O')

    print('-----ORD. DATA ASSOC-----')
    print(data_match_o)
    try:
        f_source = data_match_o['IAU_name_mc_O'].str.len() > 0
    except TypeError:
        print('PHOTOCALIBRATION,ERROR,"SExtractor has not detected close source to target IOP3."')
        return 1 
    # if len(df_mapcat[f_source].index) == 0:
    if f_source.sum() == 0:
        print(data_match_o[['id_mc_O', 'IAU_name_mc_O', 'DISTANCE_DEG_O']])
        print('PHOTOCALIBRATION,ERROR,"SExtractor has not detected close source to target IOP3."')
        return 2 
    else:
        print(f'Target = {data_match_o[f_source]}') 
    
    # Plotting MAPCAT sources
    if len(df_mc.index) > 0:
        mc_ra = data_match_o['ra2000_mc_deg_O'].values
        mc_dec = data_match_o['dec2000_mc_deg_O'].values 
        sources_mapcat_png = f'{root}_mapcat_sources.png'
        title_temp = "{}, {} ({} s)"
        title = title_temp.format(header['OBJECT'], header['DATE-OBS'], header['EXPTIME'])
        i_fits.plot(sources_mapcat_png, title=title, astroCal=True, color='magenta', \
            coords=(mc_ra, mc_dec), dictParams={'aspect':'auto', 'invert':'True'})
        print('Out PNG ->', sources_mapcat_png)
    else:
        print(f'ERROR: No closer enough MAPCAT sources found for this input FITS: {input_fits}')
        return 3
    
    # Printing info about MAPCAT-SExtractor sources
    str_match_mapcat = " MAPCAT (name, ra, dec, Rmag, Rmagerr) = ({}, {}, {}, {}, {})"
    str_match_sext = "SExtractor (ra, dec, mag_aper, magerr_aper) = ({}, {}, {}, {})"
    str_dist = "Distance = {}\n-----------------------"
    for j, row in data_match_o.iterrows():
        print(str_match_mapcat.format(row['name_mc_O'], row['ra2000_mc_deg_O'], row['dec2000_mc_deg_O'], \
            row['Rmag_mc_O'], row['Rmagerr_mc_O']))
        print(str_match_sext.format(row['ALPHA_J2000_O'], row['DELTA_J2000_O'], \
            row['MAG_APER_O'], row['MAGERR_APER_O']))
        print(str_dist.format(row['DISTANCE_DEG_O']))

    # Extraordinary counterparts location
    # rough coordinates (relative to ordinary source locations)
    df_mc_e = df_mc.copy()
    if 'MAPCAT' in input_fits:
        df_mc_e['dec2000_mc_deg'] = df_mc['dec2000_mc_deg'] - 0.0052
    else:
        df_mc_e['dec2000_mc_deg'] = df_mc['dec2000_mc_deg']
    data_match_e = assoc_sources(df_sext, df_mc_e, max_deg_dist=max_deg_dist, suffix='E')

    print('-----EXTRAORD. DATA ASSOC-----')
    print(data_match_e)

    print(type(data_match_o.index.values[0]))
    # getting common sources
    common_indexes = np.array(list(set(data_match_o.index.values).intersection(set(data_match_e.index.values))))
    if len(common_indexes) == 0:
        print(f'(len(data_match_e), len(data_match_o)) = ({len(data_match_e.index.values)}, {len(data_match_o.index.values)})')
        print('PHOTOMETRY,ERROR,"Different number of ORDINARY and EXTRAORDINARY sources."')
        return 4
    print(f'Common indexes = {common_indexes}')
    print(data_match_o.loc[common_indexes])
    print(f'indexes (data_match_o, data_match_e) = ({data_match_o.index.values}, {data_match_e.index.values})')
    data_matched = pd.concat([data_match_o.loc[common_indexes], data_match_e.loc[common_indexes]], axis=1)

    return data_matched

def main():
    parser = argparse.ArgumentParser(prog='iop3_photometric_calibration.py', \
    conflict_handler='resolve',
    description='''Main program that perfoms input FITS photometric calibration. ''',
    epilog='''''')
    parser.add_argument("config_dir", help="Configuration parameter files directory")
    parser.add_argument("output_dir", help="Output base directory for FITS calibration")
    parser.add_argument("input_fits", help="Astrocalibrated input FITS file")
    parser.add_argument("--border_image",
       action="store",
       dest="border_image",
       type=int,
       default=15,
       help="Number of pixels close to border. They will be ignored in computations [default: %(default)s].")
    parser.add_argument("--aper_pix",
       action="store",
       dest="aper_pix",
       type=float,
       default=None,
       help="Aperture (in pixels) for computing photometry [default: %(default)s].")
    parser.add_argument("--blazars_info",
       action="store",
       dest="blazars_info",
       type=str,
       default='blazar_photo_calib_last.csv',
       help="File name (located in config_dir) with information about blazar/star targets [default: %(default)s].")
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    parser.add_argument("--overwrite", dest='overwrite', action='store_true', \
        help='Pipeline overwrite previous calibrations.')
    parser.add_argument("--use_mean_fwhm", dest='use_mean_fwhm', action='store_true', \
        help='For each consecutive OBJECT observations use mean FWHM for commputing aperture.')
    parser.add_argument('-v', '--verbose', action='count', default=0,
        help="Show running and progress information [default: %(default)s].")
    args = parser.parse_args()

    # Checking input parameters
    if not os.path.exists(args.config_dir):
        str_err = 'ERROR: Config dir "{}" not available'
        print(str_err.format(args.config_dir))
        return 1

    if not os.path.isdir(args.output_dir):
        try:
            os.makedirs(args.output_dir)
        except IOError:
            str_err = 'ERROR: Could not create output directory "{}"'
            print(str_err.format(args.output_dir))
            return 2

    if not os.path.exists(args.input_fits):
        str_err = 'ERROR: Input FITS file "{}" not available'
        print(str_err.format(args.input_fits))
        return 3

    input_fits = os.path.abspath(args.input_fits)
    os.chdir(args.output_dir)
    print(f"\nWorking directory set to '{args.output_dir}'\n")

    # Reading input fits header
    print(input_fits)
    i_fits = mcFits(input_fits)
    astro_header = i_fits.header

    # search for photometry results file
    photo_res_path = glob.glob(os.path.join(args.output_dir, '*photocal_res.csv'))

    # Check for previous photometric calibration
    if 'MAGZPT' in astro_header and len(photo_res_path) > 0 and not args.overwrite:
        print(f'Input file "{input_fits}" already photo-calibrated. Nothing was done!')
        return 4
    
    text = 'OBJECT and Polarization angle = {}\nEXPTIME = {} s'
    print(text.format(astro_header['OBJECT'], astro_header['EXPTIME']))

    # ---------------------- MAPCAT sources info -----------------------
    blazar_path = os.path.join(args.config_dir, args.blazars_info)
    # blazar_path = os.path.join(args.config_dir, 'blazar_photo_calib_last.csv')
    # df_mapcat = read_blazar_file(blazar_path)

    center_fits = (astro_header['CRVAL1'], astro_header['CRVAL2'])
    print(f'center FITS coordinates = {center_fits}')
    nearest_blazar, min_dist_deg = closest_blazar(center_fits, blazar_path)
    
    print(f'Closest blazar distance to FITS\' center= {min_dist_deg} (deg)')

    if min_dist_deg > 0.5: # distance in degrees
        print('!' * 100)
        print('ERROR: Not enough close blazar or HD star found (distance <= 0.5 deg)')
        print('!' * 100)
        return 5

    # closest blazar info
    print('Closest IOP3 blazar info')
    print(nearest_blazar)
  
    ################ WORKING ON ASTROCALIBRATED FITS ################
    
    print('*' * 50)
    message = "Number of sources used in astrometric calibration: {} (of {})"
    print(message.format(astro_header['WCSMATCH'], astro_header['WCSNREF']))
    print('*' * 50)

    # Getting MAPCAT sources in area covered by input_fits
    df_mc_o = get_mapcat_sources(input_fits, blazar_path)
    
    # print("MAPCAT filtered info...")
    print(df_mc_o.info())
    print(df_mc_o)
    print(f'Number of MAPCAT sources inside FITS sky area = {len(df_mc_o.index)}')

    root, ext = os.path.splitext(input_fits)
    cat = f'{root}_photocal.cat'
    sext_conf = os.path.join(args.config_dir, 'sex.conf')
    # dir_out = os.path.split(cat)[0]
    
    # Using SExtractor for detecting sources in FITS
    mc_aper = args.aper_pix
    if not mc_aper or math.isnan(mc_aper):
        # mc_aper = nearest_blazar['aper_mc'].iloc[0]
        # if np.isnan(np.array([mc_aper])[0]):
        #     mc_aper = astro_header['FWHM']
        print(f'PHOTOCALIBRATION,ERROR,"{mc_aper} is not a valid value for Photometric APERTURE"')
        return -99
    
    # Round to 1 decimal digit
    mc_aper = round(mc_aper, 1)
    
    # Round to integer
    # mc_aper = round(mc_aper)

    res = i_fits.detect_sources(sext_conf, cat, \
        additional_params=sext_params_detection(input_fits), \
        photo_aper=mc_aper, aper_image=True, verbose=True)
    if res:
        print('PHOTOCALIBRATION,ERROR,"SExtractor did not work properly"')
        return 6

    sources_sext_png = f'{root}_sextractor_sources.png'
    if plot_cat(input_fits, sources_sext_png, cat, astro_coords=True, cat_format='FITS_LDAC'):
        print(f'PHOTOCALIBRATION,ERROR,"Could not plot {sources_sext_png}"')
        return 8
    print('Out PNG ->', sources_sext_png)

    # Merging data: MAPCAT & SExtractor
    df_SExtractor = read_sext_catalog(cat, format='FITS_LDAC')
    data_matched = merge_mapcat_sextractor(df_SExtractor, df_mc_o, input_fits, max_deg_dist=0.002)

    print('------- DATA MATCHED------')
    print(data_matched)
    try:
        source_problem = data_matched['IAU_name_mc_O'].str.len() > 0
    except TypeError:
        return 9

    print(f"Source problem = '{source_problem}'")
    info_target = data_matched[source_problem]
    print("Info target")
    print(info_target)
    source_pair_png = ''
    if len(info_target.index) > 0:
        ra_o, dec_o = info_target['ALPHA_J2000_O'].values[0], info_target['DELTA_J2000_O'].values[0]
        ra_e, dec_e = info_target['ALPHA_J2000_E'].values[0], info_target['DELTA_J2000_E'].values[0]

    #Get reference star as the one closest to the blazar
        is_blz=False
        if len(source_problem)>1:
            is_blz=True
    
        if is_blz:

            info_stars = data_matched[~source_problem]
            '''
            #Uncomment for select star with closest flux to the blazar
            flux_stars=(info_stars['FLUX_APER_O']+info_stars['FLUX_APER_E'])/2
            flux_blz=(info_target['FLUX_APER_O']+info_target['FLUX_APER_E'])/2
            diff=np.sqrt((flux_stars.values-flux_blz.values)**2)
            ref_idx=np.argmin(diff)+1
            print(ref_idx)
            source_problem[ref_idx]=True
            refstar_Rmag=info_stars['Rmag_mc_O'].values[0]
            '''

            ra_o_star, dec_o_star = info_stars['ALPHA_J2000_O'].values, info_stars['DELTA_J2000_O'].values
            ra_e_star, dec_e_star = info_stars['ALPHA_J2000_E'].values, info_stars['DELTA_J2000_E'].values
            
            dist = np.sqrt((ra_o_star-ra_o)**2+(dec_o_star-dec_o)**2)
            ref_idx=np.argmin(dist)+1
            source_problem[ref_idx]=True

        #indexes_refstar=[idx_o[source_problem][1], idx_e[source_problem][1]]
        #print(f'[Ordinary, Extraordinary] SExtractor indexes of reference star = {indexes_refstar}')

        # Plotting source problem
        # Showing detailed info about SExtractor counterparts
        if 'fits' in input_fits:
            source_pair_png = input_fits.replace('.fits', '_source_pair.png')
        else:
            source_pair_png = input_fits.replace('.fit', '_source_pair.png')

        print('Out PNG ->', source_pair_png)
        title_temp = "SExtractor Pair Detections {}, {} ({} s)"
        title = title_temp.format(astro_header['OBJECT'], astro_header['DATE-OBS'], \
                                      astro_header['EXPTIME'])
        i_fits.plot(source_pair_png, title=title, astroCal=True, \
                        coords=[(ra_o, dec_o), (ra_e, dec_e)], color=['red', 'blue']) 

    # Parameters to store...
    cal_data = defaultdict(list)
    
    # Photometric calibration
    mag_zeropoint = None
    std_mag_zeropoint = None
    num_sat = 0
    calibrators_png = ''

    # MAPCAT calibrators
    calibrators = data_matched[~source_problem]
    # If there are IOP3 calibrators in field covered by FITS
    if len(calibrators.index) > 0:                
        # Computing ZEROPOINT
        root, ext = os.path.splitext(input_fits)
        calibrators_png = root + '_photo-calibrators.png'
        # At this point, SExtractor MAG_AUTO is used for getting magnitude zero-point.
        # If a calibrator is saturated, then is rejected for mangitude zero-point computation.
        mag_zeropoint, std_mag_zeropoint, num_calibrators = compute_zeropoint(input_fits, \
            calibrators, output_png=calibrators_png)

        # getting saturated calibrators
        flags_cal_o = calibrators['FLAGS_O'].values
        flags_cal_e = calibrators['FLAGS_E'].values
        try:
            ord_sat = check_saturation(flags_cal_o)    
            ext_sat = check_saturation(flags_cal_e)    
            # If ordinary or extraordinary counterpart is saturated, source 
            # is considered as saturated.
            sat_calibrator = np.logical_or(ord_sat, ext_sat)
            num_sat = sat_calibrator.sum()
        except: 
            print(f'PHOTOCALIBRATION,ERROR,"Non valid calibrator FLAGS\nO = {flags_cal_o}\nE = {flags_cal_e}"')
            mag_zeropoint = -99
            std_mag_zeropoint = 0
            num_calibrators = 0

        # Plotting calibrators
        mc_calib_png = f'{root}_sources_mc_calib.png'
        print('Out PNG ->', mc_calib_png)
        title_temp = "Calibrators in {}, {} ({} s)"
        title = title_temp.format(astro_header['OBJECT'], astro_header['DATE-OBS'], \
            astro_header['EXPTIME'])

        # aliases
        ra_cal_o = calibrators['ra2000_mc_deg_O'].values
        dec_cal_o = calibrators['dec2000_mc_deg_O'].values
        ra_cal_e = calibrators['ra2000_mc_deg_E'].values
        dec_cal_e = calibrators['dec2000_mc_deg_E'].values

        i_fits.plot(mc_calib_png, title=title, astroCal=True, \
            color=['green', 'green'], \
            coords=[(ra_cal_o, dec_cal_o), (ra_cal_e, dec_cal_e)], \
            dictParams={'aspect':'auto', 'invert':'True'})
        cal_data['MC_CALIB_PNG'] = [mc_calib_png]
    else:
        num_calibrators = 1  
        num_sat = 1
        
        # because Mag = ZP - 2.5 * log10(Flux) => ZP = Mag + 2.5 * log10(Flux)
        # Then, as HD is a polarized source, I'll take as FLUX the sum of both
        # (Flux_o + Flux_e)
        try:
            # SExtractor AUTO measures used in photometric calibration
            # total_flux = (data['FLUX_AUTO'][indexes_target]).sum()
            if 'MAPCAT' in input_fits:
                fluxes = info_target['FLUX_AUTO_O'] + info_target['FLUX_AUTO_E']
            else:
                fluxes = info_target['FLUX_AUTO_O']

            total_flux = fluxes.values.sum()
            mag_zeropoint = info_target['Rmag_mc_O'].values[0] + \
                2.5 * np.log10(total_flux)
            std_mag_zeropoint = 0
        except ValueError:
            message = "Ordinary and extraordinary fluxes auto = ({}, {})"
            print(message.format(info_target['FLUX_AUTO_O'], info_target['FLUX_AUTO_E']))
            raise

    print(f'mag_zeropoint = {mag_zeropoint}')
    if mag_zeropoint is None or np.isnan(mag_zeropoint) or mag_zeropoint is None:
        print(f'PHOTOMETRY,ERROR,"Could not compute MAG_ZEROPOINT for \'{input_fits}\'"')
        return 8

    # date run info
    i_fits = mcFits(args.input_fits)

    dt_run = i_fits.run_date()
    date_run = dt_run.strftime("%Y-%m-%d")

    print(f"Photometric Zero-point = {round(mag_zeropoint, 2)}")
    print(f"STD(Photometric Zero-point) = {round(std_mag_zeropoint, 2)}")

    # --------------- Updating FITS header --------------------- #
    params = OrderedDict()
    params['MAGZPT'] = round(mag_zeropoint, 2)
    params['STDMAGZP'] = round(std_mag_zeropoint, 2)
    params['NSZPT'] = num_calibrators
    params['APERPIX'] = mc_aper
    if len(info_target.index) > 0:
        params['BLZRNAME'] = info_target['IAU_name_mc_O'].values[0]
    else:
        params['BLZRNAME'] = nearest_blazar['IAU_name_mc'].values[0]

    print(f'Header params = {params}')

    # Composing cards
    cards = [('MAGZPT', round(mag_zeropoint, 2), 'MAPCAT Photometric zeropoint'), \
        ('STDMAGZP', round(std_mag_zeropoint, 2), 'MAPCAT STD(Photometric zeropoint)'), \
        ('NSZPT', num_calibrators, 'MAPCAT number of calibrators used in ZEROPOINT estimation'), \
        ('APERPIX', mc_aper, 'Aperture in pixel for photometry calibration'), \
        ('BLZRNAME', params['BLZRNAME'], 'IAU name of BLAZAR'), \
        ('RUN_DATE', date_run, 'Night run date')]
    
    print(f'PHOTOCALIBRATION,INFO,"Updating calibration keywords of FITS \'{args.input_fits}\'"')
    if i_fits.update_header(cards):
        print(f'PHOTOCALIBRATION,ERROR,"Could not update photocalibration header in \'{input_fits}\'"')
        return 9

    cal_data['PATH'].append(input_fits)
    cal_data['RUN_DATE'].append(date_run)
    cal_data['CROTATION'].append(i_fits.header.get('CROTATION', ''))
    if calibrators_png and os.path.exists(calibrators_png):
        cal_data['CALIBRATORS_PNG'] = [calibrators_png]
    else:
        cal_data['CALIBRATORS_PNG'] = ''
    cal_data['N_CALIBRATORS'] = [num_calibrators]
    cal_data['N_SAT_CALIBRATORS'] = [num_sat]
    cal_data['SEXTDET_PNG'] = [sources_sext_png]
    cal_data['MAPCAT_SOURCES_PNG'] = [f'{root}_mapcat_sources.png']
    cal_data['SOURCE_PAIRS_PNG'] = [source_pair_png]
    
    df = pd.DataFrame(cal_data)
    csv_out = f'{root}_photocal_process_info.csv'    
    df.to_csv(csv_out, index=False)
    return 0
    
# -------------------------------------
if __name__ == '__main__':
    print(main())
