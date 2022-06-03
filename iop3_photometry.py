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
import astropy.wcs as wcs

# Coordinate system transformation package and modules
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import match_coordinates_sky  # Used for searching sources in catalog
from astropy.coordinates import FK5  # Low-level frames
import astropy.coordinates as coord
import astropy.units as u
from astropy.time import Time


from mcFits import mcFits

# HTML ouput template
import jinja2


# Photutils (aperture flux measurements)
from photutils.aperture import aperture_photometry, CircularAperture, CircularAnnulus

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
    
    result = subprocess.run(cmd, stdout=out, stderr=err, shell=shell, check=True)
    
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
    fits_par = i_fits.get_data(keywords=['INSTRSCL', 'FWHM', 'EXPTIME', 'OBJECT', 'DATE-OBS'])
    
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

    print(f'{com1}')
    proc1 = subprocess.Popen(com1, shell=True, stdout=subprocess.PIPE)
    out1 = proc1.stdout.read().decode('utf-8')[:-2]
    data1 = [float(d) for d in out1.split()[:2]]
    print(f'data1 = {data1}')
   
    print(f'{com2}')
    proc2 = subprocess.Popen(com2, shell=True, stdout=subprocess.PIPE)
    out2 = proc2.stdout.read().decode('utf-8')[:-2]
    data2 = [float(d) for d in out2.split()[:2]]
    print(f'data2 = {data2}')
   

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

def closest_blazar(blazar_path, path_fits):
    """"""
    # Getting header informacion
    i_fits = mcFits(path_fits)
    input_head = i_fits.header
    
    # Central FITS coordinates
    icoords = "{} {}".format(input_head['CRVAL1'], input_head['CRVAL2'])
    input_coords = SkyCoord(icoords, frame=FK5, unit=(u.deg, u.deg), \
    obstime="J2000")

    blazar_data = read_blazar_file(blazar_path)
    # Blazars subset...
    df_blazars = blazar_data[blazar_data['IAU_name_mc'].str.len() > 0]
    c  = []
    for ra, dec in zip(df_blazars['ra2000_mc_deg'], df_blazars['dec2000_mc_deg']):
        c.append("{} {}".format(ra, dec))
    blazar_coords = SkyCoord(c, frame=FK5, unit=(u.deg, u.deg), \
        obstime="J2000")
    
    # Closest MAPCAT source to FITS central coordinates
    # Distance between this center FITS and MAPCAT targets (in degrees)
    distances = input_coords.separation(blazar_coords)
    
    # Closest source in complete set...
    i_min = distances.deg.argmin()
    
    return df_blazars.iloc[i_min], distances.deg[i_min]

# def closest_blazar(astronomical_deg_coords, blazar_path):
#     """
#     Get closest blazar to astronomical_deg_coords.
    
#     Args:
#         astronomical_deg_coords (tuple or list): Astronomical coordinates (ra, dec) in degrees.
#         blazar_path (str): path to blazars info file.
        
#     Returns:
#         pandas.DataFrame: Info about closest blazar/s.
#     """
#     # read blazars file
#     data = read_blazar_file(blazar_path)
    
#     icoords = "{} {}".format(*(astronomical_deg_coords)) # tuple/list expansion
#     input_coords = SkyCoord(icoords, frame=FK5, unit=(u.deg, u.deg), \
#     obstime="J2000")
    
#     # Blazars subset...
#     df_blazars = data[data['IAU_name_mc'].notna()]  # take target sources from data
#     c  = []
#     for ra, dec in zip(df_blazars['ra2000_mc'], df_blazars['dec2000_mc']):
#         c.append("{} {}".format(ra, dec))
#     blazar_coords = SkyCoord(c, frame=FK5, unit=(u.hourangle, u.deg), \
#     obstime="J2000")
    
#     # Closest MAPCAT source to FITS central coordinates
#     # Distance between this center FITS and MAPCAT targets
#     distances = input_coords.separation(blazar_coords)
    
#     # Closest source in complete set...
#     i_min = distances.deg.argmin()
    
#     blz_name = df_blazars['IAU_name_mc'].values[i_min]
#     # print(f"Blazar closest source name = {blz_name}")
#     cond = data['IAU_name_mc'] == blz_name
    
#     return data[cond], distances.deg.min()

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
    """
    It matches catalog1 and catalog2. Coordinates for astrometric sky coordinates are given 
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
    c1 = SkyCoord(ra = ra1 * u.degree, dec = dec1 * u.degree)
    # sextractor catalog
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
    # Using SExtractor AUTO measures (not APER)
    if 'MAPCAT' in input_fits:
        calibrators_total_flux = (merged_data['FLUX_AUTO_O'] + \
                                      merged_data['FLUX_AUTO_E'])[~sat_calibrator]
    else:
        calibrators_total_flux = merged_data['FLUX_AUTO_O'][~sat_calibrator]
    
    # Computing ZEROPOINT using non-saturated calibrators (or all of them if 
    # they are all saturated)
    zps = merged_data['Rmag_mc_O'][~sat_calibrator].values + 2.5 * np.log10(calibrators_total_flux.values)
    
    return zps.mean(), zps.std(), len(zps)

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

def assoc_sources(df_sext, df_mapcat, max_deg_dist=0.006, suffix='O'):
    """_summary_

    Args:
        df_sext (pd.DataFrame): SExtractor detections Pandas DataFrame
        df_mapcat (pd.DataFrame): IOP3 sources Pandas DataFrame
        max_deg_dist (float, optional): Maximum distance in arcs between IOP3 source coordinades and SExtractor detection. Defaults to 0.006.
        suffix (str, optional): Suffix to associated field name catalogs. Defaults to 'O'.

    Returns:
        pd.DataFrame: Information from both entries (IOP3 and SExtractor sources) associated following closest distance criterium.
    """
    df_mapcat = df_mapcat.reset_index()

    # Matching SExtractor detections with closest ORDINARY MAPCAT sources: 
    # Values returned: matched source indexes, 2D-distances, 3D-distances
    ra1 = df_mapcat['ra2000_mc_deg'].values
    dec1 = df_mapcat['dec2000_mc_deg'].values
    ra2 = df_sext['ALPHA_J2000'].values
    dec2 = df_sext['DELTA_J2000'].values
    idx, d2d, d3d = match_sources(ra1, dec1, ra2, dec2, num_close=1)

    # Selecting SExtractor closest sources
    df_sext = df_sext.iloc[idx]
    df_sext = df_sext.reset_index()

    # matching catalog sources
    df_sext['DISTANCE_DEG'] = list(d2d.deg)

    # Concatenating
    df_merge = pd.concat([df_mapcat, df_sext], axis=1)
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


# def merge_mapcat_sextractor(cat, df_mc, input_fits, max_deg_dist=0.0006):
#     """_summary_

#     Args:
#         cat (_type_): _description_
#         df_mc_o (_type_): _description_
#         input_fits (_type_): _description_
#         max_deg_dist (float, optional): _description_. Defaults to 0.006.

#     Returns:
#         _type_: _description_
#     """
#     root, ext = os.path.splitext(input_fits) 
#     i_fits = mcFits(input_fits)
#     header = i_fits.header

#     data_match_o = assoc_sources(cat, df_mc, max_deg_dist=max_deg_dist, suffix='O')

#     try:
#         f_source = data_match_o['IAU_name_mc_O'].str.len() > 0
#     except TypeError:
#         print('PHOTOCALIBRATION,ERROR,"SExtractor has not detected close source to target IOP3."')
#         return 1 
#     # if len(df_mapcat[f_source].index) == 0:
#     if f_source.sum() == 0:
#         print(data_match_o[['id_mc_O', 'IAU_name_mc_O', 'DISTANCE_DEG_O']])
#         print('PHOTOCALIBRATION,ERROR,"SExtractor has not detected close source to target IOP3."')
#         return 2 
#     else:
#         print(f'Target = {data_match_o[f_source]}') 
    
#     # Plotting MAPCAT sources
#     if len(df_mc.index) > 0:
#         mc_ra = data_match_o['ra2000_mc_deg_O'].values
#         mc_dec = data_match_o['dec2000_mc_deg_O'].values 
#         sources_mapcat_png = f'{root}_mapcat_sources.png'
#         title_temp = "{}, {} ({} s)"
#         title = title_temp.format(header['OBJECT'], header['DATE-OBS'], header['EXPTIME'])
#         i_fits.plot(sources_mapcat_png, title=title, astroCal=True, color='magenta', \
#             coords=(mc_ra, mc_dec), dictParams={'aspect':'auto', 'invert':'True'})
#         print('Out PNG ->', sources_mapcat_png)
#     else:
#         print(f'ERROR: No closer enough MAPCAT sources found for this input FITS: {input_fits}')
#         return 3
    
#     # Printing info about MAPCAT-SExtractor sources
#     # str_match_mapcat = " MAPCAT (name, ra, dec, Rmag, Rmagerr) = ({}, {}, {}, {}, {})"
#     # str_match_sext = "SExtractor (ra, dec, mag_auto, magerr_auto) = ({}, {}, {}, {})"
#     # str_dist = "Distance = {}\n-----------------------"
#     # for j, row in data_match_o.iterrows():
#     #     print(str_match_mapcat.format(row['name_mc_O'], row['ra2000_mc_deg_O'], row['dec2000_mc_deg_O'], \
#     #         row['Rmag_mc_O'], row['Rmagerr_mc_O']))
#     #     print(str_match_sext.format(row['ALPHA_J2000_O'], row['DELTA_J2000_O'], \
#     #         row['MAG_AUTO_O'], row['MAGERR_AUTO_O']))
#     #     print(str_dist.format(row['DISTANCE_DEG_O']))

#     # Extraordinary counterparts location
#     # rough coordinates (relative to ordinary source locations)
#     df_mc_e = df_mc.copy()
#     df_mc_e['dec2000_mc_deg'] = df_mc['dec2000_mc_deg'] - 0.0052
#     data_match_e = assoc_sources(cat, df_mc_e, max_deg_dist=max_deg_dist, suffix='E')

#     if len(data_match_e.index) != len(data_match_o.index):
#         print(f'(len(data_match_e), len(data_match_o)) = ({len(data_match_e.index)}, {len(data_match_o.index)})')
#         print('PHOTOMETRY,ERROR,"Different number of ORDINARY and EXTRAORDINARY sources."')
#         return 4

#     data_matched = pd.concat([data_match_o, data_match_e], axis=1)

#     return data_matched

def merge_mapcat_sextractor(df_sext, df_mc, input_fits, max_deg_dist=0.006):
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
    
    print('-------- ORD. DATA MATCHED ----------')
    print(data_match_o.info())
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
    str_match_sext = "SExtractor (ra, dec, mag_auto, magerr_auto) = ({}, {}, {}, {})"
    str_dist = "Distance = {}\n-----------------------"
    for j, row in data_match_o.iterrows():
        print(str_match_mapcat.format(row['name_mc_O'], row['ra2000_mc_deg_O'], row['dec2000_mc_deg_O'], \
            row['Rmag_mc_O'], row['Rmagerr_mc_O']))
        print(str_match_sext.format(row['ALPHA_J2000_O'], row['DELTA_J2000_O'], \
            row['MAG_AUTO_O'], row['MAGERR_AUTO_O']))
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

def background_flux(x_pix, y_pix, fits_path, \
    inner_aper_radius=10., outer_aper_radius=12.):
    """Compute background noise in annular areas
    given by one or more pair of (x, y) pixel coordinates and
    anular areas given by inner and outer radius (again in pixels).

    Args:
        x_pix (float or iterable): x pixel coordinate/s.
        y_pix (float or iterable): x pixel coordinate/s. Same length than 'x_pix'.
        fits_path (str): FITS image path where aperture will be measured.
        inner_aper_radius (float): internal annulus aperture radius in pixels.
        outer_aper_radius (float): external annulus aperture radius in pixels. 
            Outer radius should be greater than inner one.
    
    Returns:
        tuple: (list(annulus_fluxes), list(annulus_areas_in_pixels))
    """
    
    positions = [(x, y) for x, y in zip(list(x_pix), list(y_pix))]
    annulus_aperture = CircularAnnulus(positions, r_in=inner_aper_radius, \
        r_out=outer_aper_radius)

    i_fits = mcFits( fits_path) 
    data = i_fits.data 

    phot_table = aperture_photometry(data, annulus_aperture)

    bkg = phot_table['aperture_sum'].value
    bkg_mean = bkg / annulus_aperture.area

    return bkg, bkg_mean

def aperture_flux(x_pix, y_pix, fits_path, aper_pix):
    """Compute aperture flux inside a circular region.

    Args:
        x_pix (float or iterable): x pixel coordinate/s.
        y_pix (float or iterable): x pixel coordinate/s. Same length than 'x_pix'.
        fits_path (str): FITS image path where aperture will be measured.
        aper_pix (float or iterable): aperture radius in pixels.
        sustract_background (bool, optional): If True, backgroun is computed and 
            sustracter form aperture flux. Defaults to False.
        inner_aper_annulus (float): Internal annulus aperture radius in pixels.
        outer_aper_annulus (float): Internal annulus aperture radius in pixels.
    
    Returns:
        tuple: (list(aperture_fluxes), list(areas_in_pixels))
    """
    positions = [(x, y) for x, y in zip(list(x_pix), list(y_pix))]
    
    i_fits = mcFits( fits_path) 
    data = i_fits.data 

    aperture = CircularAperture(positions, r=aper_pix)
    phot_table = aperture_photometry(data, aperture) 
    phot_table["aperture_sum"].info.format = "%.8g" 

    areas = [ap.area for ap in aperture]

    return list(phot_table["aperture_sum"].value), areas

def main():
    parser = argparse.ArgumentParser(prog='iop3_photometry.py', \
    conflict_handler='resolve',
    description='''Main program that perfoms input FITS aperture photometry. ''',
    epilog='''''')
    parser.add_argument("config_dir", help="Configuration parameter files directory")
    parser.add_argument("output_dir", help="Output base directory for FITS calibration")
    parser.add_argument("input_fits", help="Astrocalibrated input FITS file")
    parser.add_argument("--aper_pix",
       action="store",
       dest="aper_pix",
       type=float,
       default=None,
       help="Aperture in pixels for BLAZAR photometry measuring [default: %(default)s].")
    parser.add_argument("--blazars_info",
       action="store",
       dest="blazars_info",
       type=str,
       default='blazar_photo_calib_last.csv',
       help="File name (located in config_dir) with information about blazar/star targets [default: %(default)s].")
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
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

    if args.aper_pix is None or math.isnan(args.aper_pix):
        print(f'PHOTOMETRY,ERROR,"Not valid {args.aper_pix} value for aperture."')
        return 4

    # date run info
    # Using input path for getting observation night date... (Is this the best way or should I read FITS header?)
    dt_run = re.findall('/(\d{6})/', args.input_fits)[0]
    date_run = f'20{dt_run[:2]}-{dt_run[2:4]}-{dt_run[-2:]}'

    input_fits = os.path.abspath(args.input_fits)
    os.chdir(args.output_dir)
    print(f"\nWorking directory set to '{args.output_dir}'\n")

    # Reading input fits header
    i_fits = mcFits(input_fits)
    header = i_fits.header

    text = 'OBJECT and Polarization angle = {}\nEXPTIME = {} s'
    print(text.format(header['OBJECT'], header['EXPTIME']))

    # ---------------------- MAPCAT sources info -----------------------
    blazar_path = os.path.join(args.config_dir, args.blazars_info)
    # center_fits = (header['CRVAL1'], header['CRVAL2'])
    nearest_blazar, min_dist_deg = closest_blazar(blazar_path, input_fits)
    
    print(f'Closest blazar distance = {min_dist_deg} (deg)')
    if min_dist_deg > 0.5: # distance in degrees
        print('!' * 100)
        print('ERROR: Not enough close blazar or HD star found (distance <= 0.5 deg)')
        print('!' * 100)
        return 5

    # closest blazar info
    print('Closest IOP3 blazar info')
    print(nearest_blazar)
  
    ################ WORKING ON ASTROCALIBRATED FITS ################
    
    # Getting MAPCAT sources in area covered by input_fits
    df_mc_o = get_mapcat_sources(input_fits, blazar_path)
    
    # print("MAPCAT filtered info...")
    print(df_mc_o.info())
    print(f'Number of MAPCAT sources inside FITS sky area = {len(df_mc_o.index)}')

    root, ext = os.path.splitext(input_fits)
    cat = f'{root}_photocal.cat'
    sext_conf = os.path.join(args.config_dir, 'sex.conf')
    # dir_out = os.path.split(cat)[0]
    
    # Setting aperture for photometry
    mc_aper = float(args.aper_pix)
    # if not mc_aper or math.isnan(mc_aper):
    #     mc_aper = nearest_blazar['aper_mc'].iloc[0]
    #     if np.isnan(np.array([mc_aper])[0]):
    #         mc_aper = header['FWHM']
    # Round to 1 decimal digit
    mc_aper = round(mc_aper, 1)

    # Using SExtractor for detecting sources in FITS
    a_params = sext_params_detection(input_fits)
    a_params['PARAMETERS_NAME'] = os.path.join(args.config_dir, 'apertures.param')

    print('----------- ADDITIONAL SExtractor input parameters --------')
    print(f'additional_params={a_params}')
    print(f'cat={cat}')
    print(f'mc_aper={mc_aper}')
    print(f"mag_zeropoint={header['MAGZPT']}")
    res = i_fits.detect_sources(sext_conf, cat, additional_params=a_params, \
        photo_aper=mc_aper, aper_image=True, mag_zeropoint=header['MAGZPT'])
    if res:
        print('PHOTOMETRY,ERROR,"SExtractor did not work properly"')
        return 6

    sources_sext_png = f'{root}_sextractor_sources.png'
    if plot_cat(input_fits, sources_sext_png, cat, astro_coords=True, cat_format='FITS_LDAC'):
        print(f'PHOTOMETRY,ERROR,"Could not plot {sources_sext_png}"')
        return 7
    print('Out PNG ->', sources_sext_png)

    # Merging data: MAPCAT & SExtractor
    df_sext = read_sext_catalog(cat, format='FITS_LDAC')
    data_matched = merge_mapcat_sextractor(df_sext, df_mc_o, input_fits, max_deg_dist=0.006)

    try:
        source_problem = data_matched['IAU_name_mc_O'].str.len() > 0
    except TypeError:
        return 8

    if source_problem.sum() == 1:
            info_target = data_matched[source_problem]
            info_stars = data_matched[~source_problem]
    else:
        print(f'ASTROCALIBRATION,ERROR,"No target source found."')
        return 9

    ra_o, dec_o = info_target['ALPHA_J2000_O'].values[0], info_target['DELTA_J2000_O'].values[0]
    ra_e, dec_e = info_target['ALPHA_J2000_E'].values[0], info_target['DELTA_J2000_E'].values[0]
    ra_stars, dec_stars = info_stars['ALPHA_J2000_O'].values[0], info_stars['DELTA_J2000_O'].values[0]

    if len(source_problem)>1:
        is_blz=True
        if is_blz:
            ra_blz=ra_o
            dec_blz=dec_o
                
            dist=np.sqrt((ra_blz-ra_stars)**2+(dec_blz-dec_stars)**2)
            ref_idx=np.argmin(dist)+1
            source_problem[ref_idx]=True
            refstar_Rmag=info_stars['Rmag_mc_O'].values[0]
    
    info_target = data_matched[source_problem]
    
    # Geting X,Y coordinates
    x_o, y_o = info_target['X_IMAGE_O'].values[0], info_target['Y_IMAGE_O'].values[0]
    x_e, y_e = info_target['X_IMAGE_E'].values[0], info_target['Y_IMAGE_E'].values[0]

    # COMPUTING APERTURE FLUXES. Now we have MAG_ZEROPOINT-------------------
    # Loading FITS_LDAC format SExtractor catalog
    data = read_sext_catalog(cat, format='FITS_LDAC')
    
    
    # root, ext = os.path.splitext(input_fits)
    # fits_name = os.path.split(root)[1]
    
    pair_params = defaultdict(list)

    pair_params['ID-MC'] = [info_target['id_mc_O'].iloc[0]]
    pair_params['ID-BLAZAR-MC'] = [info_target['id_blazar_mc_O'].values[0]]
    # pair_params['TYPE'] = ['O', 'E']
    if 'INSPOROT' in header:
        angle = float(header['INSPOROT'])
    else:
        if header['FILTER']=='R':
            angle = -999.0
        elif header['FILTER']=='R_45':
            angle = float(-45)
        else:
            angle = float(header['FILTER'].replace('R',''))
            
    pair_params['ANGLE'] = [round(angle, ndigits=1)]
    pair_params['OBJECT'] = [header['OBJECT']]
    
    d_obs = Time(header['DATE-OBS'])    
    pair_params['MJD-OBS'] = [d_obs.mjd]
    pair_params['RJD-50000'] = [d_obs.mjd - 50000 + 0.5]
    pair_params['DATE-OBS'] = ['']
    if 'DATE-OBS' in header:
        pair_params['DATE-OBS'] = [header['DATE-OBS']]
    else:
        pair_params['DATE-OBS'] = [header['DATE']]
    mc_name = info_target['name_mc_O'].values[0]
    mc_iau_name = info_target['IAU_name_mc_O'].values[0]
    pair_params['MC-NAME'] = [mc_name]
    pair_params['MC-IAU-NAME'] = [mc_iau_name]
    pair_params['MAGZPT'] = [header['MAGZPT']]
    pair_params['RUN_DATE'] = [date_run]
    pair_params['EXPTIME'] = [header['EXPTIME']]
    pair_params['APERPIX'] = [mc_aper]

    pair_params['FWHM'] = [header['FWHM']]

    # Transforming from degrees coordinates (ra, dec) to ("hh mm ss.ssss", "[sign]dd mm ss.sss") representation
    print('----------- INFO TARGET ----------')
    print(info_target)
    # Ordinary
    #coordinates_O=np.array([])
    info_target['RA_J2000_O']=info_target['ALPHA_J2000_O'] #To get the same shape
    info_target['DEC_J2000_O']=info_target['DELTA_J2000_O'] #To get the same shape
   
    info_target['RA_J2000_E']=info_target['ALPHA_J2000_E'] #To get the same shape
    info_target['DEC_J2000_E']=info_target['DELTA_J2000_E'] #To get the same shape
    
    for i in range(0,info_target.shape[0]):
        coo_O = f"{info_target['ALPHA_J2000_O'].values[i]} {info_target['DELTA_J2000_O'].values[i]}"
        coordinates_O = SkyCoord(coo_O, frame=FK5, unit=(u.deg, u.deg), obstime="J2000")

        coo_E = f"{info_target['ALPHA_J2000_E'].values[i]} {info_target['DELTA_J2000_E'].values[i]}"
        coordinates_E = SkyCoord(coo_E, frame=FK5, unit=(u.deg, u.deg), obstime="J2000")
        

        #Ordinary
        info_target['RA_J2000_O'][i] = [coordinates_O.ra.to_string(unit=u.hourangle, sep=' ', \
                                                                    precision=4, pad=True)]
        info_target['DEC_J2000_O'][i] = [coordinates_O.dec.to_string(unit=u.deg, sep=' ', \
                                                                      precision=4, alwayssign=True, pad=True)]

        #Extraordinary
        info_target['RA_J2000_E'][i] = [coordinates_E.ra.to_string(unit=u.hourangle, sep=' ', \
                                                                    precision=4, pad=True)]
        info_target['DEC_J2000_E'][i] = [coordinates_E.dec.to_string(unit=u.deg, sep=' ', \
                                                                      precision=4, alwayssign=True, pad=True)]
    

    # Adding aperture (in pixels)
    print([mc_aper])
    info_target['APERPIX'] = [[mc_aper]] * info_target.shape[0]
    info_target['FWHM'] = [i_fits.header['FWHM']] * info_target.shape[0]
    if 'SECPIX' in i_fits.header:
        info_target['SECPIX'] = [i_fits.header['SECPIX']] * info_target.shape[0]
    else:
        mean_secpix = np.nanmean(np.array([i_fits.header['SECPIX1'], i_fits.header['SECPIX2']]))

    #info_target['SECPIX'] = [round(mean_secpix, 2)] * info_target.shape[0]
    #info_target['DATE-OBS'] = [i_fits.header['DATE-OBS']] * info_target.shape[0]
    #info_target['MJD-OBS'] = [pair_params['MJD-OBS']] * info_target.shape[0]
    #info_target['RJD-50000'] = [pair_params['MJD-OBS'] - 50000 + 0.5] * info_target.shape[0]
    #info_target['EXPTIME'] = [i_fits.header['EXPTIME']] * info_target.shape[0]
    #info_target['ANGLE'] = [round(angle, ndigits=1)] * info_target.shape[0]
    #info_target['MAGZPT'] = [round(header['MAGZPT'], 2)] * info_target.shape[0]

    info_target['SECPIX'] = [round(mean_secpix, 2)]
    info_target['DATE-OBS'] = [i_fits.header['DATE-OBS']]
    info_target['MJD-OBS'] = [Time(i_fits.header['DATE-OBS']).mjd]
    info_target['RJD-50000'] = [Time(i_fits.header['DATE-OBS']).mjd - 50000 + 0.5]
    info_target['EXPTIME'] = [i_fits.header['EXPTIME']]
    info_target['ANGLE'] = [round(angle, ndigits=1)]
    info_target['MAGZPT'] = [round(header['MAGZPT'], 2)]


    # df = pd.DataFrame(pair_params)
    csv_out = f'{root}_photometry.csv'
    info_target.to_csv(csv_out, index=False)

    # Imprimo el contenido del fichero
    print('Useful parameters for polarimetric computations:')
    print(info_target.keys())
    for a in info_target.values:
        print(a)
    
    # Write aperture in input FITS header
    # In any case, aperture is written in FITS header as APERPIX keyword
    new_card = [('APERPIX', round(args.aper_pix, 1), 'Aperture (pix) used in photometry')]
    i_fits.update_header(cards=new_card)
    

    return 0

# -------------------------------------
if __name__ == '__main__':
    print(main())
