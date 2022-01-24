#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu April 15 17:38:23 2021

___e-mail__ = cesar_husillos@tutanota.com
__author__ = 'Cesar Husillos'

VERSION:
    0.1 Initial version, based on CAFOS_wcstools_perfect_pruebas_revision_HD.ipynb
    0.2 (Sat April 17, 2021) MAPCAT apertures are considered.
"""


# ---------------------- IMPORT SECTION ----------------------
from dataclasses import replace
import os
import argparse
import subprocess
import re
from collections import defaultdict

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


from mcFits import mcFits

# HTML ouput template
import jinja2

# =================================
from astropy.utils.exceptions import AstropyWarning, AstropyDeprecationWarning
import warnings

# Ignore too many FITSFixedWarnings
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.simplefilter('ignore', category=AstropyDeprecationWarning)

# =================================

# ------------------------ Module functions ------------------------------ #
def plotFits(inputFits, outputImage, title=None, colorBar=True, coords=None, \
    ref_coords='world', astroCal=False, color='green', \
    dictParams={'aspect':'auto', 'vmin': 1, 'invert': True}, format='png'):
    """Plot 'inputFits' as image 'outputImage'.
    
    Args:
        inputFits (str): FITS input path.
        outputImage (str): Output plot path.
        title (str): Plot title.
        colorBar (bool): If True, colorbar is added to right side of output plot.
        coords (list or list of list): [ras, decs] or [[ras, decs], ..., [ras, decs]]
        ref_coords (str): 'world' or 'pixel'.
        astroCal (bool): True if astrocalibration was done in 'inputFits'.
        color (str or list): valid color identifiers. If 'coords' is a list of lists,
            then 'color' must be a list with length equals to lenght of 'coords' parameter.
        dictParams (dict): aplpy parameters.
        format (str): output plot file format.
        
    Return:
        0 is everything was fine. Exception in the other case.
        
    Raises:
        IndexError if list lenghts of 'coords' and 'color' are different.
        
    """
    gc = aplpy.FITSFigure(inputFits, dpi=300)
    #gc.set_xaxis_coord_type('scalar')
    #gc.set_yaxis_coord_type('scalar')
    gc.show_grayscale(**dictParams)
    #gc.recenter(512, 512)
    gc.tick_labels.set_font(size='small')
    if title:
        gc.set_title(title)
    if colorBar:
        gc.add_colorbar()
        gc.colorbar.show()
    gc.add_grid()
    gc.grid.set_color('orange')
    if astroCal:
        gc.grid.set_xspacing(1./60) # armin

    gc.grid.show()
    gc.grid.set_alpha(0.7)
    if coords:
        if type(coords[0]) == type(list()):
            for i, l in enumerate(coords):
                ra, dec = l[0], l[1]
                gc.show_markers(ra, dec, edgecolor=color[i], facecolor='none', \
                    marker='o', coords_frame=ref_coords, s=40, alpha=1)
        else:
            ra, dec = coords[0], coords[1]
            gc.show_markers(ra, dec, edgecolor=color, facecolor='none', \
                marker='o', coords_frame=ref_coords, s=40, alpha=1)
    gc.save(outputImage, format=format)
    #gc.close()

    return 0

def default_detection_params(exptime):
    """ Default values for some SExtractor detection parameters.

    They depend on exptime.

    Args:
        exptime (float): Exposure time in seconds.

    Return:
        detect_params (dict): Best test parameters for source-extractor detection

    Raises:
        ValueError, if exptime can't be cast to float.
    """
    try:
        et = float(exptime)
    except ValueError:
        raise

    det_filter = 'N'
    det_clean = 'N'
    minarea = 13
    an_thresh = 1.0
    det_thresh = 1.0
    deb_mincon = 0.1

    if et > 0.2:
        det_clean = 'N'
        deb_mincon = 0.005
    if et >= 1:
        det_filter = 'Y'
        det_clean = 'Y'
        minarea = 9
        an_thresh = 1.0
        det_thresh = 1.0
    # if et >= 80:
    if et >= 20:
        minarea = 13
        an_thresh = 2.5
        det_thresh = 2.5
    # if et >= 100:
    #     pass
    # if et >= 120:
    #     pass
    # if et >= 180:
    if et >= 120:
        minarea = 9
        an_thresh = 1.6
        det_thresh = 1.6

    detect_params = {}

    # setting config parameters
    detect_params['FILTER'] = det_filter
    detect_params['CLEAN'] = det_clean
    detect_params['DETECT_MINAREA'] = minarea
    detect_params['ANALYSIS_THRESH'] = an_thresh
    detect_params['DETECT_THRESH'] = det_thresh
    detect_params['DEBLEND_MINCONT'] = deb_mincon

    return detect_params

def relaxed_detection_params(exptime):
    """ Relaxed values for some SExtractor detection parameters.
    
        SExtractor used parameters are broader than for high exposure FITS.

    Args:
        exptime (float): Exposure time in seconds.

    Return:
        detect_params (dict): Best tested parameters for source-extractor detection.

    Raises:
        ValueError, if exptime can't be cast to float.
    """
    try:
        et = float(exptime)
    except ValueError:
        raise

    det_filter = 'N'
    det_clean = 'N'
    minarea = 10
    an_thresh = 0.1
    det_thresh = 0.1
    deb_mincon = 0.1

    # For these exptimes, values are set to given default_detection_params() function.
    if et > 0.2:
        det_clean = 'Y'
        deb_mincon = 0.005
    if et >= 1:
        det_filter = 'Y'
        det_clean = 'Y'
        minarea = 9
        an_thresh = 1.0
        det_thresh = 1.0
    if et >= 80:
        minarea = 13
        an_thresh = 2.5
        det_thresh = 2.5
    # if et >= 100:
    #     pass
    # if et >= 120:
    #     pass
    if et >= 180:
        minarea = 9
        an_thresh = 1.6
        det_thresh = 1.6

    detect_params = {}

    # setting config parameters
    detect_params['FILTER'] = det_filter
    detect_params['CLEAN'] = det_clean
    detect_params['DETECT_MINAREA'] = minarea
    detect_params['ANALYSIS_THRESH'] = an_thresh
    detect_params['DETECT_THRESH'] = det_thresh
    detect_params['DEBLEND_MINCONT'] = deb_mincon

    return detect_params


def get_radec_limits(path_fits):
    """Return alpha_J2000, delta_J2000 limits for astrometric calibrated FITS. 
    Output: dictionary with keywords 'ra_min', 'ra_max', 'dec_min', dec_max' in degrees."""
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

def read_blazar_file(path):
    """It reads blazars onfo from file given by 'path'.
    
    Output: pandas.DataFrame object.
    """

    df = pd.read_csv(path, comment='#')
    # print(df_mapcat.info())
    # getting coordinates in degrees unit
    c  = []
    for ra, dec in zip(df['ra2000_mc'], df['dec2000_mc']):
        c.append("{} {}".format(ra, dec))

    mapcat_coords = SkyCoord(c, frame=FK5, unit=(u.hourangle, u.deg), \
    obstime="J2000")
    df['ra2000_mc_deg'] = mapcat_coords.ra.deg
    df['dec2000_mc_deg'] = mapcat_coords.dec.deg
    
    return df

def closest_blazar(ra_fits, dec_fits, data):
    """"""
    icoords = "{} {}".format(ra_fits, dec_fits)
    input_coords = SkyCoord(icoords, frame=FK5, unit=(u.deg, u.deg), \
    obstime="J2000")
    
    # Blazars subset...
    df_blazars = data[data['IAU_name_mc'].notna()]  # take target sources from data
    c  = []
    for ra, dec in zip(data['ra2000_mc'], data['dec2000_mc']):
        c.append("{} {}".format(ra, dec))
    blazar_coords = SkyCoord(c, frame=FK5, unit=(u.hourangle, u.deg), \
    obstime="J2000")
    
    # Closest MAPCAT source to FITS central coordinates
    # Distance between this center FITS and MAPCAT targets
    distances = input_coords.separation(blazar_coords)
    
    # Closest source in complete set...
    i_min = distances.deg.argmin()
    blz_name = df_blazars['IAU_name_mc'].values[i_min]
    print(f"Blazar closest source name = {blz_name}")
    cond = np.logical_and(data['IAU_name_mc'] == blz_name, \
        data['Rmag_mc'] < 0)
    
    return data[cond], distances.deg.min()

def detect_sources(path_fits, cat_out, sext_conf, photo_aper):
    """"""
    i_fits = mcFits(path_fits)
    astro_header = i_fits.header
    # Getting data from ordinary-extraordinary sources in final FITS
    # 1. SExtractor calling
    fwhm_arcs = astro_header['FWHM'] * astro_header['INSTRSCL']

    com_str = "source-extractor -c {} -CATALOG_NAME {} -PIXEL_SCALE {} -SEEING_FWHM {} {}"
    com = com_str.format(sext_conf, cat_out, astro_header['INSTRSCL'], fwhm_arcs, path_fits)

    # MAPCAT aperture
    com += f" -PHOT_APERTURES {photo_aper}"

    # more Source-Extractor parameters
    additional_params = default_detection_params(astro_header['EXPTIME'])

    for k, v in additional_params.items():
        com += ' -{} {}'.format(k, v)
    print(com)
    subprocess.Popen(com, shell=True).wait()
    
    return 0


# ------------------------ MAIN FUNCTION SECTION -----------------------------
def main():
    parser = argparse.ArgumentParser(prog='iop3_photometric_calibration.py', \
    conflict_handler='resolve',
    description='''Main program that perfoms photometric calibration for input FITS. ''',
    epilog='''''')
    parser.add_argument("config_dir", help="Configuration parameter files directory")
    parser.add_argument("output_dir", help="Output base directory for FITS calibration")
    parser.add_argument("input_fits", help="Astrocalibrated input FITS file")
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    parser.add_argument("--border_image",
       action="store",
       dest="border_image",
       type=int,
       default=15,
       help="True is input file is for clusters [default: %(default)s].")
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
            print(str_err.format(args.input_dir))
            return 2

    if not os.path.exists(args.input_fits):
        str_err = 'ERROR: Input FITS file "{}" not available'
        print(str_err.format(args.input_dir))
        return 3

    input_fits = os.path.abspath(args.input_fits)
    os.chdir(args.output_dir)
    print(f"\nWorking directory set to '{args.output_dir}'\n")

    # Using input path for getting observation night date... (Is this the best way or should I read FITS header?)
    dt_run = re.findall('/(\d{6})/', args.input_fits)[0]
    date_run = f'20{dt_run[:2]}-{dt_run[2:4]}-{dt_run[-2:]}'
    fits_name = os.path.split(input_fits)[1][:-5]

    
    # Reading input fits header
    i_fits = mcFits(input_fits)
    astro_header = i_fits.header
    
    if 'MAGZPT' in astro_header:
        print(f'Input file "{input_fits}" already photo-calibrated. Nothing was done!')
        return 4
    
    text = 'OBJECT and Polarization angle = {}\nEXPTIME = {} s'
    print(text.format(astro_header['OBJECT'], astro_header['EXPTIME']))

    # ---------------------- MAPCAT sources info -----------------------
    blazar_path = os.path.join(args.config_dir, 'blazar_photo_calib_last.csv')
    df_mapcat = read_blazar_file(blazar_path)

    nearest_blazar, min_dist_deg = closest_blazar(astro_header['CRVAL1'], astro_header['CRVAL2'], df_mapcat)
    
    # closest blazar info
    alternative_ra = nearest_blazar['ra2000_mc_deg'].values[0]
    alternative_dec = nearest_blazar['dec2000_mc_deg'].values[0]    

    print(f"Blazar {nearest_blazar['IAU_name_mc'].values[0]} is the closest detection ")
    print(f" at coordinates ({alternative_ra}, {alternative_dec}")
    print('(Rmag, Rmagerr) = ({}, {})'.format(df_mapcat['Rmag_mc'][0], \
        df_mapcat['Rmagerr_mc'][0]))
    
    print(f'Distance = {min_dist_deg}')
    if min_dist_deg > 0.5: # distance in degrees
        print('!' * 100)
        print('ERROR: Not enough close blazar or HD star found (distance <= 0.5 deg)')
        print('!' * 100)
        return 4

    mc_aper = nearest_blazar['aper_mc'].values[0]
    print(f'aperture = {mc_aper} pixels')

    ################ WORKING ON ASTROMETRIC CALIBRATED FITS ################
    
    print('*' * 50)
    message = "Number of sources used in calibration: {} (of {})"
    print(message.format(astro_header['WCSMATCH'], astro_header['WCSNREF']))
    print('*' * 50)

    cat = input_fits.replace('.fits', '_photocal.cat')
    sex_conf = os.path.join(args.config_dir, 'sex.conf')
    detect_sources(input_fits, cat_out=cat, sext_conf=sex_conf, photo_aper=mc_aper)

    
    # RA,DEC limits...
    sky_limits = get_radec_limits(input_fits)

    # Loading FITS_LDAC format SExtractor catalog
    sext = fits.open(cat)
    data = sext[2].data
    
    print ("Number of detections = {}".format(data['ALPHA_J2000'].size))
    intervals = "(ra_min, ra_max, dec_min, dec_max) = ({}, {}, {}, {})"
    print(intervals.format(sky_limits['ra_min'], sky_limits['ra_max'], \
        sky_limits['dec_min'], sky_limits['dec_max']))

    # Showing detailed info about detections
    sources_sext_png = input_fits.replace('.fits', '_sextractor_sources.png')
    print('Out PNG ->', sources_sext_png)
    plotFits(input_fits, sources_sext_png, \
        title=f"SExtractor sources in {astro_header['OBJECT']}", colorBar=True, \
        coords=(data['ALPHA_J2000'], data['DELTA_J2000']), \
        astroCal=True, color='red') # , \
        # dictParams={'aspect':'auto', 'invert':'True', 'stretch': 'log', 'vmin':1})

    ##### ------ Searching for MAPCAT sources inside limits FITS coordinates ---- #####
    df_mc = df_mapcat[df_mapcat['ra2000_mc_deg'] > sky_limits['ra_min']]
    df_mc = df_mc[df_mc['ra2000_mc_deg'] < sky_limits['ra_max']]
    df_mc = df_mc[df_mc['dec2000_mc_deg'] > sky_limits['dec_min']]
    df_mc = df_mc[df_mc['dec2000_mc_deg'] < sky_limits['dec_max']]

    # print("MAPCAT filtered info...")
    # print(df_mc.info())
    print(f'Number of MAPCAT sources= {len(df_mc.index)}')

    # Plotting MAPCAT sources
    if len(df_mc.index) > 0:
        sources_mapcat_png = input_fits.replace('.fits', '_mapcat_sources.png')
        print('Out PNG ->', sources_mapcat_png)
        plotFits(input_fits, sources_mapcat_png, \
            title=f'MAPCAT sources in {fits_name}', colorBar=True, \
            coords=(df_mc['ra2000_mc_deg'].values, df_mc['dec2000_mc_deg'].values), \
            astroCal=True, color='orange', \
            dictParams={'aspect':'auto', 'invert':'True'})
    else:
        print(f'ERROR: No closer enough MAPCAT sources found for this input FITS: {input_fits}')
        return 6
    
    # mapcat catalog
    scatalog = SkyCoord(ra = df_mc['ra2000_mc_deg'].values * u.degree, \
        dec = df_mc['dec2000_mc_deg'].values * u.degree)
    # sextractor catalog
    pcatalog = SkyCoord(ra = data['ALPHA_J2000'] * u.degree, \
        dec = data['DELTA_J2000'] * u.degree)

    # Matching SExtractor detections with closest MAPCAT sources: 
    # Values returned: matched ordinary source indexes, 2D-distances, 3D-distances
    idx_o, d2d_o, d3d_o = match_coordinates_sky(scatalog, pcatalog, \
        nthneighbor=1)

    print(f'SExtractor closest ordinary detection indexes = {idx_o}')
    print(f'SExtractor closest ordinary detection distances = {d2d_o}')

    # Printing info about MAPCAT-SExtractor sources
    str_match_mapcat = " MAPCAT (name, ra, dec, Rmag, Rmagerr) = ({}, {}, {}, {}, {})"
    str_match_sext = "SExtractor (ra, dec, mag_auto, magerr_auto) = ({}, {}, {}, {})"
    str_dist = "Distance = {}\n-----------------------"
    for j in range(len(idx_o)):
        print(str_match_mapcat.format(df_mc['name_mc'].values[j], \
            df_mc['ra2000_mc_deg'].values[j], df_mc['dec2000_mc_deg'].values[j], \
            df_mc['Rmag_mc'].values[j], df_mc['Rmagerr_mc'].values[j]))
        print(str_match_sext.format(data['ALPHA_J2000'][idx_o[j]], \
            data['DELTA_J2000'][idx_o[j]], data['MAG_AUTO'][idx_o[j]], \
            data['MAGERR_AUTO'][idx_o[j]]))
        print(str_dist.format(d2d_o[j]))

    # Extraordinary counterparts location
    # rough coordinates (relative to ordinary source locations)
    ra_e = data['ALPHA_J2000'][idx_o]
    dec_e = data['DELTA_J2000'][idx_o] - 0.0052 

    scatalog_e = SkyCoord(ra = ra_e * u.degree, dec = dec_e * u.degree)
    idx_e, d2d_e, d3d_e = match_coordinates_sky(scatalog_e, pcatalog, \
        nthneighbor=1)
    print(f"pcatalog = {pcatalog}")
    print(f"scatalog_e = {scatalog_e}")

    print(f"SExtractor numbers for extraordinary counterparts = {idx_e}")
    print(f"Distances = {d2d_e}")

    # Source problem: setting filter condition that identify it.
    source_problem = None
    # If there is only source and it's R calibrated -> HD star
    if len(df_mc.index) == 1:
        source_problem = np.ones(1, dtype=bool) # filter equals to True
    else:
        # Source problem has no R filter magnitude (asigned value equals to -99)
        source_problem = df_mc['Rmag_mc'].values < 0 # negative R-mag is the source problem

    print(f'source_problem = {source_problem}')

    # Printing SExtractor indexes
    indexes = [idx_o[source_problem][0], idx_e[source_problem][0]]
    print(f'[Ordinary, Extraordinary] SExtractor indexes = {indexes}')

    # Plotting source problem
    # Muestro el nivel de detalle de las detecciones de SExtractor en la imagen final
    source_pair_png = input_fits.replace('.fits', '_source_pair.png')
    print('Out PNG ->', source_pair_png)
    plotFits(input_fits, source_pair_png, colorBar=True, \
        title=f"SExtractor Pair Detections in {astro_header['OBJECT']}", \
        coords=(data['ALPHA_J2000'][indexes], data['DELTA_J2000'][indexes]), \
        astroCal=True, color='red') # , \
        # dictParams={'aspect':'auto', 'invert':'True', 'stretch': 'log', 'vmin':1})

    # Parameters to store...
    cal_data = defaultdict(list)
    
    # Photometric calibration
    mag_zeropoint = None
    std_mag_zeropoint = None
    num_cal = len(df_mc.index) - 1
    num_sat = 0
    zps = []
    
    # If there are MAPCAT calibrators in field covered by FITS
    if len(df_mc[~source_problem].index) > 0:
        # checking for non-saturated calibrators
        index_o_cal = idx_o[~source_problem]
        index_e_cal = idx_e[~source_problem]
        # As SExtractor manual says, if some source pixel si saturated then FLAGS take
        # 3th bit of FLAG to 1. That is, value 4 = 2^2 (3th bit) is activated
        
        # getting binary string representation for each value of SExtractor FLAGS output parameter
        bin_ord = np.array([f"{format(flag, 'b') :0>8}" for flag in data['FLAGS'][index_o_cal]])
        bin_ext = np.array([f"{format(flag, 'b') :0>8}" for flag in data['FLAGS'][index_e_cal]])
        print(f'Ordinary binarian SExtractor FLAGS = {bin_ord}')
        print(f'Extraordinary binarian SExtractor FLAGS = {bin_ext}')
        
        # Checking 3th value from last character ([0|1] * 2^2) of string binary representation
        # [0|1] * 2^n + [0|1] * 2^(n - 1) + ... + [0|1] * 2^2 + [0|1] * 2^1 + [0|1] * 2^0
        ord_sat = np.array([f"{format(flag, 'b') :0>8}"[-3] == '1' for flag in data['FLAGS'][index_o_cal]], dtype=bool)
        ext_sat = np.array([f"{format(flag, 'b') :0>8}"[-3] == '1' for flag in data['FLAGS'][index_e_cal]], dtype=bool)
        
        sat_calibrator = np.logical_or(ord_sat, ext_sat)
        num_sat = sat_calibrator.sum()
        
        print(f'********************** {num_sat} calibrators are saturated in ordinary or extraordinary measurements. *********')
        if num_sat == sat_calibrator.size:
            # all calibrators are saturated, so no filtering operation will be applied
            print("----------- All calibrators are saturated. Non-saturation filter applied --------------------")
            sat_calibrator = np.zeros(data['FLAGS'].size, dtype=bool)
            
        # plotting calibrators
        calibrators_png = input_fits.replace('.fits', '_photo-calibrators.png')
        title = 'Photometric calibrators used (green) and rejected (red)'
        
        # ordinary coords
        ns_o_data = data[index_o_cal][~sat_calibrator]
        nonsat_o_coords = [ns_o_data['ALPHA_J2000'], ns_o_data['DELTA_J2000']]
        s_o_data = data[index_o_cal][sat_calibrator]
        sat_o_coords = [s_o_data['ALPHA_J2000'], s_o_data['DELTA_J2000']]
        
        # extraordinary coords
        ns_e_data = data[index_e_cal][~sat_calibrator]
        nonsat_e_coords = [ns_e_data['ALPHA_J2000'], ns_e_data['DELTA_J2000']]
        s_e_data = data[index_e_cal][sat_calibrator]
        sat_e_coords = [s_e_data['ALPHA_J2000'], s_e_data['DELTA_J2000']]
        
        coords = [nonsat_o_coords, nonsat_e_coords, sat_o_coords, sat_e_coords]
        
        plotFits(input_fits, calibrators_png, title=title, \
            colorBar=True, ref_coords='world', astroCal=True, \
            color=['green', 'green', 'red', 'red'], coords=coords, \
            dictParams={'invert':'True'})

        nonsat_index_o_cal = index_o_cal[~sat_calibrator]
        nonsat_index_e_cal = index_e_cal[~sat_calibrator]
        # non-saturated calibrators total flux (Ordinary + Extraordinary)
        calibrators_total_flux = data['FLUX_APER'][nonsat_index_o_cal] + \
        data['FLUX_APER'][nonsat_index_e_cal]
        
        
        zps = df_mc['Rmag_mc'][~source_problem][~sat_calibrator].values + \
        2.5 * np.log10(calibrators_total_flux)

        print(f'zps = {zps}')

        mag_zeropoint = zps.mean()
        std_mag_zeropoint = zps.std()

        # Plotting calibrators
        mc_calib_png = input_fits.replace('.fits', '_sources_mc_calib.png')
        print('Out PNG ->', mc_calib_png)
        plotFits(input_fits, mc_calib_png, colorBar=True, \
            title=f"MAPCAT Calibration sources in {astro_header['OBJECT']}", \
            coords=(df_mc['ra2000_mc_deg'][~source_problem].values, \
            df_mc['dec2000_mc_deg'][~source_problem].values), \
            astroCal=True, color='green', \
            dictParams={'aspect':'auto', 'invert':'True'})
        cal_data['MC_CALIB_PNG'] = [mc_calib_png]
    else:
        # Dealing with HD calibrator
        # because Mag = ZP - 2.5 * log10(Flux) => ZP = Mag + 2.5 * log10(Flux)
        # Then, as HD is a polarized source, I'll take as FLUX the sum of both
        # (Flux_o + Flux_e)
        total_flux = (data['FLUX_AUTO'][indexes]).sum()
        mag_zeropoint = df_mc[source_problem]['Rmag_mc'].values[0] + \
        2.5 * np.log10(total_flux)
        std_mag_zeropoint = 0

    print(f"Photometric Zero-point = {round(mag_zeropoint, 2)}")
    print(f"STD(Photometric Zero-point) = {round(std_mag_zeropoint, 2)}")

    # --------------- Updating FITS header --------------------- #
    if mag_zeropoint is not None:
        # Writing new information as FITS header pairs (keyword, value)
        hdul = fits.open(input_fits, mode='update')

        if 'MAGZPT' not in hdul[0].header:
            hdul[0].header.append(('MAGZPT', round(mag_zeropoint, 2), \
                'MAPCAT Photometric zeropoint'))
        else:
            hdul[0].header['MAGZPT'] = round(mag_zeropoint, 2)

        if 'STDMAGZP' not in hdul[0].header:
            hdul[0].header.append(('STDMAGZP', round(std_mag_zeropoint, 2), \
                'MAPCAT STD(Photometric zeropoint)'))
        else:
            hdul[0].header['STDMAGZP'] = round(std_mag_zeropoint, 2)

        if 'NSZPT' not in hdul[0].header:
            hdul[0].header.append(('NSZPT', len(zps), \
                'MAPCAT sources used in MAGZPT estimation'))
        else:
            hdul[0].header['NSZPT'] = len(zps)
        
        bz_name = df_mc[source_problem]['IAU_name_mc'].values[0]
        if 'BLZRNAME' not in hdul[0].header:
            hdul[0].header.append(('BLZRNAME', bz_name, \
                'IAU name for BLAZAR object'))
        else:
            hdul[0].header['BLZRNAME'] = bz_name

        hdul.flush()
        hdul.close()


    # Reading astro-photo-calibrated fits
    i_fits = mcFits(input_fits)
    astro_header = i_fits.header
    
    # Executing SExtractor again with MAG_ZEROPOINT info
    fwhm_arcs = float(astro_header['FWHM']) * float(astro_header['INSTRSCL'])
    com_str = "source-extractor -c {} -CATALOG_NAME {} -PIXEL_SCALE {} -SEEING_FWHM {} {}"
    com = com_str.format(sex_conf, cat, astro_header['INSTRSCL'], fwhm_arcs, input_fits)
    # MAPCAT aperture
    com += f" -PHOT_APERTURES {mc_aper}"
    # Magnitude ZEROPOINT
    com += f" -MAG_ZEROPOINT {mag_zeropoint}"

    # more source-extractor parameters
    additional_params = default_detection_params(astro_header['EXPTIME'])

    for k, v in additional_params.items():
        com += ' -{} {}'.format(k, v)

    print(com)
    subprocess.Popen(com, shell=True).wait()

    # Loading FITS_LDAC format SExtractor catalog
    sext = fits.open(cat)
    data = sext[2].data

    # Interesting parameters for polarimetric computation
    keywords = ['ALPHA_J2000', 'DELTA_J2000', 'FWHM_IMAGE', 'CLASS_STAR', \
        'FLAGS', 'ELLIPTICITY', 'FLUX_MAX', 'FLUX_APER', 'FLUXERR_APER', \
        'FLUX_ISO', 'FLUXERR_ISO', 'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_APER', \
        'MAGERR_APER', 'MAG_ISO', 'MAGERR_ISO', 'MAG_AUTO', 'MAGERR_AUTO']

    pair_params = defaultdict(list)

    pair_params['ID-MC'] = [df_mc[source_problem].iloc[0]['id_mc']] * 2
    pair_params['ID-BLAZAR-MC'] = [df_mc['id_blazar_mc'].values[source_problem][0]] * 2
    pair_params['TYPE'] = ['O', 'E']
    angle = float(astro_header['INSPOROT'])
    pair_params['ANGLE'] = [round(angle, ndigits=1)] * 2
    pair_params['OBJECT'] = [astro_header['OBJECT']] * 2
    pair_params['MJD-OBS'] = [astro_header['MJD-OBS']] * 2
    pair_params['DATE-OBS'] = [''] * 2
    if 'DATE-OBS' in astro_header:
        pair_params['DATE-OBS'] = [astro_header['DATE-OBS']] * 2
    else:
        pair_params['DATE-OBS'] = [astro_header['DATE']] * 2
    mc_name = df_mc['name_mc'].values[source_problem][0]
    mc_iau_name = df_mc['IAU_name_mc'].values[source_problem][0]
    pair_params['MC-NAME'] = [mc_name] * 2
    pair_params['MC-IAU-NAME'] = [mc_iau_name] * 2
    pair_params['MAGZPT'] = [mag_zeropoint] * 2
    pair_params['RUN_DATE'] = [date_run] * 2
    pair_params['EXPTIME'] = [astro_header['EXPTIME']] * 2

    # Transforming from degrees coordinates (ra, dec) to ("hh mm ss.ssss", "[sign]dd mm ss.sss") representation
    c3 = []
    for ra, dec in zip(data['ALPHA_J2000'][indexes], data['DELTA_J2000'][indexes]):
        c3.append(f"{ra} {dec}")

    coords3 = SkyCoord(c3, frame=FK5, unit=(u.deg, u.deg), obstime="J2000")

    pair_params['RA_J2000'] = coords3.ra.to_string(unit=u.hourangle, sep=' ', \
    precision=4, pad=True)
    pair_params['DEC_J2000'] = coords3.dec.to_string(unit=u.deg, sep=' ', \
    precision=3, alwayssign=True, pad=True)

    for k in keywords:
        for i in indexes:
            pair_params[k].append(data[k][i])

    df = pd.DataFrame(pair_params)
    csv_out = input_fits.replace('.fits', '_photocal_res.csv')
    df.to_csv(csv_out, index=False)

    # Imprimo el contenido del fichero
    print('Useful parameters for polarimetric computations:')
    print(df)
    
    # Parameters to store...
    # Getting useful info about calibrated fits
    some_calib_keywords = ['SOFT', 'PROCDATE', 'SOFTDET', 'MAX',  'MIN', \
        'MEAN', 'STD', 'MED', 'RA', 'DEC', 'CRVAL1', 'CRVAL2', 'EPOCH', \
        'CRPIX1', 'CRPIX2', 'SECPIX1', 'SECPIX2', 'CDELT1', 'CDELT2', 'CTYPE1', 'CTYPE2', \
        'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'WCSRFCAT', 'WCSIMCAT', 'WCSMATCH', \
        'WCSNREF', 'WCSTOL', 'CROTA1', 'CROTA2', 'WCSSEP', 'IMWCS', 'MAGZPT', \
        'STDMAGZP', 'NSZPT', 'BLZRNAME']
    
    # header = hdul = fits.open(final_fits)[0].header
    for key in some_calib_keywords:
        value = ''
        if key in astro_header:
            value = astro_header[key]
        cal_data[key].append(value)
    cal_data['PATH'].append(input_fits)
    cal_data['RUN_DATE'].append(date_run)
    cal_data['CALIBRATORS_PNG'] = [calibrators_png]
    cal_data['N_CALIBRATORS'] = [num_cal]
    cal_data['N_SAT_CALIBRATORS'] = [num_sat]
    cal_data['SEXTDET_PNG'] = [sources_sext_png]
    cal_data['MAPCAT_SOURCES_PNG'] = [sources_mapcat_png]
    cal_data['SOURCE_PAIRS_PNG'] = [source_pair_png]
    

    df = pd.DataFrame(cal_data)
    csv_out = input_fits.replace('.fits', '_photocal_process_info.csv')
    df.to_csv(csv_out, index=False)


    return 0

# -------------------------------------
if __name__ == '__main__':
    print(main())