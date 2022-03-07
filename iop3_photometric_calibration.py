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
from cmath import isnan
from dataclasses import replace
import os
import argparse
import subprocess
import re
from collections import defaultdict
from collections import OrderedDict

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
        if type(coords[0]) == type(list()) or type(coords[0]) == type((1,)):
            for i, l in enumerate(coords):
                ra, dec = l[0], l[1]
                gc.show_markers(ra, dec, edgecolor=color[i], facecolor='none', \
                    marker='o', coords_frame=ref_coords, s=40, alpha=1)
        else:
            ra, dec = coords[0], coords[1]
            gc.show_markers(ra, dec, edgecolor=color, facecolor='none', \
                marker='o', coords_frame=ref_coords, s=40, alpha=1)
    gc.save(outputImage, format=format)

    return 0

############################################ TESTING new changes ##############################

def read_sext_catalog(path, format='ASCII', verbose=False):
    """
    Read SExtractor output catalog given by 'path'.
    
    Args:
        path (str): SExtractor catalog path
        format (str): 'ASCII' or 'FTIS_LDAC' output SExtractor formats.
        verbose (bool): IT True, it prints process info.
        
    Returns:
        pandas.DataFrame: Data from SExtractor output catalog.
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
        #data_sext = pd.DataFrame(data)
    
    return data_sext

def execute_command(cmd, out=subprocess.PIPE, err=subprocess.PIPE, shell=True):
    """It executes command and checks results."""
    result = subprocess.run(cmd, stdout=out, stderr=err, shell=shell, check=True)
    
    return result

def statistics(path_fits, border=15, sat_threshold=50000):
    input_fits = mcFits(path_fits, border=border)
    head = input_fits.header
    data = input_fits.data + 0.0

    new_data = data
    if border > 0:
        new_data = data[border:-border, border:-border]
    dictStats = dict()
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
    params['DETECT_MINAREA'] = 20
    params['ANALYSIS_THRESH'] = 1.2
    params['DETECT_THRESH'] = 1.2
    params['DEBLEND_MINCONT'] = 0.005

    # getting info about FITS
    dt = statistics(path_fits, border=border, sat_threshold=sat_threshold)

    if dt['EXPTIME'] > 1:
        params['FILTER'] = 'Y'
        params['CLEAN'] = 'Y'
        # params['FILTER_NAME'] = '/home/cesar/desarrollos/Ivan_Agudo/code/iop3/conf/filters_sext/mexhat_5.0_11x11.conv'
        # params['FILTER_NAME'] = '/home/cesar/desarrollos/Ivan_Agudo/code/iop3/conf/filters_sext/gauss_5.0_9x9.conv'
        params['FILTER_NAME'] = '/home/cesar/desarrollos/Ivan_Agudo/code/iop3/conf/filters_sext/tophat_5.0_5x5.conv'
    
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
    
    plotFits(path_fits, plot_out_path, title=title, coords=coords, \
        ref_coords=ref_coords, astroCal=astrocal, color=color, \
        dictParams={'aspect':'auto', 'vmin': 1, 'stretch': 'power', 'invert': True})    
        # dictParams={'aspect':'auto', 'vmin': 1, 'invert': True}, format='png')

    return 0

def detect_sources(path_fits, sext_conf, cat_out, plot_out=None, \
    additional_params={}, photo_aper=None, mag_zeropoint=None, \
    back_image=False, segment_image=False, border=15, \
    sat_threshold=45000, verbose=True):
    """_summary_

    Args:
        path_fits (str): Path of FITS file.
        sext_conf (str): Path to SExtractor configuration file.
        cat_out (str): Path for output catalog.
        plot_out (str, optional): Path for output detected sources. Defaults to None.
        additional_params (dict, optional): Updated parameters for SExtractor. Defaults to {}.
        photo_aper (float, optional): Aperture in pixels for fotometry. Defaults to None.
        mag_zeropoint (float, optional): Photometric zero point. Defaults to None.
        back_image (bool, optional): If True, SExtractor create background map. Defaults to False.
        segment_image (bool, optional): If True, SExtractor create segmentation map. Defaults to False.
        border (int, optional): With of image close to borders that is ignored. Defaults to 15.
        sat_threshold (float, optional): Pixel threshold value. If greater, pixel is considered as saturated.
        verbose (bool, optional): If True, additional info is printed. Defaults to True.

    Returns:
        int: 0, if everything was fine.
    """
    # FITS statistic
    # First source extraction
    # masking sources and bkakground
    # second statistic over masked FITS
    # setting best extraction parameters
    # Second source extraction
    # adding data from masked sources
    # plotting
    
    o_fits = mcFits(path_fits, border=border)
    if 'MAPCAT' in path_fits:
        fits_par = o_fits.get_data(keywords=['INSTRSCL', 'FWHM', 'EXPTIME', 'OBJECT', 'DATE-OBS'])
        pixscale = fits_par['INSTRSCL']
    else:
        fits_par = o_fits.get_data(keywords=['NAXIS1', 'FWHM', 'EXPTIME', 'OBJECT', 'DATE-OBS'])
        if 'T090' in path_fits:
            pixscale = 0.387
            if fits_par['NAXIS1']==1024:
                pixscale=2*pixscale
        elif 'T150' in path_fits:
            pixscale = 0.232    
# exptime = fits_par['EXPTIME']
    fwhm_arcs = float(fits_par['FWHM']) * float(pixscale)
    
    # Adtitional ouput info
    if 'fits' in path_fits:
        back_path = path_fits.replace('.fits', '_back.fits')
        segm_path = path_fits.replace('.fits', '_segment.fits')
    else:
        back_path = path_fits.replace('.fit', '_back.fit')
        segm_path = path_fits.replace('.fit', '_segment.fit')
    # SExtractor parameters    
    params = {}
    if back_image and segment_image:
        params['CHECKIMAGE_TYPE'] = 'BACKGROUND,SEGMENTATION'
        params['CHECKIMAGE_NAME'] = f'{back_path},{segm_path}'
    elif back_image:
        params['CHECKIMAGE_TYPE'] = 'BACKGROUND'
        params['CHECKIMAGE_NAME'] = f'{back_path}'
    elif segment_image:
        params['CHECKIMAGE_TYPE'] = 'SEGMENTATION'
        params['CHECKIMAGE_NAME'] = f'{segm_path}'
    
    cmd = f"source-extractor -c {sext_conf} -CATALOG_NAME {cat_out} -PIXEL_SCALE {pixscale} -SEEING_FWHM {fwhm_arcs} "
    if photo_aper:
        cmd += f"-PHOT_APERTURES {photo_aper} "
    
    if mag_zeropoint:
        cmd += f"-MAG_ZEROPOINT {mag_zeropoint} "
    # Aditional parameters passed as arguments. They overwrite previous values
    s_params = sext_params_detection(path_fits, border, sat_threshold=sat_threshold)
    for k, v in s_params.items():
        params[k] = v
    
    for k, v in additional_params.items():
        params[k] = v
    
    # Formatting parameters to command line syntax
    com_params = [f'-{k} {v}' for k, v in params.items()]
    
    # adding parameters to command
    cmd = cmd + ' '.join(com_params)
    
    # last parameter for command
    cmd = cmd + f' {path_fits}'

    if verbose:
        print(cmd)
    
    res = execute_command(cmd)

    if res.returncode:
        print(res)
        return res.returncode

###############################################################################################

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
        
    # getting coordinates in degrees unit
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
>>>>>>> main

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
    return np.array([f"{format(flag, 'b') :0>8}"[-3] == '1' for flag in sext_flags], dtype=bool)
    
        
def compute_zeropoint(input_fits, blazar_data, sextractor_ord_data, sextractor_ext_data, \
    output_png=None):
    """Compute zeropoint and plots saturated and not-saturated calibrators location.

    Args:
        input_fits (str): FITS path.
        blazar_data (pd.DataFrame): Blazar DataFrame info.
        sextractor_ord_data (pd.DataFrame): SExtractor ordinary calibrators data.
        sextractor_ext_data (pd.DataFrame): SExtractor extraordinary calibrators data.
        
    Returns:
        tuple: (mag_zeropoint, magerr_zeropoint, num_non-saturated_calibrators)
    """ 
    i_fits = mcFits(input_fits)
     
    # checking for non-saturated calibrators
    ord_sat = check_saturation(sextractor_ord_data['FLAGS'])    
    ext_sat = check_saturation(sextractor_ext_data['FLAGS'])    
    
    # If ordinary or extraordinary counterpart is saturated, source 
    # is considered as saturated.
    sat_calibrator = np.logical_or(ord_sat, ext_sat)
    num_sat = sat_calibrator.sum()
    
    print(f'********************** {num_sat} calibrators is/are saturated. *********')
    if num_sat == ord_sat.size:
        # all calibrators are saturated, so no filtering operation will be applied
        print("----------- All calibrators are saturated. Non-saturation filter will not be applied --------------------")
        sat_calibrator = np.zeros(sat_calibrator.size, dtype=bool) # no one saturated
        
    # ordinary coords
    ns_o_data = sextractor_ord_data[~sat_calibrator]
    nonsat_o_coords = [ns_o_data['ALPHA_J2000'], ns_o_data['DELTA_J2000']]
    s_o_data = sextractor_ord_data[sat_calibrator]
    sat_o_coords = [s_o_data['ALPHA_J2000'], s_o_data['DELTA_J2000']]
    
    # extraordinary coords
    ns_e_data = sextractor_ext_data[~sat_calibrator]
    nonsat_e_coords = [ns_e_data['ALPHA_J2000'], ns_e_data['DELTA_J2000']]
    s_e_data = sextractor_ext_data[sat_calibrator]
    sat_e_coords = [s_e_data['ALPHA_J2000'], s_e_data['DELTA_J2000']]
    
    coords = [nonsat_o_coords, nonsat_e_coords, sat_o_coords, sat_e_coords]
    
    # plotting saturated and non-saturated calibrators
    if not output_png:
        root, ext = os.path.splitext(input_fits)
        output_png = root + '_photo-calibrators.png'
        
    title_temp = "Photometric calibrators in {}, {} : used (green) and rejected (red)"
    header_data = i_fits.get_data(keywords=['OBJECT', 'DATE-OBS'])
    title = title_temp.format(header_data['OBJECT'], header_data['DATE-OBS'])
    print(f'output_png = {output_png}')
    plotFits(input_fits, output_png, title=title, \
        colorBar=True, ref_coords='world', astroCal=True, \
        color=['green', 'green', 'red', 'red'], coords=coords, \
        dictParams={'invert':'True'})

    # non-saturated calibrators total flux (Ordinary + Extraordinary)
    calibrators_total_flux = sextractor_ord_data['FLUX_APER'][~sat_calibrator] + \
    sextractor_ext_data['FLUX_APER'][~sat_calibrator]
    
    # Computing ZEROPOINT using non-saturated calibrators (or all of them if 
    # they are all saturated)
    zps = blazar_data['Rmag_mc'][~sat_calibrator].values + 2.5 * np.log10(calibrators_total_flux)
    
    return zps.mean(), zps.std(), len(zps)

def update_calibration_header(input_fits, params):
    """Add or update calibration pairs (key, value) related with 
    photometric calibration information.
    
    Args:
        input_fits (str): path of FITS to update.
        params (OrderedDict): {key: values} to update in FITS header.
    Returns:
        int: 0, if everythin was fine.
    """
    # FITS to update
    hdul = fits.open(input_fits, mode='update')

    comments = {'MAGZPT': 'MAPCAT Photometric zeropoint',
                'STDMAGZP': 'MAPCAT STD(Photometric zeropoint)', 
                'NSZPT': 'MAPCAT number of calibrators used in ZEROPOINT estimation',
                'APERPIX': 'Aperture in pixel for photometry calibration',
                'BLZRNAME': 'IAU name for BLAZAR object'}
    for k, v in params.items():
        if k not in hdul[0].header:
            try:
                hdul[0].header.append((k, v, comments[k]))
            except:
                print(f'(k, v, comments) = ({k}, {v}, {comments[k]})')
                raise
        else:
            hdul[0].header[k] = v
    # commiting changes and saving       
    hdul.flush()
    hdul.close()
    
    return 0

# ------------------------ MAIN FUNCTION SECTION -----------------------------
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
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    parser.add_argument("--overwrite", dest='overwrite', action='store_true', \
        help='Pipeline overwrite previous calibrations.')
    parser.add_argument('-v', '--verbose', action='count', default=0,
        help="Show running and progress information [default: %(default)s].")
    args = parser.parse_args()
    
    #Transform overwrite parameter properly to boolean
    if args.overwrite in ('true', 'True', '1', 'y', 'yes', 'Yes'):
        args.overwrite=True
    elif args.overwrite in ('false', 'False', '0', 'n', 'no', 'No'):
        args.overwrite=False
    else:
        print("Wrong or no value for --overwrite parameter. Setting it to default (False)")
        args.overwrite=False

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
    
    if 'MAGZPT' in astro_header and not args.overwrite:
        print(f'Input file "{input_fits}" already photo-calibrated. Nothing was done!')
        return 4
    
    text = 'OBJECT and Polarization angle = {}\nEXPTIME = {} s'
    print(text.format(astro_header['OBJECT'], astro_header['EXPTIME']))

    # ---------------------- MAPCAT sources info -----------------------
    blazar_path = os.path.join(args.config_dir, 'blazar_photo_calib_last.csv')

    #df_mapcat = read_blazar_file(blazar_path)
    center_fits = (astro_header['CRVAL1'], astro_header['CRVAL2'])
    print(f'center FITS coordinates = {center_fits}')
    nearest_blazar, min_dist_deg = closest_blazar(center_fits, blazar_path)
    
    print(f'Closest blazar distance = {min_dist_deg} (deg)')

    if min_dist_deg > 0.5: # distance in degrees
        print('!' * 100)
        print('ERROR: Not enough close blazar or HD star found (distance <= 0.5 deg)')
        print('!' * 100)
        return 5

    # closest blazar info
    print('Closest IOP3 blazar info')
    print(nearest_blazar)
  
    ################ WORKING ON ASTROMETRIC CALIBRATED FITS ################
    
    print('*' * 50)
    message = "Number of sources used in calibration: {} (of {})"
    print(message.format(astro_header['WCSMATCH'], astro_header['WCSNREF']))
    print('*' * 50)

    # RA,DEC limits...
    sky_limits = get_radec_limits(input_fits)
    
    # getting blazars info inside FOV of FITS
    blazars_data = read_blazar_file(blazar_path)
    df_mc = detections_inside(blazars_data, \
        (sky_limits['ra_min'], sky_limits['ra_max']), \
        (sky_limits['dec_min'], sky_limits['dec_max']))
    
    # print("MAPCAT filtered info...")
    print(df_mc.info())
    print(f'Number of MAPCAT sources= {len(df_mc.index)}')
    
    if 'fits' in input_fits:
        cat = input_fits.replace('.fits', '_photocal.cat')
    else:
        cat = input_fits.replace('.fit', '_photocal.cat')
    sext_conf = os.path.join(args.config_dir, 'sex.conf')
    dir_out = os.path.split(cat)[0]
    
    # detecting...
    res = detect_sources(input_fits, sext_conf, cat, \
        photo_aper=nearest_blazar['aper_mc'].iloc[0], \
        border=args.border_image, sat_threshold=45000, verbose=True)
    if res:
        print('PHOTOCALIBRATION,ERROR,"SExtractor did not work properly"')
        return 6
    
    # Loading FITS_LDAC format SExtractor catalog
    data = read_sext_catalog(cat, format='FITS_LDAC')
    
    # Matching SExtractor detections with closest ORDINARY MAPCAT sources: 
    # Values returned: matched source indexes, 2D-distances, 3D-distances
    ra1 = df_mc['ra2000_mc_deg'].values
    dec1 = df_mc['dec2000_mc_deg'].values
    f_source = df_mc['IAU_name_mc'].str.len() > 0
    print(f'Target = {df_mc[f_source]}') 
    ra2 = data['ALPHA_J2000']
    dec2 =data['DELTA_J2000']
    idx_o, d2d_o, d3d_o = match_sources(ra1, dec1, ra2, dec2, num_close=1)

    print(f'Distance between target coords and closest detected source = {d2d_o.deg[f_source]} deg')

    
    # getting close enough matches
    dist = d2d_o.deg
    valid_dist = dist < 0.006 # approx. 40 pixels

    # filtering SExtractor closest sources to IOP3 sources
    idx_o = idx_o[valid_dist]
    d2d_o = d2d_o[valid_dist]
    df_mc = df_mc[valid_dist]

    data_o = data[idx_o]

    f_source = df_mc['IAU_name_mc'].str.len() > 0
    if len(df_mc[f_source].index) == 0:
        print('PHOTOCALIBRATION,ERROR,"SExtractor has not detected close source to target IOP3."')
        return 6 
    # distances = np.array([d.deg for d in d2d_o])
    # new_params = {}
    # # checking distance: 0.02 deg = 72 arcs ~ 136 pixels
    # if (distances > 0.02).sum():
    #     print(f'PHOTOMETRY,ERROR,"Could not get closer enough calibrators for {input_fits}"')
    #     return 7
    #     # # redo process, but relaxing detection params this time
    #     # # detecting but relaxing conditions...
    #     # new_params['FILTER'] = 'N'
    #     # new_params['DETECT_MINAREA'] = 15
    #     # new_params['DEBLEND_MINCONT'] = 0.1
    #     # res = detect_sources(input_fits, sext_conf, cat, additional_params=new_params, \
    #     #     photo_aper=nearest_blazar['aper_mc'].iloc[0], \
    #     #     border=args.border_image, sat_threshold=45000, verbose=True)
    #     # if res:
    #     #     print('PHOTOCALIBRATION,ERROR,"SExtractor did not work properly"')
    #     #     return 6
        
    #     # # Loading FITS_LDAC format SExtractor catalog
    #     # data = read_sext_catalog(cat, format='FITS_LDAC')
        
    #     # root, ext = os.path.splitext(input_fits)
    #     # sources_sext_png = root + '_sextractor_sources.png'
    #     # if plot_cat(input_fits, sources_sext_png, cat, astro_coords=True, cat_format='FITS_LDAC'):
    #     #     print(f'PHOTOCALIBRATION,ERROR,"Could not plot {sources_sext_png}"')
    #     #     return 8
    #     # print('Out PNG ->', sources_sext_png)
    #     # # Matching SExtractor detections with closest ORDINARY MAPCAT sources: 
    #     # # Values returned: matched source indexes, 2D-distances, 3D-distances
    #     # ra1 = df_mc['ra2000_mc_deg'].values
    #     # dec1 = df_mc['dec2000_mc_deg'].values
    #     # ra2 = data['ALPHA_J2000']
    #     # dec2 =data['DELTA_J2000']
    #     # idx_o, d2d_o, d3d_o = match_sources(ra1, dec1, ra2, dec2, num_close=1)
    #     # data_o = data[idx_o]
        
    #     # distances = np.array([d.deg for d in d2d_o])
    #     # if (distances > 0.02).sum():
    #     #     print(f'PHOTOMETRY,ERROR,"Could not get closer enough calibrators for {input_fits}"')
    #     #     return 7
    root, ext = os.path.splitext(input_fits)
    sources_sext_png = root + '_sextractor_sources.png'
    if plot_cat(input_fits, sources_sext_png, cat, astro_coords=True, cat_format='FITS_LDAC'):
        print(f'PHOTOCALIBRATION,ERROR,"Could not plot {sources_sext_png}"')
        return 8
    print('Out PNG ->', sources_sext_png)
    
    print ("Number of detections = {}".format(len(data['ALPHA_J2000'])))
    intervals = "(ra_min, ra_max, dec_min, dec_max) = ({}, {}, {}, {})"
    print(intervals.format(sky_limits['ra_min'], sky_limits['ra_max'], \
        sky_limits['dec_min'], sky_limits['dec_max']))

    # Plotting MAPCAT sources
    if len(df_mc.index) > 0:
        sources_mapcat_png = root + '_mapcat_sources.png'
        print('Out PNG ->', sources_mapcat_png)
        title_temp = "{}, {} ({} s)"
        info = i_fits.get_data(keywords=['OBJECT', 'DATE-OBS', 'EXPTIME'])
        title = title_temp.format(info['OBJECT'], info['DATE-OBS'], info['EXPTIME'])
        plotFits(input_fits, sources_mapcat_png, \
            title=title, colorBar=True, astroCal=True, color='magenta', \
            coords=(df_mc['ra2000_mc_deg'].values, df_mc['dec2000_mc_deg'].values), \
            dictParams={'aspect':'auto', 'invert':'True'})
    else:
        print(f'ERROR: No closer enough MAPCAT sources found for this input FITS: {input_fits}')
        return 6
    # # Matching SExtractor detections with closest ORDINARY MAPCAT sources: 
    # # Values returned: matched source indexes, 2D-distances, 3D-distances
    # ra1 = df_mc['ra2000_mc_deg'].values
    # dec1 = df_mc['dec2000_mc_deg'].values
    # ra2 = data['ALPHA_J2000']
    # dec2 =data['DELTA_J2000']
    # idx_o, d2d_o, d3d_o = match_sources(ra1, dec1, ra2, dec2, num_close=1)
    # data_o = data[idx_o]
    # # print(data_o)                

    # print(type(idx_o))
    #print(f'SExtractor closest ordinary detection indexes = {idx_o}')
    #print(f'SExtractor closest ordinary detection distances = {d2d_o}')

    # Printing info about MAPCAT-SExtractor sources
    str_match_mapcat = " MAPCAT (name, ra, dec, Rmag, Rmagerr) = ({}, {}, {}, {}, {})"
    str_match_sext = "SExtractor (ra, dec, mag_auto, magerr_auto) = ({}, {}, {}, {})"
    str_dist = "Distance = {}\n-----------------------"
    
    # Aliases
    name_mc = df_mc['name_mc'].values
    ra_mc = df_mc['ra2000_mc_deg'].values
    dec_mc = df_mc['dec2000_mc_deg'].values
    r_mc = df_mc['Rmag_mc'].values
    rErr_mc = df_mc['Rmagerr_mc'].values
    
    ra_se = data['ALPHA_J2000']
    dec_se = data['DELTA_J2000']
    mag_se = data['MAG_AUTO']
    magerr_se = data['MAGERR_AUTO']
    
    # filtering far IOP3 sources 
    for j in range(len(idx_o)):
        print(str_match_mapcat.format(name_mc[j], ra_mc[j], dec_mc[j], \
            r_mc[j], rErr_mc[j]))
        print(str_match_sext.format(ra_se[idx_o[j]], dec_se[idx_o[j]], \
            mag_se[idx_o[j]], magerr_se[idx_o[j]]))
        print(str_dist.format(d2d_o[j]))
    
    # if dist > 0.006: # aprox. 20 arcsec ~ 40 pixels
    #     print('PHOTOCALIBRATION,ERROR,"Distance too big for good astrocalibration"')
    #     return 7

    # Extraordinary counterparts location
    # rough coordinates (relative to ordinary source locations)
    #print(type(ra_se))
    ra_e = ra_se[idx_o]
    if 'MAPCAT' in input_fits:
        dec_e = dec_se[idx_o] - 0.0052 
    else:
        dec_e = dec_se[idx_o]
    # idx_e = []
    # d2d_e = []
    
    # for r, d in zip(ra_e, dec_e):
    #     print(f'E-coordinates = ({r}, {d})')
    #     print(f'Getting closest source from {(dec2 < (d + 0.002)).sum()}')
    #     ra2_mod = ra2[dec2 > (d + 0.002)]
    #     dec2_mod = dec2[dec2 > (d + 0.002)]
    #     ide, d2de, d3de = match_sources(r, d, ra2_mod, dec2_mod, num_close=1)    
    #     # print (type(ide[0]), type(d2de[0]))
    #     idx_e.append(ide[()]) # 0-d array index
    #     d2d_e.append(d2de[0].deg)
    
    # idx_e = np.array(idx_e)
    # d2d_e = np.array(d2d_e)
    # scatalog_e = SkyCoord(ra = ra_e * u.degree, dec = dec_e * u.degree)
    idx_e, d2d_e, d3d_e = match_sources(ra_e, dec_e, ra2, dec2, num_close=1)
    data_e = data[idx_e]
    # print(f"pcatalog = {pcatalog}")
    # print(f"scatalog_e = {scatalog_e}")

    print(f"SExtractor numbers for extraordinary counterparts = {idx_e}")
    print(f"Distances = {d2d_e}")

    # Source problem: boolena that identify target source in blazars inside FITS FOV.
    source_problem = None
    # If there is only one source and it has R magnitude measure -> HD star
    if len(df_mc.index) == 1:
        source_problem = np.ones(1, dtype=bool) # filter equals to True
    else:
        # Source problem has no R filter magnitude (asigned value equals to -99)
        source_problem = df_mc['Rmag_mc'].values < 0 # negative R-mag is the source problem

    print(f'source_problem = {source_problem}')

    print(type(idx_o), type(idx_e))
    # Printing SExtractor indexes
    indexes_target = [idx_o[source_problem][0], idx_e[source_problem][0]]
     
    print(f'[Ordinary, Extraordinary] SExtractor indexes = {indexes_target}')

    # Plotting source problem
    # Showing detailed info about SExtractor counterparts
    if 'fits' in input_fits:
        source_pair_png = input_fits.replace('.fits', '_source_pair.png')
    else:
        source_pair_png = input_fits.replace('.fit', '_source_pair.png')

    print('Out PNG ->', source_pair_png)
    title_temp = "SExtractor Pair Detections {}, {} ({} s)"
    some_values = i_fits.get_data(keywords=['OBJECT', 'DATE-OBS', 'EXPTIME'])
    title = title_temp.format(some_values['OBJECT'], some_values['DATE-OBS'], some_values['EXPTIME'])
    plotFits(input_fits, source_pair_png, colorBar=True, \
        coords=[(ra_se[indexes_target][0], dec_se[indexes_target][0]), \
        (ra_se[indexes_target][1], dec_se[indexes_target][1])], \
        title=title, astroCal=True, color=['red', 'blue']) # , \
        # dictParams={'aspect':'auto', 'invert':'True', 'stretch': 'log', 'vmin':1})

    # Parameters to store...
    cal_data = defaultdict(list)
    
    # Photometric calibration
    mag_zeropoint = None
    std_mag_zeropoint = None
    num_sat = 0
    calibrators_png = ''

    # If there are IOP3 calibrators in field covered by FITS
    if len(df_mc[~source_problem].index) > 0:
        # checking for non-saturated calibrators
        
        # SExtractor ordinary calibrators data
        sext_ord_calibrators = data_o[~source_problem]
        # SExtractor extraordinary calibrators data
        sext_ext_calibrators = data_e[~source_problem]
        
        # getting saturated calibrators
        ord_sat = check_saturation(sext_ord_calibrators['FLAGS'])    
        ext_sat = check_saturation(sext_ext_calibrators['FLAGS'])    
        # If ordinary or extraordinary counterpart is saturated, source 
        # is considered as saturated.
        sat_calibrator = np.logical_or(ord_sat, ext_sat)
        num_sat = sat_calibrator.sum()

        # Computing ZEROPOINT
        root, ext = os.path.splitext(input_fits)
        calibrators_png = root + '_photo-calibrators.png'
        mag_zeropoint, std_mag_zeropoint, num_calibrators = compute_zeropoint(input_fits, \
            df_mc[~source_problem], sext_ord_calibrators, sext_ext_calibrators, \
            output_png=calibrators_png)

        # Plotting calibrators
        if 'fits' in input_fits:
            mc_calib_png = input_fits.replace('.fits', '_sources_mc_calib.png')
        else:
            mc_calib_png = input_fits.replace('.fit', '_sources_mc_calib.png')
        print('Out PNG ->', mc_calib_png)
        title_temp = "MAPCAT Calibration sources in {}, {} ({} s)"
        title = title_temp.format(some_values['OBJECT'], some_values['DATE-OBS'], some_values['EXPTIME'])
        plotFits(input_fits, mc_calib_png, colorBar=True, \
            title=title, astroCal=True, color='green', \
            coords=(df_mc['ra2000_mc_deg'][~source_problem].values, \
            df_mc['dec2000_mc_deg'][~source_problem].values), \
            dictParams={'aspect':'auto', 'invert':'True'})
        cal_data['MC_CALIB_PNG'] = [mc_calib_png]
    else:
        num_calibrators = 1
        # Dealing with HD calibrator
        # Checking saturation
        bin_flags = np.array([f"{format(flag, 'b') :0>8}" for flag in data['FLAGS'][indexes_target]])
        sat_flags = np.array([bf[-3] == '1' for bf in bin_flags], dtype=bool)
        
        num_sat = 0
        if sat_flags.sum() > 0: # saturated source
            num_sat = 1
        
        # because Mag = ZP - 2.5 * log10(Flux) => ZP = Mag + 2.5 * log10(Flux)
        # Then, as HD is a polarized source, I'll take as FLUX the sum of both
        # (Flux_o + Flux_e)
        try:
            if 'MAPCAT' in input_fits:
                total_flux = (data['FLUX_AUTO'][indexes]).sum()
            else:
                total_flux = (data['FLUX_AUTO'][indexes]).sum() / 2

            mag_zeropoint = df_mc[source_problem]['Rmag_mc'].values[0] + \
                2.5 * np.log10(total_flux)
            std_mag_zeropoint = 0
        except ValueError:
            print(f"Ordinary and extraordinary fluxes = {data['FLUX_AUTO'][indexes_target]}")
            raise

    print(f"Photometric Zero-point = {round(mag_zeropoint, 2)}")
    print(f"STD(Photometric Zero-point) = {round(std_mag_zeropoint, 2)}")

    if np.isnan(mag_zeropoint):
        print(f'PHOTOMETRY,ERROR,"Could not compute MAG_ZEROPOINT for \'{input_fits}\'"')
        return 8
    # --------------- Updating FITS header --------------------- #
    params = OrderedDict()
    params['MAGZPT'] = round(mag_zeropoint, 2)
    params['STDMAGZP'] = round(std_mag_zeropoint, 2)
    params['NSZPT'] = num_calibrators
    params['APERPIX'] = nearest_blazar['aper_mc'].iloc[0]
    params['BLZRNAME'] = df_mc[source_problem]['IAU_name_mc'].values[0]
    print('Header params = {params}')
    
    if update_calibration_header(input_fits, params):
        print(f'PHOTOCALIBRATION,ERROR,"Could not update photocalibration header in \'{input_fits}\'"')
        return 9

    # Reading astro-photo-calibrated fits
    i_fits = mcFits(input_fits)
    astro_header = i_fits.header
    
    # Executing SExtractor again with MAG_ZEROPOINT info
    # fwhm_arcs = float(astro_header['FWHM']) * float(astro_header['INSTRSCL'])
    # detect_sources(input_fits, sext_conf, cat, additional_params=new_params, \
    detect_sources(input_fits, sext_conf, cat, \
        photo_aper=nearest_blazar['aper_mc'].iloc[0], mag_zeropoint=params['MAGZPT'])
    # com_str = "source-extractor -c {} -CATALOG_NAME {} -PIXEL_SCALE {} -SEEING_FWHM {} {}"
    # com = com_str.format(sext_conf, cat, astro_header['INSTRSCL'], fwhm_arcs, input_fits)
    # # MAPCAT aperture
    # com += f" -PHOT_APERTURES {mc_aper}"
    # # Magnitude ZEROPOINT
    # com += f" -MAG_ZEROPOINT {mag_zeropoint}"

    # # more source-extractor parameters
    # additional_params = default_detection_params(astro_header['EXPTIME'])

    # for k, v in additional_params.items():
    #     com += ' -{} {}'.format(k, v)

    # print(com)
    # subprocess.Popen(com, shell=True).wait()

    # Loading FITS_LDAC format SExtractor catalog
    data = read_sext_catalog(cat, format='FITS_LDAC')
    
    # daterun info
    # Using input path for getting observation night date... (Is this the best way or should I read FITS header?)
    dt_run = re.findall('/(\d{6})/', args.input_fits)[0]
    date_run = f'20{dt_run[:2]}-{dt_run[2:4]}-{dt_run[-2:]}'
    root, ext = os.path.splitext(input_fits)
    fits_name = os.path.split(root)[1]
    
    # Interesting parameters for polarimetric computation
    keywords = ['ALPHA_J2000', 'DELTA_J2000', 'FWHM_IMAGE', 'CLASS_STAR', \
        'FLAGS', 'ELLIPTICITY', 'FLUX_MAX', 'FLUX_APER', 'FLUXERR_APER', \
        'FLUX_ISO', 'FLUXERR_ISO', 'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_APER', \
        'MAGERR_APER', 'MAG_ISO', 'MAGERR_ISO', 'MAG_AUTO', 'MAGERR_AUTO']

    pair_params = defaultdict(list)

    pair_params['ID-MC'] = [df_mc[source_problem].iloc[0]['id_mc']] * 2
    pair_params['ID-BLAZAR-MC'] = [df_mc['id_blazar_mc'].values[source_problem][0]] * 2
    pair_params['TYPE'] = ['O', 'E']
    if 'INSPOROT' in astro_header:
        angle = float(astro_header['INSPOROT'])
    else:
        if astro_header['FILTER']=='R':
            angle = -999.0
        else:
            angle = float(astro_header['FILTER'].replace('R',''))
    pair_params['ANGLE'] = [round(angle, ndigits=1)] * 2
    pair_params['OBJECT'] = [astro_header['OBJECT']] * 2
    if 'MJD-OBS' in astro_header:
        pair_params['MJD-OBS'] = [astro_header['MJD-OBS']] * 2
    else:
        pair_params['MJD-OBS'] = [astro_header['JD']] * 2
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
    pair_params['APERPIX'] = [nearest_blazar['aper_mc'].iloc[0]] * 2

    # Transforming from degrees coordinates (ra, dec) to ("hh mm ss.ssss", "[sign]dd mm ss.sss") representation
    c3 = []
    for ra, dec in zip(data['ALPHA_J2000'][indexes_target], data['DELTA_J2000'][indexes_target]):
        c3.append(f"{ra} {dec}")

    coords3 = SkyCoord(c3, frame=FK5, unit=(u.deg, u.deg), obstime="J2000")

    pair_params['RA_J2000'] = coords3.ra.to_string(unit=u.hourangle, sep=' ', \
    precision=4, pad=True)
    pair_params['DEC_J2000'] = coords3.dec.to_string(unit=u.deg, sep=' ', \
    precision=3, alwayssign=True, pad=True)

    for k in keywords:
        for i in indexes_target:
            pair_params[k].append(data[k][i])

    df = pd.DataFrame(pair_params)
    if 'fits' in input_fits:
        csv_out = input_fits.replace('.fits', '_photocal_res.csv')
    else:
        csv_out = input_fits.replace('.fit', '_photocal_res.csv')
    df.to_csv(csv_out, index=False)
    # Imprimo el contenido del fichero
    print('Useful parameters for polarimetric computations:')
    if 'MAPCAT' in input_fits:
        print(df)
    else: 
        print(df[df['TYPE']=='O'])
    # Parameters to store...
    # Getting useful info about calibrated fits
    some_calib_keywords = ['SOFT', 'PROCDATE', 'SOFTDET', 'MAX',  'MIN', \
        'MEAN', 'STD', 'MED', 'RA', 'DEC', 'CRVAL1', 'CRVAL2', 'EPOCH', \
        'CRPIX1', 'CRPIX2', 'SECPIX1', 'SECPIX2', 'CDELT1', 'CDELT2', 'CTYPE1', 'CTYPE2', \
        'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'WCSRFCAT', 'WCSIMCAT', 'WCSMATCH', \
        'WCSNREF', 'WCSTOL', 'CROTA1', 'CROTA2', 'WCSSEP', 'IMWCS', 'MAGZPT', \
        'STDMAGZP', 'NSZPT', 'APERPIX', 'BLZRNAME']
    
    # header = hdul = fits.open(final_fits)[0].header
    for key in some_calib_keywords:
        value = ''
        if key in astro_header:
            value = astro_header[key]
        else:
            if key in ['SECPIX1', 'SECPIX2']:
                value = astro_header['SECPIX']
        cal_data[key].append(value)
    cal_data['PATH'].append(input_fits)
    cal_data['RUN_DATE'].append(date_run)
    if calibrators_png and os.path.exists(calibrators_png):
        cal_data['CALIBRATORS_PNG'] = [calibrators_png]
    else:
        cal_data['CALIBRATORS_PNG'] = ''
    cal_data['N_CALIBRATORS'] = [num_calibrators]
    cal_data['N_SAT_CALIBRATORS'] = [num_sat]
    cal_data['SEXTDET_PNG'] = [sources_sext_png]
    cal_data['MAPCAT_SOURCES_PNG'] = [sources_mapcat_png]
    cal_data['SOURCE_PAIRS_PNG'] = [source_pair_png]
    
    df = pd.DataFrame(cal_data)
    if 'fits' in input_fits:
        csv_out = input_fits.replace('.fits', '_photocal_process_info.csv')
    else:
        csv_out = input_fits.replace('.fit', '_photocal_process_info.csv')
    df.to_csv(csv_out, index=False)

    return 0

# -------------------------------------
if __name__ == '__main__':
    print(main())
