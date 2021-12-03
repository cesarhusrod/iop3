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
import shutil
import os
import argparse
import subprocess
import pprint
import re
from collections import defaultdict

# Data structures libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.ndimage import median_filter
#import seaborn

import aplpy # FITS plotting library

from astropy.io import fits # FITS library
import astropy.wcs as wcs

# Coordinate system transformation package and modules
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import match_coordinates_sky  # Used for searching sources in catalog
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
import astropy.coordinates as coord
import astropy.units as u
# Astrotime
from astropy.time import Time

# Catalogs query
from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = -1

# HTML ouput template
import jinja2

# =================================
from astropy.utils.exceptions import AstropyWarning, AstropyDeprecationWarning
import warnings

# Ignore too many FITSFixedWarnings
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.simplefilter('ignore', category=AstropyDeprecationWarning)

# =================================

# ------------------------ Module functions ------------------------------
def medianFits(input_fits, output_fits, size=5):
    """Compute pass low filter to FITS data"""
    hdul = fits.open(input_fits)
    hdul[0].data = median_filter(hdul[0].data, size).astype(np.uint16)
    hdul.writeto(output_fits, overwrite = True)
    hdul.close()
    return 0

def plotFits(inputFits, outputImage, title=None, colorBar=True, coords=None, \
    ref_coords='world', astroCal=False, color='green', \
    dictParams={'aspect':'auto', 'vmin': 1, 'invert': True}):
    """Plot 'inputFits' as image 'outputImage'.
    Return 0 is everything was fine. Exception in the other case."""
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
        ra, dec = coords[0], coords[1]
        gc.show_markers(ra, dec, edgecolor=color, facecolor='none', \
        marker='o', coords_frame=ref_coords, s=40, alpha=1)
    gc.save(outputImage)
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


# ------------------------ MAIN FUNCTION SECTION -----------------------------
def main():
    parser = argparse.ArgumentParser(prog='iop3_calibration.py', \
    conflict_handler='resolve',
    description='''Main program that perfoms astrometric and photometric calibration for input FITS. ''',
    epilog='''''')
    parser.add_argument("config_dir", help="Configuration parameter files directory")
    parser.add_argument("output_dir", help="Output base directory for FITS calibration")
    parser.add_argument("input_fits", help="Reduced input FITS file")
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

    copy_input_fits = os.path.join(args.output_dir, os.path.split(input_fits)[1])
    final_fits = copy_input_fits.replace('.fits', '_final.fits')

    if os.path.exists(final_fits):
        print("INFO: Calibration done before")
        return -1
    
    dt_run = re.findall('/(\d{6})/', args.input_fits)[0]
    date_run = f'20{dt_run[:2]}-{dt_run[2:4]}-{dt_run[-2:]}'
    fits_name = os.path.split(input_fits)[1][:-5]

    # Getting header informacion
    hdul = fits.open(input_fits)
    input_head = hdul[0].header
    input_data = hdul[0].data
    hdul.close()

    text = 'OBJECT and Polarization angle = {}\nEXPTIME = {} s'
    print(text.format(input_head['OBJECT'], input_head['EXPTIME']))

    # MAPCAT sources
    blazar_path = os.path.join(args.config_dir, 'blazar_photo_calib_last.csv')
    df_mapcat = pd.read_csv(blazar_path, comment='#')
    # print(df_mapcat.info())
    # getting coordinates in degrees unit
    c  = []
    for ra, dec in zip(df_mapcat['ra2000_mc'], df_mapcat['dec2000_mc']):
        c.append("{} {}".format(ra, dec))

    mapcat_coords = SkyCoord(c, frame=FK5, unit=(u.hourangle, u.deg), \
    obstime="J2000")
    df_mapcat['ra2000_mc_deg'] = mapcat_coords.ra.deg
    df_mapcat['dec2000_mc_deg'] = mapcat_coords.dec.deg


    # Central FITS coordinates
    icoords = "{} {}".format(input_head['RA'], input_head['DEC'])
    input_coords = SkyCoord(icoords, frame=FK5, unit=(u.deg, u.deg), \
    obstime="J2000")
    # print(input_coords)

    # Blazars subset...
    df_blazars = df_mapcat[df_mapcat['IAU_name_mc'].notna()]
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
    print(f"Blazar closest source name = {blz_name}")
    index_min = df_mapcat.index[df_mapcat['IAU_name_mc'] == blz_name].tolist()[0]
    
    # closest blazar info
    alternative_ra = df_mapcat['ra2000_mc_deg'].values[index_min]
    alternative_dec = df_mapcat['dec2000_mc_deg'].values[index_min]    

    print(f"Blazar {df_mapcat['IAU_name_mc'].values[index_min]} is the closest detection ")
    print(f" at coordinates ({alternative_ra}, {alternative_dec}")
    print('(Rmag, Rmagerr) = ({}, {})'.format(df_mapcat['Rmag_mc'][index_min], \
        df_mapcat['Rmagerr_mc'][index_min]))
    
    print(f'Distance = {distances[i_min].deg}')
    if distances[i_min].deg > 0.5: # distance in degrees
        print('!' * 100)
        print('ERROR: Not enough blazar or HD star found')
        print('!' * 100)
        return 4

    mc_aper = df_mapcat['aper_mc'][index_min]
    print(f'aperture = {mc_aper} pixels')

    # Input image PNG
    input_fits_png = '{}.png'.format(fits_name)
    print('Original FITS PNG= {}'.format(input_fits_png))
    plotFits(input_fits, input_fits_png, title=fits_name)

    copy_input_fits = os.path.join(args.output_dir, os.path.split(input_fits)[1])
    shutil.copy(input_fits, copy_input_fits)
    median_fits = None
    working_input_fits = copy_input_fits + '' # deep copy
    # If EXPTIME <= 2 seconds, median filter is applied to FITS data
    if input_head['EXPTIME'] <= 2:
        print(f"Working on {copy_input_fits}")
        median_fits = copy_input_fits.replace('.fits', '-median_filter.fits')
        if medianFits(working_input_fits, median_fits, size=5):
            str_err = "ERROR: Median filter could not be applied to input fits"
            print(str_err)
            return 5

        # Input image PNG
        median_fits_png = median_fits.replace('.fits', '.png')
        plotFits(median_fits, median_fits_png) # , dictParams={'aspect':'auto', 'stretch': 'log', 'vmin': 1})

        working_input_fits = median_fits # change working fits file to median one

    # Source detection
    back_image, segment_image, cat_image = ['back.fits', 'segment.fits', 'catalog.cat']
    sex_conf = os.path.join(args.config_dir, 'daofind.sex')

    com_str = "source-extractor -c {} -CATALOG_NAME {} -CHECKIMAGE_TYPE BACKGROUND,SEGMENTATION -CHECKIMAGE_NAME {},{} -PIXEL_SCALE {} {} "
    com = com_str.format(sex_conf, cat_image, back_image, segment_image, \
    input_head['INSTRSCL'], working_input_fits)

    # more source-extractor parameters
    additional_params = default_detection_params(input_head['EXPTIME'])

    for k, v in additional_params.items():
        com += ' -{} {}'.format(k, v)

    if median_fits:
        com = com.replace(' -ANALYSIS_THRESH 1.0 -DETECT_THRESH 1.0', ' -ANALYSIS_THRESH 0.1 -DETECT_THRESH 0.1')
        com = com.replace(' -DETECT_MINAREA 13', ' -DETECT_MINAREA 10')

    print(com)
    subprocess.Popen(com, shell=True).wait()


    # Reading SExtractor catalog
    cat = ''
    with open(cat_image) as fin:
        cat = fin.read()
    campos = re.findall(r'#\s+\d+\s+([\w_]*)', cat)
    print(f'Header catalog keywords = {campos}')

    data_sex = np.genfromtxt(cat_image, names=campos)
    # Working with pandas DataFrame
    data_sex = pd.DataFrame({k:data_sex[k] for k in campos})


    # Filtering duplicated sources
    numbers = list()
    mask = np.ones(len(data_sex.index), dtype=bool) # Nothing selected

    for index, n in enumerate(data_sex['NUMBER'].tolist()):
        tx = data_sex.iloc[index]['X_IMAGE']
        ty = data_sex.iloc[index]['Y_IMAGE']
        tm = data_sex.iloc[index]['MAG_BEST']
        if n not in numbers:
            distx = data_sex['X_IMAGE'].values - tx
            disty = np.abs(data_sex['Y_IMAGE'].values - ty)
            diffmag = np.abs(data_sex['MAG_BEST'] - tm)

            boo = (data_sex['NUMBER'].values.astype(int) != n) & (disty < 1) & \
            (distx > 0) & (distx < 38) & (diffmag < 1) # & (data['FLAGS'].astype(int) == 0)
            if boo.sum() >= 1: # Hay fuentes que han pasado el filtro anterior
                print(data_sex[boo].info())
                numbers.append(int(data_sex[boo].iloc[0]['NUMBER']))
                mask[index] = False

    mask = np.logical_not(mask) # this allows working with ordinary sources
    x = data_sex['X_IMAGE'][mask].values
    y = data_sex['Y_IMAGE'][mask].values
    mag = data_sex['MAG_BEST'][mask].values
    numb = data_sex['NUMBER'][mask].values.astype(int)

    print(f"Final duplicated number of sources = {numb.size}")
    print(numb)

    # segmentation image
    seghdul = fits.open(segment_image)
    segdata = seghdul[0].data.astype(int)
    seghdul.close()

    segment_png = '{}_segmentation.png'.format(fits_name)
    print(segdata.dtype)
    print('Segmentation -> shape: {}'.format(segdata.shape))

    plotFits(segment_image, segment_png, title=fits_name)

    # background image
    backhdul = fits.open(back_image)
    back_data = backhdul[0].data
    backhdul.close()

    back_png = '{}_background.png'.format(fits_name)
    print(back_data.dtype)
    print('Background -> shape: {}'.format(back_data.shape))

    try:
        plotFits(back_image, back_png, title=fits_name)
    except ValueError:
        print(f"WARNING: Problems plotting {back_image}")

    # Masking duplicated sources...
    mask = np.zeros(segdata.shape)
    for n in numb:
        boolm = segdata == n
        mask = np.logical_or(mask, boolm)

    # clean of duplicated sources image
    clean_fits = '{}_clean.fits'.format(fits_name)

    hdul = fits.open(working_input_fits)
    header = hdul[0].header
    data = hdul[0].data
    # replacing duplicated sources with background area given by their
    # segmentation areas
    datamasked = np.where(mask, back_data, data)
    hdul[0].data = datamasked
    hdul.writeto(clean_fits, overwrite=True)

    print("Imagen sin duplicados: {}".format(clean_fits))

    clean_png = f'{fits_name}_clean.png'
    plotFits(clean_fits, clean_png, \
        title=f'{fits_name} without duplicated detections')

    ## Rotating cleaned image 90 degrees counter clockwise
    clean_rotated_fits = clean_fits.replace('.fits', '_rotated.fits')

    hdul = fits.open(clean_fits)
    hdul[0].data = np.rot90(hdul[0].data, k = -1)
    # [:, ::-1] # pay attention to this rotacion + flipping
    hdul.writeto(clean_rotated_fits, overwrite = True)
    hdul.close()

    clean_rotated_png = clean_png.replace('.png', '_rotated.png')
    title = f'{fits_name} rotated and without duplicated detections'
    plotFits(clean_rotated_fits, clean_rotated_png, title=title)

    # Source-extraction for cleaned-rotated FITS
    cat = clean_rotated_fits.replace('.fits', '.cat')
    sex_conf = os.path.join(args.config_dir, 'daofind.sex')

    fwhm_arcs = float(input_head['FWHM']) * float(input_head['INSTRSCL'])
    com_str = "source-extractor -c {} -CATALOG_NAME {} -PIXEL_SCALE {} -SEEING_FWHM {}"
    com = com_str.format(sex_conf, cat, input_head['INSTRSCL'], \
        fwhm_arcs)

    # more source-extractor parameters
    additional_params = default_detection_params(input_head['EXPTIME'])

    for k, v in additional_params.items():
        com += ' -{} {}'.format(k, v)

    com = f"{com} {clean_rotated_fits}"
    print(com)
    subprocess.Popen(com, shell=True).wait()

    # Filtering sources too close to FITS limits
    cat_lines = ''
    with open(cat) as fin:
        cat_lines = fin.read()
    campos = re.findall(r'#\s+\d+\s+([\w_]*)', cat_lines)
    print('Campos del header del catalogo = {}'.format(campos))

    data_sex = np.genfromtxt(cat, names=campos)

    # Plotting detected sources...
    # DETECTED SOURCES
    all_detect_sext_png = clean_rotated_png.replace('.png', '_all_detect_sext.png')
    print('Out PNG ->', all_detect_sext_png)
    title_plot = f"SExtractor astrocalibration sources in {fits_name}"
    plotFits(clean_rotated_fits, all_detect_sext_png, title=title_plot, \
        colorBar=True, ref_coords='pixel', astroCal=False, color='magenta', \
        coords=(data_sex['X_IMAGE'], data_sex['Y_IMAGE'])) # , \
        # dictParams={'aspect':'auto', 'invert':'True', 'stretch': 'log', 'vmin':1})


    # Working with Pandas dataframe
    data_sex = pd.DataFrame({k:data_sex[k] for k in campos})

    # Border size (pixels)
    border = args.border_image # name simplification
    x = data_sex['X_IMAGE'].values
    y = data_sex['Y_IMAGE'].values
    inner_sources = (x > border) & (x < (int(header['NAXIS1']) - border)) \
        & (y > border) & (y < (int(header['NAXIS2']) - border))
    
    # filtering sources too closer than 100 pixels to HD stars
    if header['EXPTIME'] < 5: # HD source
        print(f"EXPTIME = {header['EXPTIME']}")
        print("\t---------------Deleting sources closer than 100 pixels to calibrator")
        index_brilliant = data_sex['FLUX_ISO'].values.argmax()
        xb = data_sex['X_IMAGE'].values[index_brilliant]
        yb = data_sex['Y_IMAGE'].values[index_brilliant]
        print(f"\tCalibrator coords (x, y) = ({xb}, {yb})")
        distances = np.sqrt(np.power(x - xb, 2) + np.power(y - yb, 2))
        indexes = (distances >= 50) | (distances < 5)
        inner_sources = inner_sources & indexes

    print("Number of sources before filtering = {}".format(y.size))
    data_sex_filtered = data_sex[inner_sources]
    # for k in campos:
    #     data_sex_filtered[k] = data_sex[k][inner_sources]

    n_sources = len(data_sex_filtered.index)
    print(f"Number of sources after filtering = {n_sources}")

    # Checking number of sources...
    if n_sources < 5:
        # relaxing SExtractor conditions
        com_str = "source-extractor -c {} -CATALOG_NAME {} -PIXEL_SCALE {} -SEEING_FWHM {}"
        com = com_str.format(sex_conf, cat, input_head['INSTRSCL'], \
            fwhm_arcs)

        # more source-extractor parameters
        additional_params = relaxed_detection_params(input_head['EXPTIME'])

        for k, v in additional_params.items():
            com += ' -{} {}'.format(k, v)

        com = f"{com} {clean_rotated_fits}"

        print(com)
        subprocess.Popen(com, shell=True).wait()

        # Plotting detected sources...
        # DETECTED SOURCES
        all_detect_sext_png = clean_rotated_png.replace('.png', '_all_detect_sext.png')
        print('Out PNG ->', all_detect_sext_png)
        title_plot = f"SExtractor astrocalibration sources in {fits_name}"
        plotFits(clean_rotated_fits, all_detect_sext_png, title=title_plot, \
            colorBar=True, ref_coords='pixel', astroCal=False, color='magenta', \
            coords=(data_sex['X_IMAGE'], data_sex['Y_IMAGE'])) # , \
            # dictParams={'aspect':'auto', 'invert':'True', 'stretch': 'log', 'vmin':1})

        # Filtering sources too close to FITS limits
        cat_lines = ''
        with open(cat) as fin:
            cat_lines = fin.read()
        campos = re.findall(r'#\s+\d+\s+([\w_]*)', cat_lines)
        print('Campos del header del catalogo = {}'.format(campos))

        data_sex = np.genfromtxt(cat, names=campos)

        # Working with Pandas dataframe
        data_sex = pd.DataFrame({k:data_sex[k] for k in campos})

        # Border size (pixels)
        border = args.border_image
        x = data_sex['X_IMAGE'].values
        y = data_sex['Y_IMAGE'].values
        inner_sources = (x > border) & (x < (int(header['NAXIS1']) - border)) \
            & (y > border) & (y < (int(header['NAXIS2']) - border))

        # filtering sources too closer than 5'-0 pixels to HD stars
        if header['EXPTIME'] < 5: # HD source
            print(f"EXPTIME = {header['EXPTIME']}")
            print("\t---------------Deleting sources closer than 50 pixels to calibrator")
            index_brilliant = data_sex['FLUX_ISO'].values.argmax()
            xb = data_sex['X_IMAGE'].values[index_brilliant]
            yb = data_sex['Y_IMAGE'].values[index_brilliant]
            print(f"\tCalibrator coords (x, y) = ({xb}, {yb})")
            distances = np.sqrt(np.power(x - xb, 2) + np.power(y - yb, 2))
            indexes = (distances >= 50) | (distances < 5)
            inner_sources = inner_sources & indexes

        print("Number of sources before filtering = {}".format(y.size))
        data_sex_filtered = data_sex[inner_sources]
        
        n_sources = len(data_sex_filtered.index)
        print(f"Number of sources after filtering = {n_sources}")

    # Plotting inner sources...
    # INNER SOURCES
    inner_detect_sext_png = clean_rotated_png.replace('.png', '_inner_detect_sext.png')
    print('Out PNG ->', inner_detect_sext_png)
    title_plot = 'SExtractor astrocalibration sources in %s' % fits_name
    plotFits(clean_rotated_fits, inner_detect_sext_png, title=title_plot, \
        colorBar=True, ref_coords='pixel', astroCal=False, color='magenta', \
        coords=(data_sex_filtered['X_IMAGE'], data_sex_filtered['Y_IMAGE'])) # , \
        # dictParams={'aspect':'auto', 'invert':'True', 'stretch': 'log', 'vmin':1})

    # Getting brightest detections
    num_sorted = 75
    index_ord = None
    dsfo = data_sex_filtered

    if input_head['EXPTIME'] <= 1:
        # low EXPTIME produces noisy images and too many false detections
        num_sorted = 25
        dsfo = data_sex_filtered.sort_values(by=['FLUX_MAX'], ascending=False)[:num_sorted]
    else:
        # Low level of noise for this exposure times
        # lower values of MAG_BEST means more brilliant sources
        dsfo = data_sex_filtered.sort_values(by=['MAG_BEST'])[:num_sorted]

    print(f'Number of brightest sources used = {len(dsfo.index)}')
    #cat_sort = cat.replace('.cat', '_sorted.cat')
    cat_sort_filtered = cat.replace('.cat', '_sorted_filtered.cat')

    # Writing to file (needed for WCSTools astrometric calibration)
    dsfo.to_csv(cat_sort_filtered, index=False, sep=' ', header=False)

    # Plotting selected brilliant sources
    out_detect_sext_png = clean_rotated_png.replace('.png', '_detect_sext.png')
    print('Out PNG ->', out_detect_sext_png)
    title_plot = f'SExtractor astrocalibration sources in {fits_name}'

    plotFits(clean_rotated_fits, out_detect_sext_png, title=title_plot, \
        colorBar=True, coords=(dsfo['X_IMAGE'], dsfo['Y_IMAGE']), \
        ref_coords='pixel', astroCal=False, \
        color='green', dictParams={'aspect':'auto', 'invert':'True'})

    # header del fichero
    print("FITS header before astrometric calibration")
    hdul = fits.open(clean_rotated_fits)
    previous_header = hdul[0].header
    hdul.close()

    # Useful information about input clean rotated FITS
    pix_scale = previous_header['INSTRSCL']
    ra_im = previous_header['RA']
    dec_im = previous_header['DEC']
    date_obs = previous_header['DATE-OBS']
    str_out = 'RA = {}\nDEC = {}\nDATE-OBS = {}\nPIX_SCALE = {}'
    print(str_out.format(ra_im, dec_im, date_obs, pix_scale))

    # Astrometric calibration process
    astrom_out_fits = clean_rotated_fits.replace('.fits', 'w.fits')

    com_str = "imwcs -wve -d {} -r 0 -y 3 -p {} -j {} {} -h {} -c {} -t 10 -o {} {}"
    com = com_str.format(cat_sort_filtered, pix_scale, ra_im, dec_im, \
        num_sorted, 'tmc', astrom_out_fits, clean_rotated_fits)

    print("astrometric calibration command")
    print("-------------------------")
    print(com)
    cal_out = '{}_imwcs_2mass.log'.format(fits_name)
    with open(cal_out, "w") as fout:
        fout.write("\n#*********************************************************\n")
        fout.write("\n#********* ASTROMETRIC CALIBRATION  ***********\n")
        fout.write("\n#*********************************************************\n")

    fout = open(cal_out, "a")
    subprocess.Popen(com, shell=True, stdout=fout, stderr=fout).wait()
    fout.close()

    # Not good calibration process if not enough sources
    # were used

    astro_header = {'WCSMATCH': 0}
    try:
        hdul_astro = fits.open(astrom_out_fits)
        astro_header = hdul_astro[0].header
    except:
        pass
    if astro_header['WCSMATCH'] < num_sorted / 2: # mala calibración
        print("Second astrometric calibration try")
        print("\tUsing new central FITS coordinates (Closest MAPCAT source)")
        # Coordinates updating to closest MAPCAT source
        ra_im = alternative_ra
        dec_im = alternative_dec
        if astro_header['WCSMATCH'] <= 4:
            # reducing "matchable" fit field ("-y 2" instead "-y 3")
            com_str = "imwcs -wve -d {} -r 0 -y 2 -p {} -j {} {} -h {} -c {} -t 10 -o {} {}"
        com = com_str.format(cat_sort_filtered, pix_scale, ra_im, dec_im, \
            num_sorted, 'tmc', astrom_out_fits, clean_rotated_fits)
        print("2nd astrometric calibration command")
        print("-" * 50)
        print(com)
        cal_out = '{}_imwcs_2mass.log'.format(fits_name)
        with open(cal_out, "w") as fout:
            fout.write("\n#*********************************************************\n")
            fout.write("\n#********* ASTROMETRIC CALIBRATION  ***********\n")
            fout.write("\n#*********************************************************\n")

        fout = open(cal_out, "a")
        subprocess.Popen(com, shell=True, stdout=fout, stderr=fout).wait()
        fout.close()

    print("2nd astrometric calibration try for file: {}".format(astrom_out_fits))
    print('-' * 100)

    astro_header = None
    try:
        calhdul = fits.open(astrom_out_fits)
        astro_header = calhdul[0].header
        calhdul.close()
    except IOError:
        # If imwcs program could'n calibrate, output FITS will not be created
        raise

    print('*' * 50)
    message = "Number of sources used in calibration: {} (of {})"
    print(message.format(astro_header['WCSMATCH'], astro_header['WCSNREF']))
    print('*' * 50)

    # Getting sources from web catalogs
    catalogs = {'2MASS': 'II/246/out', 'NOMAD': 'I/297/out', \
        'USNO-A2': 'I/252/out', 'SDSS-R12': 'V/147/sdss12'}
    for k, v in catalogs.items():
        print(f"{k}: {v}")

    center_coords = coord.SkyCoord(ra=astro_header['CRVAL1'], \
        dec=astro_header['CRVAL2'],unit=(u.deg, u.deg),frame='icrs')
    result = Vizier.query_region(center_coords, width="10m", \
        catalog=list(catalogs.values()))

    print('Web catalogs obtained')
    pprint.pprint(result)

    # Plotting web sources over our calibrated FITS
    cat_out_pngs = {}
    for k in catalogs.keys():
        wcat = None
        try:
            wcat = result[catalogs[k]]
            cat_out_pngs[k] = clean_rotated_png.replace('.png', f'_{k}.png')
            coords = None
            if k == 'SDSS-R12':
                coords = (wcat['RA_ICRS'], wcat['DE_ICRS'])
            else:
                coords = (wcat['RAJ2000'], wcat['DEJ2000'])
            print(f"Plotting data from {k} catalog")
            plotFits(astrom_out_fits, cat_out_pngs[k], \
                title=f'{k} sources in {fits_name}', colorBar=True, \
                coords=coords, astroCal=True, color='green', \
                dictParams={'aspect':'auto', 'invert':'True'})
        except TypeError:
            print(f"No data available for {k}")

    # Rotación del fichero original (90 degrees counter clockwise)
    final_fits = copy_input_fits.replace('.fits', '_final.fits')

    # Astrometrically calibrated FITS
    hdul_cal = fits.open(astrom_out_fits)

    # Copy calibrated header on rotated original FITS
    hdul = fits.open(input_fits)
    hdul[0].header = hdul_cal[0].header
    hdul[0].data = np.rot90(hdul[0].data, k = -1)# [:, ::-1] # pay attention to this rotacion + flipping
    hdul.writeto(final_fits, overwrite = True)
    hdul.close()

    hdul_cal.close()

    # Getting final FITS header
    hdul_final = fits.open(final_fits)
    header_final = hdul_final[0].header
    hdul_final.close()

    # Plotting final calibrated FITS
    final_png = final_fits.replace('.fits', '_final.png')
    title = f'{fits_name} rotated astrocalib'
    plotFits(final_fits, final_png, title=title)

    ################ WORKING ON ASTROMETRIC CALIBRATED FITS ################

    # Getting data from ordinary-extraordinary sources in final FITS
    # 1. SExtractor calling
    fwhm_arcs = astro_header['FWHM'] * astro_header['INSTRSCL']
    sex_conf = os.path.join(args.config_dir, 'sex.conf')
    final_fits = copy_input_fits.replace('.fits', '_final.fits')
    cat = final_fits.replace('.fits', '.cat')

    com_str = "source-extractor -c {} -CATALOG_NAME {} -PIXEL_SCALE {} -SEEING_FWHM {} {}"
    com = com_str.format(sex_conf, cat, input_head['INSTRSCL'], fwhm_arcs, final_fits)

    # MAPCAT aperture
    com += f" -PHOT_APERTURES {mc_aper}"

    # more source-extractor parameters
    additional_params = default_detection_params(input_head['EXPTIME'])

    for k, v in additional_params.items():
        com += ' -{} {}'.format(k, v)
    print(com)
    subprocess.Popen(com, shell=True).wait()

    # RA,DEC limits...
    center_ra = float(header_final['CRVAL1'])
    center_dec = float(header_final['CRVAL2'])
    naxis1 = int(header_final['NAXIS1'])
    naxis2 = int(header_final['NAXIS2'])
    crpix1 = float(header_final['CRPIX1'])
    crpix2 = float(header_final['CRPIX2'])
    pixscale = float(header_final['INSTRSCL'])
    ra_min = center_ra - (naxis1 - crpix1) * pixscale / 3600. # degrees
    ra_max = center_ra + (naxis1 - crpix1) * pixscale / 3600. # degrees
    dec_min = center_dec - (naxis2 - crpix2) * pixscale / 3600. # degrees
    dec_max = center_dec + (naxis2 - crpix2) * pixscale / 3600. # degrees
    # Loading FITS_LDAC format SExtractor catalog
    sext = fits.open(cat)
    data = sext[2].data
    # ra_min = data['ALPHA_J2000'].min()
    # ra_max = data['ALPHA_J2000'].max()
    # dec_min = data['DELTA_J2000'].min()
    # dec_max = data['DELTA_J2000'].max()

    print ("Number of detections = {}".format(data['ALPHA_J2000'].size))
    intervals = '(ra_min, ra_max, dec_min, dec_max) = ({}, {}, {}, {})'
    print(intervals.format(ra_min, ra_max, dec_min, dec_max))

    # Showing detail of detections
    sources_sext_png = cat.replace('.cat', '.png')
    print('Out PNG ->', sources_sext_png)
    plotFits(final_fits, sources_sext_png, \
        title=f'MAPCAT sources in {fits_name}', colorBar=True, \
        coords=(data['ALPHA_J2000'], data['DELTA_J2000']), \
        astroCal=True, color='red') # , \
        # dictParams={'aspect':'auto', 'invert':'True', 'stretch': 'log', 'vmin':1})

    # Searching for MAPCAT sources inside limits FITS coordinates
    df_mc = df_mapcat[df_mapcat['ra2000_mc_deg'] > ra_min]
    df_mc = df_mc[df_mc['ra2000_mc_deg'] < ra_max]
    df_mc = df_mc[df_mc['dec2000_mc_deg'] > dec_min]
    df_mc = df_mc[df_mc['dec2000_mc_deg'] < dec_max]

    print("MAPCAT filtered info...")
    print(df_mc.info())
    print(f'Number of MAPCAT sources= {len(df_mc.index)}')

    # Plotting MAPCAT sources
    if len(df_mc.index) > 0:
        sources_mapcat_png = final_fits.replace('.fits', '_mapcat_sources.png')
        print('Out PNG ->', sources_mapcat_png)
        plotFits(final_fits, sources_mapcat_png, \
            title=f'MAPCAT sources in {fits_name}', colorBar=True, \
            coords=(df_mc['ra2000_mc_deg'].values, df_mc['dec2000_mc_deg'].values), \
            astroCal=True, color='orange', \
            dictParams={'aspect':'auto', 'invert':'True'})

    # Closest SExtractor detections
    scatalog = SkyCoord(ra = df_mc['ra2000_mc_deg'].values * u.degree, \
        dec = df_mc['dec2000_mc_deg'].values * u.degree)
    pcatalog = SkyCoord(ra = data['ALPHA_J2000'] * u.degree, \
        dec = data['DELTA_J2000'] * u.degree)

    idx_o, d2d_o, d3d_o = match_coordinates_sky(scatalog, pcatalog, \
        nthneighbor=1)

    print(f'SExtractor closest ordinary detection indexes = {idx_o}')
    print(f'SExtractor closest ordinary detection distances = {d2d_o}')

    # Printing info about MAPCAT-SExtractor sources
    str_match = " MAPCAT (name, ra, dec, Rmag, Rmagerr) = ({}, {}, {}, {}, {})"
    str_match += "\nSExtractor (ra, dec, mag_auto, magerr_auto) = ({}, {}, {}, {})"
    str_dist = "Distance = {}\n-----------------------"
    for j in range(len(idx_o)):
        print(str_match.format(df_mc['name_mc'].values[j], \
            df_mc['ra2000_mc_deg'].values[j], df_mc['dec2000_mc_deg'].values[j], \
            df_mc['Rmag_mc'].values[j], df_mc['Rmagerr_mc'].values[j], \
            data['ALPHA_J2000'][idx_o[j]], data['DELTA_J2000'][idx_o[j]], \
            data['MAG_AUTO'][idx_o[j]], data['MAGERR_AUTO'][idx_o[j]]))
        print(str_dist.format(d2d_o[j]))

    # Extraordinary counterparts location
    # rough coordinates
    ra_e = data['ALPHA_J2000'][idx_o]
    dec_e = data['DELTA_J2000'][idx_o] - 0.0052

    scatalog_e = SkyCoord(ra = ra_e * u.degree, dec = dec_e * u.degree)
    idx_e, d2d_e, d3d_e = match_coordinates_sky(scatalog_e, pcatalog, \
        nthneighbor=1)
    print(f"pcatalog = {pcatalog}")
    print(f"scatalog_e = {scatalog_e}")

    print(f"SExtractor numbers for extraordinary counterparts = {idx_e}")
    print(f"Distances = {d2d_e}")

    # Source problem...
    source_problem = None
    if len(df_mc.index) == 1:
        # If there is only source and it's R calibrated -> HD star
        source_problem = np.ones(1, dtype=bool)
    else:
        # Source problem has no R filter magnitude
        source_problem = df_mc['Rmag_mc'].values < 0

    print(f'source_problem = {source_problem}')

    # Printing SExtractor indexes
    indexes = [idx_o[source_problem][0], idx_e[source_problem][0]]
    print(f'[Ordinary, Extraordinary] SExtractor indexes = {indexes}')

    # Plotting source problem
    # Muestro el nivel de detalle de las detecciones de SExtractor en la imagen final
    source_pair_png = sources_mapcat_png.replace('.png', '_source_pair.png')
    print('Out PNG ->', source_pair_png)
    plotFits(final_fits, source_pair_png, colorBar=True, \
        title=f'SExtractor Pair Detections in {os.path.split(final_fits)[1]}', \
        coords=(data['ALPHA_J2000'][indexes], data['DELTA_J2000'][indexes]), \
        astroCal=True, color='red') # , \
        # dictParams={'aspect':'auto', 'invert':'True', 'stretch': 'log', 'vmin':1})


    # Parameters to store...
    # Getting useful info about calibrated fits
    some_calib_keywords = ['SOFT', 'PROCDATE', 'SOFTDET', 'MAX',  'MIN', \
        'MEAN', 'STD', 'MED', 'RA', 'DEC', 'CRVAL1', 'CRVAL2', 'EPOCH', \
        'CRPIX1', 'CRPIX2', 'SECPIX1', 'SECPIX2', 'CDELT1', 'CDELT2', 'CTYPE1', 'CTYPE2', \
        'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'WCSRFCAT', 'WCSIMCAT', 'WCSMATCH', \
        'WCSNREF', 'WCSTOL', 'CROTA1', 'CROTA2', 'WCSSEP', 'IMWCS', 'MAGZPT', \
        'STDMAGZP', 'NSZPT', 'BLZRNAME']
    
    cal_data = {}
    header = hdul = fits.open(final_fits)[0].header
    for key in some_calib_keywords:
        cal_data[key] = [header.get(key, '')]
    cal_data['PATH'] = [final_fits]
    cal_data['RUN_DATE'] = [date_run]

    # Photometric calibration
    mag_zeropoint = None
    std_mag_zeropoint = None
    num_cal = None

    # If there are MAPCAT calibrators in field covered by FITS
    if len(df_mc[~source_problem].index) > 0:
        # calibrators total flux (Ordinary + Extraordinary)
        total_flux = data['FLUX_APER'][idx_o[~source_problem]] + \
        data['FLUX_APER'][idx_e[~source_problem]]
        zps = df_mc['Rmag_mc'][~source_problem].values + \
        2.5 * np.log10(total_flux)

        print(f'zps = {zps}')

        mag_zeropoint = zps.mean()
        std_mag_zeropoint = zps.std()
        num_cal = zps.size

        # Plotting calibrators
        mc_calib_png = final_fits.replace('.fits', '_sources_mc_calib.png')
        print('Out PNG ->', mc_calib_png)
        plotFits(final_fits, mc_calib_png, colorBar=True, \
            title=f'MAPCAT Calibration sources in {fits_name}', \
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
        num_cal = 1


    print(f"Photometric Zero-point = {round(mag_zeropoint, 2)}")
    print(f"STD(Photometric Zero-point) = {round(std_mag_zeropoint, 2)}")

    # Updating FITS header
    if mag_zeropoint is not None:
        # Lo pongo en la cabecera del FITS
        hdul = fits.open(final_fits, mode='update')

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
            hdul[0].header.append(('NSZPT', num_cal, \
                'MAPCAT sources used in MAGZPT estimation'))
        else:
            hdul[0].header['NSZPT'] = num_cal
        
        bz_name = df_mc[source_problem]['IAU_name_mc'].values[0]
        if 'BLZRNAME' not in hdul[0].header:
            hdul[0].header.append(('BLZRNAME', bz_name, \
                'IAU name for BLAZAR object'))
        else:
            hdul[0].header['BLZRNAME'] = bz_name

        hdul.flush()
        hdul.close()

        # Executing SExtractor again
        com_str = "source-extractor -c {} -CATALOG_NAME {} -PIXEL_SCALE {} -SEEING_FWHM {} {}"
        com = com_str.format(sex_conf, cat, input_head['INSTRSCL'], fwhm_arcs, final_fits)

        # MAPCAT aperture
        com += f" -PHOT_APERTURES {mc_aper}"

        # Magnitude ZEROPOINT
        com += f" -MAG_ZEROPOINT {mag_zeropoint}"

        # more source-extractor parameters
        additional_params = default_detection_params(input_head['EXPTIME'])

        for k, v in additional_params.items():
            com += ' -{} {}'.format(k, v)

        print(com)
        subprocess.Popen(com, shell=True).wait()

        # Loading FITS_LDAC format SExtractor catalog
        sext = fits.open(cat)
        data = sext[2].data
    
    cal_data['CLEAN_ROT_PNG'] = [clean_rotated_png]
    cal_data['INNER_SEXTDET_PNG'] = [inner_detect_sext_png]
    cal_data['OUT_SEXTDET_PNG'] = [out_detect_sext_png]
    for k, v in cat_out_pngs.items():
        cal_data[f"CALIB_{k}_PNG"] = [v]
    cal_data['FINAL_PNG'] = [final_png]
    cal_data['SEXTDET_PNG'] = [sources_sext_png]
    cal_data['MAPCAT_SOURCES_PNG'] = [sources_mapcat_png]
    cal_data['SOURCE_PAIRS_PNG'] = [source_pair_png]
    

    df = pd.DataFrame(cal_data)
    csv_out = final_fits.replace('.fits', '_info.csv')
    df.to_csv(csv_out, index=False)


    # Interesting parameters for polarimetric computation
    keywords = ['ALPHA_J2000', 'DELTA_J2000', 'FWHM_IMAGE', 'CLASS_STAR', \
        'FLAGS', 'ELLIPTICITY', 'FLUX_MAX', 'FLUX_APER', 'FLUXERR_APER', \
        'FLUX_ISO', 'FLUXERR_ISO', 'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_APER', \
        'MAGERR_APER', 'MAG_ISO', 'MAGERR_ISO', 'MAG_AUTO', 'MAGERR_AUTO']

    pair_params = defaultdict(list)

    pair_params['ID-MC'] = [df_mc[source_problem].iloc[0]['id_mc']] * 2
    pair_params['ID-BLAZAR-MC'] = [df_mc['id_blazar_mc'].values[source_problem][0]] * 2
    pair_params['TYPE'] = ['O', 'E']
    angle = float(header_final['INSPOROT'])
    pair_params['ANGLE'] = [round(angle, ndigits=1)] * 2
    pair_params['OBJECT'] = [header_final['OBJECT']] * 2
    pair_params['MJD-OBS'] = [header_final['MJD-OBS']] * 2
    pair_params['DATE-OBS'] = [header_final['DATE-OBS']] * 2
    mc_name = df_mc['name_mc'].values[source_problem][0]
    mc_iau_name = df_mc['IAU_name_mc'].values[source_problem][0]
    pair_params['MC-NAME'] = [mc_name] * 2
    pair_params['MC-IAU-NAME'] = [mc_iau_name] * 2
    pair_params['MAGZPT'] = [mag_zeropoint] * 2
    pair_params['RUN_DATE'] = [date_run] * 2
    pair_params['EXPTIME'] = [header_final['EXPTIME']] * 2

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
    csv_out = final_fits.replace('.fits', '.csv')
    df.to_csv(csv_out, index=False)

    # Imprimo el contenido del fichero
    print('Useful parameters for polarimetric computations:')
    print(df)

    return 0

# -------------------------------------
if __name__ == '__main__':
    print(main())