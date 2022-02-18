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
import glob
import re
from collections import defaultdict, OrderedDict

# Data structures libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyrsistent import v
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

from mcFits import mcFits
Vizier.ROW_LIMIT = -1

# HTML ouput template
import jinja2

# Our IOP3 FITS module
from mcFits import mcFits


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
    ref_coords='world', astroCal=False, color='green', format='png', \
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


def get_skylimits(path_fits):
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


def read_sext_catalog(path, format='txt', verbose=False):
    cat = ''
    with open(path) as fin:
        cat = fin.read()
    campos = re.findall(r'#\s+\d+\s+([\w_]*)', cat)
    if verbose:
        print(f'Header catalog keywords = {campos}')

    data_sext = np.genfromtxt(path, names=campos)
    # Working with pandas DataFrame
    data_sext = pd.DataFrame({k:data_sext[k] for k in campos})
    
    return data_sext
    
def get_duplicated(data):
    
    numbers = list()
    mask = np.ones(len(data.index), dtype=bool) # Nothing selected

    for index, n in enumerate(data['NUMBER'].tolist()):
        tx = data.iloc[index]['X_IMAGE']
        ty = data.iloc[index]['Y_IMAGE']
        tm = data.iloc[index]['MAG_BEST']
        if n not in numbers:
            distx = data['X_IMAGE'].values - tx
            disty = np.abs(data['Y_IMAGE'].values - ty)
            diffmag = np.abs(data['MAG_BEST'] - tm)

            #Original pair-sources conditions
            #  boo = (data_sex['NUMBER'].values.astype(int) != n) & (disty < 1) & \
            # (distx > 0) & (distx < 38) & (diffmag < 1) # & (data['FLAGS'].astype(int) == 0)

            # relaxed pair-sources conditions
            boo = (data['NUMBER'].values.astype(int) != n) & (disty < 10) & \
            (distx > 0) & (distx < 45) & (diffmag < 2) # & (data['FLAGS'].astype(int) == 0)
            if boo.sum() >= 1: # Hay fuentes que han pasado el filtro anterior
                # print(data_sex[boo].info())
                numbers.append(int(data[boo].iloc[0]['NUMBER']))
                mask[index] = False

    mask = np.logical_not(mask) # this allows working with ordinary sources

    return mask

def mask_duplicated_simple(input_fits, output_fits, sext_catalog, segmentation_fits):
    fits_name = os.path.split(input_fits)[1].replace('.fits', '')
    
    # segmentation image
    seg = mcFits(segmentation_fits)
    segdata = seg.data

    # Reading text format SExtractor catalog
    data_sex = read_sext_catalog(sext_catalog)

    # Filtering duplicated sources
    mask = get_duplicated(data_sex)
    print(f"Duplicated number of sources = {mask.size}")
    
    numb = data_sex['NUMBER'][mask].values

    ##### --------- Masking duplicated sources -------- #####
    mask = np.zeros(segdata.shape)
    for n in numb:
        boolm = segdata == n
        mask = np.logical_or(mask, boolm)

    hdul = fits.open(input_fits)
    data = hdul[0].data
    # replacing duplicated sources with background area given by their
    # segmentation areas
    datamasked = np.where(mask, int(np.median(data)), data)
    hdul[0].data = datamasked
    hdul.writeto(output_fits, overwrite=True)

    return 0

def mask_duplicated(input_fits, output_fits, sext_catalog, segmentation_fits, background_fits):
    fits_name = os.path.split(input_fits)[1].replace('.fits', '')
    
    # segmentation image
    seg = mcFits(segmentation_fits)
    segdata = seg.data

    # background image
    back = mcFits(background_fits)
    back_data = back.data.astype(np.uint16)
    # Reading text format SExtractor catalog
    data_sex = read_sext_catalog(sext_catalog)

    # Filtering duplicated sources
    mask = get_duplicated(data_sex)
    print(f"Duplicated number of sources = {mask.size}")
    
    numb = data_sex['NUMBER'][mask].values.astype(int)

    ##### --------- Masking duplicated sources -------- #####
    mask = np.zeros(segdata.shape)
    for n in numb:
        boolm = segdata == n
        mask = np.logical_or(mask, boolm)

    hdul = fits.open(input_fits)
    data = hdul[0].data
    # replacing duplicated sources with background area given by their
    # segmentation areas
    datamasked = np.where(mask, back_data, data)
    hdul[0].data = datamasked
    hdul.writeto(output_fits, overwrite=True)

    return 0

def rotation(input_fits, output_fits):
    
    hdul_in = fits.open(input_fits)
    if 'MAPCAT' in input_fits:
        hdul_out = fits.PrimaryHDU(data=np.rot90(hdul_in[0].data, k = -1).astype(np.uint16), header=hdul_in[0].header)
    else:
        #images of OSN are mirrored in X axis, and must be rotated 90 degrees to be equal to mapcat
        #check for header parameter "FLIPSTAT"
        hdul_out=fits.PrimaryHDU(data=hdul_in[0].data[:,::-1].astype(np.uint16) , header=hdul_in[0].header)
        hdul_out = fits.PrimaryHDU(data=np.rot90(hdul_out.data, k = -1).astype(np.uint16), header=hdul_out.header)
        hdul_out = fits.PrimaryHDU(data=np.rot90(hdul_out.data, k = -1).astype(np.uint16), header=hdul_out.header)
    hdul_out.writeto(output_fits, overwrite = True)
    hdul_in.close()
    
    return 0

def execute_command(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, \
        stderr=subprocess.PIPE, shell=True, check=True)
    return result


def get_sources(input_fits, sext_conf, add_params={}, threshold_exptime=2, \
    back_image=False, segment_image=False, verbose=True):

    i_fits = mcFits(input_fits)
    if 'MAPCAT' in input_fits:
        pixscale = i_fits.header['INSTRSCL']
    elif 'T090' in input_fits:
        pixscale=0.387
    elif 'T150' in input_fits:
        pixscale=0.232
    exptime = i_fits.header['EXPTIME']
    fwhm_arcs = float(i_fits.header['FWHM']) * float(pixscale)
    
    if 'fits' in input_fits:
        out_cat = input_fits.replace('.fits', '.cat')
    else:
        out_cat = input_fits.replace('.fit', '.cat')

    # Adtitional ouput info
    if 'fits' in input_fits:
        back_path = input_fits.replace('.fits', '_back.fits')
        segm_path = input_fits.replace('.fits', '_segment.fits')
    else:
        back_path = input_fits.replace('.fit', '_back.fit')
        segm_path = input_fits.replace('.fit', '_segment.fit')

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
    
    cmd = f"source-extractor -c {sext_conf} -CATALOG_NAME {out_cat} -PIXEL_SCALE {pixscale} -SEEING_FWHM {fwhm_arcs} "
    
    # more source-extractor parameters
    additional_params = default_detection_params(exptime)
    for k, v in additional_params.items():
        params[k] = v
    
    # Additipnal parameters passed as arguments. They overwrite previous values
    for k, v in add_params.items():
        params[k] = v
    
    # Formatting parameters to command line syntax
    com_params = [f'-{k} {v}' for k, v in params.items()]
    
    # adding parameters to command
    cmd = cmd + ' '.join(com_params)
    
    # last parameter for command
    cmd = cmd + f' {input_fits}'

    # Extra parametrization
    if exptime <= threshold_exptime:
        # tunning detection parameters
        cmd = cmd.replace(' -ANALYSIS_THRESH 1.0', ' -ANALYSIS_THRESH 0.1')
        cmd = cmd.replace(' -DETECT_THRESH 1.0', ' -DETECT_THRESH 0.1')
        cmd = cmd.replace(' -DETECT_MINAREA 13', ' -DETECT_MINAREA 10')

    if verbose:
        print(cmd)
    
    return execute_command(cmd)
    
def filter_detections(data, path_fits, border=15, exptime_threshold=5, \
    max_distance=50, min_distance=5):
    """
    Reject sources closer to image borders (and to most brilliant source).
    
    Sources closer to image border than 'border' pixels are rejected.
    In case of short exposure images (lower than 'exptime_threshold' 
    seconds), detections closer than 'distance' to most brilliant source
    are rejected also.
    """
    fits = mcFits(path_fits)
    input_header = fits.header
    
    size_x = int(input_header['NAXIS1'])
    size_y = int(input_header['NAXIS2'])
    exptime = input_header['EXPTIME']

    # Getting inner FITS area (inside borders of image)
    x = data['X_IMAGE'].values
    y = data['Y_IMAGE'].values
    # print(f'x = {x}')
    # print(x.dtype)
    inner_sources = (x > border) & (x < (size_x - border)) \
        & (y > border) & (y < (size_y - border))
    
    # Filtering sources too closer to saturated HD stars (distance lower than 50 pixels)
    if exptime < exptime_threshold: # EXPTIME is the criterium for HD source
        print(f"EXPTIME = {exptime}")
        print("\t----------- Deleting sources closer than 50 pixels to calibrator")
        # getting most brilliant source dataframe index
        index_brilliant = data['FLUX_ISO'].values.argmax()
        # getting (x, y) pixel coordinates
        xb = x[index_brilliant]
        yb = y[index_brilliant]
        print(f"\tTarget source coords (x, y) = ({xb}, {yb})")
        # computing distances between calibrator an other sources...
        distances = np.sqrt(np.power(x - xb, 2) + np.power(y - yb, 2))
        # setting validity criterium
        indexes = (distances >= max_distance) | (distances < min_distance)
        # final validity boolean array for detected sources in dataframe
        inner_sources = inner_sources & indexes

    print("Sources before filtering = {}".format(y.size))
    data_filtered = data[inner_sources]
    
    return data_filtered

def get_brilliant_sources(data, exptime, max_number=75, exptime_threshold=1):
    # Getting brightest detections
    df_sorted = None

    if exptime <= exptime_threshold:
        # low EXPTIME produces noisy images and too many false detections
        num_sorted = 25
        df_sorted = data.sort_values(by=['FLUX_MAX'], ascending=False)[:max_number]
    else:
        # Low level of noise for this exposure times
        # lower values of MAG_BEST means more brilliant sources
        df_sorted = data.sort_values(by=['MAG_BEST'])[:max_number]

    return df_sorted


def astrocal(path_fits, coord_csv, closest_blazar_coords={}):
    
    # header del fichero
    fits = mcFits(path_fits)

    # Useful information about input clean rotated FITS
    if 'MAPCAT' in path_fits:
        pix_scale = fits.header['INSTRSCL']
    elif 'T090' in path_fits:
        pix_scale=0.387
    elif 'T150' in path_fits:
        pix_scale=0.232
    date_obs = ''
    if 'DATE-OBS' in fits.header:
        date_obs = fits.header['DATE-OBS']
    else:
        date_obs = fits.header['DATE']
    
    # FITS or external center coordinates
    if 'MAPCAT' in path_fits:
        ra_im = fits.header['RA']
        dec_im = fits.header['DEC']
    else:
        icoords = icoords = "{} {}".format(fits.header['OBJCTRA'], fits.header['OBJCTDEC'])
        input_coords = SkyCoord(icoords, frame=FK5, unit=(u.hourangle, u.deg), \
                                    obstime="J2000")
        ra_im = input_coords.ra.value
        dec_im = input_coords.dec.value
    if 'RA' in closest_blazar_coords:
        ra_im = closest_blazar_coords['RA']
    if 'DEC' in closest_blazar_coords:
        dec_im = closest_blazar_coords['DEC']
        
    str_out = 'ASTROCALIBRATION,INFO,"Trying calibration with data: RA = {}, DEC = {}, DATE-OBS = {}, PIX_SCALE = {}"'
    print(str_out.format(ra_im, dec_im, date_obs, pix_scale))

    # reading 'coord_csv' file
    data_csv = pd.read_csv(coord_csv)
    
    # Astrometric calibration process
    # IF MODEL IS PROVIDED, no calibration process will be done................
    if 'fits' in path_fits:
        astrom_out_fits = path_fits.replace('.fits', 'w.fits')
    else:
        astrom_out_fits = path_fits.replace('.fit', 'w.fits')
    # Composing astrometric calibraton command with 'imwcs' from WCSTools
    # com_str = "imwcs -wve -d {} -r 0 -y 3 -p {} -j {} {} -h {} -c {} -t 10 -o {} {}"
    matchable_fit = 3
    com_str = "imwcs -wve -d {} -r 0 -y {} -p {} -j {} {} -h {} -c {} -t 10 {}"
    if closest_blazar_coords:
        # reducing "matchable" fit field ("-y 2" instead "-y 3")
        matchable_fit = 2
    # cmd = com_str.format(coord_csv, pix_scale, ra_im, dec_im, \
    #     len(data_csv.index), 'tmc', astrom_out_fits, path_fits)
    cmd = com_str.format(coord_csv, matchable_fit, pix_scale, ra_im, dec_im, \
        len(data_csv.index), 'tmc', path_fits)

    print("astrometric calibration command")
    print("-" * 50)
    print(cmd)
    
    result = execute_command(cmd)
    
    if 'fits' in path_fits:
        cal_out = path_fits.replace('.fits', '_imwcs_2mass.log')
    else:
        cal_out = path_fits.replace('.fit', '_imwcs_2mass.log')
    # writing output astrometric calibration log file
    with open(cal_out, "w") as fout:
        fout.write("\n#*********************************************************\n")
        fout.write("\n#********* ASTROMETRIC CALIBRATION  ***********\n")
        fout.write("\n#*********************************************************\n")
        fout.write("---------------- STDOUT ----------------")
        fout.write(result.stdout.decode('utf-8'))
        fout.write("---------------- STDERR ----------------")
        fout.write(result.stderr.decode('utf-8'))
    
    return result


def calibrate(path_fits, sext_conf, closest_iop3_source, min_astrocal_sources=5, border=15):
    
    input_fits = mcFits(path_fits)
    input_head = input_fits.header
    fits_object = input_head['OBJECT']
    dateobs = input_head['DATE-OBS']
    exptime = input_head['EXPTIME']
    
    # detecting sources and getting additional info
    res_get_sources = get_sources(path_fits, sext_conf, back_image=True, segment_image=True)
    if res_get_sources.returncode:
        message = f'ASTROCALIBRATION,ERROR,"Could not get sources for {path_fits}"'
        print(message)
        print(f'STDOUT = {res_get_sources.stdout.decode("UTF-8")}')
        print(f'STDERR = {res_get_sources.stderr.decode("UTF-8")}')
        return 1
    
    # Plotting background and segmentation FITS
    # segmentation image
    if 'fits' in path_fits:
        segmentation_fits = path_fits.replace('.fits', '_segment.fits')
        segment_png = path_fits.replace('.fits', '_segmentation.png')
    else:
        segmentation_fits = path_fits.replace('.fit', '_segment.fit')
        segment_png = path_fits.replace('.fit', '_segmentation.png')
    title = 'Segmentation for OBJECT={}, EXPTIME={}, DATE-OBS={}'
    title = title.format(fits_object, exptime, dateobs)
    plotFits(segmentation_fits, segment_png, title=title)

    # background image
    if 'fits' in path_fits:
        background_fits = path_fits.replace('.fits', '_back.fits')
        back_png = path_fits.replace('.fits', '_background.png')
    else:
        background_fits = path_fits.replace('.fit', '_back.fit')
        back_png = path_fits.replace('.fit', '_background.png')
    title = 'Background for OBJECT={}, EXPTIME={}, DATE-OBS={}'
    title = title.format(fits_object, exptime, dateobs)
    try:
        plotFits(background_fits, back_png, title=title)
    except ValueError:
        print(f'ASTROCALIBRATION,WARNING,"Problems plotting blackground FITS {background_fits}"')

    # masking duplicated sources
    if 'MAPCAT' in path_fits:
        clean_fits = path_fits.replace('.fits', '_clean.fits')
        cat_image = path_fits.replace('.fits', '.cat')
        segment_image = path_fits.replace('.fits', '_segment.fits')
        back_image = path_fits.replace('.fits', '_back.fits')
    # res_masking = mask_duplicated(path_fits, clean_fits, cat_image, segment_image, back_image)
        res_masking = mask_duplicated_simple(path_fits, clean_fits, cat_image, segment_image)
        if res_masking:
            message = 'ASTROCALIBRATION,ERROR,"Error masking duplicated sources in {}"'
            print(message.format(path_fits))
            return 2
        clean_png = clean_fits.replace('.fits', '.png')
    else:
        clean_fits = path_fits.replace('.fit', '_clean.fit')
        cat_image = path_fits.replace('.fit', '.cat')
        segment_image = path_fits.replace('.fit', '_segment.fit')
        back_image = path_fits.replace('.fit', '_back.fit')
        print("Image without duplicated: {}".format(clean_fits))
        clean_png = clean_fits.replace('.fit', '.png')
        shutil.copy(path_fits, clean_fits)
    plotFits(clean_fits, clean_png, title=f'{fits_object} {dateobs} {exptime}s without duplicated')

    ##### ----- Rotating cleaned image 90 degrees counter clockwise ---- #####
    if 'fits' in clean_fits:
        clean_rotated_fits = clean_fits.replace('.fits', '_rotated.fits')
    else:
        clean_rotated_fits = clean_fits.replace('.fit', '_rotated.fit')
    if rotation(clean_fits, clean_rotated_fits):
        message = 'ASTROCALIBRATION,ERROR,"Error rotating FITS {}"'
        print(message.format(clean_fits))
        return 3
    
    clean_rotated_png = clean_png.replace('.png', '_rotated.png')
    title = f'{fits_object} {dateobs} {exptime}s no-duplicated and rotated'
    plotFits(clean_rotated_fits, clean_rotated_png, title=title)

    ##### ----------- Source-extraction for cleaned-rotated FITS ----------- #####
    res_get_sources = get_sources(clean_rotated_fits, sext_conf, verbose=True)
    if res_get_sources.returncode:
        message = f'ASTROCALIBRATION,ERROR,"Could not get sources for {clean_rotated_fits}"'
        print(message)
        print(f'STDOUT = {res_get_sources.stdout.decode("UTF-8")}')
        print(f'STDERR = {res_get_sources.stderr.decode("UTF-8")}')
        return 4
    # Reading text format SExtractor output catalog
    if 'fits' in clean_rotated_fits:
        cat = clean_rotated_fits.replace('.fits', '.cat')
    else:
        cat = clean_rotated_fits.replace('.fit', '.cat')
    data_sext = read_sext_catalog(cat)
    
    ##### -------- Plotting valid detected sources ----- #####
    all_detect_sext_png = clean_rotated_png.replace('.png', '_all_detect_sext.png')
    print('Out PNG ->', all_detect_sext_png)
    title_plot = f"SExtractor sources in {fits_object} {dateobs} {exptime}s"
    plotFits(clean_rotated_fits, all_detect_sext_png, title=title_plot, \
        ref_coords='pixel', color='magenta', \
        coords=(data_sext['X_IMAGE'], data_sext['Y_IMAGE'])) # , \
        # dictParams={'aspect':'auto', 'invert':'True', 'stretch': 'log', 'vmin':1})
        
    if len(data_sext.index) < 3:
        message = 'ASTROCALIBRATION,WARNING,"Low number of sources ({}) in {}".'
        print(message.format(len(data_sext.index), clean_rotated_fits))

    ##### -------- Filtering sources too close to FITS limits ------ #####
    data_sext_filtered = filter_detections(data_sext, path_fits, border)
    
    # Number of filtered detections
    n_sources = len(data_sext_filtered.index)

    # If number of filtered sources is lower than 5, validity conditions are relaxed
    if n_sources < min_astrocal_sources:
        # relaxing SExtractor input parameters
        additional_params = relaxed_detection_params(input_head['EXPTIME'])
        res_get_sources = get_sources(clean_rotated_fits, sext_conf, add_params=additional_params)
        if not res_get_sources.returncode:
            message = f'ASTROCALIBRATION,ERROR,"Could not get sources for {clean_rotated_fits}"'
            print(message)
            print(f'STDOUT = {res_get_sources.stdout.decode("UTF-8")}')
            print(f'STDERR = {res_get_sources.stderr.decode("UTF-8")}')
            return 5
        
        # read output SExtractor catalog
        data_sext = read_sext_catalog(cat)
        
        # Re-plotting detected sources...
        all_detect_sext_png = clean_rotated_png.replace('.png', '_all_detect_sext.png')
        print('Out PNG ->', all_detect_sext_png)
        title_plot = f"SExtractor astrocalibration sources in {clean_rotated_fits}"
        plotFits(clean_rotated_fits, all_detect_sext_png, title=title_plot, \
            ref_coords='pixel', color='magenta', \
            coords=(data_sext['X_IMAGE'], data_sext['Y_IMAGE'])) # , \
            # dictParams={'aspect':'auto', 'invert':'True', 'stretch': 'log', 'vmin':1})

        # Re-filtering sources too close to FITS limits
        data_sext_filtered = filter_detections(data_sext, path_fits, border)
        
    n_sources = len(data_sext_filtered.index)
    print(f"Number of sources after filtering = {n_sources}")

    # Plotting inner sources...
    # INNER SOURCES
    inner_detect_sext_png = clean_rotated_png.replace('.png', '_inner_detect_sext.png')
    print('Out PNG ->', inner_detect_sext_png)
    title_plot = f'SExtractor astrocalibration sources in {clean_rotated_fits}'
    plotFits(clean_rotated_fits, inner_detect_sext_png, title=title_plot, \
        ref_coords='pixel', color='magenta', \
        coords=(data_sext_filtered['X_IMAGE'], data_sext_filtered['Y_IMAGE'])) # , \
        # dictParams={'aspect':'auto', 'invert':'True', 'stretch': 'log', 'vmin':1})

    # get better SExtractor detections for astrocalibration
    df_sorted = get_brilliant_sources(data_sext_filtered, exptime)    
    
    print(f'Number of brightest sources used = {len(df_sorted.index)}')
    #cat_sort = cat.replace('.cat', '_sorted.cat')
    cat_sort_filtered = cat.replace('.cat', '_sorted_filtered.cat')

    # Writing to file (needed for WCSTools astrometric calibration)
    df_sorted.to_csv(cat_sort_filtered, index=False, sep=' ', header=False)

    # Plotting selected brilliant sources
    out_detect_sext_png = clean_rotated_png.replace('.png', '_detect_sext.png')
    print('Out PNG ->', out_detect_sext_png)
    title_plot = f'Valid SExtractor sources for astrocalibration in {fits_object} {dateobs} {exptime}s'

    plotFits(clean_rotated_fits, out_detect_sext_png, title=title_plot, \
        coords=(df_sorted['X_IMAGE'], df_sorted['Y_IMAGE']), \
        ref_coords='pixel', color='green') #, dictParams={'aspect':'auto', 'invert':'True'})
    # ----- First astrometric calibration try ----- #
    if 'fits' in clean_rotated_fits:
        shutil.copy(clean_rotated_fits, clean_rotated_fits.replace('.fits', '_copy.fits'))
        clean_rotated_fits_copy=clean_rotated_fits.replace('.fits', '_copy.fits')
    else:
        shutil.copy(clean_rotated_fits, clean_rotated_fits.replace('.fit', '_copy.fit'))
        clean_rotated_fits_copy=clean_rotated_fits.replace('.fit', '_copy.fit')
    res_astrocal = astrocal(clean_rotated_fits, cat_sort_filtered)
    if res_astrocal.returncode:
        message = 'ASTROCALIBRATION,ERROR,"Failed astrometric calibration for FITS {}"'
        print(message.format(clean_rotated_fits))
        print(f'STDOUT = {res_astrocal.stdout}')
        print(f'STDERR = {res_astrocal.stderr}')
        return 6

    if 'fits' in clean_rotated_fits:
        astrom_out_fits = clean_rotated_fits.replace('.fits', 'w.fits')
    else:
        astrom_out_fits = clean_rotated_fits.replace('.fit', 'w.fits')
    # Not good calibration process if not enough sources
    # were used
    wcsmatch = 0
    try:
        astro_fits = mcFits(astrom_out_fits)
        astro_header = astro_fits.header
        wcsmatch = astro_header['WCSMATCH']
    except:
        message = f'ASTROCALIBRATION,WARNING,"First astrometric calibration failed. No *w.fits file found ({astrom_out_fits})"'
        print(message) 
    # ----- Second astrometric calibration try ----- #
    if wcsmatch < (len(df_sorted.index) / 2): # bad calibration
        print("\nSecond astrometric calibration try")
        print("\tUsing new central FITS coordinates (Closest MAPCAT source)")
        closest_blazar_coords = {'RA': closest_iop3_source['ra2000_mc_deg'], \
            'DEC': closest_iop3_source['dec2000_mc_deg']}
        #if 'fits' in clean_rotated_fits:
        #    shutil.copy(clean_rotated_fits, clean_rotated_fits.replace('.fits', '_copy.fits'))
        #    clean_rotated_fits_copy=clean_rotated_fits.replace('.fits', '_copy.fits')
        #else:
        #    shutil.copy(clean_rotated_fits, clean_rotated_fits.replace('.fit', '_copy.fit'))
        #    clean_rotated_fits_copy=clean_rotated_fits.replace('.fit', '_copy.fit')
        res_astrocal = astrocal(clean_rotated_fits_copy, cat_sort_filtered, closest_blazar_coords=closest_blazar_coords)
        if 'fits' in clean_rotated_fits_copy:
            shutil.copy(clean_rotated_fits_copy.replace('.fits', 'w.fits'), astrom_out_fits)
        else:
            shutil.copy(clean_rotated_fits_copy.replace('.fit', 'w.fits'), astrom_out_fits)
        if res_astrocal.returncode:
            return 7

    return 0

def get_best_astrocal(calibration_dir):
    """"""
    cal_fits = glob.glob(os.path.join(calibration_dir, '*/*final.fit*'))
    print(f'Calibrations done = {len(cal_fits)}')
    cal_results = defaultdict(list)
    for cf in cal_fits:
        astro_fits = mcFits(cf)
        cal_results['PATH'] = cf
        for k in ['WCSMATCH', 'EXPTIME']:
            cal_results[k].append(astro_fits.header[k])
    # transform to pandas DataFrame
    df_cal = pd.DataFrame(cal_results)
    
    # getting best calibration according to maching sources number
    df_cal_best = df_cal.sort_values(by=['EXPTIME', 'WCSMATCH'], ascending=[False, False]).head(1)
    
    return df_cal_best

def center_coordinates(pixel_obj_coords, deg_astro_coords, pixel_fits_size, arcs_per_pixel=0.53):
    """Return central FITS coordinates (ra, dec) in degrees given 
    pixel_obj_coords and deg_astro_coords.
    
    Args:
        pixel_obj_coords (tuple): 
            Star coordinates in pixels (x, y).
        deg_astro_coords (tuple): 
            IOP3 closest source sky coordinates in degrees (ra, dec).
        pixel_fits_size (tuple): 
            FITS image dimensions in pixels (width, height).
        arcs_per_pixel (float):
            arcsecs covered per pixel in FITS.
        
    Return:
        deg_central_coords (dict): 
            Estimated central FITS coordinates in degrees. Keywords ['RA', 'DEC'].
    """
    deg_central_coords = {}
    x_obj, y_obj = pixel_obj_coords
    size_x, size_y = pixel_fits_size
    ra, dec = deg_astro_coords
    
    deg_central_coords['RA'] = (x_obj - (size_x / 2)) * arcs_per_pixel / 3600.0 + ra
    deg_central_coords['DEC'] = ((size_y / 2) - y_obj) * arcs_per_pixel / 3600.0 + dec
    
    return deg_central_coords
    
def calibrate_star(path_fits, sext_conf, closest_iop3_source, min_astrocal_sources=5, border=15):
    """"""
    input_fits = mcFits(path_fits)
    input_head = input_fits.header
    fits_object = input_head['OBJECT']
    dateobs = input_head['DATE-OBS']
    exptime = input_head['EXPTIME']
    
    # detecting sources and getting additional info
    res_get_sources = get_sources(path_fits, sext_conf, back_image=True, segment_image=True)
    if res_get_sources.returncode:
        message = f'ASTROCALIBRATION,ERROR,"Could not get sources for {path_fits}"'
        print(message)
        print(f'STDOUT = {res_get_sources.stdout.decode("UTF-8")}')
        print(f'STDERR = {res_get_sources.stderr.decode("UTF-8")}')
        return 1
    
    # Plotting background and segmentation FITS
    # segmentation image
    if 'fits' in path_fits:
        segmentation_fits = path_fits.replace('.fits', '_segment.fits')
        segment_png = path_fits.replace('.fits', '_segmentation.png')
    else:
        segmentation_fits = path_fits.replace('.fit', '_segment.fit')
        segment_png = path_fits.replace('.fit', '_segmentation.png')
    title = 'Segmentation for OBJECT={}, EXPTIME={}, DATE-OBS={}'
    title = title.format(fits_object, exptime, dateobs)
    plotFits(segmentation_fits, segment_png, title=title)

    # background image
    if 'fits' in path_fits:
        background_fits = path_fits.replace('.fits', '_back.fits')
        back_png = path_fits.replace('.fits', '_background.png')
    else:
        background_fits = path_fits.replace('.fit', '_back.fit')
        back_png = path_fits.replace('.fit', '_background.png')
    title = 'Background for OBJECT={}, EXPTIME={}, DATE-OBS={}'
    title = title.format(fits_object, exptime, dateobs)
    try:
        plotFits(background_fits, back_png, title=title)
    except ValueError:
        print(f'ASTROCALIBRATION,WARNING,"Problems plotting background FITS {background_fits}"')


    # masking duplicated sources
    if 'MAPCAT' in path_fits:
        clean_fits = path_fits.replace('.fits', '_clean.fits')
        cat_image = path_fits.replace('.fits', '.cat')
        segment_image = path_fits.replace('.fits', '_segment.fits')
        back_image = path_fits.replace('.fits', '_back.fits')
    # res_masking = mask_duplicated(path_fits, clean_fits, cat_image, segment_image, back_image)
        res_masking = mask_duplicated_simple(path_fits, clean_fits, cat_image, segment_image)
        if res_masking:
            message = 'ASTROCALIBRATION,ERROR,"Error masking duplicated sources in {}"'
            print(message.format(path_fits))
            return 2
        clean_png = clean_fits.replace('.fits', '.png')
    else:
        clean_fits = path_fits.replace('.fit', '_clean.fit')
        cat_image = path_fits.replace('.fit', '.cat')
        segment_image = path_fits.replace('.fit', '_segment.fit')
        back_image = path_fits.replace('.fit', '_back.fit')
        print("Image without duplicated: {}".format(clean_fits))
        clean_png = clean_fits.replace('.fit', '.png')
        shutil.copy(path_fits, clean_fits)
    plotFits(clean_fits, clean_png, title=f'{fits_object} {dateobs} {exptime}s without duplicated')

    ##### ----- Rotating cleaned image 90 degrees counter clockwise ---- #####
    if 'fits' in clean_fits:
        clean_rotated_fits = clean_fits.replace('.fits', '_rotated.fits')
    else:
        clean_rotated_fits = clean_fits.replace('.fit', '_rotated.fit')
    if rotation(clean_fits, clean_rotated_fits):
        message = 'ASTROCALIBRATION,ERROR,"Error rotating FITS {}"'
        print(message.format(clean_fits))
        return 3
    
    clean_rotated_png = clean_png.replace('.png', '_rotated.png')
    title = f'{fits_object} {dateobs} {exptime}s no-duplicated and rotated'
    plotFits(clean_rotated_fits, clean_rotated_png, title=title)

    ##### ----------- Source-extraction for cleaned-rotated FITS ----------- #####
    res_get_sources = get_sources(clean_rotated_fits, sext_conf, verbose=True)
    if res_get_sources.returncode:
        message = f'ASTROCALIBRATION,ERROR,"Could not get sources for {clean_rotated_fits}"'
        print(message)
        # print(f'STDOUT = {res_get_sources.stdout.decode("UTF-8")}')
        # print(f'STDERR = {res_get_sources.stderr.decode("UTF-8")}')
        return 4
    # Reading text format SExtractor output catalog
    if 'fits' in clean_rotated_fits:    
        cat = clean_rotated_fits.replace('.fits', '.cat')
    else:
        cat = clean_rotated_fits.replace('.fit', '.cat')
    data_sext = read_sext_catalog(cat)
    
    ##### -------- Plotting valid detected sources ----- #####
    all_detect_sext_png = clean_rotated_png.replace('.png', '_all_detect_sext.png')
    print('Out PNG ->', all_detect_sext_png)
    title_plot = f"SExtractor sources in {fits_object} {dateobs} {exptime}"
    plotFits(clean_rotated_fits, all_detect_sext_png, title=title_plot, \
        ref_coords='pixel', color='magenta', \
        coords=(data_sext['X_IMAGE'], data_sext['Y_IMAGE'])) # , \
        # dictParams={'aspect':'auto', 'invert':'True', 'stretch': 'log', 'vmin':1})
    

    if len(data_sext.index) < 3:
        message = 'ASTROCALIBRATION,WARNING,Low number of sources ({}) in "{}"'
        print(message.format(len(data_sext.index), clean_rotated_fits))

    ##### -------- Filtering sources too close to FITS limits ------ #####
    data_sext_filtered = filter_detections(data_sext, path_fits, border)
    
    # Number of filtered detections
    n_sources = len(data_sext_filtered.index)

    # getting most brilliant source. This is our STAR
    dt_star = data_sext_filtered.sort_values(['FLUX_MAX','FLUX_AUTO'], ascending=False).head(1)
    x_star = dt_star['X_IMAGE'].values[0]
    y_star = dt_star['Y_IMAGE'].values[0]
    
    plotFits(clean_rotated_fits, 'selected_star.png', title='SELECTED STAR', \
        ref_coords='pixel', color='magenta', \
        coords=(dt_star['X_IMAGE'].values, dt_star['Y_IMAGE'].values))
    
    # get best calibration of the night
    calibration_dir = "/".join(path_fits.split("/")[:-2])
    df_best = get_best_astrocal(calibration_dir)
    print('Best calibration')
    for k in ['PATH', 'WCSMATCH', 'EXPTIME']:
        print(f'{k} = {df_best[k].iloc[0]}')
    
    best_fits = mcFits(df_best['PATH'].iloc[0])
    
    # set new/updated header keywords
    new_keys = OrderedDict()
    
    # new_keys['CRVAL1'] = closest_iop3_source['ra2000_mc_deg']
    # # Change DEC coordinate. I add -50 pixels to closest IOP3 target
    # new_keys['CRVAL2'] = closest_iop3_source['dec2000_mc_deg'] - (50 * best_fits.header['SECPIX2'] / 3600.0) 
    
    # New procedure
    center_coords = center_coordinates((x_star, y_star), \
        (closest_iop3_source['ra2000_mc_deg'], closest_iop3_source['dec2000_mc_deg']), \
        (input_head['NAXIS1'], input_head['NAXIS2']), arcs_per_pixel=0.53)
    new_coords = SkyCoord(center_coords['RA'], center_coords['DEC'], unit="deg")
    new_keys['CRVAL1'] = center_coords['RA']
    new_keys['CRVAL2'] = center_coords['DEC']
    new_keys['EPOCH'] = 2000
    new_keys['CRPIX1'] = input_head['NAXIS1'] / 2
    new_keys['CRPIX2'] = input_head['NAXIS1'] / 2
    new_keys['CDELT1'] = best_fits.header['CDELT1']
    new_keys['CDELT2'] = best_fits.header['CDELT2']
    new_keys['CTYPE1'] = 'RA---TAN'
    new_keys['CTYPE2'] = 'DEC--TAN'
    new_keys['CD1_1'] = best_fits.header['CD1_1']
    new_keys['CD1_2'] = best_fits.header['CD1_2']
    new_keys['CD2_1'] = best_fits.header['CD2_1']
    new_keys['CD2_2'] = best_fits.header['CD2_2']
    new_keys['WCSRFCAT'] = 'tmc'
    new_keys['WCSIMCAT'] = ''
    new_keys['WCSMATCH'] = 1
    new_keys['WCSNREF'] = 1
    new_keys['WCSTOL'] = 0
    # toks = closest_iop3_source['ra2000_mc'].strip().split(' ')
    # ra_star = '{}:{}:{}'.format(toks[0], toks[1], toks[2][6])
    # new_keys['RA'] = ra_star
    # toks = closest_iop3_source['dec2000_mc'].strip().split(' ')
    # dec_star = '{}:{}:{}'.format(toks[0], toks[1], toks[2][5])
    # new_keys['DEC'] = dec_star
    
    # new_keys['RA'] = closest_iop3_source['ra2000_mc'].replace(' ', ':')[:-1]
    # new_keys['DEC'] = closest_iop3_source['dec2000_mc'].replace(' ', ':')[:-1]
    
    t_ra = new_coords.ra.hms
    t_dec = new_coords.dec.dms
    new_keys['RA'] = f'{int(t_ra.h)} {int(t_ra.m)} {t_ra.s:.4f}'
    op = '+'
    if t_dec.d < 0:
        op = ''
    new_keys['DEC'] = f'{op}{int(t_dec.d)} {int(t_dec.m)} {t_dec.s:.5f}'
    new_keys['EQUINOX'] = 2000
    new_keys['CROTA1'] = best_fits.header['CROTA1']
    new_keys['CROTA2'] = best_fits.header['CROTA2']
    new_keys['SECPIX1'] = best_fits.header['SECPIX1']
    new_keys['SECPIX2'] = best_fits.header['SECPIX2']
    new_keys['WCSSEP'] = 0
    new_keys['IMWCS'] = 'None'
    
    print(new_keys)
    # using clean_rotated_fits as FITS base    
    hdul = fits.open(clean_rotated_fits)
    header = hdul[0].header
    # delete some keywords
    for k in ['PC001001', 'PC001002', 'PC002001', 'PC002002']:
        header.remove(k, ignore_missing=True)
    
    # special keywords
    if 'RA' in header:
        header.rename_keyword('RA', 'WRA')
        header.rename_keyword('DEC', 'WDEC')
    else:
        header.rename_keyword('OBJCTRA', 'WRA')
        header.rename_keyword('OBJCTDEC', 'WDEC')
    # header['BLANK'] = 32768

    # calibration keywords
    for k, v in new_keys.items():
        header.remove(k, ignore_missing=True)
        header.append(card=(k, v, ''), end=True) # set
        
    # Finally, saving/updating astrocalibreated file with star coordinates and astrometric keywords from best fit
    if 'fits' in clean_rotated_fits:
        astrom_out_fits = clean_rotated_fits.replace('.fits', 'w.fits')
    else:
        astrom_out_fits = clean_rotated_fits.replace('.fit', 'w.fits')
    hdul[0].header = header
    hdul.writeto(astrom_out_fits, output_verify='fix', overwrite=True)

    return 0

def read_blazar_file(blazar_csv):

    df_mapcat = pd.read_csv(blazar_csv, comment='#')
    # (df_mapcat.info())
    # getting coordinates in degrees unit
    c  = []
    for ra, dec in zip(df_mapcat['ra2000_mc'], df_mapcat['dec2000_mc']):
        c.append("{} {}".format(ra, dec))

    mapcat_coords = SkyCoord(c, frame=FK5, unit=(u.hourangle, u.deg), \
    obstime="J2000")
    df_mapcat['ra2000_mc_deg'] = mapcat_coords.ra.deg
    df_mapcat['dec2000_mc_deg'] = mapcat_coords.dec.deg

    return df_mapcat

def closest_blazar(blazar_data, path_fits):
    #Getting telescope type
    tel_type=re.findall('(\w+/)', 
                        path_fits)[len(re.findall('(\w+/)', 
                                                  path_fits))-2][:-1]
    # Getting header informacion
    i_fits = mcFits(path_fits)
    input_head = i_fits.header
    # Central FITS coordinates
    if tel_type=='MAPCAT':
        icoords = "{} {}".format(input_head['RA'], input_head['DEC'])
        input_coords = SkyCoord(icoords, frame=FK5, unit=(u.deg, u.deg), \
                                    obstime="J2000")
    else:
        if 'OBJCTRA' in input_head:
            icoords = "{} {}".format(input_head['OBJCTRA'], input_head['OBJCTDEC'])
        else:
            print('Object coordinates are missing from header of {}'.format(path_fits) )
            icoords = "0 0"
        input_coords = SkyCoord(icoords, frame=FK5, unit=(u.hourangle, u.deg), \
                                    obstime="J2000")
    # Blazars subset...
    df_blazars = blazar_data[blazar_data['IAU_name_mc'].notna()]
    c  = []
    for ra, dec in zip(df_blazars['ra2000_mc'], df_blazars['dec2000_mc']):
        c.append("{} {}".format(ra, dec))
    blazar_coords = SkyCoord(c, frame=FK5, unit=(u.hourangle, u.deg), \
    obstime="J2000")
    # Closest MAPCAT source to FITS central coordinates
    # Distance between this center FITS and MAPCAT targets (in degrees)
    distances = input_coords.separation(blazar_coords)
    
    # Closest source in complete set...
    i_min = distances.deg.argmin()
    
    return df_blazars.iloc[i_min], distances.deg[i_min]

def query_external_catalog(path_fits, catalog={}):
    # Getting sources from web catalogs
    astro_fits = mcFits(path_fits)
    astro_header = astro_fits.header
    
    ra = astro_header['CRVAL1']
    dec = astro_header['CRVAL2']
    center_coords = coord.SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')

    # Plotting web sources over our calibrated FITS
    for cat_name, cat_code in catalog.items():
        result = Vizier.query_region(center_coords, width="10m", catalog=cat_code)
        print('Web catalog obtained')
        pprint.pprint(result)
        
        wcat = None
        try:
            wcat = result[cat_code]
            # print(f'wcat = {wcat}')
            
            if cat_name == 'SDSS-R12':
                coords = (wcat['RA_ICRS'], wcat['DE_ICRS'])
            else:
                coords = (wcat['RAJ2000'], wcat['DEJ2000'])    
            # print(f'coords = {coords}')
            
            cat_out_png = path_fits.replace('.fits', f'_{cat_name}.png')
            print(f"Plotting data from {cat_code} catalog")
            print(f'outplot = {cat_out_png}')
            plotFits(path_fits, cat_out_png, \
                title=f'{cat_name} sources in {path_fits}', \
                coords=coords, astroCal=True, color='green', \
                dictParams={'aspect':'auto', 'invert':'True'})
        except TypeError:
            return 1
    
    return 0


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
    parser.add_argument("--max_dist_deg",
       action="store",
       dest="max_dist_deg",
       type=float,
       default=0.5,
       help="Max distance beteen FITS center and closest IOP3 source [default: %(default)s].")
    # parser.add_argument("--exptime_median",
    #    action="store",
    #    dest="exptime_median",
    #    type=float,
    #    default=2,
    #    help="Max exptime for applying median data FITS correction in order to minimize sky noise [default: %(default)s].")
    parser.add_argument("--fits_astrocal",
       action="store",
       dest="fits_astrocal",
       type=str,
       default=None,
       help="FITS astrometrically calibrated that will be use a model. [default: %(default)s].")
    parser.add_argument("--is_star",
       action="store",
       dest="is_star",
       type=bool,
       default=False,
       help="If True, star astrometric calibration process is applied. [default: %(default)s].")
    parser.add_argument("--overwrite",
       action="store",
       dest="overwrite",
       type=str,
       default='False',
       help="If True, previous astrometric calibration is ignored and done again. [default: %(default)s].")
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
        str_err = 'ASTROCALIBRATION,ERROR,"Config dir {} not available."'
        print(str_err.format(args.config_dir))
        return 1

    if not os.path.isdir(args.output_dir):
        try:
            os.makedirs(args.output_dir)
        except IOError:
            str_err = 'ASTROCALIBRATION,ERROR,"Could not create output directory {}"'
            print(str_err.format(args.input_dir))
            return 2

    if not os.path.exists(args.input_fits):
        str_err = 'ASTROCALIBRATION,ERROR,"Input FITS file {} not available"'
        print(str_err.format(args.input_dir))
        return 3

    input_fits = os.path.abspath(args.input_fits)
    copy_input_fits = os.path.join(args.output_dir, os.path.split(input_fits)[1])
    if 'fits' in input_fits:
        final_fits = copy_input_fits.replace('.fits', '_final.fits')
    else:
        final_fits = copy_input_fits.replace('.fit', '_final.fit')

    if os.path.exists(final_fits) and not args.overwrite:
        print(f'ASTROCALIBRATION,INFO,"Calibration done before for {final_fits}"')
        return -1
    
    # Using input path for getting observation night date... (Is this the best way or should I read FITS header?)
    dt_run = re.findall('/(\d{6})/', input_fits)[0]
    date_run = f'20{dt_run[:2]}-{dt_run[2:4]}-{dt_run[-2:]}'
    if 'fits' in input_fits:
        fits_name = os.path.split(input_fits)[1].replace('.fits', '')
    else:
        fits_name = os.path.split(input_fits)[1].replace('.fit', '')
    # Setting current working directory
    os.chdir(args.output_dir)
    print(f"\nWorking directory set to '{args.output_dir}'\n")

    # Getting header informacion
    i_fits = mcFits(input_fits)
    input_head = i_fits.header
    
    obj = input_head['OBJECT']
    exptime = input_head['EXPTIME']

    text = 'ASTROCALIBRATION,INFO,"OBJECT and Polarization angle = {}, EXPTIME = {} s."'
    print(text.format(obj, exptime))

    # Reading blazar CSV file
    blazar_path = os.path.join(args.config_dir, 'blazar_photo_calib_last.csv')
    blazar_data = read_blazar_file(blazar_path)
    
    # getting closest blazar to input FITS centre
    closest_iop3_source, min_deg_dist = closest_blazar(blazar_data, input_fits)
    print('Closest IOP3 source')
    print('------------')
    print(closest_iop3_source)
    print('------------')
    
    print(f'Minimum distance to IOP3 source (deg)= {min_deg_dist}')
    if min_deg_dist > args.max_dist_deg: # distance in degrees
        message = 'ASTROCALIBRATION,ERROR,"Closest IOP3 source too far ({} deg)."'
        print(message.format(min_deg_dist))
        return 4

    mc_aper = closest_iop3_source['aper_mc']
    print(f'ASTROCALIBRATION,INFO,"SExtractor aperture = {mc_aper} pixels"')

    # Input image PNG
    input_fits_png = f'{fits_name}.png'
    print('Original FITS PNG= {}'.format(input_fits_png))
    title = "OBJECT = {}, EXPTIME = {} s, DATE-OBS = {}"
    title = title.format(input_head['OBJECT'], input_head['EXPTIME'], input_head['DATE-OBS'])
    print(text.format(input_head['OBJECT'], input_head['EXPTIME']))
    plotFits(input_fits, input_fits_png, title=title)

    copy_input_fits = os.path.join(args.output_dir, os.path.split(input_fits)[1])
    shutil.copy(input_fits, copy_input_fits)
    # If EXPTIME <= 2 seconds, median filter to minimize pixel noise is applied to FITS data
    # if exptime <= args.exptime_median:
    if args.is_star:
        print(f"Working on {copy_input_fits}")
        if medianFits(copy_input_fits, copy_input_fits, size=5):
            str_err = f'ASTROCALIBRATION,ERROR,"Median filter could not be applied to fits {working_input_fits}"'
            print(str_err)
            return 5

        # Input image PNG
        if 'fits' in copy_input_fits:
            copy_input_fits_png = copy_input_fits.replace('.fits', '.png')
        else:
            copy_input_fits_png = copy_input_fits.replace('.fit', '.png')
        plotFits(copy_input_fits, copy_input_fits_png, title=f"Median filtered FITS: {title}") 
        # , dictParams={'aspect':'auto', 'stretch': 'log', 'vmin': 1})

    # file names (Please: pay attention to use of relative paths given by 'fits_name')
    if 'fits' in input_fits:
        clean_fits = f'{fits_name}_clean.fits'
        clean_rotated_fits = clean_fits.replace('.fits', '_rotated.fits')
        astrom_out_fits = clean_rotated_fits.replace('.fits', 'w.fits')
    else:
        clean_fits = f'{fits_name}_clean.fit'
        clean_rotated_fits = clean_fits.replace('.fit', '_rotated.fit')
        astrom_out_fits = clean_rotated_fits.replace('.fit', 'w.fits')
        
    clean_png = f'{fits_name}_clean.png'
    clean_rotated_png = clean_png.replace('.png', '_rotated.png')
    inner_detect_sext_png = clean_rotated_png.replace('.png', '_inner_detect_sext.png')
    out_detect_sext_png = clean_rotated_png.replace('.png', '_detect_sext.png')
    
    sext_conf = os.path.join(args.config_dir, 'daofind.sex')
    if not args.is_star: # blazar object
        if args.fits_astrocal is None:
            res_cal = calibrate(copy_input_fits, sext_conf, closest_iop3_source, min_astrocal_sources=5, border=args.border_image)
            if res_cal:
                print(f'ASTROCALIBRATION,ERROR,"Could not calibrate astrometrically FITS {clean_rotated_fits}"')
                return 6
        else:
            # Get astrometric calibration keywords and values
            fits_model = mcFits(args.fits_astrocal)
            model_astrovalues = fits_model.get_astroheader()
            
            # creating new astrometric calibrated FITS
            print(f'clean_rotated_fits = {os.path.abspath(clean_rotated_fits)}')
            print(f'astrom_out_fits = {os.path.abspath(astrom_out_fits)}')
            shutil.copy(clean_rotated_fits, astrom_out_fits)
            
            # Editing for adding astrometric pairs of keyword-value to output astrometric calibrated fits
            with fits.open(astrom_out_fits, 'update') as fout:
                hdr = fout[0].header
                for k, v in model_astrovalues.items():
                    if k in hdr:
                        hdr[k] = v
                    else:
                        hdr.append((k, v, ''), end=True)
    else: 
        # Working with star
        res_cal = calibrate_star(copy_input_fits, sext_conf, closest_iop3_source, min_astrocal_sources=5, border=args.border_image)  
        if res_cal:
            print(f'ASTROCALIBRATION,ERROR,"Could not calibrate astrometrically FITS {clean_rotated_fits}"')
            return 7
    
    # Checking astrometric calibrated ouput FITS
    if not os.path.exists(astrom_out_fits):
        fits_cal = os.path.abspath(astrom_out_fits)
        print(f'ASTROCALIBRATION,ERROR,"Calibrated FITS {fits_cal} not found."')
        return 8

    # reading astrocalibrated FITS
    astro_fits = mcFits(astrom_out_fits)
    astro_header = astro_fits.header
    
    # print(f'astro_header clean rotated = {astro_header}')

    print('*' * 50)
    message = 'ASTROCALIBRATION,INFO,"Number of sources used in calibration: {} (of {})"'
    print(message.format(astro_header['WCSMATCH'], astro_header['WCSNREF']))
    print('*' * 50)
    
    # Query to external catalogs
    # Getting sources from web catalogs
    catalogs = {'2MASS': 'II/246/out', 'NOMAD': 'I/297/out', \
        'USNO-A2': 'I/252/out', 'SDSS-R12': 'V/147/sdss12'}
    
    cat_out_pngs = {}
    for name_cat, code_cat in catalogs.items():        
        query_cat = {f"{name_cat}": code_cat}
        print(query_cat)
        if not query_external_catalog(astrom_out_fits, query_cat):
            cat_out_pngs[name_cat] = clean_rotated_fits.replace('.fits', f'_{name_cat}.png')
        else:
            print(f'ASTROCALIBRATON,WARNING,"No data available for {name_cat} external catalog."')

    # original FITS rotation
    tmp_rotation = 'tmp_rotation.fits'
    rotation(input_fits, tmp_rotation)
    
    # Composing final FITS
    if 'fits' in copy_input_fits:
        final_fits = copy_input_fits.replace('.fits', '_final.fits')
    else:
        final_fits = copy_input_fits.replace('.fit', '_final.fit')
    tmp_fits = mcFits(tmp_rotation)
    hdul_final = fits.PrimaryHDU(data=tmp_fits.data, header=astro_header)
    hdul_final.writeto(final_fits, output_verify='fix', overwrite=True)
   
    os.remove(tmp_rotation) 
   
    header_cal = mcFits(final_fits).header
    # print('-' * 100)
    # print(f'Final FITS header = {header_cal}')
    # print('-' * 100)
    
    # Plotting final calibrated FITS
    if 'fits' in final_fits:
        final_png = final_fits.replace('.fits', '_final.png')
    else:
        final_png = final_fits.replace('.fit', '_final.png')
    title = '{}, {}, {} rotated astrocalib'
    title = title.format(header_cal['DATE-OBS'], header_cal['OBJECT'], \
        header_cal['EXPTIME'])
    plotFits(final_fits, final_png, title=title)

    # Parameters to store...
    i_fits = mcFits(final_fits)
    astro_header = i_fits.header
    # Getting useful info about calibrated fits
    some_calib_keywords = ['SOFT', 'PROCDATE', 'SOFTDET', 'MAX',  'MIN', \
        'MEAN', 'STD', 'MED', 'RA', 'DEC', 'CRVAL1', 'CRVAL2', 'EPOCH', \
        'CRPIX1', 'CRPIX2', 'SECPIX1', 'SECPIX2', 'CDELT1', 'CDELT2', 'CTYPE1', 'CTYPE2', \
        'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'WCSRFCAT', 'WCSIMCAT', 'WCSMATCH', \
        'WCSNREF', 'WCSTOL', 'CROTA1', 'CROTA2', 'WCSSEP', 'IMWCS']
    
    cal_data = {}
    
    cal_data['PATH'] = [final_fits]
    cal_data['RUN_DATE'] = [date_run]    
    
    for key in some_calib_keywords:
        if key in astro_header:
            cal_data[key] = astro_header[key]
    
    cal_data['CLEAN_ROT_PNG'] = [clean_rotated_png]
    cal_data['INNER_SEXTDET_PNG'] = [inner_detect_sext_png]
    cal_data['OUT_SEXTDET_PNG'] = [out_detect_sext_png]
    for k, v in cat_out_pngs.items():
        cal_data[f"CALIB_{k}_PNG"] = v
    cal_data['FINAL_PNG'] = final_png
    df = pd.DataFrame(cal_data)
    if 'fits' in final_fits:
        csv_out = final_fits.replace('.fits', '_astrocal_process_info.csv')
    else:
        csv_out = final_fits.replace('.fit', '_astrocal_process_info.csv')
    df.to_csv(csv_out, index=False)
    

    return 0

# -------------------------------------
if __name__ == '__main__':
    print(main())
