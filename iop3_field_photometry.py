from dataclasses import replace
import os
import argparse
import subprocess
import re
from collections import defaultdict
import glob
from typing import DefaultDict

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

import iop3_photometric_calibration as photo
import iop3_polarimetry as pol

date_run='211106'

conf='/home/users/dreg/misabelber/GitHub/iop3/conf/'
path="/home/users/dreg/misabelber/GitHub/data/calibration/T090/"+date_run
files=glob.glob(path+"/**/*_final.fit")
data_res=pd.DataFrame()


for input_fits in files:
    print(input_fits)
    i_fits = mcFits(input_fits)
    astro_header = i_fits.header

    blazar_path = conf+'/unpolarized_field_stars.csv'
    df_mapcat = photo.read_blazar_file(blazar_path)
    nearest_blazar, min_dist_deg = photo.closest_blazar(astro_header['CRVAL1'], astro_header['CRVAL2'], df_mapcat)


    print(f'Distance = {min_dist_deg}')
    if min_dist_deg > 0.5: # distance in degrees
        print('!' * 100)
        print('ERROR: Not enough close blazar or HD star found (distance <= 0.5 deg)')
        print('!' * 100)
        
    # closest blazar info
    alternative_ra = nearest_blazar['ra2000_mc_deg'].values[0]
    alternative_dec = nearest_blazar['dec2000_mc_deg'].values[0]    

    print(f"Blazar {nearest_blazar['IAU_name_mc'].values[0]} is the closest detection ")
    print(f" at coordinates ({alternative_ra}, {alternative_dec}")
    print('(Rmag, Rmagerr) = ({}, {})'.format(nearest_blazar['Rmag_mc'].values[0], \
                                                  nearest_blazar['Rmagerr_mc'].values[0]))
    
    mc_aper = nearest_blazar['aper_mc'].values[0]
    print(f'aperture = {mc_aper} pixels')

    cat='test.cat'
    sex_conf = os.path.join(conf, 'sex.conf')
    photo.detect_sources(input_fits, cat_out=cat, sext_conf=sex_conf, photo_aper=mc_aper)
    
    # RA,DEC limits...
    sky_limits = photo.get_radec_limits(input_fits)

    # Loading FITS_LDAC format SExtractor catalog
    sext = fits.open(cat)
    data = sext[2].data
    
    print ("Number of detections = {}".format(data['ALPHA_J2000'].size))
    intervals = "(ra_min, ra_max, dec_min, dec_max) = ({}, {}, {}, {})"
    print(intervals.format(sky_limits['ra_min'], sky_limits['ra_max'], \
                                   sky_limits['dec_min'], sky_limits['dec_max']))
        
    df_mc = df_mapcat[df_mapcat['ra2000_mc_deg'] > sky_limits['ra_min']]
    df_mc = df_mc[df_mc['ra2000_mc_deg'] < sky_limits['ra_max']]
    df_mc = df_mc[df_mc['dec2000_mc_deg'] > sky_limits['dec_min']]
    df_mc = df_mc[df_mc['dec2000_mc_deg'] < sky_limits['dec_max']]
    
    mag_zeropoint = astro_header['MAGZPT']
    pixscale = 0.387
    fwhm_arcs = float(astro_header['FWHM']) * pixscale
        
    com_str = "source-extractor -c {} -CATALOG_NAME {} -PIXEL_SCALE {} -SEEING_FWHM {} {}"
    com = com_str.format(sex_conf, cat, pixscale, fwhm_arcs, input_fits)
    com += f" -PHOT_APERTURES {mc_aper}"
    com += f" -MAG_ZEROPOINT {mag_zeropoint}"
    additional_params = photo.default_detection_params(astro_header['EXPTIME'])
    
    print(com)
    subprocess.Popen(com, shell=True).wait()
    
    sext = fits.open(cat)
    data = sext[2].data
        
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

    keywords = ['ALPHA_J2000', 'DELTA_J2000', 'FWHM_IMAGE', 'CLASS_STAR', \
                    'FLAGS', 'ELLIPTICITY', 'FLUX_MAX', 'FLUX_APER', 'FLUXERR_APER', \
                    'FLUX_ISO', 'FLUXERR_ISO', 'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_APER', \
                    'MAGERR_APER', 'MAG_ISO', 'MAGERR_ISO', 'MAG_AUTO', 'MAGERR_AUTO']
    
    source_problem = df_mc['Rmag_mc'].values < 0
    
    nsources=df_mc.shape[0]

    pair_params = photo.defaultdict(list)
    pair_params['ID-MC'] = np.repeat(df_mc['id_mc'].values,2)
    pair_params['ID-BLAZAR-MC'] = np.repeat(df_mc['id_blazar_mc'].values,2)
    pair_params['TYPE'] = [item for sublist in [['O', 'E']] * nsources for item in sublist]

    if 'INSPOROT' in astro_header:
        angle = float(astro_header['INSPOROT'])
    else:
        if astro_header['FILTER']=='R':
            angle = -999.0
        else:
            angle = float(astro_header['FILTER'].replace('R',''))

    pair_params['ANGLE'] = [round(angle, ndigits=1)] * 2 * nsources
    pair_params['OBJECT'] = np.repeat(df_mc['IAU_name_mc'].values,2)
    if 'MJD-OBS' in astro_header:
        pair_params['MJD-OBS'] = [astro_header['MJD-OBS']] * 2 *nsources
    else:
        pair_params['MJD-OBS'] = [astro_header['JD']] * 2 * nsources
        pair_params['DATE-OBS'] = [''] * 2 * nsources
    if 'DATE-OBS' in astro_header:
        pair_params['DATE-OBS'] = [astro_header['DATE-OBS']] * 2 * nsources
    else:
        pair_params['DATE-OBS'] = [astro_header['DATE']] * 2 * nsources
    mc_name = df_mc['name_mc'].values
    mc_iau_name = df_mc['IAU_name_mc'].values
    pair_params['MC-NAME'] = np.repeat(mc_name,2)
    pair_params['MC-IAU-NAME'] = np.repeat(mc_iau_name,2)
    pair_params['MAGZPT'] = [mag_zeropoint] * 2 * nsources
    pair_params['EXPTIME'] = [astro_header['EXPTIME']] * 2 *nsources
    pair_params['APERPIX'] = [mc_aper] * 2 *nsources
    
    indexes=idx_o
    
    for k in keywords:
        for i in indexes:
            pair_params[k].append([data[k][i]] * 2)
        pair_params[k] = [item for sublist in pair_params[k] for item in sublist]

    df = pd.DataFrame(pair_params)
    csv_out = 'photocal_res.csv'
    df.to_csv(csv_out, index=False)
    data_res=pd.concat([data_res,df])

data_res = data_res.sort_values(by=['MJD-OBS'])
object_names = data_res['MC-IAU-NAME'].unique()
print('^' * 100)
print('OBJECTS = ', object_names)
print('^' * 100)

pol_rows = []

# dictionary for storing results...
pol_data = DefaultDict(list)

# Processing each target object
for name in object_names:
    data_sets = pol.object_measures(data_res, name)
            
    for data_object in data_sets:
        print('group')
        print('-' * 60)
        print(data_object[['TYPE','OBJECT','ANGLE','MAG_APER', 'MAGERR_APER', 'FLUX_APER', 'FLUXERR_APER']])

        if len(data_object.index) < 8:
            print(f'POLARIMETRY,WARNING,"Not enough observations for compute object \'{name}\' polarimetry"')
            continue

        try:
            res_pol = pol.compute_polarimetry(data_object)
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
        #mean_secpix = (data_object[is_ord]['SECPIX1'].mean() + data_object[is_ord]['SECPIX2'].mean()) / 2
        #res_pol['APERAS'] = res_pol['APERPIX'] * mean_secpix
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
                         res_pol['APERPIX']]
        pol_rows.append(row)

cols = ['EPOCH','RJD-50000','OBJECT', 'P', 'dP', 'Theta', 'dTheta', 'Q', 'dQ', 'U', 'dU','R','Sigma','APERPIX']
dfpol = pd.DataFrame(pol_rows, columns=cols)
selection=dfpol[(dfpol.P > 7.0) & (dfpol.P < 10.0) & (dfpol.Q>0) & (dfpol.U>0)]
selection.to_csv('/home/users/dreg/misabelber/%s_field.csv' % date_run)