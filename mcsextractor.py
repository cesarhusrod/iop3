#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains routines for managing SExtractor (Source Extractor) Software.

Module created for working on
     * FITS files.

Created on Mon Wed 02 11:15:01 2020.

___e-mail__ = cesar_husillos@tutanota.com
__author__ = 'Cesar Husillos'

VERSION: 0.1
"""

import os
import subprocess
import copy
import re
from io import StringIO
import itertools

# Astronomical packages
from astropy.io import fits  # FITS library

# Numerical packages
import numpy as np
import pandas as pd

# plotting package
from matplotlib import pyplot as plt
from astropy.visualization import astropy_mpl_style

# Coordinate system transformation package and modules
from astropy import units as u
from astropy.coordinates import SkyCoord  # High-level coordinates

#import statsmodels.api as smapi  # Linear fitting module
#from scipy import optimize

from mcfits import MCFits

class NonvalidCatalog(Exception):
    pass

class NonvalidFITS(Exception):
    pass

class MandatoryParam(Exception):
    pass

class MCSExtractor():
    def __init__(self, fits_path=None, cat_path=None, detect_params=None):
        """Class Constructor.

        This class can be used for detect sources or read catalogs. Because of that,
        parameters are optional.

        Configuration parameters for optimal MAPCAT images detection are given.
        If any of them are not provided in detect_parameters, they are set
        depending on FITS exposure time value (FITS header 'EXPTIME' keyword).

        Parameters
        ----------
        fits_path : str
            Path to FITS file (the default is None).
        cat_path : str
            Path to existing SExtractor catalog (the default is None).
        detect_parameters : dict
            SExtractor parameters dictionary for detecting sources in 'fits_path'.
            If 'CATALOG_NAME' keyword is included, it overwrites 'cat_path'
            parameter (the default is None).
        ofits: MCFits object.


        Examples
        -------
        Examples should be written in doctest format, and
        should illustrate how to use the function/class.
        >>>

        """

        # Catalog data
        self._data = None

        # Checking 'fits_path' input parameter validity
        if fits_path and not os.path.exists(fits_path):
            str_error = f"ERROR: Optional input FITS path '{fits_path}' is not valid."
            raise NonvalidFITS(str_error)
        
        self._fits_path = fits_path
        self._ofits = MCFits(self._fits_path)
        self._cat_path = cat_path
        self._detect_params = detect_params

        # Overwriting _cat_path attribute
        if 'CATALOG_NAME' in self._detect_params:
            self._cat_path = self._detect_params['CATALOG_NAME']

        if self._cat_path is None:
            raise NonvalidCatalog(f"ERROR: SExtractor output catalog was not set!")

        # Providing default values for SExtractor detection over MAPCAT FITS files
        self.__config_default()

    @property
    def fits_path(self):
        return self._fits_path
    
    @fits_path.setter
    def fits_path(self, new_value):
        self._fits_path = new_value

    @property
    def ofits(self):
        return self._ofits
    
    @property
    def cat_path(self):
        return self._cat_path

    @cat_path.setter
    def cat_path(self, new_value):
        self._cat_path = new_value
    
    @property
    def detect_params(self):
        return self._detect_params
    
    @detect_params.setter
    def detect_params(self, key, new_value):
        if key not in self._detect_params:
            print(f"WARNING: keyword {key} not set before")
        self._detect_params[key] = new_value

    def __config_default(self):
        """ Defautl values for some SExtractor detection parameters.

        They depend on exptime.
        
        Parameters
        ----------
        
        Examples
        -------
        Examples should be written in doctest format, and
        should illustrate how to use the function/class.
        >>>

        """
        # Preconfigured values depending on exptime
        # Getting FITS EXPTIME header value
        if 'EXPTIME' not in self.ofits._header:
            raise NonvalidFITS(f"ERROR: Couldn't get access to 'EXPTIME' keyword on FITS '{self._fits_path}'")
        
        exptime = int(self._ofits._header['EXPTIME'])

        det_filter = 'N'
        det_clean = 'N'
        minarea = 13
        an_thresh = 1.0
        det_thresh = 1.0
        deb_mincon = 0.1

        if exptime >= 0.2:
            # spurious detections for Y_IMAGE = 1 (lower border)
            if 'CLEAN' not in self._detect_params:
                det_clean = 'Y'
            if 'DEBLEND_MINCONT' not in self._detect_params:
                deb_mincon = 0.005
        if exptime >= 1:
            if 'FILTER' not in self._detect_params:
                det_filter = 'Y'
            if 'CLEAN' not in self._detect_params:
                det_clean = 'Y'
            if 'DETECT_MINAREA' not in self._detect_params:
                minarea = 9
            if 'ANALYSIS_THRESH' not in self._detect_params:
                an_thresh = 1.0
            if 'DETECT_THRESH' not in self._detect_params:
                det_thresh = 1.0
        if exptime >= 80:
            if 'DETECT_MINAREA' not in self._detect_params:
                minarea = 13
            if 'ANALYSIS_THRESH' not in self._detect_params:
                an_thresh = 2.5
            if 'DETECT_THRESH' not in self._detect_params:
                det_thresh = 2.5
        # if float(self._header['EXPTIME']) >= 100:
        #     pass
        # if float(self._header['EXPTIME']) >= 120:
        #     pass
        if exptime >= 180:
            if 'DETECT_MINAREA' not in self._detect_params:
                minarea = 9
            if 'ANALYSIS_THRESH' not in self._detect_params:
                an_thresh = 1.6
            if 'DETECT_THRESH' not in self._detect_params:
                det_thresh = 1.6
         
        # setting config parameters
        self._detect_params['FILTER'] = det_filter
        self._detect_params['CLEAN'] = det_clean
        self._detect_params['DETECT_MINAREA'] = minarea
        self._detect_params['ANALYSIS_THRESH'] = an_thresh
        self._detect_params['DETECT_THRESH'] = det_thresh
        self._detect_params['DEBLEND_MINCONT'] = deb_mincon

    def read(self):
        """It reads catalog given by 'cat_path' attribute.
        Only available SExtractor ouput formats are readable. More information
        (https://www.astromatic.net/software/sextractor)

        Returns
        -------
        pandas.dataframe
            Information contained in catalog.

        Raises
        -------
        NonvalidCatalog
            If catalog path doesn't exist.

        Examples
        -------
        Examples should be written in doctest format, and
        should illustrate how to use the function/class.
        >>>

        """

        if not os.path.exists(self._cat_path):
            raise NonvalidCatalog(f"ERROR: Non-valid input catalog {self._cat_path}")

        # cat_ext = os.path.splitext(self.cat_path)[1]

        if self._cat_path.endswith('.fits'):
            hdul = fits.open(self._cat_path)
            self._data = hdul[2].data  # assuming first extension is a table
            # cols = hdul[2].columns
            # cols.info()
        else: # text formatted
            text_cat = open(self._cat_path).read()
            # getting headers...
            fields = re.findall(r'#\s+\d+\s+([A-Z0-9_]+).*', text_cat)
            # reading data
            self._data = np.genfromtxt(self._cat_path, names=fields)

        return self._data

    def extract(self, overwrite=True, show_info=True):
        """SExtractor program call for getting sources from FITS file.

        Parameters
        ----------
        overwrite : bool
            If True, previous catalog and images are overwritten (the default is True).
        show_info : bool
            If True, additional information is printed (the default is True).

        Returns
        -------
        int
            0 if every process was fine.

        Raises
        -------
        MandatoryParam
            If mandatory 'CONFIG_FILE' in 'self.detect_parameters' or
            'self.cat_path' attributes are not set.

        Examples
        -------
        Examples should be written in doctest format, and
        should illustrate how to use the function/class.
        >>>

        """
        # Source Extraction command
        com = ["source-extractor"]

        out_dirs = list()  # detection products directory

        print('----------DETECT PARAMS --------')
        print(self.detect_params)

        print('---------- CAT PATH ------------')
        print(self.cat_path)

        # Mandatory KEYWORDS
        if 'CONFIG_FILE' not in self._detect_params:
            raise MandatoryParam("ERROR: 'CONFIG_FILE' keyword not found in input dictionary.\nABORTING execution!!")
        if  self._cat_path is None:
            raise MandatoryParam("ERROR: 'CATALOG_NAME' keyword not found in input dictionary.\nABORTING execution!!")

        if os.path.exists(self._cat_path) and not overwrite:
            print("INFO: Aborting execution.")
            print(f"\tOutput catalog {self._cat_path} already found and 'overwrite' param is False.")
            return 0

        # Mandatory KEYWORDS
        com.append(f'-c {self._detect_params["CONFIG_FILE"]}')
        com.append(f'-CATALOG_NAME {self._cat_path}')

        detection_keywords = "-CHECKIMAGE_TYPE "
        detection_values = "-CHECKIMAGE_NAME "
        keys = list()
        values = list()

        for k, v in self._detect_params.items():
            # Special format keywords
            if k in ['BACKGROUND', 'SEGMENTATION', 'APERTURES']:
                keys.append(k)
                values.append(v)
                d = os.path.split(v)[0]
                if len(d) and not os.path.exists(d): # output directories for SExtractor products
                    out_dirs.append(d)
            else:
                # The others
                com.append("-%s %s" % (k, str(v)))

        # Keywords that may be present in header
        if 'PIXEL_SCALE' not in self._detect_params.keys():
            if 'INSTRSCL' in self._ofits._header:
                com.append(f"-PIXEL_SCALE {self._ofits._header['INSTRSCL']}")
        if 'SEEING_FWHM' not in self._detect_params.keys():
            if 'FWHM' in self._ofits._header:  # in arcsecs
                fw = float(self._ofits._header['FWHM'])
                ps = float(self._ofits._header['INSTRSCL'])
                com.append("-SEEING_FWHM %(fw)4.2f " % {'fw': fw * ps})
        if 'MAG_ZEROPOINT' not in self._detect_params.keys():
            if 'MAGZPT' in self._ofits._header:
                com.append("-MAG_ZEROPOINT %(zp)4.2f " % {
                    'zp': str(self._ofits._header['MAGZPT'])})
        # Adding special format keyword:value pairs
        if keys:
            com.append("%s %s" % (detection_keywords, ','.join(keys)))
            com.append("%s %s" % (detection_values, ','.join(values)))

        # Create product directories
        for dire in out_dirs:
            if not os.path.exists(dire):
                try:
                    os.makedirs(dire)
                except IOError:
                    print(f"ERROR: Couldn't create output directory '{dire}'")
                    raise

        com.append(self._fits_path)  # Final SExtractor command
        if show_info:
            print(' '.join(com))

        # Executing SExtractor call
        subprocess.Popen(' '.join(com), shell=True).wait()

        return 0

    def fwhm(self, min_num_sources=3, overwrite=True, show_info=True):
        """
        This function estimates FWHM of image.

        For doing that, it calls SExtractor and filter detections to minimum
        excentricity and better FLAG values (isolated, non saturated...).

        Args:
            detect_params (dict): Valid parameters for SExtractor.
            min_num_sources (int): Minum number of sources for FWHM estimation.
            overwrite (bool): It True, output SExtractor catalog is overwritten.
            show_info (bool): It True, process info is displayed.

        Returns:
            dict: result about FWHM computation and statistics.

        """
        # TODO: Sustituir diccionarios por namedtuple (collections)

        if self.extract(overwrite, show_info):
            return 1

        # field_names = ['X_IMAGE', 'Y_IMAGE', 'MAG_BEST', 'NUMBER',
        #               'ELLIPTICITY', 'FLAGS', 'CLASS_STAR', 'FWHM_IMAGE']
        # data_sextractor = np.genfromtxt(detect_params['CATALOG_NAME'],
        #                                names=field_names)
        data_sextractor = self.read()

        # Selecting best detections
        ellipticity = 0
        flag_value = 0
        condition1 = data_sextractor['FLAGS'] == 0
        flag_ellip = itertools.product([0, 1], np.arange(0, 1, 0.1))
        for flag_value, ellipticity in flag_ellip:
            # Preferred sources are non-saturated and non-deblended ones.
            condition1 = condition1 | (data_sextractor['FLAGS'] == flag_value)
            # circular sources
            condition2 = data_sextractor['ELLIPTICITY'] < ellipticity
            data_fwhm = data_sextractor[condition1 & condition2]
            if data_fwhm.shape[0] >= min_num_sources:
                # there are more than three sources for compute statistics
                break
        if data_fwhm.shape[0] == 0:
            return {'MEAN': None,
                    'STD': None,
                    'MIN': None,
                    'MAX': None,
                    'MEDIAN': None,
                    'MAX_ELLIP': ellipticity,
                    'FLAG': flag_value}
        # printing info
        print("Original number of sources ->", data_sextractor['X_IMAGE'].size)
        print("Filtered number of sources ->", data_sextractor['X_IMAGE'].size)
        if show_info:
            print("FWHM_IMAGE    CLASS_STAR    ELLIPTICITY")
            print("---------------------------------------")
            for fwhm, class_star, ellipticity in zip(data_fwhm['FWHM_IMAGE'],
                                                     data_fwhm['CLASS_STAR'],
                                                     data_fwhm['ELLIPTICITY']):
                print(f"{fwhm}  --  {class_star} --  {ellipticity}")

        # appending info to FITS Header
        new_cards = list()
        candidateCards = [('SOFTDET', 'SExtractor', 'Source detection soft'),
                          ('FWHM', round(data_fwhm['FWHM_IMAGE'].mean(), 2),
                           'Mean FWHM [pix]'),
                          ('FWHMSTD', round(data_fwhm['FWHM_IMAGE'].std(), 2),
                           'Std FWHM [pix]'),
                          ('FWNSOURC', data_fwhm['FWHM_IMAGE'].size,
                           'FWHM number of sources used'),
                          ('FWHMFLAG', flag_value, 'SExtractor source FLAG'),
                          ('FWHMELLI', ellipticity, 'SExtractor max ELLIP'),
                          ('PIXSCALE', self._ofits._header['INSTRSCL'], '[arcs/pix]')]
        for card in candidateCards:
            if card[0] in self._ofits._header:
                self._ofits._header[card[0]] = card[1]
            else:
                new_cards.append(card)
        # ------------ Writing FWHM computed ---------------------
        self._ofits._header.extend(new_cards, end=True)
        if show_info:
            print(f'INFO: Saving new cards in header {self._fits_path}')
        self._ofits.save()

        return {'MEAN': data_fwhm['FWHM_IMAGE'].mean(),
                'STD': data_fwhm['FWHM_IMAGE'].std(),
                'MIN': data_fwhm['FWHM_IMAGE'].min(),
                'MAX': data_fwhm['FWHM_IMAGE'].max(),
                'MEDIAN': np.median(data_fwhm['FWHM_IMAGE']),
                'MAX_ELLIP': ellipticity,
                'FLAG': flag_value
                }

    def _mask(self, x_centers, y_centers, pix_radius=100):
        """Mask sources given by (x_center, y_centers) pixel coordinates.

        In a radius of pix_radius pixels, these areas are filled with 0's.
        
        Args:
            x_centers (np.array or iterable): x pixel coordinates.
            y_centers (np.array or iterable): y pixel coordinates.
            pix_radius (int): circular mask pixel radius. played.

        Returns:
            np.array (2D): FITS data masked.
        """
        x_dim, y_dim = self.data.shape
        copy_data = np.copy(self.data)
        for xc, yc in zip(x_centers, y_centers):
            for x in range(x_dim):
                for y in range(y_dim):
                    if np.sqrt(np.power(x - xc, 2) + np.power(y - yc, 2)) <= pix_radius:
                        copy_data[y, x] = 0
        return copy_data

    def mask_sat_sources(self, pix_radius=100, overwrite=True,
                         show_info=True):
        """
        A short description.

        A bit longer description.

        Args:
            variable (type): description

        Returns:
            type: description

        Raises:
            Exception: description

        """

        if self.extract(overwrite=overwrite, show_info=show_info):
            return 1

        # SExtractor catalog
        data_sextractor = self.read()

        # Masked saturated sources output FITS
        out_fits = self._fits_path.replace('.fits', '_masked_sat.fits')
        copy_data = np.copy(self.data)

        sat_sources = data_sextractor['FLAGS'] == 4
        if sat_sources.sum() > 0:
            #mask each saturated sources
            data_sat = data_sextractor[sat_sources]
            if show_info:
                print(f'\t{sat_sources.sum()} sources saturated.')
            for x_center, y_center in zip(data_sat['X_IMAGE'],
                                          data_sat['Y_IMAGE']):
                if show_info:
                    print(f'\t\t({x_center}, {y_center})')
                copy_data = self._mask(data_sat['X_IMAGE'],
                                       data_sat['Y_IMAGE'], pix_radius)

        if show_info:
            print(f"Output saturated sources FITS: {out_fits}")
        
        hdu = fits.PrimaryHDU(header=self._ofits._header, data=copy_data)
        hdul = fits.HDUList([hdu])
        try:
            hdul.writeto(out_fits, overwrite=True, output_verify='ignore')
        except IOError:
            print(f"ERROR: Write output NON saturated FITS file '{out_fits}'")
            raise

        return 0


    def mask_supersources(self, out_masked_fits, detect_params,
                          min_area=400, factor=2,
                          overwrite=True, show_info=True):
        """
        It finds and masks wide saturated sources.

        This function generates new FITS from bigger and/or saturated sources masked.

        Args:
            out_masked_fits (str): Output masked FITS path.
            detect_params (dict): SExtractor parameter for detecting sources.
            min_area (int): minimun area for masking source.
            factor (float): scale from masked radius sources. if 1.0 mask will be equal to .
            overwrite (bool): If True, previous detection catalog is overwritten.
            show_info (bool): Show extra info about process.

        Returns:
            int: ', if everything was fine.

        Raises:
            IOError: if output FITS couldn't be written.

        """
        if 'SEGMENTATION' not in detect_params:
            detect_params['SEGMENTATION'] = self._fits_path.replace('.fits', '_segment_supersources.fits')
        # BACKGOUND is contaminated by original data values
        # if 'BACKGROUND' not in detect_params:
        #    detect_params['BACKGROUND'] = self.path.replace('.fits', '_back_supersources.fits')

        if self.extract(overwrite=overwrite, show_info=show_info):
            # TODO: Crear excepción propia
            return 1

        # SExtractor catalog
        data_sextractor = self.read()

        # segmentation image
        mc_segment = MCFits(self._detect_params['SEGMENTATION'])
        # mc_background = mcFits(detect_params['BACKGROUND'])

        # masked detections counter
        counter = 0
        # print(data['NUMBER'])
        for index, number in enumerate(data_sextractor['NUMBER'].tolist()):
            # condition_segmented = mc_segment.data == int(number)
            binary_segmented = np.where(mc_segment.data == int(number), 1, 0)
            # print(int(d), 'num. pix =', binary_segmented.sum())
            if binary_segmented.sum() > min_area:
                # masking
                x_image = int(data_sextractor['X_IMAGE'][index])
                y_image = int(data_sextractor['Y_IMAGE'][index])

                w, h = self._ofits._data.shape
                Y, X = np.ogrid[:h, :w]
                dist_from_center = np.sqrt((X - x_image)**2 + (Y-y_image)**2)
                radius = max(data_sextractor['A_IMAGE'][index],
                             data_sextractor['B_IMAGE'][index])
                # Replace supersource values by 0
                self.data = np.where(dist_from_center <= radius * factor,
                                     0, self.data)
                counter += 1

        hdu = fits.PrimaryHDU(data=self.data)
        hdu.header = self._ofits._header
        hdul = fits.HDUList([hdu])
        try:
            hdul.writeto(out_masked_fits, overwrite=True,
                         output_verify='ignore')
        except IOError:
            print(f"ERROR: Write output FITS file '{out_masked_fits}'")
            raise

        if show_info:
            print(f'Masked {counter} super-sources in "{self._fits_path}"')

        return 0

    def _filter_duplicated(self, 
                           searching_params={'min_dist_x': 25, 'max_dist_x': 38, 
                                             'min_dist_y': 0, 'max_dist_y': 1, 
                                             'max_mag_dif': 1},
                           overwrite=False, show_info=True):
        """
        This private function detect and reject duplicated sources.

        For each source, it looks for twin detection in area given by
        ['min_dist_x', 'max_dist_x', 'min_dist_y', 'max_dist_y'].

        Args:
            searching_params (dict): area and maximum difference in MAGNITUDE for consider a source as twin.
                min_dist_x (int): minimum distance in axis x (pixels) between source a twin candidate.
                max_dist_x (int): maximum distance in axis x (pixels) between source a twin candidate.
                min_dist_y (int): minimum distance in axis y (pixels) between source a twin candidate.
                max_dist_y (int): maximum distance in axis y (pixels) between source a twin candidate.
            overwrite (bool): It True, existing output SExtractor catalog is overwritten.
            show_info (bool): If True, additional info is displayed.

        Returns:
            array: SExtractor IDs NUMBER sources.

        """

        if self.extract(overwrite=overwrite, show_info=show_info):
            # TODO: Crear excepción propia
            return 1

        # Reading catalog
        data_sext = self.read()
        print('Detections number = ', data_sext['NUMBER'].size)

        # List of selected SExtractor NUMBERs
        numbers = list()
        mask = np.ones(data_sext['NUMBER'].size, dtype=bool)

        for index, number in enumerate(data_sext['NUMBER']):
            if number not in numbers:
                # Computing distance in axis x and y for each detected source
                distance_x = data_sext['X_IMAGE'] - data_sext['X_IMAGE'][index]
                distance_y = np.abs(data_sext['Y_IMAGE'] - data_sext['Y_IMAGE'][index])
                # taking into account magnitude diferences also
                diffmag = np.abs(data_sext['MAG_BEST'] - data_sext['MAG_BEST'][index])

                # Filtering
                my_filter = (data_sext['NUMBER'].astype(np.int) != number)
                my_filter &= (distance_y >= searching_params['min_dist_y'])
                my_filter &= (distance_y <= searching_params['max_dist_y'])
                my_filter &= (distance_x >= searching_params['min_dist_x'])
                my_filter &= (distance_x <= searching_params['max_dist_x'])
                my_filter &= (diffmag <= searching_params['max_mag_dif'])

                # Checking filter results
                if my_filter.sum() >= 1:
                    # print("Clone points number ->", boo.sum())
                    # if more than one similar detections are closed, only the
                    # first one is considered. (Maybe it's not the best solution)
                    numbers.append(data_sext['NUMBER'][my_filter][0])
                    # TODO: think about it
                    mask[index] = False

        # After checking each source, only not masked sources are taken
        # into account
        return data_sext['NUMBER'][mask]

    def mask_duplicated(self, out_fits,
                        searching_params={'min_dist_x': 25, 'max_dist_x': 38,
                                          'min_dist_y': 0, 'max_dist_y': 1,
                                          'max_mag_dif': 1},
                        overwrite=False, show_info=True):
        """
        This function detects duplicated sources and gretes a new one
        without them.

        It employs criteria taken into account relative position and
        magnitud for each detection.

        Args:
            out_fits (str): Output clean FITS path.
            detect_params (dict): SExtractor config file for detecting sources.
            searching_params (dict): magnitude an position criteria for set duplicated source.
            overwrite (bool): It true, previous SExtractor catalog and FITS are overwritten.
            show_info (bool): If true, additional info is printed.

        Returns:
            int: 0 if everything was fine.

        Raises:
            Exception: description

        """

        source_ids = self._filter_duplicated(searching_params=searching_params,
                                             overwrite=overwrite,
                                             show_info=show_info)
        
        try:
            oseg = MCFits(self._detect_params['SEGMENTATION'])
        except IOError:
            print(f"ERROR: Reading segment map {self._detect_params['SEGMENTATION']}")
            raise

        # printing info about segmentation data
        if show_info:
            print(oseg._data.dtype)
            print(f"Segmentation -> (shape:) {oseg._data.shape} (size:) {oseg._data.size}")

        # segmentation_data = np.ma.array(segmentation_data)
        mask = np.zeros(oseg._data.shape)  # Mask to False values (any source masked)

        for n in source_ids:
            condition_mask = oseg._data == n
            mask = np.logical_or(mask, condition_mask)
            # mask = np.logical_and(mask, segmentation_data[segmentation_data in numb]
        # print('(mask shape)', mask.shape)

        # Now the script uses SExtractor background image to hide duplicated sources
        # background
        # try:
        #     oback = MCFits(self._detect_params['BACKGROUND'])
        # except IOError:
        #     print(f"ERROR: Reading background map {self._detect_params['BACKGROUND']}")
        #     raise

        # Output cleaned for duplicated sources FITS
        if overwrite:
            directory = os.path.split(out_fits)[0]
            if len(directory) and not os.path.exists(directory):
                try:
                    os.makedirs(directory)
                except IOError:
                    print(f"ERROR: Making output directory '{directory}'")
                    raise

            # masking duplicated sources to 0!!!
            # data_masked = np.where(mask, oback._data.mean(), self.data)
            data_masked = np.where(mask, 0, self.data)
            hdu = fits.PrimaryHDU(data=data_masked.astype(np.uint16),
                                  header=self._ofits._header)
            try:
                hdu.writeto(out_fits, overwrite=True, output_verify='ignore')
            except IOError:
                print(f"ERROR: Couldn't write output masked duplicated sources '{out_fits}'")
                raise

        return 0
