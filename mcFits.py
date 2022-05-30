#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains routines oriented to deal with FITS format files.

It allows to users read, write, update and create new files in FITS format.

Created on Mon March 28 17:43:15 2020.

___e-mail__ = cesar_husillos@tutanota.com
__author__ = 'Cesar Husillos'

VERSION:
    1.1 Cleaned initial version (2021-11-03)
    1.2 Refactoring of code
"""


from collections import defaultdict
import os
import subprocess
import re
from datetime import datetime,timedelta
from io import StringIO

# Astronomical packages
import aplpy  # FITS plotting library
from astropy.io import fits  # FITS library
from astropy.wcs import WCS
from astropy.time import Time

# Numerical packages
import numpy as np
import pandas as pd

# plotting package
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Coordinate system transformation package and modules
from astropy import units as u
from astropy.coordinates import SkyCoord  # High-level coordinates

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

def _linear_fit(x, ord):
    """
    Simple linear fitting with SLOPE = 1.

    A bit longer description.

    Args:
        variable (type): description

    Returns:
        type: description

    Raises:
        Exception: description

    """

    return x + ord


class mcFits:
    def __init__(self, path, index=0, border=0, saturation=45000):
        self._path = path
        self._index = index
        self._save = False
        self._hdu = None
        self._header = None
        self._data = None
        try:
            self._hdu = fits.open(self._path)
            self._header = self._hdu[self._index].header
            self._data = self._hdu[self._index].data
        except FileNotFoundError:
            print(f"ERROR: '{path}' not found.")
            raise

        # Checking and setting border parameter
        self._border = 0
        if int(border) < 0: 
            print('REDUCTION,WARNING,"Border can not set to negative values. Set to 0."')
            self._border = 0
        elif (self._header['NAXIS1'] < 2 * int(border)) or \
            (self._header['NAXIS2'] < 2 * int(border)):
            print('REDUCTION,WARNING,"Border too high. Set to 0."')
            self._border = 0
        else:
            self._border = int(border)

        # Checking and setting saturation
        try:
            self.saturation = int(saturation)
            if self.saturation < 0:
                print('REDUCTION,WARNING,"Saturation can not set to negative values. Set to 45000."')
                self.saturation = 45000
        except ValueError:
            raise

    @property
    def hdu(self):
        return self._hdu

    @property
    def path(self):
        return self._path

    @property
    def header(self):
        return self._header

    @header.setter
    def header(self, new_value):
        self._header = new_value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_value):
        self._data = new_value

    @property
    def index(self):
        return self._index

    @property
    def border(self):
        return self._border

    @property
    def save(self):
        return self._save

    @save.setter
    def save(self, new_value):
        self._save = new_value

    def __del__(self):
        """Destructor checks if header has been changed. In that case,
        it saves file."""
        if self._save:
            print('INFO: Saving FITs changes.')
            self._hdu.writeto(self._path, overwrite=True, output_verify='ignore')
        if self._hdu:
            self._hdu.close()

    def __repr__(self):
        info = f'{self.__class__.__name__}('
        info += f'{self._path!r}, {self._index!r}, {self._border!r}, '
        info += f'{self.saturation!r})'
        return info

    def run_date(self):
        """Returns Python datetime object of FITS night run

        Returns:
            datetime.datetime: Run night observation date.
        """
        ofits = mcFits(self._path)
        t = Time(ofits.header["DATE-OBS"], format="isot", scale="utc")
        t_dt = t.to_datetime()
        
        if t_dt.hour < 12: # fits taken in previous day
            t_dt = t_dt - timedelta(days=1)
    
        return t_dt

    def stats(self, exclude_borders=True):
        """
        Basic statistical parameters from FITS data.

        minimum, maximum, mean, std, median and data type are computed.

        Args:
            exclude_borders (bool): If True, image border are excluded.

        Returns:
            dict: keywords are ['MIN', 'MAX', 'MEAN', 'STD', 'MEDIAN', 'TYPE']

        Raises:
            Exception: description

        """

        """Return a dictionary with min, max, mean, std and median of data."""
        
        new_data = self.data
        if exclude_borders and self._border > 0:
            new_data = self.data[self._border:-self._border,
                                 self._border:-self._border]
        # print(f'Stats from shaped data = {new_data.shape}')
        dictStats = dict()
        dictStats['MIN'] = new_data.min()
        dictStats['MAX'] = new_data.max()
        dictStats['MEAN'] = new_data.mean()
        dictStats['STD'] = new_data.std()
        dictStats['MEDIAN'] = np.median(new_data)
        dictStats['TYPE'] = new_data.dtype.name

        return dictStats

    def get_data(self, keywords=[]):
        """
        Get values from header and return them as dictionary.
        If keyword not in header, value associated is set to None

        Args:
            keywords (list): list of keywords.

        Returns:
            dict: dictionary which keys are the keywords. 
                    Value is None if keyword not in self.header
        """
        return {k:self.header.get(k, None) for k in keywords}
        
    def plot_histogram(self, out_plot, title='', log_y=True, dpi=300,
                       exclude_borders=True, histogram_params=dict()):
        """
        Plot histogram of input FITS given by 'inputFits'.

        If 'log_y' is True, scale of y-axis will be logaritmic.
        Other plot parameters are configurable though
        'histogram_params' parameter.
        It is passed to matplotlib 'hist' function.

        Args:
            out_plot (str): output FITS histogram.
            title (str): Histogram title.
            log_y (bool): if True, y-axis is given in logarithmic scale.
            dpi (int): plot resolution in "dots per inch" units.
            exclude_borders (bool): If True, FITS data borders are excluded 
                from histogram plot.
            histogram_params (dict): matplotlib.pyplot.hist valid parameters.

        Returns:
            int: 0, if everything was fine.

        Raises:
            Exception: if something failed.

        """
        new_data = self.data
        if exclude_borders and self._border > 0:
            new_data = self.data[self._border:-self._border,
                                 self._border:-self._border]

        # Generation of pd.Series object
        sdata = pd.Series(new_data.flatten())
        my_figure, my_axis = plt.subplots()
        # pd.Series histogram function
        sdata.hist(**histogram_params)
        if title:
            my_axis.set_title(title)
        my_axis.set_xlabel('Counts (#)')
        my_axis.set_ylabel(r'Pix Value (#)')
        if log_y:
            my_axis.set_ylabel(r'$log_{10}(Counts)$ (#)')
            my_axis.set_yscale('log')
        
        my_figure.savefig(out_plot, dpi=dpi)
        plt.close()

        return 0

    def plot(self, outputImage=None, title=None, colorBar=True, coords=None, \
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
        gc = aplpy.FITSFigure(self.path, dpi=300)
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
        gc.tick_labels.set_xposition('bottom')
        gc.tick_labels.set_yposition('left')
        gc.tick_labels.show()
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
        if outputImage is None:
            root, ext = os.path.splitext(self.path)
            outputImage = f'{root}.{format}'
        gc.save(outputImage, format=format)
        gc.close()

        return 0

    def statistics(self, border=15, sat_threshold=50000):
        """_summary_

        Args:
            sat_threshold (int, optional): _description_. Defaults to 50000.

        Returns:
            _type_: _description_
        """
        new_data = self.data
        if border > 0:
            new_data = self.data[border:-border, border:-border]
        dictStats = dict()
        dictStats['MINIMUM'] = new_data.min()
        dictStats['MAXIMUM'] = new_data.max()
        dictStats['MEAN'] = new_data.mean()
        dictStats['STD'] = new_data.std()
        dictStats['MEDIAN'] = np.median(new_data)
        dictStats['NPIX'] = self.header['NAXIS1'] * self.header['NAXIS2']
        dictStats['NSAT'] = (new_data >= sat_threshold).sum()
        dictStats['EXPTIME'] = self.header['EXPTIME']
        dictStats['FILENAME'] = os.path.split(self.path)[1]
        dictStats['STD/MEAN'] = dictStats['STD'] / dictStats['MEAN']
        mean = dictStats['MEAN']
        median = dictStats['MEDIAN']
        dictStats['MEAN_MEDIAN'] = np.round((mean - median) / (mean + median) * 100, 3)

        return dictStats

    def sext_params_detection(self, border=15, sat_threshold=45000):
        """It analyzes FITS data and return best input parameters for maximize detection.
        
        Args:
            border (int): Border size. It won't be used in statistics computation.
            sat_threshold (int): threshold pixel value considered as saturated.
            
        Returns:
            dict: Dictionary with best detection parameters for SExtractor.
        """
        
        params = {}
        # default values
        params['FILTER'] = 'N'
        params['CLEAN'] = 'N'
        params['DETECT_MINAREA'] = 25
        params['ANALYSIS_THRESH'] = 1.0
        params['DETECT_THRESH'] = 1.0
        params['DEBLEND_MINCONT'] = 0.005
        if 'CCDGAIN' in self.header:
            params['GAIN'] = self.header['CCDGAIN']

        # getting info about FITS
        dt = self.statistics(border=border, sat_threshold=sat_threshold)

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

    def detect_sources(self, sext_conf, cat_out, \
        additional_params={}, photo_aper=None, mag_zeropoint=None, \
        back_image=False, segment_image=False, aper_image=False, \
        verbose=True):
        """SExtractor call for FITS source detection.

        Args:
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
        pixscale = self.header['INSTRSCL']

        # Adtitional ouput info
        root, ext = os.path.splitext(self.path)
        back_path = f'{root}_back.fits'
        segm_path = f'{root}_segment.fits'
        aper_path = f'{root}_apertures.fits'

        check_types = ['BACKGROUND', 'SEGMENTATION', 'APERTURES']
        check_names = [back_path, segm_path, aper_path]
        options = [back_image, segment_image, aper_image]
        
        # SExtractor parameters    
        params = {}

        if back_image or segment_image or aper_image:
            params['CHECKIMAGE_TYPE'] = ','.join([p for p, o in zip(check_types, options) if o == True])
            params['CHECKIMAGE_NAME'] = ','.join([p for p, o in zip(check_names, options) if o == True])
        
        cmd = f"source-extractor -c {sext_conf} -CATALOG_NAME {cat_out} -PIXEL_SCALE {pixscale} "
        if photo_aper:
            cmd += f"-PHOT_APERTURES {photo_aper} "
        
        
        fwhm = self.header.get('FWHM', None)
        if fwhm:
            fwhm_arcs = float(fwhm) * float(pixscale)
            cmd += f"-SEEING_FWHM {fwhm_arcs} "
        
        if mag_zeropoint:
            cmd += f"-MAG_ZEROPOINT {mag_zeropoint} "
        
        for k, v in additional_params.items():
            params[k] = v
        
        # Formatting parameters to command line syntax
        com_params = [f'-{k} {v}' for k, v in params.items()]
        
        # adding parameters to command
        cmd = cmd + ' '.join(com_params)
        
        # last parameter for command
        cmd = f'{cmd} {self.path}'

        if verbose:
            print(cmd)
        
        res = execute_command(cmd)

        if res.returncode:
            print(res)
            return res.returncode

        return 0

    def fwhm_from_cat(self, cat_out, cat_format='FITS_LDAC'):
        # Filtering detections
        data = read_sext_catalog(cat_out, format=cat_format)
         
        # checking Saturation    
        fboo = check_saturation(data['FLAGS']) # saturated booleans
        if ~fboo.sum() > 15:
            data = data[~fboo]
        # filtering by ellipticity
        foo = data['ELLIPTICITY'] < 0.1
        if foo.sum() > 10:
            print(f'{foo.sum()} sources passed ELLIPTICITY filter')
            data = data[foo]
        # filtering to isolated sources
        foo = data['FLAGS'] == 0
        if foo.sum() > 5:
            print(f'{foo.sum()} sources passed FLAGS filter')
            data = data[foo]
        # filtering to STARS
        foo = data['CLASS_STAR'] > 0.8
        if foo.sum() > 3:
            print(f'{foo.sum()} sources passed CLASS_STAR filter')
            data = data[foo]

        cards = [('SOFTDET', 'SExtractor', 'Source detection software'), \
            ('FWHM', round(data['FWHM_IMAGE'].mean(), 2), 'Mean pix FWHM'), \
            ('FWHMSTD', round(data['FWHM_IMAGE'].std(), 2), 'Std pix FWHM'), \
            ('FWNSOURC', data['FWHM_IMAGE'].size, 'FWHM number of sources used'), \
            ('FWHMFLAG', data['FLAGS'].max(), 'SExtractor max source FLAG'), \
            ('FWHMELLI', round(data['ELLIPTICITY'].max(), 2), 'SExtractor max ELLIP'), \
            ('PIXSCALE', self._header['INSTRSCL'], 'Scale [arcs/pix]'),
            ('CSTARMIN', round(data['CLASS_STAR'].min(), 2), 'SExtractor min CLASS_STAR')]

        return cards

    def get_fwhm(self, sext_conf, cat_out, cat_format='FITS_LDAC', plot=True, other_params={}):
        """FWHM FITS computation.

        Args:
            sext_conf (str): Path to SExtractor configuration file.
            cat_out (str): SExtractor output path file.
            cat_format (str, optional): SExtractor output file format. 
                Valid values: 'FITS_LDAC' ans 'ASCII'. Defaults to 'FITS_LDAC'.
            plot (bool, optional): If True, FITS plot is generated. Sources used in
                FWHM computations are plotted in. Defaults to True.
            other_params (dict, optional): Parameters that overwrite default ones in
                SExtractor configuration file. Defaults to {}.

        Returns:
            list: List of cards (keyword, value, comment). It can be used for updating 
                FITS header.
        """
        cards = []
        data = None

        res = self.detect_sources(sext_conf, cat_out, additional_params=other_params)

        if not res:
            cards = self.fwhm_from_cat(cat_out, cat_format=cat_format)
            
        if plot:
            root, ext = os.path.splitext(cat_out)
            plot_out = f'{root}.png'
            coords = [data['X_IMAGE'], data['Y_IMAGE']]
            
            self.plot(plot_out, 'FWHM selected sources', coords=coords, ref_coords='pixel')

        return cards
  
    def update_header_fwhm(self, sext_conf, cat_out, cat_format='FITS_LDAC', other_params={}):
        """Add or update calibration pairs (key, value) related with 
        FITS FWHM.

        Args:
            sext_conf (str): SExtractor configuration file path. 
            cat_out (str): Output SExtractor file.
            cat_format (str, optional): Output SExtractor file format. Defaults to 'FITS_LDAC'.
            other_params (dict, optional): Parameters and their values that overwrite 
                SExtractor configuration. Defaults to {}.

        Returns:
            int: 0, if everything was fine.
        """
        cards = self.get_fwhm(sext_conf, cat_out, cat_format=cat_format, other_params=other_params)
        print(cards)
        
        return self.update_header(cards)
    
    def update_header(self, cards):
        """Add or update FITS header with keywords values given by 'cards' param.

        Args:
            cards (list): List of tuples with following format (keyword, value, comment)
            
        Returns:
            int: 0, if everything was fine."""

        with fits.open(self.path, mode='update') as hdul:
            hdr = hdul[0].header
            
            new_cards = []
            for card in cards:
                if card[0] in hdr:
                    # updating existing keywords
                    hdr[card[0]] = card[1]
                else:
                    # appending new info to FITS Header
                    new_cards.append(card)
                
            # ------------ Writing FWHM computed ---------------------
            hdr.extend(new_cards, end=True)
            hdul.flush()

        return 0

    def _filter_duplicated(self, detect_params,
                           searching_params={'min_dist_x': 25, 'max_dist_x': 38,
                                             'min_dist_y': 0, 'max_dist_y': 1,
                                             'max_mag_dif': 1},
                           overwrite=False, show_info=True):
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

        if self.extract_sources(detect_params, overwrite=overwrite,
                                show_info=show_info):
            # TODO: Crear excepción propia
            return 1

        # Reading input catalog
        fields = ['X_IMAGE', 'Y_IMAGE', 'MAG_BEST', 'NUMBER', 'ELLIPTICITY',
                  'FLAGS', 'CLASS_STAR', 'FWHM_IMAGE']
        data_sextractor = np.genfromtxt(detect_params['CATALOG_NAME'], names=fields)
        print('Detections number = ', data_sextractor['NUMBER'].size)

        # List of selected SExtractor NUMBERs
        numbers = list()
        mask = np.ones(data_sextractor['NUMBER'].size, dtype=bool)

        for index, number in enumerate(data_sextractor['NUMBER']):
            if number not in numbers:
                # Computing distance in axis x and y for each detected source
                distance_x = data_sextractor['X_IMAGE'] - data_sextractor['X_IMAGE'][index]
                distance_y = np.abs(data_sextractor['Y_IMAGE'] - data_sextractor['Y_IMAGE'][index])
                # taking into account magnitude diferences also
                diffmag = np.abs(data_sextractor['MAG_BEST'] - data_sextractor['MAG_BEST'][index])

                # Filtering
                my_filter = (data_sextractor['NUMBER'].astype(np.int) != number)
                my_filter &= (distance_y >= searching_params['min_dist_y'])
                my_filter &= (distance_y <= searching_params['max_dist_y'])
                my_filter &= (distance_x >= searching_params['min_dist_x'])
                my_filter &= (distance_x <= searching_params['max_dist_x'])
                my_filter &= (diffmag <= searching_params['max_mag_dif'])

                # Checking filter results
                if my_filter.sum() >= 1:
                    # print("Clone points number ->", boo.sum())
                    # if more than one close similar source the first one is taken
                    # Maybe it's not the best solution
                    numbers.append(data_sextractor['NUMBER'][my_filter][0])
                    # TODO: think about it
                    mask[index] = False

        # After checking each source, only not masked sources are taken into account
        return data_sextractor['NUMBER'][mask]

    def mask_duplicated(self, out_fits, detect_params,
                        searching_params={'min_dist_x': 25, 'max_dist_x': 38,
                                          'min_dist_y': 0, 'max_dist_y': 1,
                                          'max_mag_dif': 1},
                        overwrite=False, show_info=True):
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
        source_ids = self._filter_duplicated(detect_params=detect_params,
                                             searching_params=searching_params,
                                             overwrite=overwrite,
                                             show_info=show_info)
        if 'SEGMENTATION' in detect_params.keys():
            try:
                # TODO: Should I use mcFits class in mcFits method?
                segmentation_hdul = fits.open(detect_params['SEGMENTATION'])
                segmentation_data = segmentation_hdul[0].data
                segmentation_hdul.close()
            except IOError:
                print("ERROR: Couldn't read segmentation map '%s'" %
                      detect_params['SEGMENTATION'])
                raise

        # printing info about segmentation data
        if show_info:
            print(segmentation_data.dtype)
            print('Segmentation -> (shape:)', segmentation_data.shape,
                  '(size:)', segmentation_data.size)

        # masegmentation_data = np.ma.array(segmentation_data)
        mask = np.zeros(segmentation_data.shape)  # Mask to False values (any source masked)

        for n in source_ids:
            condition_mask = segmentation_data == n
            mask = np.logical_or(mask, condition_mask)
            # mask = np.logical_and(mask, segmentation_data[segmentation_data in numb]
        # print('(mask shape)', mask.shape)

        # Now the script uses SExtractor background image to hide duplicated sources
        # background
        if 'BACKGROUND' in detect_params.keys():
            try:
                background_hdul = fits.open(detect_params['BACKGROUND'])
                background_data = background_hdul[0].data
                background_hdul.close()
            except IOError:
                print("ERROR: Couldn't read background map '%s'" %
                      detect_params['BACKGROUND'])
                raise

        # Output cleaned for duplicated sources FITS
        if overwrite:
            directory, file_name = os.path.split(out_fits)
            if len(directory) and not os.path.exists(directory):
                try:
                    os.makedirs(directory)
                except IOError:
                    print("ERROR: Couldn't create output directory '%s'" %
                          directory)
                    raise

            data_masked = np.where(mask, background_data, self.data)
            hdu = fits.PrimaryHDU(data=data_masked.astype(np.uint16),
                                  header=self._header)
            try:
                hdu.writeto(out_fits, overwrite=True, output_verify='ignore')
            except IOError:
                print("ERROR: Couldn't write output no duplicated FITS sources '%s'" %
                      out_fits)
                raise

        return 0

    def rotate(self, overwrite=False):
        """
        FITS image rotation.

        It rotates image 90 degrees counter clockwise.

        Args:
            overwrite (bool): If True, FITS data is rotated and overwritten.

        Returns:
            fits.PrimaryHDU: object with header and rotated data.

        Raises:
            Exception: type depending on failing line of code.

        """
        # matrix rotation
        rot_data = np.rot90(self.data, k=-1)
        # new rotated fits
        hdu = fits.PrimaryHDU(data=rot_data, header=self._header)

        if overwrite:
            hdu.writeto(self._path, overwrite=overwrite, output_verify='ignore')

        return hdu

    def mask_supersources(self, out_masked_fits, detect_params,
                          min_area=400, square_size=180,
                          overwrite=True, show_info=True):
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
        if 'SEGMENTATION' not in detect_params:
            detect_params['SEGMENTATION'] = self.path.replace('.fits', '_segment_supersources.fits')
        # BACKGOUND is contaminated by original data values
        # if 'BACKGROUND' not in detect_params:
        #    detect_params['BACKGROUND'] = self.path.replace('.fits', '_back_supersources.fits')

        if self.extract_sources(detect_params, overwrite=overwrite,
                                show_info=show_info):
            # TODO: Crear excepción propia
            return 1

        # SExtractor catalog
        root, ext = os.path.splitext(self.path)
        cat_path = f'{root}.cat'
        data_sextractor = read_sext_catalog(cat_path, format='ASCII')

        # segmentation image
        mc_segment = mcFits(detect_params['SEGMENTATION'])
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
                dist = int(square_size / 2)
                # Replace supersource values for background ones
                # self.data[x-dist:x+dist, y-dist:y+dist] = mc_background.data[x-dist:x+dist, y-dist:y+dist]
                self.data[x_image - dist: x_image + dist,
                          y_image - dist: y_image + dist] = 0
                counter += 1

        hdu = fits.PrimaryHDU(data=self.data)
        hdu.header = self.header
        hdul = fits.HDUList([hdu])
        try:
            hdul.writeto(out_masked_fits, overwrite=True, output_verify='ignore')
        except IOError:
            print("ERROR: Write output FITS file '%s'" % out_masked_fits)
            raise

        if show_info:
            print('Masked %d super-sources in %s' % (counter, self.path))

        return 0

    
    
    def get_astroheader(self):
        """Get header info used for set relation between pixel (x,y) and 
        sky (ra,dec) coordinates.

        Returns:
            dict: keywords related with astrometric FITS calibration.
        """
        astro_keys = ['CRVAL1', 'CRVAL2', 'EPOCH', 'CRPIX1', 'CRPIX2', 'SECPIX', \
            'SECPIX1', 'SECPIX2', 'CDELT1', 'CDELT2', 'CTYPE1', 'CTYPE2', \
            'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'WCSRFCAT', 'WCSIMCAT', \
            'WCSMATCH', 'WCSNREF', 'WCSTOL', 'RA', 'DEC', 'EQUINOX', \
            'CROTA1', 'CROTA2', 'WCSSEP', 'IMWCS']

        return {k:self.header.get(k, None) for k in astro_keys}
