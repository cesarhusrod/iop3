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
"""


from collections import defaultdict
import os
import subprocess
# import copy
import re
from io import StringIO

# Astronomical packages
import aplpy  # FITS plotting library
from astropy.io import fits  # FITS library
from astropy.wcs import WCS

# Numerical packages
import numpy as np
import pandas as pd

# plotting package
from matplotlib import pyplot as plt

# Coordinate system transformation package and modules
from astropy import units as u
from astropy.coordinates import SkyCoord  # High-level coordinates
# from PyAstronomy import pyasl # for changing coordinates format

# import statsmodels.api as smapi  # Linear fitting module
# from scipy import optimize

def _read_sextractor_catalog(catalog):
    """
    It reads SExtractor output catalog.

    This function reads files in text or FITS/LDAC format
    produced by SExtractor.

    Args:
        catalog (str): path to SExtractor output file.

    Returns:
        ndarray: numpy.ndarray with SExtractor detection parameters.

    Raises:
        Exception: Any type, depending on failing code line.

    """
    data_sextractor = None
    
    if os.path.splitext(catalog)[1] == '.fits':
        hdul = fits.open(catalog)
        data_sextractor = hdul[2].data  # assuming first extension is a table
    else:
        fields = list()
        with open(catalog) as f_input:
            line = f_input.readline()
            while line.startswith('#'):
                try:
                    fields.append(line.split()[2])
                except:
                    pass
                line = f_input.readline()

        data_sextractor = np.genfromtxt(catalog, names=fields)

    return data_sextractor


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
    def __init__(self, path, index=0, border=0, saturation=40000):
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
        if (np.array(self._data.shape) - 2 * int(border)).all():
            self._border = int(border)

        # Checking and setting saturation
        try:
            self.saturation = int(saturation)
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
        dictStats = dict()
        dictStats['MIN'] = new_data.min()
        dictStats['MAX'] = new_data.max()
        dictStats['MEAN'] = new_data.mean()
        dictStats['STD'] = new_data.std()
        dictStats['MEDIAN'] = np.median(new_data)
        dictStats['TYPE'] = new_data.dtype.name

        return dictStats

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

    def plotFits(self, plot_path, title=None, colorbar=True, \
        coords=None, ref_coords='world', astrocal=False, \
        colors=['green', 'red', 'blue'], dparams={'aspect':'auto'}):
        """
        It creates 'plot_path' image file using data from current FITS.

        Args:
            plotpath (str): path to output plot.
            title (str): title of plot.
            colorbar (bool): If True, colorbar is added to plot.
            coords (list): List of tuples of arrays (o lists). Each element of list
                is an independent region for plotting on image.
            ref_coords (str): Allowed values are:
                * 'world', if FITS is calibrated astrometically,
                * 'pixel', in other case
            colors (list): Each element is a string with name of color.
            dparams (dict): Additional shape parameters of plot.

        Returns:
            int: 0 if everything was fine.

        Raises:
            Exception: depending on failing line of code.

        """

        gc = aplpy.FITSFigure(self.path, dpi=300)
        #gc.set_xaxis_coord_type('scalar')
        #gc.set_yaxis_coord_type('scalar')
        gc.show_grayscale(**dparams)
        #gc.recenter(512, 512)
        gc.tick_labels.set_font(size='small')
        if title:
            gc.set_title(title)
        if colorbar:
            gc.add_colorbar()
            gc.colorbar.show()
        gc.add_grid()
        gc.grid.set_color('orange')
        if astrocal:
            gc.grid.set_xspacing(1./60) # armin

        gc.grid.show()
        gc.grid.set_alpha(0.7)
        if coords:
            for i, c in coords.enumerate():
                ra, dec = c[0], c[1]
                gc.show_markers(ra, dec, edgecolor=colors[i], facecolor='none',
                                marker='o', coords_frame=ref_coords, s=40, alpha=1)
        gc.save(plot_path)
        gc.close()

        return 0


    def plot(self, title=None, out_path=None, colorBar=True, regions=list(),
             coords=list(), coord_color='magenta', dpi=300,
             parameters={'aspect': 'auto'}):
        """
        Another plotting FITS data function.

        It is a wrapper of aplpy.FITSFigure class that plots FITS data.
        Output plot is located at the same path of input image. Extension
        is changed to '.png'.

        Args:
            title (str): Title plot.
            colorBar (bool): If True, color bar is added to plot.
            coords (list): [[x_coords], [y_coords]] list of points 
                to draw in plot.
            coord_color (str): valid color name.
            dpi (int): plot resolution in "dot per inch" units.
            parameters (dict): valid aplpy.FITSFigure.show_grayscale 
                parameters dictionary.

        Returns:
            int: 0, if everything was fine.

        Raises:
            Exception: depending on failing line of code.

        """
        gc = aplpy.FITSFigure(self._path, dpi=dpi)
        # gc.set_xaxis_coord_type('scalar')
        # gc.set_yaxis_coord_type('scalar')
        gc.show_grayscale(**parameters)
        # gc.recenter(512, 512)
        gc.tick_labels.set_font(size='small')
        gc.tick_labels.set_xposition('bottom')
        gc.tick_labels.set_yposition('right')
        # gc.tick_labels.set_xformat('d.ddddd')
        # gc.tick_labels.set_yformat('d.ddddd')
        if title:
            gc.set_title(title)
        if colorBar:
            gc.add_colorbar()
            gc.colorbar.show()
        for reg in regions:
            gc.show_regions(reg)
        if len(coords) > 0:  # only wcs coordinates
            gc.show_circles(coords[0], coords[1], radius=5. / 60 / 60,
                            edgecolor=coord_color)
        gc.add_grid()
        gc.grid.set_color('orange')
        # gc.grid.set_xspacing('tick')
        # gc.grid.set_yspacing('tick')
        gc.grid.show()
        gc.grid.set_alpha(0.7)
        if out_path:
            gc.save(out_path)
        else:    
            gc.save(self._path.replace('.fits', '.png'))
        gc.close()

        return 0


    def plot_higher_than(self, out_plot, max_value=4000, title=None,
                         show_center=True, dpi=300, color="orange",
                         parameters={'markersize': 2, 'origin': 'lower'}):
        """
        This function plot sparse 'matrix2D' with 'markersize'
        and saves it in 'out_plot' after setting 'title'.

        A bit longer description.

        Args:
            variable (type): description

        Returns:
            type: description

        Raises:
            Exception: description

        """
        mat = np.flip(self.data, 0)
        saturated = np.where(mat >= max_value, 1, 0)
        my_figure, my_axis = plt.subplots()
        my_axis.spy(saturated, **parameters)
        my_axis.set_title(title)
        # grid properties
        if show_center:
            x_center = int(int(self._header['NAXIS1']) / 2)
            y_center = int(int(self._header['NAXIS2']) / 2)
            my_axis.set_xticks(np.array([x_center]), minor=False)  # center of image
            my_axis.set_yticks(np.array([y_center]), minor=False)  # center of image
        my_axis.grid(color=color, alpha=0.7, which='both')
        plt.savefig(out_plot, dpi=dpi)
        plt.close()

        return 0

    def plot_lower_than(self, out_plot, low_value=0, title=None,
                        show_center=True, dpi=300, color="orange",
                        parameters={'markersize': 2, 'origin': 'lower'}):
        """
        It plots FITS matrix values lower than 'low_value'.

        A bit longer description.

        Args:
            out_plot (str): Output plot path
            title (str): Plot title (default = None)
            low_value (int): Pixel values lower than this are plotted
                            (default = None)
            show_center (bool): If True, a plus symbol marks image center
                                (default = True)
            dpi (int): Plot dots per inche resolution (default = 300)
            color (str): Color name or hexadecimal code for center mark
                            (default = 'orange')
            parameters (dict): Params to pass to 'spy' matplotlib function.
                                (default = {'markersize':2, 'origin':'lower'})

        Returns:
            type: description

        Raises:
            Exception: description

        """

        """This function plot sparse 'matrix2D' with 'markersize'
        and saves it in 'out_plot' after setting 'title'."""
        mat = np.flip(self.data, 0)
        saturated = np.where(mat < 0, 1, 0)
        my_figure, my_axis = plt.subplots()
        my_axis.spy(saturated, **parameters)
        my_axis.set_title(title)
        # grid properties
        if show_center:
            x_center = int(int(self._header['NAXIS1']) / 2)
            y_center = int(int(self._header['NAXIS2']) / 2)
            my_axis.set_xticks(np.array([x_center]), minor=False)  # center of image
            my_axis.set_yticks(np.array([y_center]), minor=False)  # center of image
        my_axis.grid(color=color, alpha=0.7, which='both')
        plt.savefig(out_plot, dpi=dpi)
        plt.close()

        return 0

    def fwhm_def_params(self):
        """
        SExtractor default detection parameters for FWHM estimation.
        
        They depends on EXPTIME header keyword value.
        
        Args:
            None
        
        Returns:
            Dictionay with SExtractor relevant detection parameters and their values.
            
        """
        detect_params = {}
        
        if float(self._header['EXPTIME']) >= 0:
            detect_params['FILTER'] = detect_params.get('FILTER', 'N')
            detect_params['CLEAN'] = detect_params.get('CLEAN', 'N')
            detect_params['DETECT_MINAREA'] = detect_params.get('DETECT_MINAREA', 13)
            detect_params['ANALYSIS_THRESH'] = detect_params.get('ANALYSIS_THRESH', 1.0)
            detect_params['DETECT_THRESH'] = detect_params.get('DETECT_THRESH', 1.0)
            detect_params['DEBLEND_MINCONT'] = detect_params.get('DEBLEND_MINCONT', 0.1)
        
        if float(self._header['EXPTIME']) >= 0.2:
            # spurious detections for Y_IMAGE = 1 (lower border)
            detect_params['CLEAN'] = detect_params.get('CLEAN', 'Y')
            detect_params['DEBLEND_MINCONT'] = detect_params.get('DEBLEND_MINCONT', 0.005)
            
        if float(self._header['EXPTIME']) >= 1:
            detect_params['FILTER'] = detect_params.get('FILTER', 'Y')
            detect_params['CLEAN'] = detect_params.get('CLEAN', 'Y')
            detect_params['DETECT_MINAREA'] = detect_params.get('DETECT_MINAREA', 9)
            detect_params['ANALYSIS_THRESH'] = detect_params.get('ANALYSIS_THRESH', 1.0)
            detect_params['DETECT_THRESH'] = detect_params.get('DETECT_THRESH', 1.0)
            
        if float(self._header['EXPTIME']) >= 80:
            detect_params['DETECT_MINAREA'] = detect_params.get('DETECT_MINAREA', 13)
            detect_params['ANALYSIS_THRESH'] = detect_params.get('ANALYSIS_THRESH', 2.5)
            detect_params['DETECT_THRESH'] = detect_params.get('DETECT_THRESH', 2.5)
        
        if float(self._header['EXPTIME']) >= 180:
            detect_params['DETECT_MINAREA'] = detect_params.get('DETECT_MINAREA', 9)
            detect_params['ANALYSIS_THRESH'] = detect_params.get('ANALYSIS_THRESH', 1.6)
            detect_params['DETECT_THRESH'] = detect_params.get('DETECT_THRESH', 1.6)
            
        return detect_params
    

    def extract_sources(self, detect_params, overwrite=False,
                        show_info=True):
        """
        Source extraction from FITS file.

        It parse input parameters and set best parameters depending on
        FITS exposure time ('EXPTIME') in order to improve SExtractor 
        source detection.

        Args:
            detect_params (dict): valid SExtractor input parameters.
            overwrite (bool): if True, ouput SExtractor catalog is overwritten.
            show_info (bool): If True, additional processing info is produced.

        Returns:
            int: 0, if everythong was fine.

        Raises:
            Exception: type depends on failing line of code.

        """
        
        # Preconfigured values depending on exptime
        def_params = self.fwhm_def_params()
        
        # If not explicitily set, they are append to input argument called 'detect_params'
        for k, v in def_params.items():
            if k not in detect_params:
                detect_params[k] = v
                
        com = ["sex"]

        out_dirs = list()

        # Mandatory KEYWORDS
        mandatory_keywords = ['CONFIG_FILE', 'CATALOG_NAME']
        for k in mandatory_keywords:
            if k not in detect_params.keys():
                print(f"ERROR: Mandatory keyword '{k}' not found in input parameter list.")
                print("ABORTING execution!!")
                return 1

        if os.path.exists(detect_params['CATALOG_NAME']) and not overwrite:
            print('INFO: Output catalog "%s" already found. Nothing done.' %
                  detect_params['CATALOG_NAME'])
            return 0

        # Mandatory KEYWORDS
        com.append('-c %s' % detect_params["CONFIG_FILE"])
        com.append('-CATALOG_NAME %s' % detect_params["CATALOG_NAME"])

        detection_keywords = "-CHECKIMAGE_TYPE "
        detection_values = "-CHECKIMAGE_NAME "
        keys = list()
        values = list()

        for k, v in detect_params.items():
            # Special format keywords
            if k in ['BACKGROUND', 'SEGMENTATION', 'APERTURES']:
                keys.append(k)
                values.append(v)
                dire = os.path.split(v)[0]
                if len(dire) and not os.path.exists(dire):
                    out_dirs.append(dire)
            elif k in mandatory_keywords:
                pass # already processed
            else:
                # The others
                com.append(f"-{k} {v}")

        # Keywords that may be present in header
        if 'PIXEL_SCALE' not in detect_params.keys():
            if 'INSTRSCL' in self._header:
                com.append(f"-PIXEL_SCALE {str(self._header['INSTRSCL'])} ")
        if 'SEEING_FWHM' not in detect_params.keys():
            if 'FWHM' in self._header:  # in arcsecs
                fw = float(self._header['FWHM']) * float(self._header['INSTRSCL'])
                com.append(f"-SEEING_FWHM {fw} ")
        if 'MAG_ZEROPOINT' not in detect_params.keys():
            if 'MAGZPT' in self._header:
                com.append(f"-MAG_ZEROPOINT {self._header['MAGZPT']}")
        # Adding special format keyword:value pairs
        if keys:
            com.append("%s %s" % (detection_keywords, ','.join(keys)))
            com.append("%s %s" % (detection_values, ','.join(values)))

        # Creating directories for SExtractor output files
        for odir in out_dirs:
            if not os.path.exists(odir):
                try:
                    os.makedirs(odir)
                except IOError:
                    print(f"ERROR: Couldn't create output directory '{odir}'")
                    raise

        com.append(self._path)  # Last SExtractor parameter

        str_com = ' '.join(com)
        if show_info:
            print(str_com)

        # Executing SExtractor call
        subprocess.Popen(str_com, shell=True).wait()

        return 0

    def plot_pix_sources(self, detect_params, number_brigthest_sources=1000,
                         overwrite=True, color='magenta', show_info=True):
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
        if overwrite:
            if self.extract_sources(detect_params=detect_params,
                                    overwrite=overwrite, show_info=show_info):
                return 1
        data_sextractor = _read_sextractor_catalog(detect_params['CATALOG_NAME'])

        # sorting by ascending MAGNITUDE
        order = data_sextractor['MAG_BEST'].argsort()
        if data_sextractor['X_IMAGE'].size > 0:
            if order.size > number_brigthest_sources:
                order = order[:number_brigthest_sources]
        # Plotting
        wcs = WCS(self.header)
        my_axis = plt.subplot(projection=wcs)
        # The following line makes it so that the zoom level no longer changes,
        # otherwise Matplotlib has a tendency to zoom out when adding overlays.
        my_axis.set_autoscale_on(False)
        # Add three markers at (40, 30), (100, 130), and (130, 60). The facecolor is
        # a transparent white (0.5 is the alpha value).
        my_axis.scatter(data_sextractor['X_IMAGE'][order],
                        data_sextractor['Y_IMAGE'][order], s=100,
                        edgecolor='magenta', facecolor=(1, 1, 1, 0))  # transparent
        out_plot = self.path.replace('.fits', '_pix_detections.png')
        plt.savefig(out_plot, dpi=300)
        plt.close()

        return 0

    def make_region_sextractor(self, detect_params, number_brigthest_sources=1000,
                               overwrite=True, color='magenta', show_info=True):
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
            return 1

        # # Plotting detected SExtractor sources
        # dire, astro_fits_name = os.path.split(astrom_out_fits)
        # aper_image_png = aper_image.replace('.fits', '_aper.png')
        # detections_png = os.path.join(metafiles_dir, astro_png_name)
        # plot(aper_image, aper_image_png,
        #          title='Apertures image for %s' % input_name, colorBar=False)

        # SExtractor catalog
        if show_info:
            print('Detection catalog =', detect_params['CATALOG_NAME'])

        data_sextractor = self.__read_cat(detect_params['CATALOG_NAME'])

        if data_sextractor['X_IMAGE'].size > 0:
            # sorting by ascending MAGNITUDE
            order = data_sextractor['MAG_BEST'].argsort()
            if order.size > number_brigthest_sources:
                order = order[:number_brigthest_sources]
            lines_out = ['# # Region file format: DS9 version 4.1']
            lines_out.append('global color="%s" ' % color +
                             'dashlist=8 3 width=1 ' +
                             'font="helvetica 10 normal roman" select=1 ' +
                             'highlite=1 dash=0 fixed=0 edit=1 move=1 ' +
                             'delete=1 include=1 source=1')
            lines_out.append('physical')
            for x_image, y_image in zip(data_sextractor['X_IMAGE'][order],
                                        data_sextractor['Y_IMAGE'][order]):
                lines_out.append('circle(%(x)d,%(y)d,13) # width=2' %
                                 {'x': int(x_image), 'y': int(y_image)})
            out_region_file = self._path.replace('.fits', '.reg')
            with open(out_region_file, 'w') as fout:
                fout.write("\n".join(lines_out))
            print('INFO: Output region file in "%s"' % out_region_file)
        else:
            return 1

        return 0

    def compute_fwhm(self, detect_params, save=True, overwrite=True,
                     show_info=True):
        """
        It computes Full Width and Half Maximum value for FITS image.

        It executes SExtractor, filter best detections taking into account
           - ellipticity,
           - far from border and isolated 
           - and star flux profile 
        and compute statistics for getting image FWHM.

        Ellipticity and flags are output SExtractor parameters than change if
        minimum number of sources is not reached.

        Args:
            detect_params (dict): valid SExtractor parameters.
            save (bool): If True, FWHM parameters are written into FITS header.
            overwrite (bool): If True, SExtractor is executed again, even 
                previous detection was done.
            show_info (bool): If True, aditional processing info is printed by console.

        Returns:
            dict: As keywords (referred to FWHM) are
                ['MIN', 'MAX', 'MEAN', 'STD', 'MEDIAN', 'MAX_ELLIP', 'FLAG']

        Raises:
            Exception: type depends on failing line of code.

        """
        self._save = save  # This property is used for overwriting FITS info
        if self.extract_sources(detect_params, overwrite, show_info):
            return 1

        field_names = ['X_IMAGE', 'Y_IMAGE', 'MAG_BEST', 'NUMBER',
                       'ELLIPTICITY', 'FLAGS', 'CLASS_STAR', 'FWHM_IMAGE']
        data_sextractor = np.genfromtxt(detect_params['CATALOG_NAME'],
                                        names=field_names)

        cond = np.ones(data_sextractor.shape[0], dtype=bool)
        # data_fwhm = copy.deepcopy(data_sextractor)
        flag_value = data_sextractor['FLAGS'].max()
        ellipticity = data_sextractor['ELLIPTICITY'].max()
        if data_sextractor.shape[0] > 10:
            # Selecting best detections
            ellipticity = 0
            flag_value = 0
            for flag_value in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
                # non-saturated, non-deblended sources
                condition1 = data_sextractor['FLAGS'] == flag_value
                if data_sextractor[condition1].shape[0] > 10:
                    for ellipticity in np.arange(0, 1.1, 0.1):    
                        # circular sources
                        condition2 = data_sextractor['ELLIPTICITY'] < ellipticity
                        if data_sextractor[condition1 & condition2].shape[0] > 2:
                            # there are more than three sources for compute statistics
                            cond = condition1 & condition2
                            break

        if data_sextractor[cond].shape[0] == 0:
            print('WARNING: No values after filtering process.')
            print(f'Detection parameters = {detect_params}')
            return {'MEAN': None,
                    'STD': None,
                    'MIN': None,
                    'MAX': None,
                    'MEDIAN': None,
                    'MAX_ELLIP': ellipticity,
                    'FLAG': flag_value}
        
        if show_info:
            # printing info
            print("Original sources number ->", data_sextractor['X_IMAGE'].size)
            print("Filtered sources number->", data_sextractor[cond]['X_IMAGE'].size)
            print("FWHM_IMAGE", "CLASS_STAR", "ELLIPTICITY")
            for fwhm, class_star, ellip in zip(data_sextractor[cond]['FWHM_IMAGE'],
                                                     data_sextractor[cond]['CLASS_STAR'],
                                                     data_sextractor[cond]['ELLIPTICITY']):
                print(fwhm, '--', class_star, '--', ellip)

        # appending info to FITS Header
        new_cards = list()
        candidateCards = [('SOFTDET', 'SExtractor', 'Source detection software'),
                          ('FWHM', round(data_sextractor[cond]['FWHM_IMAGE'].mean(), 2),
                           'Mean pix FWHM'),
                          ('FWHMSTD', round(data_sextractor[cond]['FWHM_IMAGE'].std(), 2),
                           'Std pix FWHM'),
                          ('FWNSOURC', data_sextractor[cond]['FWHM_IMAGE'].size,
                           'FWHM number of sources used'),
                          ('FWHMFLAG', flag_value, 'SExtractor source FLAG'),
                          ('FWHMELLI', round(ellipticity, 2), 'SExtractor max ELLIP'),
                          ('PIXSCALE', self._header['INSTRSCL'], 'Scale [arcs/pix]')]
        for card in candidateCards:
            if card[0] in self._header:
                self._header[card[0]] = card[1]
            else:
                new_cards.append(card)
                
        # ------------ Writing FWHM computed ---------------------
        self._header.extend(new_cards, end=True)
        self._save = True

        return {'MEAN': data_sextractor[cond]['FWHM_IMAGE'].mean(),
                'STD': data_sextractor[cond]['FWHM_IMAGE'].std(),
                'MIN': data_sextractor[cond]['FWHM_IMAGE'].min(),
                'MAX': data_sextractor[cond]['FWHM_IMAGE'].max(),
                'MEDIAN': np.median(data_sextractor[cond]['FWHM_IMAGE']),
                'MAX_ELLIP': ellipticity,
                'FLAG': flag_value
                }

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
        data_sextractor = _read_sextractor_catalog(self.path.replace('.fits',
                                                                     '.cat'))

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

    
    # ---------------------- Astrometric calibration ----------------------
    @staticmethod
    def mc_pointing_sources(input_file):
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

        sources = pd.read_csv(input_file)
        right_ascencions = list()
        declinations = list()

        for cra, cdec in zip(sources['RA'].values, sources['DEC'].values):
            coordinates = SkyCoord('%s %s' % (cra, cdec), unit=(u.hourangle, u.deg))
            right_ascencions.append(coordinates.ra.deg)
            declinations.append(coordinates.dec.deg)

        sources['RADEG'] = np.array(right_ascencions)
        sources['DECDEG'] = np.array(declinations)

        return sources

    def astrometric_calibration(self, mc_sources_file, detect_params={},
                                num_sources=40, tol_arcs=10, overwrite=False,
                                show_info=True):
        """
        A short description.

        A bit longer description.

        Args:
            variable (type): description

        Returns:
            dict: astrometric fitting parameters (pr None if ti was not possible)

        """
        if overwrite or not os.path.exists(detect_params['CATALOG_NAME']):
            if self.extract_sources(detect_params, overwrite=overwrite):
                return 1

        # Sorting
        cat_sort = detect_params['CATALOG_NAME'].replace('.cat', '_sorted.cat')
        # Sorting by magnitude
        com = "sort -n +2.0 %(c1)s | head -%(ns)d > %(c2)s" % {
            'c1': detect_params['CATALOG_NAME'], 'c2': cat_sort, 'ns': num_sources}
        print('Sorting command ->', com)
        subprocess.Popen(com, shell=True).wait()

        # self.plot_pix_sources(detect_params=detect_params, number_brigthest_sources=40,
        #                            overwrite=False)

        # Blazar sources file
        mc_sources = self.mc_pointing_sources(mc_sources_file)
        # Getting pointing coordinates
        obj = self.header['OBJECT'].split()[0]
        source = None
        # by default ra, dec from File
        right_ascencion, declination = self.header['RA'], self.header['DEC']
        for index, k in enumerate(mc_sources['IAU Name'].values):
            if obj == k:
                source = mc_sources.iloc[index]
                break
        if source is not None:
            print('%s: Changing RA,DEC from FITS (%s, %s) to blazar Catalog (%s, %s)' % (
                source['IAU Name'], str(self.header['RA']), str(self.header['DEC']),
                str(source['RADEG']), str(source['DECDEG'])))
            right_ascencion, declination = source['RADEG'], source['DECDEG']

        # Fixed name for astrometric FITS result
        calfits = self.path.replace('.fits', '_astrocal.fits')

        # First calibration try: using FITS coordinates (RA, DEC)
        print('-- First calibration try: using FITS coordinates ({right_ascencion}, {declination}) --')
        str_com = 'imwcs -vew -d {} -y 3 -p {} -j {} {} -c tmc -t {} -o {} {}'
        com = str_com.format(cat_sort, self.header['INSTRSCL'], right_ascencion,
                             declination, tol_arcs, calfits, self.path)

        print('Executing astrometric calibration command:')
        print(com)
        print('-' * 60)

        # delete previous log files
        for ntry in [1, 2]:
            log = self.path.replace('.fits', '_imwcs_stderr_%d.log' % ntry)
            if os.path.exists(log):
                os.remove(log)
            log2 = self.path.replace('.fits', '_imwcs_stdout_%d.log' % ntry)
            if os.path.exists(log2):
                os.remove(log2)

        stderr_cal_log_file = self.path.replace('.fits', '_imwcs_stderr_1.log')
        stdout_cal_log_file = self.path.replace('.fits', '_imwcs_stdout_1.log')

        with open(stdout_cal_log_file, 'w') as fout:
            with open(stderr_cal_log_file, 'w') as ferr:
                subprocess.Popen(com, shell=True, stdout=fout, stderr=ferr).wait()
        print(f'stderr calibration process log file: {stderr_cal_log_file}')
        print(f'stdout calibration process log file: {stdout_cal_log_file}')

        # Check first astrometric calibration try
        fpars = self.parse_calibration_logs(stderr_cal_log_file,
                                            stdout_cal_log_file)

        calib2 = (fpars is None) or (('NMATCHCAL' in fpars) and (int(fpars['NMATCHCAL']) < 15) and (source is not None))

        if calib2:
            # try FITS central coordinates
            right_ascencion, declination = self.header['RA'], self.header['DEC']
            print('-- SECOND CALIBRATION TRY: FITS coordinates ({right_ascencion}, {declination}) --')
            if show_info:
                print('INFO: Not enough sources used for good calibration results.')
                print('\tTrying MAPCAT coordinates for this OBJECT')
            com = f"imwcs -v -e -d {cat_sort} -w -y 2 -p {self.header['INSTRSCL']} "
            com += f"-j {right_ascencion} {declination} -c tmc -t {tol_arcs} "
            com += f"-o {calfits} {self.path}"

            print('Executing astrometric calibration command:')
            print(f"\t{com}")
            print('-' * 60)
            # Registering result in log files
            stderr_cal_log_file = self.path.replace('.fits', '_imwcs_stderr_2.log')
            stdout_cal_log_file = self.path.replace('.fits', '_imwcs_stdout_2.log')
            with open(stdout_cal_log_file, 'w') as fout:
                with open(stderr_cal_log_file, 'w') as ferr:
                    subprocess.Popen(com, shell=True, stdout=fout, stderr=ferr).wait()

            print(f'stderr calibration process log file: {stderr_cal_log_file}')
            print(f'stdout calibration process log file: {stdout_cal_log_file}')

            fpars = self.parse_calibration_logs(stderr_cal_log_file,
                                                stdout_cal_log_file)

        return fpars

    def parse_calibration_logs(self, log_stderr, log_stdout):
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
        # Log files resulting after astrometric calibration process

        # log_stderr = self.path.replace('.fits', f'_astrocal_imwcs_stderr_{n_try}.log')
        # log_stdout = self.path.replace('.fits', f'_astrocal_imwcs_stdout_{n_try}.log')
        if not os.path.exists(log_stdout):
            print(f'ERROR: No STDOUT calibration log available: \n\t{log_stdout}')
            return None
        if not os.path.exists(log_stderr):
            print(f'ERROR: No STDOUT calibration log available: \n\t{log_stderr}')
            return None
        fit_params = dict()  # output parameters

        # -------------------------- first file (imwcs STDOUT) -----------------------------
        # log_file = self.path.replace('.fits', '_imwcs2.log')
        text_out = open(log_stdout).read()

        # working on regular expressions for extracting calibration info
        scale_rotation = re.findall(r'Arcsec/Pixel=\s+([\d.-]+)\s+([\d.-]+)\s+Rotation=\s+([\d.-]+)\s+degrees', text_out)[0]
        fit_params['CDELT1'],  fit_params['CDELT2'], fit_params['ROTDEG'] = [float(sr) for sr in scale_rotation]

        fit_params['CRVAL1'], fit_params['CRVAL2'], fit_params['EQUINOX'], fit_params['CRPIX1'], fit_params['CRPIX2'] = re.findall(r'Optical axis=\s+([\d.:+-]+)\s+([\d:.+-]+)\s+(J2000)\s+x=\s+([\d.-]+)\s+y=\s+([\d.-]+)', text_out)[0]
        fit_params['CRPIX1'] = float(fit_params['CRPIX1'])
        fit_params['CRPIX2'] = float(fit_params['CRPIX2'])

        sources = ['2mass_id,ra2000,dec2000,MagJ,X,Y,magi,dra,ddec,sep']
        matches = re.findall("([\d.]+)\s+([\d.:]+)\s+([\d.:-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)", text_out)
        for m in matches:
            sources.append(','.join(list(m)))
        data_out = StringIO('\n'.join(sources))
        fit_params['SOURCES'] = pd.read_csv(data_out, sep=",")

        res = re.findall(r"dxy=\s+([\d.]+)", text_out)
        fit_params['DISTARCS'] = res[0]

        res = re.findall(r'nmatch=\s+(\d+)\s+nstars=\s+(\d+) between tmc and \S+\s+niter=\s+(\d+)', \
            text_out)
        fit_params['NMATCH'], fit_params['NSTARS'], fit_params['NITER'] = [int(r) for r in res[0]]

        # ---------------------------- Second file (imwcs STDERR) ---------------------------
        text_err = open(log_stderr).read()

        patterns = {
            'CTYPE1': r'CTYPE1\s+=\s+(\S+)',
            'CTYPE2': r'CTYPE2\s+=\s+(\S+)',
            'CRVAL1': r'CRVAL1\s+=\s+([\d.-]+)',
            'CRVAL2': r'CRVAL2\s+=\s+([\d.-]+)',
            'CRPIX1': r'CRPIX1\s+=\s+([\d.-]+)',
            'CRPIX2': r'CRPIX2\s+=\s+([\d.-]+)',
            'CD1_1': r'CD1_1\s+=\s+([\d.-]+)',
            'CD1_2': r'CD1_2\s+=\s+([\d.-]+)',
            'CD2_1': r'CD2_1\s+=\s+([\d.-]+)',
            'CD2_2': r'CD2_2\s+=\s+([\d.-]+)'
        }
        for key, pattern in patterns.items():
            if key in ['CTYPE1', 'CTYPE2']:
                fit_params[key] = re.findall(pattern, text_err)[-1]
            else:
                fit_params[key] = float(re.findall(pattern, text_err)[-1])
        
        return fit_params

    def get_astroheader(self):

        astro_keys = ['CRVAL1', 'CRVAL2', 'EPOCH', 'CRPIX1', 'CRPIX2', 'SECPIX', \
            'SECPIX1', 'SECPIX2', 'CDELT1', 'CDELT2', 'CTYPE1', 'CTYPE2', \
            'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'WCSRFCAT', 'WCSIMCAT', \
            'WCSMATCH', 'WCSNREF', 'WCSTOL', 'RA', 'DEC', 'EQUINOX', \
            'CROTA1', 'CROTA2', 'WCSSEP', 'IMWCS']

        return {k:self.header[k] for k in astro_keys if k in self.header}

        
#     def get_sources(self, source="2MASS", format='csv', show_info=True):
#         """
#         A short description.

#         A bit longer description.

#         Args:
#             variable (type): description

#         Returns:
#             type: description

#         Raises:
#             Exception: description

#         """
#         viz_params = None
#         if source.upper() == "SDSS":
#             viz_params = """<<====End
# -c=%(ra)s %(dec)s
# -c.bm=8.4x8.4
# -out.max=unlimited
# -source=V/139/sdss9
# -c.eq=J2000
# -out=q_mode
# -out=cl
# -out=SDSS9
# -out=m_SDSS9
# -out=Im
# -out=RA_ICRS
# -out=DE_ICRS
# -out=ObsDate
# -out=Q
# -out=umag
# -out=e_umag
# -out=gmag
# -out=e_gmag
# -out=rmag
# -out=e_rmag
# -out=imag
# -out=e_imag
# -out=zmag
# -out=e_zmag
# -out=zsp
# -oc.form=d
# ====End""" % {'ra': self._header['CRVAL1'], 'dec': self._header['CRVAL2']}
#         elif source.upper() == "USNO":
#             viz_params = """<<====End
# -c=%(ra)s %(dec)s
# -oc.form=dec
# -out.max=unlimited
# -order=I
# -c.eq=J2000
# -c.r=8
# -c.u=arcmin
# -c.geom=r
# -out.src=I/252/out
# -source=I/252/out
# -out.orig=standard
# -out=USNO-A2.0
# -out=RAJ2000
# -out=DEJ2000
# -out=ACTflag
# -out=Mflag
# -out=Bmag
# -out=Rmag
# -out=Epoch
# ====End""" % {'ra': self._header['CRVAL1'], 'dec': self._header['CRVAL2']}
#         elif source.upper() == "2MASS":
#             viz_params = """<<====End
# -c=%(ra)s %(dec)s
# -oc.form=dec
# -out.max=unlimited
# -order=I
# -out.src=II/246/out
# -c.eq=J2000
# -c.r=8
# -c.u=arcmin
# -c.geom=r
# -source=II/246/out
# -out.orig=standard
# -out=RAJ2000
# -out=DEJ2000
# -out=2MASS
# -out=Jmag
# -out=e_Jmag
# -out=Hmag
# -out=e_Hmag
# -out=Kmag
# -out=e_Kmag
# -out=Qflg
# -out=Rflg
# -out=Bflg
# -out=Cflg
# -out=Xflg
# -out=Aflg
# ====End""" % {'ra': self._header['CRVAL1'], 'dec': self._header['CRVAL2']}
#         else:
#             print(f"Unknown catalog name '{source}'")
#             return None

#         com = 'vizquery -mime={format} {viz_params}'

#         out_catalog = self._path.replace('.fits', f'_{source}.{format}')
#         fout = open(out_catalog, 'w')
#         subprocess.Popen(com, shell=True, stdout=fout, stderr=fout).wait()
#         fout.close()
#         if show_info:
#             print(com)
#             print('Output catalog available in "%s"' % out_catalog)

#         return 0

#     def read_viz_cat(self, source='2MASS', format='csv', show_info=True):
#         """
#         A short description.

#         A bit longer description.

#         Args:
#             variable (type): description

#         Returns:
#             type: description

#         Raises:
#             Exception: description

#         """
#         sources_catalog = self._path.replace('.fits', f'_{source}.{format}')
#         if not os.path.exists(sources_catalog):
#             self.get_sources(source=source, show_info=show_info)

#         # defining primary variables...
#         skip_header = None
#         delimiter = ';'

#         if source.upper() == '2MASS':
#             # Region file for 2MASS catalog
#             head = "RAJ2000;DEJ2000;2MASS;Jmag;e_Jmag;Hmag;e_Hmag;"
#             head += "Kmag;e_Kmag;Qflg;Rflg;Bflg;Cflg;Xflg;Aflg"
#             skip_header = 66
#         elif source.upper() == 'SDSS':
#             head = "q_mode;cl;SDSS9;m_SDSS9;Im;RA_ICRS;DE_ICRS;ObsDate;Q;umag;"
#             head += "e_umag;gmag;e_gmag;rmag;e_rmag;v;e_imag;zmag;e_zmag;zsp"
#             skip_header = 69
#         elif source.upper() == 'USNO':
#             head = "USNO-A2.0;RAJ2000;DEJ2000;ACTflag;Mflag;Bmag;Rmag;Epoch"
#             skip_header = 52
#         else:
#             print("Unknown catalog name '%s'" % source)
#             return 1

#         # fnames = head.replace(delimiter, ',')
#         # out_cat = self._path.replace(f'_{source}', '')
#         vizdata = np.genfromtxt(sources_catalog, skip_header=skip_header,
#                                 skip_footer=1,
#                                 names=head.replace(delimiter, ','),
#                                 delimiter=delimiter, missing_values="-999",
#                                 comments='#', autostrip=True)
#         # Location keywords homogeneization
#         if source == 'SDSS':
#             vizdata['RAJ2000'] = copy.deepcopy(vizdata['RA_ICRS'])
#             vizdata['DEJ2000'] = copy.deepcopy(vizdata['DE_ICRS'])
#             del(vizdata['RA_ICRS'])
#             del(vizdata['DE_ICRS'])

#         return vizdata

#     def make_vizquery_region(self, source="2MASS", format='csv',
#                              number_brigthest_sources=50, color='magenta',
#                              show_info=True):
#         """
#         A short description.

#         A bit longer description.

#         Args:
#             variable (type): description

#         Returns:
#             type: description

#         Raises:
#             Exception: description

#         """
#         vizdata = self.read_viz_cat(source=source, format='csv',
#                                     show_info=show_info)
#         if not vizdata.size:
#             print(f'Input source "{source}" not valid.')
#             return 1

#         key_mag = None
#         if source.upper() == '2MASS':
#             key_mag = 'Jmag'
#         elif source.upper() == 'SDSS':
#             key_mag = 'rmag'
#         elif source.upper() == 'USNO':
#             key_mag = 'Rmag'

#         if vizdata['RAJ2000'].size > 0:
#             # sorting sources catalog by magnitude
#             index = vizdata[key_mag].argsort()
#             if vizdata['RAJ2000'].size > number_brigthest_sources:
#                 brightest_ra = vizdata['RAJ2000'][index[:number_brigthest_sources]]
#                 brightest_dec = vizdata['DEJ2000'][index[:number_brigthest_sources]]
#             else:
#                 brightest_ra = vizdata['RAJ2000']
#                 brightest_dec = vizdata['DEJ2000']
#             lines_out = ['# Region file format: DS9 version 4.0']
#             lines_out.append('global color="%s" ' % color +
#                              'font="helvetica 10 normal roman" edit=1 ' +
#                              'move=1 delete=1 highlite=1 include=1 wcs=wcs')
#             for right_ascension, declination in zip(brightest_ra, brightest_dec):
#                 lines_out.append(f'fk5;circle({right_ascension},{declination},5")')
#             # writing ouptu region file
#             sources_catalog = self._path.replace('.fits',
#                                                  f'_{source}.{format}')
#             out_reg = sources_catalog.replace('.csv', '.reg')
#             with open(out_reg, 'w') as fout:
#                 fout.write("\n".join(lines_out))
#             print(f'INFO: Output region file in "{out_reg}"')
#         else:
#             return 2

#         return 0

#     #  ---------------- PHOTOMETRIC CALIBRACION -------------------
#     def photometric_calibration(self, mapcat_catalog, detect_params,
#                                 max_dist_arcs=5, overwrite=True,
#                                 show_info=True):
#         """
#         A short description.

#         A bit longer description.

#         Args:
#             variable (type): description

#         Returns:
#             type: description

#         Raises:
#             Exception: description

#         """
#         self.extract_sources(detect_params=detect_params,
#                              overwrite=overwrite, show_info=show_info)

#         # # Plotting detected SExtractor sources
#         # dire, astro_fits_name = os.path.split(astrom_out_fits)
#         # aper_image_png = aper_image.replace('.fits', '_aper.png')
#         # detections_png = os.path.join(metafiles_dir, astro_png_name)
#         # plot(aper_image, aper_image_png,
#         #          title='Apertures image for %s' % input_name, colorBar=False)

#         # SExtractor catalog
#         if os.path.splitext(detect_params['CATALOG_NAME'])[1] == '.fits':
#             hdul = fits.open(detect_params['CATALOG_NAME'])
#             # assuming the first extension is a table
#             data_sextractor = hdul[2].data
#             # cols = hdul[2].columns
#             # cols.info()
#         else:
#             lines = open(detect_params['CATALOG_NAME'], 'r').readlines(size=50)
#             fields = list()
#             for line in lines:
#                 if line.startswith('#'):
#                     fields.append(line.split()[2])
#                 else:
#                     break
#             data_sextractor = np.genfromtxt(detect_params['CATALOG'],
#                                             names=fields)
#         # data_sextractor = data_sextractor[data_sextractor['FLAGS'] == 0]
#         # data_sextractor['MAG_AUTO']
#         # print("Clean SExtractor detections = ", data_sextractor['MAG_AUTO'].size)

#         # ---------- Reading MAPCAT sources -----------------
#         mapcat_data = pd.read_csv(mapcat_catalog, comment="#")

#         # Transforming astronomical coordinates: from sexagesimal to decimal degrees
#         dra2000 = list()
#         ddec2000 = list()
#         for right_ascencion, declination in zip(mapcat_data['ra2000'], mapcat_data['dec2000']):
#             coordinates = SkyCoord('%s %s' % (right_ascencion, declination),
#                                    unit=(u.hourangle, u.deg))
#             dra2000.append(coordinates.ra.deg)
#             ddec2000.append(coordinates.dec.deg)

#         # Creating arrays of coordinates
#         mapcat_data['dra2000'] = np.array(dra2000)
#         mapcat_data['ddec2000'] = np.array(ddec2000)

#         # Cross-matching catalogs
#         coo_photoCat = SkyCoord(mapcat_data['dra2000'] * u.deg,
#                                 mapcat_data['ddec2000'] * u.deg)
#         coo_sexCat = SkyCoord(data_sextractor['ALPHA_J2000'] * u.deg,
#                               data_sextractor['DELTA_J2000'] * u.deg)
#         if show_info:
#             print("Size of BLAZARs catalog = ", mapcat_data['dra2000'].size)
#             print("Size of SExtractor catalog = ",
#                   data_sextractor['ALPHA_J2000'].size)

#         idx, d2d, d3d = coo_sexCat.match_to_catalog_sky(coo_photoCat)
#         # idx = Indices into coo_photoCat to get the matched points for each
#         #    of this object’s coordinates.
#         # d2d = The on-sky separation between the closest match for each
#         #    element in this object in coo_photoCat.
#         # d3d = The 3D distance between the closest match for each element in
#         #    this object in coo_photoCat

#         if show_info:
#             print('Minimum distance sources = %6.2f arcsec (%6.2f armin)' %
#                   (d2d.arcsec.min(), d2d.arcsec.min() / 60))
#         # Checking closest sources
#         if (d2d.arcsec <= max_dist_arcs).sum() == 0:
#             print("$" * 100)
#             print("WARNING: There are no calibration sources close enough" +
#                   " to image detections.\nImpossible photometric calibration!!")
#             print("\t- Closest source found at %f arcsecs -" % d2d.arcsec.min())
#             print("$" * 100)
#             return 1

#         if show_info:
#             print("Size of matching sources result = ", idx.size)

#         png_hist_match = self._path.replace('.fits', '_histMatchSources.png')
#         plt.hist(d2d.arcsec, histtype='step', range=(0, max_dist_arcs))
#         plt.xlabel('Distance [arcsec]')
#         plt.ylabel('Sources number [#]')
#         plt.tight_layout()
#         plt.savefig(png_hist_match, dpi=300)
#         plt.close()

#         # print(SkyCoord(data['dra2000'][0]*u.deg, data['ddec2000'][0]*u.deg))
#         # print(SkyCoord(data['dra2000'][0], data['ddec2000'][0], unit=u.deg))

#         # Filtering closest matches between photometrical reference stars
#         # and SExtractor detections (distance lower than 5 arsec)
#         index_sextractor = list()
#         index_photo_cal = list()
#         blazar_rmag = mapcat_data['Rmag'].values
#         blazar_rmagerr = mapcat_data['Rmagerr'].values
#         for index, value in enumerate(zip(idx, d2d)):
#             # print(val[0])
#             if (value[1].arcsec < max_dist_arcs) and (blazar_rmag[value[0]] >= 0):
#                 print(value[0], blazar_rmag[value[0]], value[1].arcsec)
#                 index_sextractor.append(index)
#                 index_photo_cal.append(value[0])
#                 print("\t(SExtractor index, Blazard index) = (%d, %d)" % (index, value[0]))
#             # Plotting MAPCAT sources
#         # Checking plot
#         png_photo_cal_sources = self.path.replace('.fits', '_MAPCAT_photoCal_sources.png')
#         print('Cross-matching catalogs in PNG ->', png_photo_cal_sources)
#         self.plot('MAPCAT Sources',
#                  coords=(mapcat_data['dra2000'][index_photo_cal],
#                          mapcat_data['ddec2000'][index_photo_cal]),
#                  parameters={'aspect': 'auto', 'invert': 'True'})

#         mag_sex = data_sextractor['MAG_AUTO'][index_sextractor].astype(np.float)
#         magerr_sex = data_sextractor['MAGERR_AUTO'][index_sextractor].astype(np.float)
#         mag_cal = blazar_rmag[index_photo_cal].astype(np.float)
#         magerr_cal = blazar_rmagerr[index_photo_cal].astype(np.float)
#         if show_info:
#             keywords = ['MAG_AUTO', 'MAGERR_AUTO', 'MAG_CAL', 'magerr_cal']
#             print('  '.join(keywords))
#             for index in mag_sex.size:
#                 print(mag_sex[index], magerr_sex[index], mag_cal[index],
#                       magerr_cal[index])
#         magerr_cal[magerr_cal < 0] = 0  # if no error available, it's set to zero

#         # Fitting photometric measures for calibrating input_fits
#         # linear fitting with slope = 1
#         popt, pcov = optimize.curve_fit(_linear_fit, mag_sex, mag_cal)
#         if show_info:
#             print('SLOPE 1 linear fitting results', popt, pcov)

#         # Add FITS keyword for ZEROPOINT photometric calibration
#         # Now copy header to reducted FITS
#         new_cards = [('MAGZPT', popt[0], 'MAGNITUDE ZEROPOINT')]
#         self._header.extend(new_cards, strip=False, end=True)
#         self._save = True

#         # Plotting fit
#         plt.errorbar(mag_sex, mag_cal, xerr=magerr_sex, yerr=magerr_cal,
#                      fmt="r.")
#         plt.xlabel('MAG SExtractor')
#         plt.ylabel('MAG R filter')
#         plt.title('MAG R = MAG SEXTRACTOR + %6.2f' % popt[0])
#         png_photo_calibration = self._path.replace('.fits', '_photoFit.png')
#         plt.savefig(png_photo_calibration, dpi=300)
#         plt.close()

#         # Other fitting routines
#         # Fitting routines
#         print('Other fitting routines (linear with y-axis magnitude errors)')
#         xfit = smapi.add_constant(mag_sex)
#         yfit = mag_cal
#         weights = 1. / (magerr_cal ** 2)
#         print(xfit.shape, yfit.shape, weights.shape)
#         wls = smapi.WLS(yfit, xfit, weights)
#         results = wls.fit(cov_type='fixed scale')
#         print('Summary fitting:', results.summary())
#         print('\t\tp-values:', results.pvalues)
#         print('\t\tt-values:', results.tvalues)
#         print('\t\tResulting parameters:', results.params)
#         print('\t\tConfidence intervals:', results.conf_int())
#         print('\t\t-' * 30)

#         # Plotting linear fit
#         png_plotfit2 = self._path.replace('.fits', '_photoFit_method2.png')
#         plt.errorbar(mag_sex, mag_cal, xerr=magerr_sex, yerr=magerr_cal,
#                      fmt="r.", label='Exp. mags')
#         # plt.plot(mag_sex, results.params.const + results.params.x1 * mag_sex,
#         # 'g-', label='linear fit')
#         plt.plot(mag_sex, results.predict(), 'g-', label='linear fit')
#         plt.xlabel('MAG SExtractor')
#         plt.ylabel('MAG R filter')
#         plt.legend()
#         plt.title('$R^2=%4.3f$' % results.rsquared)
#         plt.savefig(png_plotfit2, dpi=300)
#         plt.close()

#         return 0
