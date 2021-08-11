#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains routines for astronomy reduction FITS.

Module created for working on generic astronomy reduction process.
Based on nightly reductions, a image directory path is enough for beginning the process.

It allows
* Classifying images based on FITS header keywords.
 * Generate SUPERFLATs based on median of each image pixels.
 * Generate SUPERBIAS by median combination of BIAS images.
 * Generate cleaned images for science images.

Created on Mon March 30 16:51:58 2020.

___e-mail__ = cesar_husillos@tutanota.com
__author__ = 'Cesar Husillos'

VERSION:
    0.1 Initial version
"""


# ---------------------- IMPORT SECTION ----------------------
import os
import argparse
import glob
import subprocess
import math
import pickle
import pprint
import datetime
import re

#import warnings

# Numerical packages
import numpy as np
import pandas as pd
import statsmodels.api as sm # Linear fitting module
from scipy import optimize

# Plotting packages
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
#import seaborn

# HTML templates packages for documentation and logging
import jinja2 # templates module

# Astronomical packages
import aplpy # FITS plotting library
from astropy.io import fits # FITS library
from astropy.io.fits import Header

# Coordinate system transformation package and modules
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
import astropy.units as u
import astropy.wcs as wcs
#from PyAstronomy import pyasl # for changing coordinates format

from mcFits import mcFits

class mcReduction:

    def __init__(self, input_dir, out_dir, border=0):
        self.input_dir = input_dir
        self.out_dir = out_dir
        self.border = border # image border not considered
        self.date = re.findall('MAPCAT_(\d{4}-\d{2}-\d{2})', self.input_dir)[0]
        self.path_info_fits = None
        self.info_fits = None
        self.bias = None
        self.masterBIAS = None
        self.masterFLAT = dict() # one masterBIAS for each polarization angle
        self.flats = None
        self.science = None
        self.template = None
        self.process = None

        self.__getInfo()
        self.__classify()


    def __getInfo(self):
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

        # Getting input_dir FITS
        filelist = glob.glob(os.path.join(self.input_dir, '*.fits'))
        filelist.sort()
        pathFileList = os.path.join(self.input_dir, 'Filelist.txt')
        if len(filelist) > 0:
            print("Number of FITS files to process =", len(filelist))
            with open(pathFileList, 'w') as fout:
                fout.write("\n".join(filelist) + "\n")
        else:
            print("ERROR: No FITS file found in directory ", self.input_dir)
            return 5

        # Keywords for recovering useful information
        keywords = ['OBJECT', 'EXPTIME', 'INSPOROT', 'NAXIS1', 'NAXIS2',
                    'RA', 'DEC', 'MJD-OBS', 'DATE-OBS', 'IMAGETYP']
        keyFile = os.path.join(self.input_dir, 'Keywordlist.txt')
        with open(keyFile, 'w') as fout:
            fout.write("\n".join(keywords) + "\n")

        # WCSTools command for getting information about FITS image list
        # Directory naming convention: MAPCAT_YYMMDD
        strDate = os.path.split(self.input_dir)[-1].split("_")[-1]
        infoFitsFile = os.path.join(self.input_dir, strDate + '.cat')
        com = 'gethead -jh -n 10 -t @%(fl)s @%(kl)s > %(inf)s' % {
            'fl': pathFileList, 'kl': keyFile, 'inf': infoFitsFile}
        print("File with information about each FITS file ->", infoFitsFile)
        print(com)
        # Executing command
        subprocess.Popen(com, shell=True).wait()
        self.path_info_fits = infoFitsFile

        return 0

    def __classify(self):
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

        df = pd.read_csv(self.path_info_fits, sep='\t')[1:]
        # sorting by field 'FILENAME'
        self.info_fits = df.sort_values('FILENAME') # FILENAME contains image date

        # Adding absolute route for FILENAME field
        filenames = list()
        for fn in self.info_fits['FILENAME'].values:
            filenames.append(os.path.join(self.input_dir, fn))
        self.info_fits['FILENAME'] = filenames
        # Raw input images classification by their OBJECT names
        procOBJ = list()
        for index, row in self.info_fits.iterrows():
            if row['OBJECT'].find('bias') != -1:
                procOBJ.append('bias')
            elif row['OBJECT'].find('flat') != -1:
                procOBJ.append('flat')
            else:
                toks = row['OBJECT'].split()
                if len(toks) == 3: # TODO: review this condition
                    procOBJ.append(toks[0])
                else:
                    procOBJ.append(row['OBJECT'])
        # compute statistics for each FITS file
        stat_keys = ['MIN', 'MAX', 'MEAN', 'STD', 'MEDIAN']
        stats = {k:list() for k in stat_keys}
        for fn in self.info_fits['FILENAME'].values:
            ofits = mcFits(fn, border=self.border)
            inf = ofits.stats()
            for k in stat_keys:
                stats[k].append(inf[k])
        # appending a new column to original dataframe
        self.info_fits['procOBJ'] = pd.array(procOBJ)
        for k in ['MIN', 'MAX', 'MEAN', 'STD', 'MEDIAN']:
            self.info_fits[k] = pd.array(stats[k])

        # Classifying
        self.bias = self.info_fits[self.info_fits['procOBJ'] == 'bias']
        self.flats = self.info_fits[self.info_fits['procOBJ'] == 'flat']
        bflats = self.info_fits['procOBJ'] == 'flat'
        bbias = self.info_fits['procOBJ'] == 'bias'
        dfbool = ~(bflats | bbias)
        self.science = self.info_fits[dfbool]

        return

    def createMasterBIAS(self, show_info=True):
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

        """Function for Master BIAS generation. If operation='mean', mean BIAS
        will be computed. Median BIAS in other case.
        It returns 0 if everything was fine. An Exception in the other case."""
        bias_data = list()
        for fn in self.bias['FILENAME'].values:
            ofits = mcFits(fn, border=self.border)
            print('Bias file "%s" INFO -> (Min, Max, Mean, Std, Median, dtype) ='
                  % fn, ofits.stats())
            bias_data.append(ofits.data)

        matrix_bias = np.array(bias_data)
        if show_info:
            print("Number of input images ->", len(bias_data))
            print("Internal matrix shape ->", matrix_bias.shape)

        # Median BIAS computation
        mmat = np.median(matrix_bias, axis=0)
        if self.border > 0:
            # Area out of borders given by 'borderSize' are set to zero
            mmat[:self.border, :] = 0
            mmat[-self.border:, :] = 0
            mmat[:, :self.border] = 0
            mmat[:, -self.border:] = 0

        if show_info:
            print("final matrix shape ->", mmat.shape)

        # new header tags
        #hbias['DATAMAX'] = mmat.astype(np.int32).max()
        #hbias['DATAMIN'] = mmat.astype(np.int32).min()
        #hbias['DATAMEAN'] = mmat.astype(np.int32).mean()

        #hbias.remove('BLANK', ignore_missing=True, remove_all=False)
        #hbias['BZERO'] = '0.0'
        #hbias['OBJECT'] = 'Master BIAS'
        #hbias['BITPIX'] = 32 # 32-bit twos-complement binary integer
        if self.border > 0:
            inner_mmat = mmat[self.border:-self.border, self.border:-self.border]
        else:
            inner_mmat = mmat
        
        newCards = [('SOFT', 'IOP^3 Pipeline v1.0', 'Software used'),
                    ('PROCDATE', datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                                 'Master BIAS writing date'),
                    ('PXBORDER', self.border , 'Pixel border size'),
                    ('BIASOP', 'median', 'Operation used for combination'),
                    ('DIRRAW', self.input_dir, 'Raw images dir'),
                    ('DIROUT', self.out_dir, 'MasterBIAS root dir'),
                    ('MAX', inner_mmat.max(), 'Max value'),
                    ('MIN', inner_mmat.min(), 'Min value'),
                    ('MEAN', round(inner_mmat.mean(), 2), 'Mean value'),
                    ('STD', round(inner_mmat.std(), 2), 'Std value'),
                    ('MED', round(np.median(inner_mmat), 2), 'Median value'),
                    ('OBJECT', 'Master BIAS', '')
                   ]
        for ind, val in enumerate(self.bias['FILENAME'].values):
            newCards.append(('BIAS%d' % ind, os.path.split(val)[1],
                             'FITS combined in MasterBIAS'))
        #hbias.extend(newCards, strip=False, end=True)

        hdr = fits.Header()
        hdr.extend(newCards, strip=False, end=True)

        #mmat2 = mmat - int(hbias['BZERO'])
        #hdu = fits.PrimaryHDU(data=mmat2.astype(np.int16), header=hbias)
        hdu = fits.PrimaryHDU(data=mmat.astype(np.uint16), header=hdr)
        dateOBS = self.input_dir.split('_')[-1]
        self.masterBIAS = os.path.join(self.out_dir, 'masterBIAS_{}.fits'.format(self.date))
        hdu.writeto(self.masterBIAS, overwrite=True)

        return 0


    def createMasterFLAT(self, show_info=True):
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

        """Function for creating MasterFlat image.
        Return 0 if everything was fine. Raise exception in the other case."""
        if show_info:
            print('\tNumber of flats =', len(self.flats['FILENAME'].values))

        # Getting Master BIAS data
        oMB = mcFits(self.masterBIAS, border=self.border)

        if show_info:
            print("\tMaster BIAS info -> (Min, Max, Mean, Std, Median, dtype) =",
                  oMB.stats())

        # getting polarization Angles
        pol_angles = self.flats['INSPOROT'].unique()
        print('Polarization angle availables ->', pol_angles)

        # One masterFLAT for each polarization angle
        for pa in pol_angles:
            # filtering and combining FLATS for each polarization angle
            print('\n+++++++++++++++ Working on polarization angle -> %s'
                  % pa, "+" * 15, "\n")
            # selecting flats with this polarization angle
            dff = self.flats[self.flats['INSPOROT'] == pa]
            if len(dff.index) == 0:
                print("WARNING: Not found FITS for polarization angle = %s" % pa)
                continue

            print('\tNumber of flats =', len(dff.index))

            #Collecting data FLATS...
            flat_data = list()
            for fn in dff['FILENAME'].values:
                oflat = mcFits(fn, border=self.border)
                # substracting BIAS operation for each FLAT raw image
                matres = 1.0 * oflat.data - oMB.data
                # negative values are non-sense physically
                matres = np.where(matres < 0, 0, matres)
                flat_data.append(matres.astype(np.uint16))
                # changing array list to hiper-array
            matrix_flats = np.array(flat_data)

            # computing median for each pixel coordinates
            mmat = np.median(matrix_flats, axis=0)

            # inner area of median flat matrix
            inner_mmat = mmat + 0 # deep matrix copying
            if self.border > 0:
                inner_mmat = mmat[self.border:-self.border, self.border:-self.border]
                maxinner = inner_mmat.max()
                # Area out of borders given by 'borderSize' are set to zero
                mmat[:self.border, :] = maxinner
                mmat[-self.border:, :] = maxinner
                mmat[:, :self.border] = maxinner
                mmat[:, -self.border:] = maxinner
            
            if show_info:
                print("\t\tfinal matrix shape ->", mmat.shape)

            # last flat: header will be used in masterFLAT
            oflat = mcFits(dff['FILENAME'].values[-1], border=self.border)

            newCards = [('SOFT', 'IOP^3  Pipeline v1.0', 'Software used'),
                        ('PROCDATE', datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                                     'Master FLAT writing date'),
                        ('PXBORDER', self.border , 'Pixel border size'),
                        ('MBIAS', self.masterBIAS, 'MasterBIAS employed'),
                        ('FLATOP', 'median', 'Operation used for combination'),
                        ('DIRRAW', self.input_dir, 'Raw images Directory'),
                        ('DIROUT', self.out_dir, 'MasterFLAT root directory'),
                        ('INSPOROT', pa, 'rotator position'),
                        ('MAX', inner_mmat.max(),'Max value'),
                        ('MIN', inner_mmat.min(),'Min value'),
                        ('MEAN', round(inner_mmat.mean(), 2),'Mean value'),
                        ('STD', round(inner_mmat.std(), 2),'Std value'),
                        ('MED', round(np.median(inner_mmat), 2),'Median value'),
                        ('OBJECT', 'Master FLAT','')
                        ]
            for ind, val in enumerate(dff['FILENAME'].values):
                newCards.append(('FLAT%d' % ind, os.path.split(val)[1],
                                 'FITS combined in MasterFLAT'))
            hdr = fits.Header()
            hdr.extend(newCards, strip=False, end=True)
            # saving median array
            hdures = fits.PrimaryHDU(data=mmat.astype(np.uint16), header=hdr)
            # MasterFLAT path
            nameFLAT = "flt_{}_{:03.1f}".format(self.date,float(pa))
            masterFLATPath = os.path.join(self.out_dir, "{}.fits".format(nameFLAT))
            self.masterFLAT[round(float(pa), 1)] = masterFLATPath
            hdures.writeto(masterFLATPath, overwrite=True)

        return 0


    def reduce(self, show_info=True):
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

        """This function reduce 'image' using 'bias' and 'flat' images.
        It returns an error code:
            0 if everything was fine.
        An Exception is raised any other case."""

        # bias
        oBIAS = mcFits(self.masterBIAS, border=0)

        # ALL SCIENCE FITS Reduction
        for sciFITS in self.science['FILENAME'].values:
            print("{0} Working on '{1}' {0}".format('-' * 6, sciFITS))
            # image
            oSCIENCE = mcFits(sciFITS, border=0)
            # External border of image
            #frameOfImage = np.zeros(oSCIENCE.data.shape, dtype=np.bool)
            # internal part mask
            #frameOfImage[self.border:-self.border, self.border:-self.border] = True

            if show_info:
                message = "Polarization angle set to -> {}"
                print(message.format(oSCIENCE.header['INSPOROT']))
            if float(oSCIENCE.header['INSPOROT']) > 70:
                str_err = 'ERROR: filter value not valid'
                print(str_err)
                continue
            # flat
            flat = self.masterFLAT[round(float(oSCIENCE.header['INSPOROT']), 1)]
            oFLAT = mcFits(flat, border=0)
            data_flat_norm = 1.0 * oFLAT.data / oFLAT.data.max()

            # main processing image routine
            try: 
                data_final = (1.0 * oSCIENCE.data - oBIAS.data) / data_flat_norm
            except ValueError:
                print(f"ERROR: while reducing '{sciFITS}'")
                continue

            # Negative values are not considered...
            scaled = np.where(data_final < 0, 0, data_final)

            # scaling image to UINT16. Inner max value are considered.
            #inner_max = data_final[self.border:-self.border, self.border:-self.border].max()
            # scaled = (scaled * (np.iinfo(np.uint16).max)) / data_final.max()
            scaled = np.where(scaled > np.iinfo(np.uint16).max, np.iinfo(np.uint16).max, scaled)

            
            # Out of border zone is set to original science values
            #scaled = np.where(frameOfImage, scaled, oSCIENCE.data)
            #inner_mmat = scaled[self.border:-self.border, self.border:-self.border]
            newCards = [('SOFT', 'IOP^3 Pipeline v1.0', 'Software used'),
                        ('PROCDATE', datetime.datetime.now().isoformat(), 'Reduction process date'),
                        ('PXBORDER', self.border , 'Pixel border size'),
                        ('FLAT', os.path.split(flat)[1], 'MasterFLAT'),
                        ('BIAS', os.path.split(self.masterBIAS)[1], 'Master BIAS'),
                        ('DIRRAW', os.path.split(flat)[0], 'Root Dir'),
                        ('DIROUT', self.out_dir, 'Reduction directory'),
                        ('MAX', round(scaled.max(), 2), 'Max value'),
                        ('MIN', round(scaled.min(), 2), 'Min value'),
                        ('MEAN', round(scaled.mean(), 2), 'Mean value'),
                        ('STD', round(scaled.std(), 2), 'Standar deviation value'),
                        ('MED', round(np.median(scaled), 2), 'Median value')
                        ]

            # ------------ WRITING REDUCED IMAGE ON DISK ---------------------
            #hdr = fits.Header()
            #hdr.extend(redCards + newCards, end=True)
            head = oSCIENCE.header
            head.extend(newCards, end=True)
            hdu = fits.PrimaryHDU(data=scaled.astype(np.uint16), header=head)
            out_image = os.path.join(self.out_dir, os.path.split(sciFITS)[1])
            hdu.writeto(out_image, overwrite=True, output_verify='ignore')

            if show_info:
                print("*" * 50)
                print("INPUT image stats({}): {}".format(oSCIENCE.path, oSCIENCE.stats()))
                oBIAS = mcFits(self.masterBIAS, border=self.border)
                print("BIAS stats ({}): {}".format(oBIAS.path, oBIAS.stats()))
                oFLAT = mcFits(flat, border=self.border)
                print("FLAT stats ({}): {}".format(oFLAT.path, oFLAT.stats()))
                # if self.border > 0:
                #     inner_norm_flat = data_flat_norm[self.border:-self.border,
                #                                      self.border:-self.border]
                # else:
                #     inner_norm_flat = data_flat_norm
                # print("NORMALIZED FLAT \n\t%6.2f, %6.2f, %6.2f, %6.2f, %6.2f %s"
                #       % (inner_norm_flat.min(), inner_norm_flat.max(),
                #          inner_norm_flat.mean(), inner_norm_flat.std(),
                #          np.median(inner_norm_flat), inner_norm_flat.dtype.name))
                oRED = mcFits(out_image, border=0)
                print("OUT image stats ({})".format(out_image, oRED.stats()))
                #inner_data = oRED.data[self.border:-self.border,
                #                       self.border:-self.border]
                print("Scaled (previous to uint casting) ->", scaled.min(),
                      scaled.max(), scaled.mean(), scaled.std(),
                      np.median(scaled), scaled.dtype.name)
                print("Data previous scaling process ->", data_final.min(),
                      data_final.max(), data_final.mean(), data_final.std(),
                      np.median(data_final), data_final.dtype.name)
                print("Num of negative pixel (reduced fits) -> {}".format((oRED.data < 0).sum()))
                print("Num of negative pixels (scaled matrix) ->".format((scaled < 0).sum()))
                print("Num of negative pixels (data_final matrix) ->".format((data_final < 0).sum()))
                print("*" * 50)

        return 0
