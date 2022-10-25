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
import glob
import subprocess
import datetime
import re
from collections import defaultdict

# Numerical packages
import numpy as np
import pandas as pd

from astropy.io import fits # FITS library

from mcFits import mcFits

class mcReduction:

    def __init__(self, input_dir, out_dir, border=0):
        self.input_dir = input_dir
        self.out_dir = out_dir
        self.border = border # image border not considered
        dt_run = re.findall('(\d{6})', self.input_dir)[-1]
        self.date = f'20-{dt_run[:2]}-{dt_run[2:4]}-{dt_run[-2:]}'
        self.path_info_fits = None
        self.info_fits = None
        self.bias = None
        self.masterBIAS = None
        self.masterFLAT = dict() # one masterBIAS for each polarization angle
        self.flats = None
        self.science = None
        self.template = None
        self.process = None
        self.tel_type= None

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
        #Getting telescope type
        self.tel_type=re.findall('(\w+/)', 
                                 self.input_dir)[len(re.findall('(\w+/)', 
                                                                self.input_dir))-1][:-1]
        # Getting input_dir FITS        
        if self.tel_type=='MAPCAT':
            filelist = glob.glob(os.path.join(self.input_dir, '*.fits'))
        else:
            filelist = glob.glob(os.path.join(self.input_dir, '*.fit'))
        filelist.sort()
        pathFileList = os.path.join(self.input_dir, 'Filelist.txt')
        if len(filelist) > 0:
            print(f"Number of FITS files to process = {len(filelist)}")
            with open(pathFileList, 'w') as fout:
                fout.write("\n".join(filelist) + "\n")
        else:
            print("ERROR: No FITS file found in directory ", self.input_dir)
            return 5

        # Keywords for recovering useful information
        keywords = ['OBJECT', 'EXPTIME', 'INSPOROT', 'NAXIS1', 'NAXIS2',
                    'RA', 'DEC', 'MJD-OBS', 'DATE-OBS', 'IMAGETYP', 'FILTER']
        keyFile = os.path.join(self.input_dir, 'Keywordlist.txt')
        with open(keyFile, 'w') as fout:
            fout.write("\n".join(keywords) + "\n")

        # WCSTools command for getting information about FITS image list
        # Directory naming convention: MAPCAT_YYMMDD
        strDate = os.path.split(self.input_dir)[-1].split("_")[-1]
        infoFitsFile = os.path.join(self.input_dir, strDate + '.cat')
        com = 'gethead -jh -n 10 -t @%(fl)s @%(kl)s > %(inf)s' % {
            'fl': pathFileList, 'kl': keyFile, 'inf': infoFitsFile}
        print(com)
        # Executing command
        subprocess.Popen(com, shell=True).wait()
        print(f"File with information about each FITS -> {infoFitsFile}")
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
        print(f"Initial number of FITS = {len(df)}")

        #############################################################
        ## STEP 1: In order to avoid problems, FITS whose dimensions 
        ##     NAXIS1 x NAXIS2
        ## are out of common values for this run are discarded.
        #############################################################

        # checking unique image NAXIS1 value/s
        s_n1 = df['NAXIS1'].value_counts()
        # If more than one matrix dimension was found...
        if len(s_n1) > 1:
            # filter using more repeated naxis1 value
            df = df[df['NAXIS1'] == s_n1.index[s_n1.argmax()]]
        
        # same procedure for NAXIS2 keyword
        s_n2 = df['NAXIS2'].value_counts()
        # If more than one matrix dimension was found...
        if len(s_n2) > 1:
            # filter using more repeated naxis1 value
            df = df[df['NAXIS2'] == s_n2.index[s_n2.argmax()]]

        print(f"Final filtered number of FITS = {len(df)}")
        
        #############################################################
        ## STEP 2: Getting FITS info
        #############################################################

        # sorting by observation datetime
        self.info_fits = df.sort_values('DATE-OBS')

        # Adding absolute route for FILENAME field
        self.info_fits['FILENAME'] = [os.path.join(self.input_dir, fn) \
            for fn in self.info_fits['FILENAME'].values]

        # Raw input images classification by their OBJECT names
        procOBJ = list()
        for index, row in self.info_fits.iterrows():
            obj = row['OBJECT'].lower()
            imtype = row['IMAGETYP'].strip().lower()
            if imtype.find('bias') != -1 or imtype.find('dark') != -1:
                procOBJ.append('bias')
            elif imtype.find('flat') != -1 or obj.find('flat')!= -1 or obj.find('dome') != -1 or imtype.find('dome') != -1:
                procOBJ.append('flat')
            else:
                toks = row['OBJECT'].split()
                if len(toks) == 3: # TODO: review this condition
                    procOBJ.append(toks[0])
                else:
                    procOBJ.append(row['OBJECT'])
        
        # appending a new column to original dataframe
        self.info_fits['procOBJ'] = pd.array(procOBJ)
        
        # compute statistics for each FITS file
        stat_keys = ['MIN', 'MAX', 'MEAN', 'STD', 'MEDIAN']
        stats = defaultdict(list)
        for fn in self.info_fits['FILENAME'].values:
            print(fn)
            ofits = mcFits(fn, border=self.border)
            inf = ofits.stats()
            for k in stat_keys:
                stats[k].append(inf[k])
        
        # and adding FITS data statistics also
        for k in ['MIN', 'MAX', 'MEAN', 'STD', 'MEDIAN']:
            self.info_fits[k] = pd.array(stats[k])

        #############################################################
        ## STEP 3: Classification according to FITS content
        #############################################################
        
        # Classifying
        bflats = self.info_fits['procOBJ'] == 'flat'
        bbias = self.info_fits['procOBJ'] == 'bias'
        bscience = ~(bflats | bbias) # different to flat or bias = science

        self.bias = self.info_fits[bbias]
        self.flats = self.info_fits[bflats]
        self.science = self.info_fits[bscience]

        return {'bias': self.bias, 'flats': self.flats, 'science': self.science}

    def createMasterBIAS(self, show_info=True):
        """
        Class method for generating MasterBIAS.

        It takes FITS classified as 'bias' and combine them using
        median operation over data to get masterBIAS.

        Args:
            show_info (bool): If True, information during process is printed.

        Returns:
            int: 0, if everything was fine.

        Raises:
            Exception: any king depending on failing line of code.

        """

        bias_data = list()
        for fn in self.bias['FILENAME'].values:
            ofits = mcFits(fn, border=self.border)
            if show_info:
                print(f'Bias file "{fn}"')
                print(f'Bias statistics -> {ofits.stats()}')
            bias_data.append(ofits.data)

        matrix_bias = np.array(bias_data)
        if show_info:
            print(f"Number of input images -> {len(bias_data)}")
            print(f"Internal matrix shape -> {matrix_bias.shape}")

        # Median BIAS computation
        mmat = np.median(matrix_bias, axis=0)
        inner_mmat = mmat
        if self.border > 0:
            # Area out of borders given by 'borderSize' are set to zero
            mmat[:self.border, :] = 0
            mmat[-self.border:, :] = 0
            mmat[:, :self.border] = 0
            mmat[:, -self.border:] = 0
            inner_mmat = mmat[self.border:-self.border, self.border:-self.border]
        
        # FITS header
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
            newCards.append((f'BIAS{ind}', os.path.split(val)[1], \
                'FITS combined in MasterBIAS'))
        
        hdr = fits.Header()
        hdr.extend(newCards, strip=False, end=True)

        # Creating and saving new MasterBIAS FITS
        hdu = fits.PrimaryHDU(data=mmat.astype(np.uint16), header=hdr)
        self.masterBIAS = os.path.join(self.out_dir, f'masterBIAS_{self.date}.fits')
        hdu.writeto(self.masterBIAS, overwrite=True)

        return 0


    def createMasterFLAT(self, show_info=True):
        """
        Class method for creating MasterFLAT.

        It takes and combines (mean pixel values) reduced FITS of flat 
        type taking into account polarization angle (INSPOROT keyword).

        Args:
            show_info (bool): If True, process information is printed.

        Returns:
            int: 0, if everything was fine.

        Raises:
            Exception: Any exception type depending on failing line of code.

        """

        if show_info:
            print(f"\tNumber of flats = {len(self.flats['FILENAME'].values)}")

        # Getting Master BIAS data
        oMB = mcFits(self.masterBIAS, border=self.border)

        if show_info:
            print(f"\tMaster BIAS info -> {oMB.stats()}")

        print(self.flats.info)
        # getting polarization Angles
        if self.tel_type=='MAPCAT':
            pol_angles = self.flats['INSPOROT'].unique()
        else:
            pol_angles = self.flats['FILTER'].unique()

        print(f'Available polarization angles -> {pol_angles}')

        # One masterFLAT for each polarization angle
        for pa in pol_angles:
            if show_info:
                print('\n{0} Working on polarization angle -> {1} {0}\n'.format("+" * 15, pa))
            # selecting flats with this polarization angle
            if self.tel_type=='MAPCAT':
                dff = self.flats[self.flats['INSPOROT'] == pa]
            else:
                dff = self.flats[self.flats['FILTER'] == pa]
            if len(dff.index) == 0:
                print(f"WARNING: Not found FITS for polarization angle = {pa}")
                continue

            if show_info:
                print(f'\tNumber of flats = {len(dff.index)}')

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
                # inner_mat contains only the center of the image (statistical purpose)
                inner_mmat = mmat[self.border:-self.border, self.border:-self.border]

                maxinner = inner_mmat.max()
                # Area out of borders given by 'borderSize' are set to zero
                mmat[:self.border, :] = maxinner
                mmat[-self.border:, :] = maxinner
                mmat[:, :self.border] = maxinner
                mmat[:, -self.border:] = maxinner
            
            if show_info:
                print(f"\t\tfinal matrix shape -> {mmat.shape}")

            # last flat: header will be used in masterFLAT
            oflat = mcFits(dff['FILENAME'].values[-1], border=self.border)
            
            try:
                if round(float(pa)) == 360: # some FLATS have INSPOROT = 359.98 as value
                    pa = 0.0
            except:
                print("WARNING: Polarization angle given as string (this is probably OSN)")
                                 
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
                newCards.append((f'FLAT{ind}', os.path.split(val)[1], \
                    'FITS combined in MasterFLAT'))
            
            hdr = fits.Header()
            hdr.extend(newCards, strip=False, end=True)
            
            # saving median array
            hdu_res = fits.PrimaryHDU(data=mmat.astype(np.uint16), header=hdr)
            
            # MasterFLAT path
            if self.tel_type=='MAPCAT':
                name_FLAT = "flt_{}_{:03.1f}".format(self.date,float(pa))
                masterFLAT_path = os.path.join(self.out_dir, f"{name_FLAT}.fits")
                self.masterFLAT[round(float(pa), 1)] = masterFLAT_path
            else:
                name_FLAT = "flt_{}_{}".format(self.date,pa)
                masterFLAT_path = os.path.join(self.out_dir, f"{name_FLAT}.fits")
                self.masterFLAT[pa] = masterFLAT_path
            
            hdu_res.writeto(masterFLAT_path, overwrite=True)

        return 0


    def reduce(self, show_info=True):
        """
        Class method for FITS reduction process.

        This function reduce each classified as 'science' FITS in
        directory using masterBIAS and masterFLAT images. 
        
        There are several masterFLAT so it is used the one whose 
        INSPOROT angle has the same value than 'science' image.

        Args:
            show_info (bool): if True, additional process info is printed.

        Returns:
            int: 0, if everything was fine.

        Raises:
            Exception: any type, according to failing line of code.

        """

        # bias
        oBIAS = mcFits(self.masterBIAS, border=0)

        # ALL SCIENCE FITS Reduction
        for sciFITS in self.science['FILENAME'].values:
            print("{0} Working on '{1}' {0}".format('-' * 6, sciFITS))
            # image
            oSCIENCE = mcFits(sciFITS, border=0)
            
            if self.tel_type=='MAPCAT':
                pol_angle = oSCIENCE.header['INSPOROT']
                if float(pol_angle) > 70:
                    print(f'ERROR: instrument angle value ({pol_angle}) is not valid')
                    continue
                flat = self.masterFLAT[round(float(pol_angle), 1)]
            else:
                pol_angle = oSCIENCE.header['FILTER']
                print(pol_angle)
                flat = self.masterFLAT[pol_angle]
            if show_info:
                print(f"Polarization angle set to -> {pol_angle}")
            
            # flat
            print(pol_angle)
            def isfloat(num):
                try:
                    float(num)
                    return True
                except ValueError:
                    return False

            if isfloat(pol_angle):
                try:
                    flat = self.masterFLAT[round(float(pol_angle), 1)]
                except:
                    flat = self.masterFLAT[pol_angle]
                    print(f'self.masterFLAT = {self.masterFLAT}')
                    raise
            else:
                flat = self.masterFLAT[pol_angle]    
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
            scaled = np.where(scaled > np.iinfo(np.uint16).max, np.iinfo(np.uint16).max, scaled)

            
            # Out of border zone is set to original science values
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
            head = oSCIENCE.header
            head.extend(newCards, end=True)
            hdu = fits.PrimaryHDU(data=scaled.astype(np.uint16), header=head)
            out_image = os.path.join(self.out_dir, os.path.split(sciFITS)[1])
            hdu.writeto(out_image, overwrite=True, output_verify='ignore')

            if show_info:
                print("*" * 50)
                print(f"INPUT image stats({oSCIENCE.path}): {oSCIENCE.stats()}")
                oBIAS = mcFits(self.masterBIAS, border=self.border)
                print(f"BIAS stats ({oBIAS.path}): {oBIAS.stats()}")
                oFLAT = mcFits(flat, border=self.border)
                print(f"FLAT stats ({oFLAT.path}): {oFLAT.stats()}")
                oRED = mcFits(out_image, border=0)
                print("OUT image stats ({out_image}): {oRED.stats()}")

                print("Scaled (previous to uint casting) ->", scaled.min(),
                      scaled.max(), scaled.mean(), scaled.std(),
                      np.median(scaled), scaled.dtype.name)
                print("Data previous scaling process ->", data_final.min(),
                      data_final.max(), data_final.mean(), data_final.std(),
                      np.median(data_final), data_final.dtype.name)
                print(f"Num of negative pixel (reduced fits) -> {(oRED.data < 0).sum()}")
                print(f"Num of negative pixels (scaled matrix) -> {(scaled < 0).sum()}")
                print(f"Num of negative pixels (data_final matrix) -> {(data_final < 0).sum()}")
                print("*" * 50)

        return 0
