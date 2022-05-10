#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu April 09 09:49:53 2020

___e-mail__ = cesar_husillos@tutanota.com
__author__ = 'Cesar Hiop3_pousillos'

VERSION:
    1.0 Initial version
"""


# ---------------------- IMPORT SECTION ----------------------
from cmath import e
import os
import argparse
from collections import defaultdict
from pprint import pprint
from tkinter import N
from typing import DefaultDict
from urllib.parse import quote

import pandas as pd
import jinja2 # HTML templates packages for documentation and logging

import mcReduction
from mcFits import *

def report(csv_file, output_dir, template_file, title=''):
    """Create output HTML report with CSV input data.

    Args:
        csv_file (string): Input CSV FITS data file.
        output_dir (string): Output directory report.
        template_file (string): HMTL template file.
        title (str): Report title.

    Returns:
        int:
            0, if everything was OK.
            1, if reading csv_file fault.
            2, if file_template was not found.
    
    Exception:
        Exception: If output html report could not be written.
    """
    # rendering Jinja2 web page template
    results = None
    try:
        results = pd.read_csv(csv_file)
    except:
        print(f"ERROR: Reading CVS input file '{csv_file}'.")
        return 1

    # results = results.sort(columns='DATE-OBS')
    results['FILENAME'] = [os.path.split(r)[1] for r in results['PATH'].values]
    # results['FILEPNG'] = [quote(r.replace('.fits', '.png'), safe='') for r in results['FILENAME'].values]
    # results['HISTOPNG'] = [quote(r.replace('.fits', '_histogram.png'), safe='') for r in results['FILENAME'].values]
    if 'MJD-OBS' in results.columns:
        results['MJDOBS'] = results['MJD-OBS']
    if 'DATE-OBS' in results.columns:
        results['DATE-OBS'] = results['DATE-OBS']
    

    if not os.path.exists(template_file):
        print(f"ERROR: Template file '{template_file}' not found.")
        return 2

    fits_dir = os.path.split(results['PATH'].values[0])[0]
    template_dir, template_name = os.path.split(template_file)
    templateLoader = jinja2.FileSystemLoader(searchpath=template_dir)
    templateEnv = jinja2.Environment(loader=templateLoader)
    template = templateEnv.get_template(template_name)

    dictOfLists = results.to_dict(orient='list')
    listOfDicts = [dict(zip(dictOfLists,t)) for t in zip(*dictOfLists.values())]
    print(listOfDicts[0])

    outputText = template.render(results=listOfDicts, \
        title = title, fits_dir=fits_dir, \
        num_fits = len(results['FILENAME']))  # this is where to put args to the template renderer
    outputText = outputText.replace('assets/', \
        os.path.join(template_dir, 'assets/'))

    # saving QC HTML results page
    outHTML = os.path.join(output_dir, template_name)
    # print(f"outHTML = {outputText}")
    try:
        with open(outHTML, 'w') as fout:
            fout.write(outputText)
    except:
        print(f"ERROR: output '{template_file}' could not be written.")
        raise
    
    return 0

# ------------------------ MAIN FUNCTION SECTION -----------------------------
def main():
    parser = argparse.ArgumentParser(prog='mcPipe_OK.py', \
    conflict_handler='resolve',
    description='''Main program that reads, classify and reduces data from 
    FITS located at input directory. ''',
    epilog='''''')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')
    parser.add_argument("config_dir", help="Configuration parameter files directory")
    parser.add_argument("reduction_dir", help="Reduction output directory")
    parser.add_argument("input_dir", help="Input directory")
    parser.add_argument("--border_image",
       action="store",
       dest="border_image",
       type=int,
       default=0,
       help="True is input file is for clusters [default: %(default)s].")
    parser.add_argument('-v', '--verbose', action='count', default=0,
        help="Show running and progress information [default: %(default)s].")
    args = parser.parse_args()

    # Checking input parameters
    if not os.path.exists(args.config_dir):
        str_err = 'ERROR: Config dir "{}" not available'
        print(str_err.format(args.config_dir))
        return 1
    
    if not os.path.isdir(args.reduction_dir):
        try:
            os.makedirs(args.reduction_dir)
        except IOError:
            str_err = 'ERROR: Could not create output reduction directory "{}"'
            print(str_err.format(args.reduction_dir))
            return 2

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        str_err = 'ERROR: Input directory "{}" not available'
        print(str_err.format(input_dir))
        return 3
    
    border_image = args.border_image
    if args.border_image < 0:
        border_image = 0
    elif args.border_image > 300:
        str_err = 'ERROR: Border image "{}" too high for getting realistic results.'
        print(str_err.format(args.border_image))
        return 4
    
    # Getting run date
    dt_run = re.findall('(\d{6})', input_dir)[-1]
    date_run = f'20{dt_run[:2]}-{dt_run[2:4]}-{dt_run[-2:]}'

    if not os.path.exists(args.reduction_dir):
        try:
            os.makedirs(args.reduction_dir)
        except IOError:
            str_err = 'ERROR: Output reduction directory "{}" couldn\'t be created'
            print(str_err.format(args.reduction_dir))
            return 4
        else:
            message = "Output reduction directory set to '{}'"
            print(message.format(args.reduction_dir))

    # Raw input data
    data_raw_input = defaultdict(list)

    # Create generic object of class mcReduction
    oReduction = mcReduction.mcReduction(input_dir, args.reduction_dir,
                                         border=border_image)

    # Input FITS classification printing
    print(f'Bias number = {len(oReduction.bias)}')
    print(f'Flats number = {len(oReduction.flats)}')
    print(f'Science number = {len(oReduction.science)}')
    # print('Description = {}'.format(oReduction.science.describe))

    # Getting input FITS info
    some_raw_keys = ['INSFLNAM', 'NAXIS1', 'NAXIS2', 'RA', 'DEC', 'OBJECT', \
        'EXPTIME', 'DATE-OBS', 'EQUINOX', 'MJD-OBS', 'INSTRSCL', 'IMAGETYP']

    input_fits = oReduction.bias['FILENAME'].tolist() + oReduction.flats['FILENAME'].tolist() + \
        oReduction.science['FILENAME'].tolist()

    for fi in input_fits:
        # reading input FITS
        ff = mcFits(fi, border=args.border_image)

        # if ff.header['NAXIS1'] < 1024:
        #     continue
        
        # getting formated info
        data_raw_input['PATH'].append(fi)
        data_raw_input['RUN_DATE'].append(date_run)
        
        pol_ang = ''
        if 'INSPOROT' in ff.header:
            try:
                pol_ang = round(float(ff.header['INSPOROT']), 1)
            except:
                pol_ang=ff.header['INSPOROT']
        data_raw_input['POLANGLE'].append(pol_ang)
        
        for s_key in some_raw_keys:
            data_raw_input[s_key].append(ff.header.get(s_key, 'FILTER'))
        
        # adding FITS data statistics
        for key, value in ff.stats().items():
            data_raw_input[key].append(value)
        
    
    # Saving input raw data by using a CSV file
    csv_data_raw = os.path.join(args.reduction_dir, 'input_data_raw.csv')
    df_raw_out = pd.DataFrame(data_raw_input)
    # print('df_raw_out.info()')
    # print(df_raw_out.info())
    df_raw_out.to_csv(csv_data_raw, index=False)


    # ------------------- MasterBIAS generation -------------------
    if oReduction.createMasterBIAS() != 0:
        print("MasterBIAS generation failed.")
        # without MasterBIAS, reduction process has no sense. So script ends here.
        return 6

    # Plotting MasterBIAS and histogram
    mcBIAS = mcFits(oReduction.masterBIAS, border=oReduction.border)
    
    if 'fits' in oReduction.masterBIAS:
        plotBIAS = oReduction.masterBIAS.replace('.fits', '.png')
    else:
        plotBIAS = oReduction.masterBIAS.replace('.fit', '.png')
    title = f'{oReduction.date} masterBIAS'
    mcBIAS.plot(title)

    if 'fits' in oReduction.masterBIAS:
        plotBIASHist = oReduction.masterBIAS.replace('.fits', '_histogram.png')
    else:
        plotBIASHist = oReduction.masterBIAS.replace('.fit', '_histogram.png')
    histo_par = {'xlabelsize':8,
                 'ylabelsize': 8,
                 'bins': 100,
                 'color':'#6022aa',
                 'alpha':0.6}
    mcBIAS.plot_histogram(plotBIASHist, title=title, histogram_params=histo_par)


    # Getting masterBIAS info (there is only one by night)
    data_masterbias_info = {}
    
    data_masterbias_info['RUN_DATE'] = [date_run]
    data_masterbias_info['PATH'] = oReduction.masterBIAS # this attribute is a list
    data_masterbias_info['PLOT'] = [plotBIAS]
    data_masterbias_info['HISTOGRAM'] = [plotBIASHist]

    some_bias_keys = ['NAXIS1', 'NAXIS2', 'SOFT', 'PROCDATE', 'BIASOP', 'PXBORDER', \
        'OBJECT', 'MAX', 'MIN', 'MEAN', 'STD', 'MED']

    for s_key in some_bias_keys:
        data_masterbias_info[s_key] = [mcBIAS.header.get(s_key, '')]
    
    # Number of FITS used in MasterBIAS composition is variable.
    counter = 0
    while f'BIAS{counter}' in mcBIAS.header:
        data_masterbias_info[f'BIAS{counter}'] = [mcBIAS.header[f'BIAS{counter}']]
        counter += 1

    # Saving masterbias data in CSV file
    csv_masterbias_data = os.path.join(args.reduction_dir, 'masterbias_data.csv')
    df_masterbias_out = pd.DataFrame(data_masterbias_info)
    df_masterbias_out.to_csv(csv_masterbias_data, index=False)
    # pprint.pprint(data_masterbias_info)
    

    # --------------- MasterFLATs generation ---------------------
    data_masterflats_info = defaultdict(list)

    if oReduction.createMasterFLAT():
        print("MasterFLATs generation failed.")
        return 7

    num_fits_in_mf = 0    

    for pol_ang, mf_path in oReduction.masterFLAT.items():
        print('- ' * 5 + mf_path)
        mcFLAT = mcFits(mf_path, border=oReduction.border)

        # Plotting FLAT and histogram
        if 'fits' in oReduction.masterFLAT[pol_ang]:
            plotFLAT = oReduction.masterFLAT[pol_ang].replace('.fits', '.png')
        else:
            plotFLAT = oReduction.masterFLAT[pol_ang].replace('.fit', '.png')
        title = f"MasterFLAT (date, pol.angle)=({oReduction.date}, {pol_ang})"
        mcFLAT.plot(title)
        if 'fits' in oReduction.masterFLAT[pol_ang]:
            plotFLATHist = oReduction.masterFLAT[pol_ang].replace('.fits', '_histogram.png')
        else:
            plotFLATHist = oReduction.masterFLAT[pol_ang].replace('.fit', '_histogram.png')

        mcFLAT.plot_histogram(plotFLATHist, title=title, \
            histogram_params=histo_par)
        
        # Getting FLATS info
        data_masterflats_info['PATH'].append(mf_path)
        data_masterflats_info['RUN_DATE'].append(date_run)
        pol_ang = ''
        if 'INSPOROT' in mcFLAT.header:
            try:
                pol_ang = round(float(mcFLAT.header['INSPOROT']), 1)
            except:
                pol_ang = mcFLAT.header['INSPOROT']
        else:
            pol_ang = mcFLAT.header['FILTER']
        data_masterflats_info['POLANGLE'].append(pol_ang)
        data_masterflats_info['PLOT'].append(plotFLAT)
        data_masterflats_info['HISTOGRAM'].append(plotFLATHist)

        some_flats_keys = ['NAXIS1', 'NAXIS2', 'SOFT', 'PROCDATE', 'FLATOP', \
            'PXBORDER', 'OBJECT', 'MBIAS', 'MAX', 'MIN', 'MEAN', 'STD', 'MED']

        for sf_key in some_flats_keys:
            data_masterflats_info[sf_key].append(mcFLAT.header.get(sf_key, ''))

        # FITS number used for generating FLAT is variable
        counter = 0
        while f"FLAT{counter}" in mcFLAT.header:
            data_masterflats_info[f"FLAT{counter}"].append(mcFLAT.header[f"FLAT{counter}"])
            counter += 1
        num_fits_in_mf = max([counter, num_fits_in_mf])

    for n in range(num_fits_in_mf):
        while len(data_masterflats_info[f'FLAT{n}']) < len(oReduction.masterFLAT.keys()):
            data_masterflats_info[f'FLAT{n}'].append('')
        
    # Saving masterbias data
    # Ouput CSV data file info
    csv_masterflats_data = os.path.join(args.reduction_dir, 'masterflats_data.csv')
    df_masterflats_out = pd.DataFrame.from_dict(data_masterflats_info, orient='index')
    df_masterflats_out=df_masterflats_out.transpose()
    df_masterflats_out.to_csv(csv_masterflats_data, index=False)

    # --------------- Making FITS set reduction operations --------------#
    
    # info container
    data_red_out = defaultdict(list)

    # Computing Science FITS
    oReduction.reduce()

    #print('Plotting figures. Please wait...')
    for index in range(oReduction.science.shape[0]):
        sci = oReduction.science.iloc[index]
        path_red = os.path.join(oReduction.out_dir, os.path.split(sci['FILENAME'])[1])
        if not os.path.exists(path_red):
            print(f"WARNING: '{path_red}' not available.")
            continue
        mcRED = mcFits(path_red)

        # Estimating FWHM from extracted sources
        dictSEx = {}
        if 'fits' in mcRED.path:
            dictSEx['CATALOG_NAME'] = mcRED.path.replace('.fits', '.cat')
        else:
            dictSEx['CATALOG_NAME'] = mcRED.path.replace('.fit', '.cat')
        dictSEx['CONFIG_FILE'] = os.path.join(args.config_dir, 'daofind.sex')
        try:
            mcRED.compute_fwhm(dictSEx)
            mcRED = 0 # Forcing to write FWHM dataSEX
            mcRED = mcFits(path_red, border=border_image)
        except Exception as e:
            print(f'REDUCTION,ERROR,"Could not compute FWHM in FITS \'{path_red}\'."')
            print(e)

        # Plotting Science image...
        if 'fits' in mcRED.path:
            plotSCI = mcRED.path.replace('.fits', '.png')
        else:
            plotSCI = mcRED.path.replace('.fit', '.png')
        title_pattern = "{} - {} - {:.1f} s"
        title = title_pattern.format(sci['DATE-OBS'], \
            sci['OBJECT'], float(sci['EXPTIME']))
        mcRED.plot(title)

        # Plotting histogram
        if 'fits' in mcRED.path:
            plotSCIHist = mcRED.path.replace('.fits', '_histogram.png')
        else:
            plotSCIHist = mcRED.path.replace('.fit', '_histogram.png')
        mcRED.plot_histogram(plotSCIHist, title=title, histogram_params=histo_par)

        # Getting REDUCED FITS info
        data_red_out['PATH'].append(mcRED.path)
        data_red_out['RUN_DATE'].append(date_run)
        pol_ang = ''
        if 'INSPOROT' in mcRED.header:
            pol_ang = round(float(mcRED.header['INSPOROT']), 1)
        else:
            try:
                pol_ang = round(float(mcRED.header['FILTER'][1:]), 1)
            except:
                pol_ang = 99.0
        data_red_out['POLANGLE'].append(pol_ang)
        data_red_out['PLOT'].append(plotSCI)
        data_red_out['HISTOGRAM'].append(plotSCIHist)

        some_reduction_keys = ['NAXIS1', 'NAXIS2', 'SOFT', 'PROCDATE', \
            'PXBORDER', 'INSFLNAM', 'RA', 'DEC', 'OBJECT', 'EXPTIME', \
            'DATE-OBS', 'EQUINOX', 'MJD-OBS', 'BIAS', 'FLAT', 'FWHM', \
            'FWHMSTD', 'FWNSOURC', 'FWHMFLAG', 'FWHMELLI', \
            'MAX', 'MIN', 'MEAN', 'STD', 'MED']
        
        for sr_key in some_reduction_keys:
            data_red_out[sr_key].append(mcRED.header.get(sr_key, ''))

    # Saving output reduced data
    # Output CSV data file
    csv_data_red = os.path.join(args.reduction_dir, 'output_data_red.csv')
    df_red_out = pd.DataFrame(data_red_out)
    df_red_out.to_csv(csv_data_red, index=False)

    # Reporting results...
    dir_templates = os.path.join(args.config_dir, 'templates')

    # Raw images
    template = os.path.join(dir_templates, 'raw_fits.html')
    res_rep_RawFits = report(csv_data_raw, \
        output_dir=args.reduction_dir, template_file=template, \
        title=f'{date_run} raw FITS')
    
    # MasterBIAS
    template = os.path.join(dir_templates, 'masterbias_fits.html')
    rep_MB = report(csv_masterbias_data, \
        output_dir=args.reduction_dir, template_file=template, \
        title=f'{date_run} MASTERBIAS result')
    
    # MasterFlats
    template = os.path.join(dir_templates, 'masterflat_fits.html')
    print(csv_masterflats_data)
    rep_MF = report(csv_masterflats_data, \
        output_dir=args.reduction_dir, template_file=template, \
        title=f'{date_run} MASTERFLATS result')
    
    # Reduced FITS
    template = os.path.join(dir_templates, 'reduced_fits.html')
    rep_RedFits = report(csv_data_red, \
        output_dir=args.reduction_dir, template_file=template, \
        title=f'{date_run} reduced FITS')
    
    return 0

if __name__ == '__main__':
    if not main():
        print("*" * 80)
        print("Reduction done successfully!!")
        print("*" * 80)
