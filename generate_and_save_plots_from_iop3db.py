#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting and saving polarimetry object measurements (and magnitudes) using IOP^3 database queryies.

___e-mail__ = cesar_husillos@tutanota.com
__author__ = 'Cesar Husillos'

VERSION: 0.1
"""

import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from astropy.time import Time

import mysql.connector
from mysql.connector import errorcode
from helpers_for_nice_axes import *
import matplotlib
matplotlib.use('Agg')

def main():
    parser = argparse.ArgumentParser(prog='generate_and_save_plots_from_iop3db.py', \
    conflict_handler='resolve',
    description='''Plot object photometry and magnitude throughout time. ''',
    epilog="")
    parser.add_argument("--out_dir",
       action="store",
       dest="out_dir",
       type=str,
       default='./',
       help="Ouput plot directory. [default: %(default)s].")
    
    parser.add_argument("--date_start",
                        action="store",
                        dest="date_start",
                        type=str,
                        default="2022-01-01",       
                        help="Start date to plot in iso format. [default: %(default)s].")
    parser.add_argument("--date_end",
                        action="store",
                        dest="date_end",
                        default="2022-10-25",
                        type=str,
                        help="End date to plot in iso format. [default: %(default)s].")
    parser.add_argument("--full_range",
                        action="store",
                        dest="full_range",
                        type=bool,
                        default=False,
                        help="If True, plot all data. [default: %(default)s].")
    parser.add_argument("blazar_name", help="Blazar name used for query IOP3 database.") # mandatory argument
    
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        try:
            os.makedirs(args.out_dir)
        except IOError:
            print(f'ERROR: could not create output directory "{args.out_dir}"')
            return 1
    
    try:
        # create database connection
        cnx = mysql.connector.connect(user='iop3admin', password='IOP3_db_admin', \
            host='127.0.0.1', database='iop3db', ssl_disabled=True)
        # cursor for executing queries
        cursor = cnx.cursor()
        #Get dates in MJD
        if args.full_range==False:
            rjd_start=Time(f'{args.date_start} 12:00:00.000', format='iso').jd-2400000-50000
            rjd_end=Time(args.date_end, format='iso').jd-2400000-50000+1.5 #Get the full night 
        # query
            query1 = f"SELECT name, alternative_name, telescope, P, dP, `rjd-50000`,`mjd_obs`, Theta, dTheta, R, dR FROM polarimetry WHERE NAME='{args.blazar_name}' AND `rjd-50000`>='{rjd_start}' AND `rjd-50000`<='{rjd_end}'"
        else:
            query1 = f"SELECT name, alternative_name, telescope, P, dP, `rjd-50000`,`mjd_obs`, Theta, dTheta, R, dR FROM polarimetry WHERE NAME='{args.blazar_name}'"
        # query execution
        cursor.execute(query1)

        # getting all results
        table_rows = cursor.fetchall()

        # casting to Pandas DataFrame object
        df = pd.DataFrame(table_rows, columns=['name', 'alternative_name', 'telescope', 'P','dP', 'RJD-50000', 'MJD-OBS', 'Theta', 'dTheta', 'R', 'dR'])
        #df['jyear'] = Time(df['MJD-OBS'], format='mjd').jyear
        
        if len(df.index) == 0:
            print(f'WARNING: No data stored in IOP^3 database for object called "{args.blazar_name}"')
            cursor.close()
            cnx.close()
            return 2
        df['jyear'] = Time(df['RJD-50000']+2400000+50000, format='jd').jyear
        #Get reference stars
        for i in range(0,df.shape[0]):
            alt_name=df.alternative_name.values[i]
            if alt_name!=None:
                break

        if args.full_range==False:
            query2 = f"SELECT name, alternative_name, telescope, P, dP, `rjd-50000`,`mjd_obs`, Theta, dTheta, R, dR, Rmag_lit FROM polarimetry_reference_stars WHERE ALTERNATIVE_NAME LIKE '%{alt_name}%' AND `rjd-50000`>='{rjd_start}' AND `rjd-50000`<='{rjd_end}'"
        else:
            query2 = f"SELECT name, alternative_name, telescope, P, dP, `rjd-50000`,`mjd_obs`, Theta, dTheta, R, dR, Rmag_lit FROM polarimetry_reference_stars WHERE ALTERNATIVE_NAME LIKE '%{alt_name}%'"
        # query execution
        cursor.execute(query2)

        # getting all results
        table_rows_refstars = cursor.fetchall()

        # casting to Pandas DataFrame object
        df_stars = pd.DataFrame(table_rows_refstars, columns=['name', 'alternative_name', 'telescope', 'P','dP', 'RJD-50000', 'MJD-OBS', 'Theta', 'dTheta', 'R', 'dR', 'Rmag_lit'])
        if len(df_stars.index) == 0:
            print(f'WARNING: This is not a blazar "{alt_name}", quitting')
            cursor.close()
            cnx.close()
            return 2
        #df_stars['jyear'] = Time(df_stars['RJD-50000'], format='mjd').jyear
        df_stars['jyear'] = Time(df_stars['RJD-50000']+2400000+50000, format='jd').jyear
        if len(df_stars.index) == 0:
            print(f'WARNING: No data stored in IOP^3 database for object called "{alt_name}"')
            cursor.close()
            cnx.close()
            return 2
        df_stars['jyear'] = Time(df_stars['RJD-50000']+2400000+50000, format='jd').jyear
        
        #Get Rmag_lit
        for i in range(0,df_stars.shape[0]):
            Rmag_lit=df_stars['Rmag_lit'].values[i]
            if Rmag_lit!=None:
                break
        for i in range(0,df_stars.shape[0]):
            alt_name_star=df_stars.alternative_name.values[i]
            if alt_name_star!=None:
                break        
        #to start plotting
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12,9), sharex=True, gridspec_kw={'hspace':0.09})
        telescopes=["T090", "T150", "MAPCAT"]
        colors=["green", "blue", "red"]
        shapes=["*","p","d"]
        count=0
        for i, telescope in enumerate(telescopes): 
            blazar_df = df[df['telescope'] == telescope]
            if blazar_df.shape[0]>0:
                axes[0].errorbar(blazar_df['jyear'], blazar_df['P'], markersize=10, yerr=blazar_df['dP'], color=colors[i], fmt='.', alpha=0.5, label=f'{alt_name} {telescope}')
                axes[1].errorbar(blazar_df['jyear'], blazar_df['Theta'], markersize=10, yerr=blazar_df['dTheta'], color=colors[i],fmt='.', alpha=0.5)
                axes[2].errorbar(blazar_df['jyear'], blazar_df['R'], markersize=10, yerr=blazar_df['dR'], color=colors[i],fmt='.', alpha=0.5)
            
                star_df = df_stars[df_stars['telescope'] == telescope]
                axes[0].errorbar(star_df['jyear'], star_df['P'], markersize=10, yerr=star_df['dP'], color='orange',fmt='.', marker=shapes[i], alpha=0.5)
                axes[1].errorbar(star_df['jyear'], star_df['Theta'], markersize=10, yerr=star_df['dTheta'], color='orange',fmt='.', marker=shapes[i], alpha=0.5)
                axes[2].errorbar(star_df['jyear'], star_df['R'], markersize=10, yerr=star_df['dR'], color='orange', marker=shapes[i],fmt='.', alpha=0.5, label=f'{alt_name_star} (ref. star) {telescope}')
                if count==0:
                    h7 = axes[2].axhline(y=Rmag_lit, color='r', linestyle='-', linewidth=1, alpha=0.5, label=f"{alt_name_star} (ref.star) R LIT")
                    count=count+1

        axes[0].legend()
        axes[2].legend()
        axes[2].invert_yaxis()
        axes[0].set_ylabel('P (%)')
        axes[1].set_ylabel('Theta (deg)')
        axes[2].set_ylabel('R (mag)')
        axes[2].set_xlabel('MJD')
        fig.suptitle(f'{alt_name} ({args.blazar_name})', y=1)
            
        axes[0].grid(axis='y')
        axes[1].grid(axis='y')
        axes[2].grid(axis='y')
    
        #format axes labels for jyear and date
        set_ax_jyear(axes[-1])
        for i in range(1, len(axes)):
            set_ax_dates(axes[i], axes[i].twiny(), labels=False).grid(axis='x')
        set_ax_dates(axes[0], axes[0].twiny(), labels=True).grid(axis='x')

        # saving plot in file
        if args.full_range==True:
            png_path = os.path.join(args.out_dir, f'{args.blazar_name}_full_range.png')
        elif args.date_start==args.date_end:
            png_path = os.path.join(args.out_dir, f'{args.blazar_name}_{args.date_start}.png')
        else:
            png_path = os.path.join(args.out_dir, f'{args.blazar_name}_{args.date_start}_{args.date_end}.png')
        plt.savefig(png_path, dpi=300)
        #plt.show()
        # Closing cursor. No more queryies could be executed from now
        cursor.close()
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
    else:
        cnx.close()

    return 0


if __name__ == '__main__':
    print(main())
