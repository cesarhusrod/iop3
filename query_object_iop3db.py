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

from matplotlib import pyplot as plt
import pandas as pd

import mysql.connector
from mysql.connector import errorcode

def main():
    parser = argparse.ArgumentParser(prog='query_object_iop3db.py', \
    conflict_handler='resolve',
    description='''Plot object photometry and magnitude throughout time. ''',
    epilog="")
    parser.add_argument("--out_dir",
       action="store",
       dest="out_dir",
       type=str,
       default='./',
       help="Ouput plot directory. [default: %(default)s].")
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
            host='127.0.0.1', database='iop3db')
        # cursor for executing queries
        cursor = cnx.cursor()

        # query
        query1 = f"SELECT P, dP, `rjd-50000`, Theta, dTheta, R, dR FROM polarimetry WHERE NAME='{args.blazar_name}'"
        
        
        # query execution
        cursor.execute(query1)

        # getting all results
        table_rows = cursor.fetchall()

        # casting to Pandas DataFrame object
        df = pd.DataFrame(table_rows, columns=['P','dP', 'RJD-50000', 'Theta', 'dTheta', 'R', 'dR'])
        if len(df.index) == 0:
            print(f'WARNING: No date stored in IOP^3 database for object called "{args.blazar_name}"')
            cursor.close()
            cnx.close()
            return 2

        print(df.info())
        print(df)

        #to start plotting
        fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)

        # first plot
        df.plot(ax=axes[0], kind='scatter',x='RJD-50000',y='P', yerr='dP', color='red', s=3)
        axes[0].set_title(args.blazar_name)
        axes[0].grid()
        
        # second plot
        df.plot(ax=axes[1], kind='scatter',x='RJD-50000',y='Theta', yerr='dTheta', color='green', s=3)
        axes[1].grid()
        
        # third plot
        df.plot(ax=axes[2], kind='scatter',x='RJD-50000',y='R', yerr='dR', color='blue', s=3)
        axes[2].grid()
        
        # Plotting as scatter plot
        # plt.grid()

        # saving plot in file
        png_path = os.path.join(args.out_dir, f'{args.blazar_name}.png')
        plt.savefig(png_path, dpi=300)
        
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