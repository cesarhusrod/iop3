#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon June 25 17:50:04 2021

___e-mail__ = cesar_husillos@tutanota.com
__author__ = 'Cesar Husillos'

VERSION:
    1.0 Initial version
"""

# ---------------------- IMPORT SECTION ----------------------

import os
import argparse
from getpass import getpass

from numpy import integer
import pandas as pd

import mysql.connector

"""
database url is dialect+driver://username:password@db_address/db_name
To connect to mysql/mariadb, pymysql module is required to install.
The purpose of using SqlAlchemy is to abstract sql syntax from the programmer/scripter,
hence there should be no sql syntax used, to use sql syntax use the execute method of the create_engine object.
"""

def main():
    parser = argparse.ArgumentParser(prog='iop3_add_blazarinfo_to_database.py', \
    conflict_handler='resolve',
    description='''Set MAPCAT blazar parameters into `blazar_source` table. ''',
    epilog='''''')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    parser.add_argument("blazar_csv", help="Path to CSV format file with information about MAPCAT blazars.")
    # parser.add_argument("input_dir", help="Input directory")
    parser.add_argument("--db_server",
        action="store",
        dest="db_server",
        default="localhost",
        help="Database server name or IP [default: %(default)s]")
    parser.add_argument("--db_name",
        action="store",
        dest="db_name",
        default="iop3db",
        help="Database name [default: %(default)s]")
    parser.add_argument("--db_user",
        action="store",
        dest="db_user",
        default="iop3admin",
        help="Database admin name [default: %(default)s]")
    parser.add_argument("--db_password",
        action="store",
        dest="db_password",
        default="IOP3_db_admin",
        help="Database admin user password [default: %(default)s]")
    parser.add_argument('-v', '--verbose', action='count', default=0,
        help="Show running and progress information [default: %(default)s].")
    
    args = parser.parse_args()

    db_user = args.db_user
    if not len(db_user):
        db_user = input(f"Write login for allowed user to create tables and relationships on database '{args.db_name}': ")
    
    db_password = args.db_password
    if not len(db_password):
        db_password = getpass(f"Password of {db_user}@{args.db_server}: ")

    # Database connection
    mydb = mysql.connector.connect(host=args.db_server, \
        user=db_user, password=db_password, database=args.db_name, ssl_disabled=True)

    db_cursor = mydb.cursor()

    # Selecting database for table creation
    db_cursor.execute(f"USE {args.db_name};")

    # reading input CSV blazar file
    if not os.path.exists(args.blazar_csv):
        print(f'ERROR: Blazars file "{args.blazar_csv}" does not exists')
        return 1
    
    data = pd.read_csv(args.blazar_csv)    
    
    query_insert = 'INSERT INTO blazar_source(`aper_pix_sext`, `name`, `name_IAU`, `ra2000`, `dec2000`, `rmag`, `rmagerr`, `P`, `dP`, `Theta`, `dTheta`) VALUES '
    values = []
    for index, row in data.iterrows():
        # check for new blazar
        query_check = "SELECT id FROM blazar_source WHERE name = '{}'".format(row['name_mc'])
        db_cursor.execute(query_check)
        res_search = db_cursor.fetchall()
        if db_cursor.rowcount == 0:
            line = " ({}, '{}', '{}', '{}', '{}', {}, {}, {}, {}, {}, {})"
            line = line.format(row['aper_mc'], row['name_mc'], row['IAU_name_mc'], \
                row['ra2000_mc'], row['dec2000_mc'], row['Rmag_mc'], row['Rmagerr_mc'], \
                row['PolDeg_mc'], row['ErrPolDeg_mc'], row['PolAngle_mc'], row['ErrPolAngle_mc'])
            # Insert new blazar or calibrator        
            values.append(line.replace('nan', 'NULL').replace("'NULL'", "NULL"))
    
    if values:
        query_insert = query_insert + ', '.join(values)
        print(query_insert.replace('), ', '), \n'))
        # Executing INSERT command
        db_cursor.execute(query_insert)

        mydb.commit()

    # Disconnecting
    mydb.close()

    return 0

if __name__ == '__main__':
    print(main())
