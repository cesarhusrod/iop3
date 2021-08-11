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

import mysql.connector

"""
database url is dialect+driver://username:password@db_address/db_name
To connect to mysql/mariadb, pymysql module is required to install.
The purpose of using SqlAlchemy is to abstract sql syntax from the programmer/scripter,
hence there should be no sql syntax used, to use sql syntax use the execute method of the create_engine object.
"""

def main():
    parser = argparse.ArgumentParser(prog='iop3_create_database.py', \
    conflict_handler='resolve',
    description='''Testing SQLAlchemy capacities. ''',
    epilog='''''')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    parser.add_argument("sql_file", help="Path to SQL file with tables creation commands.")
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
        user=db_user, password=db_password, database=args.db_name)

    mycursor = mydb.cursor()

    # Selecting database for table creation
    mycursor.execute(f"USE {args.db_name};")

    # Executing command for table creation in database
    with open(args.sql_file) as sql:
        print(f"INFO: Executing SQL commands given in '{args.sql_file}'")
        mydb.cmd_query_iter(sql.read())

    # Disconnecting
    mydb.close()

    return 0

if __name__ == '__main__':
    print(main())
