#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 23:05:26 2025
@author: micha
Main function to perform LAST analysis
Written originally by Ron ARAD
Modified and enahnced by Micha
imports the functions package
Argument: path to the toml config file
Reads an toml file with program prameters and runs the analysis
"""
# import iniParser as ip
import tomllib as tom
import sys
import pyLAST as pla
import pandas as pd
import time
from datetime import datetime
import os
# import clickhouse_connect
# from clickhouse_driver import Client
# from pydantic import BaseModel

configfile = sys.argv[1]


with open(configfile, "rb") as f:
    cfg = tom.load(f)
general = cfg['general']
euclid = pla.LastDatabase(**cfg['databases']['euclid'])
last0 = pla.LastDatabase(**cfg['databases']['last0'])
inpath = general['input_path']
outpath = general['output_path']
N_days = general['Ndays']
N_show = general['Nshow']
localrun =  general['localrun']
run = cfg['run']
plots = cfg['plots']

if localrun==False:
    if general['database']=='euclid':
        client = euclid.connect()
        print("connecting to %s\n"%euclid.name)
    elif general['database']=='last0':
        client = last0.connect()
        print("connecting to %s\n"%last0.name)
    else:
        raise Exception('Unknown or badly defined database location')
else:
    sys.exit()
    #TODO  read from a local file


# eclid = cfg['databases']['euclid']
# last0 = cfg['databases']['last0']
     # LAST_0 as client using clickhouse_connect
 # if database == 'LAST_0':
 #     client = clickhouse_connect.get_client(host='10.23.1.25', port=8123, \
 #              username='last_user', password='physics', database='observatory_operation')        
 # elif database == 'euclid':
 #     # euclid as client using clickhouse_driver
    
 #     client = Client(host=euclidhost, port=euclidport, \
 #              user=eucliduser, password=euclidpw, database='observatory_operation')  
#%%
tic = time.time()
timestamp = datetime.now().strftime('%y%m%d%H%M')
print('started queries\n ***********************\n')
if run['tracking_analysis']:
    if localrun:
        # tracking = load_tracking_csv(input_file_tracking, fraction_to_read, start_reading_at_frac)
        sys.exit()
    else:
        tracking = pla.read_DB(N_days, N_show, client,
                               'operation_strings', 'XerxesMountBinary.get.Status', "value LIKE 'tracking'" )
        #tracking = read_DB(N_days,N_show,'operation_strings', 'XerxesMountBinary.get.Status' )
    if tracking.empty:
        print('There are NO observations during your requested interval - stopping')
        sys.exit()
    tracking = pla.filter_N_days(pla.basic_processing_tracking(tracking), N_days, N_show)
    tracking.to_csv(os.path.join(outpath,timestamp,'tracking.csv'))
    tracking_groups = pla.tracking_windows(pla.separate_by_mount(tracking)) 
    if localrun:
        # df_RA = load_tracking_csv(input_file_RA, fraction_to_read, start_reading_at_frac)
        sys.exit()
    else:
        df_RA = pla.read_DB(N_days, N_show, client,
                            'operation_numbers', 'XerxesMountBinary.get.RA')
    df_RA = pla.filter_N_days(pla.basic_processing_tracking(df_RA), N_days, N_show) 
    RA_groups = pla.separate_by_mount(df_RA)
    if localrun:
        # df_Dec = load_tracking_csv(input_file_Dec, fraction_to_read, start_reading_at_frac)
        sys.exit()
    else:
        df_Dec = pla.read_DB(N_days, N_show, client,
                            'operation_numbers', 'XerxesMountBinary.get.Dec')
    df_Dec = pla.filter_N_days(pla.basic_processing_tracking(df_Dec), N_days, N_show)
    Dec_groups = pla.separate_by_mount(df_Dec)
    if localrun:
        # df_Az = load_tracking_csv(input_file_Az, fraction_to_read, start_reading_at_frac)
        sys.exit()
    else:
        df_Az = pla.read_DB(N_days, N_show, client,
                            'operation_numbers', 'XerxesMountBinary.get.Az')
    df_Az = pla.filter_N_days(pla.basic_processing_tracking(df_Az), N_days, N_show)
    Az_groups = pla.separate_by_mount(df_Az)
    if localrun:
        # df_Alt = pla.load_tracking_csv(input_file_Alt, fraction_to_read, start_reading_at_frac)
        sys.exit()
    else:
        df_Alt = pla.read_DB(N_days, N_show, client, 
                             'operation_numbers', 'XerxesMountBinary.get.Alt')
    df_Alt = pla.filter_N_days(pla.basic_processing_tracking(df_Alt), N_days, N_show)
    Alt_groups = pla.separate_by_mount(df_Alt)
#  
if run['FWHM_analysis']:
     print('reading FWHM analysis')
     if  localrun:
         sys.exit() # for now
         #TODO read from local file df_FWHM = pla.load_FWHM_csv(input_file_FWHM, fraction_to_read, start_reading_at_frac)
         
     else:
         df_FWHM = pla.read_DB(N_days, N_show, client,
                               'operation_strings', 'camera.set.FWHMellipse:')
     if df_FWHM.empty:
         print('There are NO observations during your requested interval - stopping')
         sys.exit()  
     df_FWHM = pla.filter_N_days(pla.basic_processing_FWHM(df_FWHM), N_days, N_show)
     df_FWHM, empty_cols = pla.filter_columns_by_nan_or_empty_lists(df_FWHM, 0.5) 
                     #optional 2nd param: frac determines threshold from total rows (0.1)
     print('empty columns removed:', empty_cols)
     FWHM_groups = pla.separate_by_mount(df_FWHM)

#
if run['focus_analysis']:
    if localrun:
        sys.exit()
        # df_focus = load_FWHM_csv(input_file_focus, fraction_to_read, start_reading_at_frac)
    else: 
        df_focus = pla.read_DB(N_days, N_show, client, 
                               'operation_strings', 'unitCS.set.FocusData:', "value like '%LoopCompleted%:%true%' ")
    if df_focus.empty:
        print('There are NO focusings during your requested interval - stopping')
        sys.exit()
    df_focus = pla.filter_N_days(pla.basic_processing_FWHM(df_focus), N_days, N_show)
    df_focus["time"] = pd.to_datetime(df_focus["time"], errors="coerce")
    df_focus = df_focus.sort_values(by=[df_focus.index.name, "TimeStarted"])
    # if run['tracking_analysis']:
    #     if 'df_Alt' in locals():
    #         df_focus = pla.add_Alt_to_focus(df_focus, df_Alt)
    focus_groups = pla.separate_by_mount(df_focus)  
#
if run['Temp_analysis']:    
    if localrun:
        # df_Temp_1 = load_tracking_csv(input_file_Temp_1, fraction_to_read, start_reading_at_frac)
        sys.exit()
    else:
        df_Temp_1 = pla.read_DB(N_days, N_show, client,
                                'operation_numbers', 'unitCS.get.Temperature:', "endsWith(rediskey, '.1')")     
    df_Temp_1 = pla.filter_N_days(pla.basic_processing_tracking(df_Temp_1), N_days, N_show)
    df_Temp_1_groups = pla.separate_by_mount(df_Temp_1)
    
    if localrun:
        # df_Temp_2 = load_tracking_csv(input_file_Temp_2, fraction_to_read, start_reading_at_frac)
        sys.exit()
    else:
        df_Temp_2 = pla.read_DB(N_days, N_show, client,
                                'operation_numbers', 'unitCS.get.Temperature:', "endsWith(rediskey, '.2')")     
    df_Temp_2 = pla.filter_N_days(pla.basic_processing_tracking(df_Temp_2), N_days, N_show)
    df_Temp_2_groups = pla.separate_by_mount(df_Temp_2)
    df_Temp_groups = []
    for i in range(10):
        df_Temp_1_groups[i].drop('scope',axis=1) # for the temperatures, we don't have scope info only mount
        df_Temp_2_groups[i].drop('scope',axis=1) # for the temperatures, we don't have scope info only mount
        '''Fill df_Temp_groups with the data of the smoother temperature sensor {1 or 2} for each mount'''
        df_Temp_groups.append(pla.smoother_df(df_Temp_1_groups[i],df_Temp_2_groups[i], col="value"))

# End timer
toc = time.time()
print(f"\nElapsed time: {toc - tic:.3f} seconds")
print('\nfinished loading data')
#%% Plots cell    
if  plots['FWHM_no_filter']:
    pla.plot_FWHM(FWHM_groups,outpath)
print('Finished')