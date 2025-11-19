#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 22:34:41 2025
Tracking analysis
Written originally by Ron ARAD
Modified and enahnced by Micha
imports the functions package
Argument: path to the toml config file
Reads an toml file with program prameters and runs the analysis
@author: micha
"""
import tomllib as tom
import sys
import pyLAST as pla
import numpy as np
# import pandas as pd
import time
from datetime import datetime
import os
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
tic = time.time()
timestamp = datetime.now().strftime('%y%m%d%H%M')
print('started queries\n ***********************\n')
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
# tracking.to_csv(os.path.join(outpath,timestamp,'tracking.csv'))
tracking_groups = pla.tracking_windows(pla.separate_by_mount(tracking)) 

df_RA = pla.read_DB(N_days, N_show, client,
                    'operation_numbers', 'XerxesMountBinary.get.RA')
df_RA = pla.filter_N_days(pla.basic_processing_tracking(df_RA), N_days, N_show) 
RA_groups = pla.separate_by_mount(df_RA)
df_Dec = pla.read_DB(N_days, N_show, client,
                    'operation_numbers', 'XerxesMountBinary.get.Dec')
df_Dec = pla.filter_N_days(pla.basic_processing_tracking(df_Dec), N_days, N_show)
Dec_groups = pla.separate_by_mount(df_Dec)

df_Az = pla.read_DB(N_days, N_show, client,
                        'operation_numbers', 'XerxesMountBinary.get.Az')
df_Az = pla.filter_N_days(pla.basic_processing_tracking(df_Az), N_days, N_show)
Az_groups = pla.separate_by_mount(df_Az)

df_Alt = pla.read_DB(N_days, N_show, client, 
                         'operation_numbers', 'XerxesMountBinary.get.Alt')
df_Alt = pla.filter_N_days(pla.basic_processing_tracking(df_Alt), N_days, N_show)
Alt_groups = pla.separate_by_mount(df_Alt)
#
tracking_results_groups = pla.analyze_tracking(tracking_groups, RA_groups, Dec_groups, Az_groups, Alt_groups)
#Note, in tracking_results the std values are already in arcsec

# if plot_RA_Dec_std:    
pla.plot_tracking_results_1(tracking_results_groups,outpath) #tracking std vs hour
# if plot_RA_Dec:
pla.plot_tracking_results_2(tracking_results_groups,outpath) #RA and Dec vs hour

'''count the number of active observation days'''
indexes = pla.find_noon_rollover_indexes(tracking)
print('The analysis includes', len(indexes), ' days')
temp = indexes[1:] + [indexes[-1]]
N=300  #minimum number of observations per night
print('Number of days with active observations: ', np.sum(np.diff(indexes) > N)+1)
print('finished tracking analysis')
