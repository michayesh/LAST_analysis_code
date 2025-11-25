#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 21:46:57 2025
FWHM analysis
Written originally by Ron ARAD
Modified and enahnced by Micha
imports the functions package
Argument: path to the toml config file
Reads an toml file with program prameters and runs the analysis
@author: micha
"""
#import iniParser as ip
import tomllib as tom
import sys
import pyLAST as pla
# import pandas as pd
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
tic = time.time()
timestamp = datetime.now().strftime('%y%m%d%H%M')
# generate a folder for the current run:
outdir = os.path.join(outpath,timestamp + '_output')
if  not os.path.isdir(outdir):
    os.mkdir(outdir)

print('started queries\n ***********************\n')
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
# End timer
toc = time.time()
print(f"\nElapsed time: {toc - tic:.3f} seconds")
print('\nfinished loading data')
pla.plot_FWHM(FWHM_groups,outdir)
print('Finished')
