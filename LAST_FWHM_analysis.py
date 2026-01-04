#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Nov 18 21:46:57 2025
FWHM analysis
Written originally by Ron ARAD
Modified and enhanced by Micha
imports the functions package
Argument: path to the toml config file
Reads an toml file with program prameters and runs the analysis
@author: micha
"""
import tomllib as tom
import sys
import pyLAST as pla
import pandas as pd
import time
from datetime import datetime
import os
from matplotlib import pyplot as plt
# import plotly.tools as tls
# import clickhouse_connect
# from clickhouse_driver import Client
# from pydantic import BaseModel

configfile = sys.argv[1]
with open(configfile, "rb") as f:
    cfg = tom.load(f)

runcfg =  pla.runconfig(cfg)
euclid = pla.LastDatabase(runcfg.databases.euclid)
last0 = pla.LastDatabase(runcfg.databases.last0)
inpath = runcfg.general.input_path
outpath = runcfg.general.output_path
dbpath = runcfg.general.database_path
N_days = runcfg.general.Ndays
N_show = runcfg.general.Nshow
startdate =runcfg.general.startdate
enddate = runcfg.general.enddate
nmounts = runcfg.general.nmounts
run = runcfg.run
plots = runcfg.plots    
general = cfg['general']


if runcfg.general.localrun==False:
    if general['database']=='euclid':
        client = euclid.connect()
        print("connecting to %s\n"%euclid.name)
    elif general['database']=='last0':
        client = last0.connect()
        print("connecting to %s\n"%last0.name)
    else:
        raise Exception('Unknown or badly defined database location')
# else:
#     sys.exit()
    #TODO  read from a local file
tic = time.time()
timestamp = datetime.now().strftime('%y%m%d%H%M')
# generate a directory for the current run:
    # for plots 
outdir = os.path.join(runcfg.general.output_path, timestamp + '_output')
if  not os.path.isdir(outdir):
    os.mkdir(outdir)
# for databases qeury results csv files
time_span_stamp = \
    pla.generate_time_span_str(N_days,N_show,startdate,enddate)
db_out_path = os.path.join(dbpath,time_span_stamp[0])
if  not os.path.isdir(db_out_path):
    os.mkdir(db_out_path)
print('The database path is: %s \n'% db_out_path)
if  runcfg.general.localrun:
    if os.path.exists(db_out_path):
        # retrieve the csv file name
        df_FWHM_csv_file_name = pla.save_df_to_csv(df = None,
                                                   dfname='df_FWHM',
                                                   outdir= db_out_path,
                                                   time_span_stamp = time_span_stamp[0])
        df_FWHM = pd.read_csv(os.path.join(db_out_path,df_FWHM_csv_file_name),
                              dtype={'rediskey':str},
                              index_col='rediskey')
        nrecs = len(df_FWHM)
        print(f'Finished reading {nrecs} records from the csv file.')
    else:
        print(f'Database path {db_out_path} does not exist.')
        sys.exit() # for now
    #TODO read from local file df_FWHM = pla.load_FWHM_csv(input_file_FWHM, fraction_to_read, start_reading_at_frac)
    
else:
    print('starts queries\n ***********************\n')
    df_FWHM = pla.read_DB1(client,
                          'operation_strings', 'camera.set.FWHMellipse:', 
                          None, startdate, enddate, N_days, N_show)
    if df_FWHM.empty:
        print('There are NO observations during your requested interval - stopping')
        sys.exit()  
    df_FWHM = pla.basic_processing_FWHM(df_FWHM)
    # df_FWHM = pla.filter_N_days(pla.basic_processing_FWHM(df_FWHM), N_days, N_show)
    # df_FWHM = pla.filter_N_days(df_FWHM, N_days, N_show)
    df_FWHM, empty_cols = pla.filter_columns_by_nan_or_empty_lists(df = df_FWHM,
                                                               frac_nans = 0.5) 
    #optional 2nd param: frac determines threshold from total rows (0.1)
    print('empty columns removed:', empty_cols)
    toc = time.time()
    # End timer
    print(f"\nElapsed time: {toc - tic:.3f} seconds")
    print('\nfinished loading data')
    # if loaded from online db save the df to csv for local run
    df_FWHM_csv_file_name = pla.save_df_to_csv(df = df_FWHM,
                                           dfname='df_FWHM',
                                           outdir= db_out_path,
                                           time_span_stamp = time_span_stamp[0])
    print(f'saved data in: \n {df_FWHM_csv_file_name}\n')
#%% Start analysing the data
print('Starting data analysis.\n')
# generate list of dfs by mount
FWHM_groups = pla.separate_by_mount(df = df_FWHM)

if plots.FWHM_no_filter:
    pla.plot_FWHM(FWHM_groups = FWHM_groups,
              output_directory = outdir,
              showplots=False)
#%%  Generate FWHM stats by telescope  
FWHM_tel_stats=df_FWHM.groupby('rediskey')['minor'].agg(['mean','std'])
tels = FWHM_tel_stats.index.values
telmeans = FWHM_tel_stats['mean'].values
telstds = FWHM_tel_stats['std'].values
# plot telescope stats
fig0,ax0 = plt.subplots(figsize=(20,8))
ax0.bar(tels,telmeans, color = 'skyblue', capsize =4)
ax0.errorbar(tels,telmeans,
            yerr = telstds,
            fmt='none', 
            capsize =0, 
            color='red')
ax0.grid(axis='y')
ax0.tick_params(axis='x',labelsize=10)
ax0.tick_params(axis='y',labelsize=12)
ax0.set_ylabel('FWHM Minor Axis [pix]',fontsize=14)
ax0.set_xlabel('Telescope Label',fontsize =14)
ax0.set_title(f'Telescope Stats {time_span_stamp[1]} - {time_span_stamp[2]}', fontsize = 18)
plt.savefig(os.path.join(outdir,'FWHM_telescope_stats.png'))
plt.show()
print('Finished')
