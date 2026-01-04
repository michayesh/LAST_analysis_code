#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 09:27:17 2025


Focus best position vs. temperature analysis
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
# import clickhouse_connect
# from clickhouse_driver import Client
# from pydantic import BaseModel

configfile = sys.argv[1]
with open(configfile, "rb") as f:
    cfg = tom.load(f)

runcfg =  pla.runconfig(cfg)


#%%
# define data base objects
euclid = pla.LastDatabase(runcfg.databases.euclid)
last0 = pla.LastDatabase(runcfg.databases.last0)
inpath = runcfg.general.input_path
outpath = runcfg.general.output_path
dbpath = runcfg.general.database_path
# general.setdefault('Ndays')
N_days = runcfg.general.Ndays
# general.setdefault('Nshow')
N_show = runcfg.general.Nshow
# general.setdefault('startdate')
startdate =runcfg.general.startdate
# general.setdefault('enddate')
enddate = runcfg.general.enddate
# localrun =  runcfg.general.localrun
nmounts = runcfg.general.nmounts
run = runcfg.run
plots = runcfg.plots
# regular_fit =  run.regular_fit
# constraints = run.constraints
if runcfg.general.localrun==False:
    if runcfg.general.database =='euclid':
        client = euclid.connect()
        print("connecting to %s\n"%euclid.name)
    elif runcfg.general.database =='last0':
        client = last0.connect()
        print("connecting to %s\n"%last0.name)
    else:
        raise Exception('Unknown or badly defined database location')
else:
    sys.exit()
    #TODO  read from a local file
tic = time.time()
timestamp = datetime.now().strftime('%y%m%d%H%M')
# generate a directory for the current run:
    # for plots 
outdir = os.path.join(runcfg.general.output_path, timestamp + '_output')
if  not os.path.isdir(outdir):
    os.mkdir(outdir)
# for databases qeury results csv files
time_span_stamp,_ = pla.generate_time_span_str(N_days,N_show,startdate,enddate)
db_out_path = os.path.join(dbpath,time_span_stamp)
if  not os.path.isdir(db_out_path):
    os.mkdir(db_out_path)
print('The database out path is: %s \n'% db_out_path)
print('starting queries\n ***********************\n')
# end of general preamble
#%%
if  runcfg.general.localrun:
    sys.exit() # for now
    #TODO read from local file df_FWHM = pla.load_FWHM_csv(input_file_FWHM, fraction_to_read, start_reading_at_frac)
    
else:
    #%% focus data
    print('Reading focus data...\n')
    df_focus = pla.read_DB1(client,
                         db_name = 'operation_strings', 
                         rediskey_prefix = 'unitCS.set.FocusData:', 
                         extra = "value like '%LoopCompleted%:%true%' ",
                         startdate=startdate,
                         enddate = enddate,
                         N_days = N_days,
                         N_read = N_show)
    df_focus = pla.basic_processing_FWHM(df_focus)
    # df_focus = pla.filter_N_days(df_focus, N_days, N_show)
    df_focus["time"] = pd.to_datetime(df_focus["time"], errors="coerce")
    df_focus = df_focus.sort_values(by=[df_focus.index.name, "TimeStarted"])
    df_focus_csv_file_name = pla.save_df_to_csv(dfname='df_focus',
                                                outdir= db_out_path,
                                                time_span_stamp = time_span_stamp,
                                                df= df_focus)
    print('Saved %s \n'%df_focus_csv_file_name)
    #%% tracking data
    print('Reading tracking data... \n')
    df_tracking = pla.read_DB1(client=client,
                        db_name ='operation_strings', 
                        rediskey_prefix ='XerxesMountBinary.get.Status', 
                        extra = "value LIKE 'tracking'", 
                        startdate=startdate,
                        enddate = enddate,
                        N_days = N_days,
                        N_read = N_show)
    df_tracking =  pla.basic_processing_tracking(df_tracking)
    # tracking = pla.filter_N_days(tracking, N_days, N_show)
    df_tracking_csv_file_name = pla.save_df_to_csv(
                                df= df_tracking,
                                dfname='df_tracking',
                                outdir = db_out_path,
                                time_span_stamp = time_span_stamp)
    print('Saved %s \n'%df_tracking_csv_file_name)
    df_tracking = pla.separate_by_mount(df_tracking)
    df_tracking_groups = pla.tracking_windows(df_tracking) 
    
    #%% RA data
    print('Reading RA data... \n')
    df_RA = pla.read_DB1(client = client,
                     db_name ='operation_numbers',
                     rediskey_prefix = 'XerxesMountBinary.get.RA',
                     extra =None,
                     startdate = startdate,
                     enddate = enddate,
                     N_days = N_days,
                     N_read = N_show)
    df_RA = pla.basic_processing_tracking(df_RA)
    df_RA_csv_file_name = pla.save_df_to_csv(
                                    df = df_RA,
                                    dfname = 'df_RA',
                                    outdir = db_out_path,
                                    time_span_stamp = time_span_stamp)
    print('Saved %s \n'%df_RA_csv_file_name)
    # df_RA = pla.filter_N_days(df_RA, N_days, N_show) 
    RA_groups = pla.separate_by_mount(df_RA)
    
    #%% Dec data
    print('Reading Dec data... \n')
    df_Dec = pla.read_DB1(client= client,
                      db_name = 'operation_numbers',
                      rediskey_prefix = 'XerxesMountBinary.get.Dec',
                      extra = None, 
                      startdate = startdate, 
                      enddate = enddate,
                      N_days = N_days,
                      N_read = N_show)
    df_Dec = pla.basic_processing_tracking(df_Dec)
    # df_Dec = pla.filter_N_days(df_Dec, N_days, N_show)
    df_Dec_csv_file_name = pla.save_df_to_csv(
                            df = df_Dec,
                            dfname = 'df_Dec',
                            outdir = db_out_path,
                            time_span_stamp = time_span_stamp)
    print('Saved %s \n'%df_Dec_csv_file_name)
    df_Dec_groups = pla.separate_by_mount(df_Dec)
    #%% Az data
    print('Reading Az data... \n')
    df_Az = pla.read_DB1(client = client,
                     db_name='operation_numbers',
                     rediskey_prefix = 'XerxesMountBinary.get.Az',
                     extra = None, 
                     startdate = startdate, 
                     enddate = enddate,
                     N_days = N_days,
                     N_read =N_show)
    df_Az = pla.basic_processing_tracking(df_Az)
    # df_Az = pla.filter_N_days(df_Az, N_days, N_show)
    df_Az_csv_file_name = pla.save_df_to_csv(
                                df = df_Az,
                                dfname = 'df_Az',
                                outdir = db_out_path,
                                time_span_stamp = time_span_stamp)
    print('Saved %s \n'%df_Az_csv_file_name)
    Az_groups = pla.separate_by_mount(df_Az)
    #%% Alt data
    print('Reading Alt data... \n')
    df_Alt = pla.read_DB1(client = client,
                      db_name= 'operation_numbers',
                      rediskey_prefix = 'XerxesMountBinary.get.Alt',
                      extra = None, 
                      startdate = startdate, 
                      enddate = enddate,
                      N_days = N_days,
                      N_read =N_show)
    df_Alt = pla.basic_processing_tracking(df_Alt)
    # df_Alt = pla.filter_N_days(df_Alt, N_days, N_show)
    df_Alt_csv_file_name = pla.save_df_to_csv(
                                df = df_Alt,
                                dfname = 'df_Alt',
                                outdir = db_out_path,
                                time_span_stamp = time_span_stamp)
    print('Saved %s \n'%df_Alt_csv_file_name)
    Alt_groups = pla.separate_by_mount(df_Alt) 
    df_focus = pla.add_Alt_to_focus(df_focus, df_Alt)
    df_focus_csv_file_name = pla.save_df_to_csv(
                                df = df_focus,
                                dfname = 'df_focus',
                                outdir = db_out_path,
                                time_span_stamp = time_span_stamp)
    print('Saved %s \n'%df_focus_csv_file_name)
    
    focus_groups = pla.separate_by_mount(df_focus)
    #%% temperature data
    print('Reading mount temperatures...\n')
    df_Temp_1 = pla.read_DB1(client = client,
                            db_name = 'operation_numbers',
                            rediskey_prefix ='unitCS.get.Temperature:',
                            extra = "endsWith(rediskey, '.1')",
                            startdate = startdate, 
                            enddate = enddate,
                            N_days = N_days,
                            N_read =N_show)
    df_Temp_1 = pla.basic_processing_tracking(df_Temp_1)
    df_Temp_1_csv_file_name = pla.save_df_to_csv(
                                df = df_Temp_1,
                                dfname = 'df_Temp_1',
                                outdir = db_out_path,
                                time_span_stamp = time_span_stamp)
    print('Saved %s \n'%df_Temp_1_csv_file_name)
    # df_Temp_1 = pla.filter_N_days(df_Temp_1, N_days, N_show)
    df_Temp_1_groups = pla.separate_by_mount(df_Temp_1)
    
    df_Temp_2 = pla.read_DB1(client = client,
                            db_name = 'operation_numbers', 
                            rediskey_prefix = 'unitCS.get.Temperature:', 
                            extra = "endsWith(rediskey, '.2')",
                            startdate = startdate, 
                            enddate = enddate,
                            N_days = N_days,
                            N_read =N_show)
    df_Temp_2 = pla.basic_processing_tracking(df_Temp_2)
    # df_Temp_2 = pla.filter_N_days(df_Temp_2, N_days, N_show)
    df_Temp_2_csv_file_name = pla.save_df_to_csv(
                                df = df_Temp_2,
                                dfname = 'df_Temp_2',
                                outdir = db_out_path,
                                time_span_stamp = time_span_stamp)
    print('Saved %s \n'%df_Temp_2_csv_file_name)
    df_Temp_2_groups = pla.separate_by_mount(df_Temp_2)    
    df_Temp_groups = []
    for i in range(nmounts):
        df_Temp_1_groups[i].drop('scope',axis=1) # for the temperatures, we don't have scope info only mount
        df_Temp_2_groups[i].drop('scope',axis=1) # for the temperatures, we don't have scope info only mount
        '''Fill df_Temp_groups with the data of the smoother temperature sensor {1 or 2} for each mount'''
        df_Temp_groups.append(pla.smoother_df(df_Temp_1_groups[i],df_Temp_2_groups[i], col="value"))
# end of database reads
#%% preprocess the data
 # Create a filtered df_focus that only has data before 19:00
 # Extract hour (first two chars before '-')
df_focus = df_focus.dropna(subset=['HH-MM-DD-mm'])   # this protects agains nan in the following line
 #df_focus["hour"] = df_focus["HH-MM-DD-mm"].str.split("_").str[0].astype(int)
df_focus['hour'] = df_focus['HH-MM-DD-mm'].str.split('_').str[0].astype(float) + df_focus['HH-MM-DD-mm'].str.split('_').str[1].astype(float)/60.
df_focus['adjusted_hour'] = df_focus['hour']
df_focus.loc[df_focus['hour'] < 5, 'adjusted_hour'] += 24
# Keep only rows where 16 < hour < 19
df_focus_filtered = df_focus[(df_focus["hour"] > 15) & (df_focus["hour"] < 19)].drop(columns="hour")
# make sure Temperature is numeric
df_focus['Temperature'] = pd.to_numeric(df_focus['Temperature'], errors='coerce')
df_focus['BestPos'] = pd.to_numeric(df_focus['BestPos'], errors='coerce')

# compute delta_Temperature per mount.scope group (index)
df_focus['delta_Temperature'] = df_focus.groupby(df_focus.index)['Temperature']   \
                                .transform(lambda x: x - x.iloc[0])
# create BestPos_Temp as BestPos + delta_Temperature * 18
df_focus['BestPos_Temp'] = df_focus['BestPos'] - df_focus['delta_Temperature'] * 18

#%% start plots
if plots.fraction_bad_focus:
    pla.plot_bad_fit_fraction(df=df_focus,
                              output_directory=outdir)
    #%%
if plots.Focus_temperature_slope:
    df_medians = pla.plot_bestpos_vs_temp_by_mount(df = df_focus,
                                               output_directory= outdir,
                                               plots=plots,
                                               regular_fit = run.regular_fit,
                                               x_axis='Temperature',
                                               y_axis='BestPos')
    # df_focus = pla.filter_N_days(df_focus, N_days, N_show)
    temp = pla.plot_bestpos_vs_temp_by_mount(df = df_focus,
                                             output_directory= outdir,
                                             plots=plots,
                                             regular_fit = run.regular_fit,
                                             x_axis='adjusted_hour',
                                             y_axis='BestPos')
    temp2 =pla.plot_bestpos_vs_temp_by_mount(df=df_focus,
                                             output_directory= outdir,
                                             plots=plots,
                                             regular_fit = run.regular_fit,
                                             x_axis='adjusted_hour',
                                             y_axis='BestPos_Temp')
    #%%
if plots.Alt_for_each_Focus:
    pla.plot_alt_vs_hour(focus_groups = focus_groups, 
                         output_directory = outdir) #plots the Alt at which focus was performed for each mount vs time