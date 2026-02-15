#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 8 2026
Get data from the  last.visit.images and generate a report of image quality metrics for the LAST telescopes
Te report will be based on statistics over a defined pariod and will consist of:
FWHM bar graph per telescope
A table of the problematic telescopes
FWHM maps of the problematic telescopes and a map of a good (best) one.
The script reads a date range in a toml config file or just a date range
@author: micha
"""
import tomllib as tom
import sys
import pyLAST as pla
import pandas as pd
# import time
# import numpy as np
from datetime import datetime
import os
from tqdm import tqdm
import matplotlib
matplotlib.use('Qt5Agg') # or 'TkAgg'
configfile = sys.argv[1]

with open(configfile, "rb") as f:
    cfg = tom.load(f)

runcfg =  pla.runconfig(cfg)
inpath = runcfg.general.input_path
outpath = runcfg.general.output_path
dbpath = runcfg.general.database_path
startdate =runcfg.general.startdate
enddate = runcfg.general.enddate
nmounts = runcfg.general.nmounts
run = runcfg.run
plots = runcfg.plots
localrun = runcfg.general.localrun
fwhm_percentile = runcfg.general.fwhm_percentile
timestamp = datetime.now().strftime('%y%m%d%H%M')
# generate a directory for the current run:
    # for plots
outdir = os.path.join(runcfg.general.output_path, timestamp + '_output')
if  not os.path.isdir(outdir):
    os.mkdir(outdir)
# for databases query results csv files
time_span_stamp = \
    pla.generate_time_span_str(startdate,enddate)
db_out_path = os.path.join(dbpath,time_span_stamp[0])
if  not os.path.isdir(db_out_path):
    os.mkdir(db_out_path)
print('The database path is: %s \n'% db_out_path)
# TODO generate a function for this section
if localrun:
    vimg_csv_file_name = pla.save_df_to_csv(df=None,
                                            dfname='vimg_df',
                                            outdir=db_out_path,
                                            time_span_stamp=time_span_stamp[0])
    vimg_df = pd.read_csv(os.path.join(db_out_path, vimg_csv_file_name))

else:
    science = pla.LastDatabase(runcfg.databases.science)
    client = science.connect()

    # df=client.query_df('''SELECT dateobs,mountnum,camnum,ra,dec,cropid,fwhm,med_a,med_b,med_th,airmass
    #                    FROM last.visit_images
    #                    WHERE dateobs > '2025-11-20 00:00:00'
    #                    AND dateobs < '2025-11-23 00:00:00' '''
    #                    )

    vimg_df = pla.read_visitDB(client, startdate=startdate,
                               enddate=enddate)

    vimg_df_csv_file_name = pla.save_df_to_csv(df=vimg_df,
                                               dfname='vimg_df',
                                               outdir=db_out_path,
                                               time_span_stamp=time_span_stamp[0])
    print(f'saved data in: \n {vimg_df_csv_file_name}\n')
