#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 23:14:46 2025
Get data from the  last.visit.images and plot image quality metrics for
 telescopes
@author: micha
"""
# import matplotlib
# matplotlib.use('QtAgg')
import tomllib as tom
import sys
import pyLAST as pla
import pandas as pd
from datetime import datetime
import os
from matplotlib import pyplot as plt
# Read config file
configfile = sys.argv[1]
with open(configfile, "rb") as f:
    cfg = tom.load(f)
# generate a runconfig object
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
timestamp = datetime.now().strftime('%y%m%d%H%M')
# generate a directory for the current run for plots
outdir = os.path.join(runcfg.general.output_path, timestamp + '_output')
if  not os.path.isdir(outdir):
    os.mkdir(outdir)
# for databases qeury results csv files
time_span_stamp = \
    pla.generate_time_span_str(startdate = startdate,
                               enddate = enddate)
db_out_path = os.path.join(dbpath,time_span_stamp[0])
if  not os.path.isdir(db_out_path):
    os.mkdir(db_out_path)
print('The database path is: %s \n'% db_out_path)
#TODO generate a function for this section
if localrun:
    vimg_csv_file_name = pla.save_df_to_csv(df = None,
                                               dfname='vimg_df',
                                               outdir= db_out_path,
                                               time_span_stamp = time_span_stamp[0])
    vimg_df = pd.read_csv(os.path.join(db_out_path,vimg_csv_file_name))
                          
else:
    science = pla.LastDatabase(runcfg.databases.science)
    client = science.connect()
    # the database columns that are read are listed in the function
    vimg_df = pla.read_visitDB(client,startdate=startdate,
                          enddate=enddate)

    vimg_df_csv_file_name = pla.save_df_to_csv(df = vimg_df,
                                       dfname='vimg_df',
                                       outdir= db_out_path,
                                       time_span_stamp = time_span_stamp[0])
    print(f'saved data in: \n {vimg_df_csv_file_name}\n')

mountlist = [group for name, group in vimg_df.groupby('mountnum')]
imq =pd.DataFrame(columns =['mount','tel','crop','meanfwhm','stdfwhm'])
# loop on mounts and generate sub groups by telescopes and by fov regions (keys)
telmean=[]
telstd=[]
tel_labels=[]
for mount in mountlist:
    mountnum = int(mount['mountnum'].mean())
    print(f'analysing mount {mountnum}')
    tel_list = [group for name, group in mount.groupby('camnum')]
    cropfwhm =[]
    cropsfwhm =[] 
    axratio =[]
    telfwhm =[]
    telairmass =[]
    
    try:
        for tel in tel_list: # loop on the telescopes of each mount
            telnum = int(tel['camnum'].mean())
            print(f'Analysing tel{telnum}')
            # TODO filter here for 5% percentile of fwhm and airmass
            # fill in miising croipids with nans
            tel['cropid'] =pd.Categorical(tel['cropid'],categories=range(1,25))
            # generate a data frame of the different observations 
            # generate a dataframe with FWHM mean and std values per crop id
            cropfwhm_df = tel.groupby('cropid',observed = False)['fwhm'].agg(['mean','std'])
            # genrate a list of the 24 cropid data per this telescope
            cropid_list = [group for name, group in tel.groupby('cropid',observed = False)]
            # plt.scatter(cropid_list[8]['airmass'],cropid_list[8]['fwhm'])
            grouped_by_obsdate = tel.groupby('dateobs',observed = False)
            tel_obs_df = grouped_by_obsdate.agg({
                                                'fwhm':['mean','std'],
                                                'airmass':['mean']})
            telfwhm.append(tel_obs_df['fwhm']['mean'])
            telairmass.append(tel_obs_df['airmass']['mean'])
            tel_labels.append(f'{mountnum}.{telnum}')
            telmean.append(cropfwhm_df['mean'].agg('mean'))
            telstd.append(cropfwhm_df['std'].agg('mean'))
            cropxvals = cropfwhm_df.index.values
            cropfwhm.append(cropfwhm_df['mean'].values) # mean values of cropids per telescope
            cropsfwhm.append(cropfwhm_df['std'].values)
            # calculate mean axis ratio for each telescope
            cropaxes_df = tel.groupby('cropid', observed=
            False)[['med_a', 'med_b']].agg('mean')
            cropaxes_df['axratio'] = cropaxes_df['med_a'] / cropaxes_df['med_b']
            axratio.append(cropaxes_df['axratio'].values)


        #plots per mount
        # plot FWHM errorbar plot vs cropid per telescope for a mount use telescope colors
        if plots.vimg_FWHM_vs_cropid:
            pla.plot_property_vs_cropid_per_mount(
                    vals=cropfwhm,
                    val_stds =cropsfwhm,
                    mountnum = mountnum,
                    property_name=('FWHM','[arcsec]'),
                    time_span_stamp= time_span_stamp,
                    outdir= outdir)

        # Plot fwhm maps for four telescopes on a mount
        if plots.vimg_FWHM_tel_maps:
           pla.plot_mount_telescope_maps(mountnum= mountnum,
                                         vals=cropfwhm, 
                                         property_name = ('mean FWHM','[arcsec]'),
                                         time_span_stamp = time_span_stamp,
                                         outdir= outdir)
           plt.close('all')


        # Plot axis ratio maps for four telescopes on a mount
        if plots.vimg_axis_ratio_tel_maps:  
           pla.plot_mount_telescope_maps(mountnum= mountnum,
                                         vals=axratio, 
                                         property_name = ('mean axis ratio',''),
                                         time_span_stamp = time_span_stamp,
                                         outdir= outdir)
        
    except Exception as e:
        print(e)
        print(f'Problem with mapping mount {mountnum} - {e}')
        continue
    
if plots.vimg_FWHM_tel_stats:
    pla.plot_tel_stats(vals=telmean,
                       val_stds=telstd,
                       tel_labels=tel_labels,
                       property_name=('vimgFWHM', '[arcsec]'),
                       time_span_stamp=time_span_stamp,
                       outdir=outdir,
                       colorindex=None)

    

plt.close('all')
print('Finished')