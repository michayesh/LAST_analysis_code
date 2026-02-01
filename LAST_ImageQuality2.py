#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 2026
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
# import time
# import numpy as np
from datetime import datetime
import os
from tqdm import tqdm
import matplotlib
matplotlib.use('Qt5Agg') # or 'TkAgg'
from matplotlib import pyplot as plt

from pyLAST import debugger_is_active

# import clickhouse_connect
# from clickhouse_driver import Client
# from pydantic import BaseModel

configfile = sys.argv[1]

with open(configfile, "rb") as f:
    cfg = tom.load(f)

runcfg =  pla.runconfig(cfg)

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
localrun = runcfg.general.localrun
fwhm_percentile = runcfg.general.fwhm_percentile
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

# df=client.query_df('''SELECT dateobs,mountnum,camnum,ra,dec,cropid,fwhm,med_a,med_b,med_th,airmass
#                    FROM last.visit_images 
#                    WHERE dateobs > '2025-11-20 00:00:00'
#                    AND dateobs < '2025-11-23 00:00:00' '''
#                    )

    vimg_df = pla.read_visitDB(client,startdate=startdate,
                          enddate=enddate)

    vimg_df_csv_file_name = pla.save_df_to_csv(df = vimg_df,
                                       dfname='vimg_df',
                                       outdir= db_out_path,
                                       time_span_stamp = time_span_stamp[0])
    print(f'saved data in: \n {vimg_df_csv_file_name}\n')

# grouped_dfs = {group_key: group_df for group_key, group_df in df.groupby('Category')}

# Accessing a specific sub-DataFrame
# print(grouped_dfs['B'])
# if localrun:
#     imq_csv_file_name = pla.save_df_to_csv(df = None,
#                                                dfname='imq_df',
#                                                outdir= db_out_path,
#                                                time_span_stamp = time_span_stamp[0])
#     imq_df = pd.read_csv(os.path.join(db_out_path,imq_csv_file_name))
# else:

if not runcfg.general.imq_df_exists:
    obslist = [group for name, group in vimg_df.groupby('dateobs')]
    imqdictlist =[] # empty list of dicts
    for index,curobs in tqdm(enumerate(obslist),total=len(obslist)):
        #for each obsdate in principle cam be several mounts and telescopes in the group
        # separate to mounts
        mountlist = [group for name, group in curobs.groupby('mountnum')]
        for mount in mountlist:
            mountnum = int(mount['mountnum'].mean())
            # if pla.debugger_is_active():
            # print(f'Analysing mount {mountnum}')
            tel_list = [group for name, group in mount.groupby('camnum')]
            for tel in tel_list:
                telnum = int(tel['camnum'].mean())
                # if pla.debugger_is_active():
                # print(f'Analysing tel {telnum}')
                # find if some of the cropid's are missing
                missing_crops = pla.find_missing_crops(tel['cropid'].values)
                if len(missing_crops)>0:
                    # print(f'M{mountnum}T{telnum}:Missing crops on {tel['dateobs'].iloc[0]}: {missing_crops}')
                    continue
                #  generate the cropid 24 element fwhm vector
                fwhmvec = tel['fwhm'].values
                # generate the various metrics from this vector
                metrics = pla.telmoments(fwhmvec)
                # populate a dict with the data variables
                rowdict ={'dateobs':tel['dateobs'].iloc[0],
                          'mountnum':mountnum,
                          'telnum':telnum,
                          'ra_mean':tel['ra'].mean(),
                          'ra_std':tel['ra'].std(),
                          'dec_mean':tel['dec'].mean(),
                          'dec_std':tel['dec'].std(),
                          'airmass': tel['airmass'].mean(),
                          'fwhm_mean':metrics[0],
                          'fwhm_center':metrics[1],
                          'fwhm_depth':metrics[2],
                          'fwhm_perivar':metrics[3],
                          'fwhm_updown':metrics[4],
                          'fwhm_leftright':metrics[5],
                          'fwhm_xuldr':metrics[6],
                          'fwhm_xurdl':metrics[7]
                          }
        imqdictlist.append(rowdict)
        # i+=1
        # print('%.1f \n'%(i/num_of_obs*100))
    imq_df = pd.DataFrame(imqdictlist)
    imq_df_csv_file_name = pla.save_df_to_csv(df = imq_df,
                                       dfname='imq_df',
                                       outdir= db_out_path,
                                       time_span_stamp = time_span_stamp[0])
    print(f'saved data in: \n {imq_df_csv_file_name}\n')
else:
    imq_df_csv_file_name = pla.save_df_to_csv(df=None,
                                          dfname='imq_df',
                                          outdir= db_out_path,
                                          time_span_stamp = time_span_stamp[0])
    imq_df = pd.read_csv(os.path.join(db_out_path,imq_df_csv_file_name))
if plots.imq_metrics_vs_airmass:
    pla.plot_imq_metrics_vs_airmass(imq_df,
                                    fwhm_percentile = fwhm_percentile,
                                    time_span_stamp =time_span_stamp,
                                    outdir = outdir,
                                    condition = runcfg.general.filter_condition)

print('Finished')


# plot the metrics vs airmass
# imq =pd.DataFrame(columns =['mount','tel','crop','meanfwhm','stdfwhm'])
# mountdict = {group_key: group_df for group_key, group_df in vimg_df.groupby('mountnum')}
# mountlist =[mountdict[key] for key in mountdict.keys()]
#%% loop on mounts and generate sub groups by telescopes and by fov regions (keys)
# newrows=[]
# telmean=[]
# telstd=[]
# teldelta =[]
# tel_labels=[]
# for mount in mountlist:
#     mountnum = int(mount['mountnum'].mean())
#     print(f'analysing mount {mountnum}')
#     tel_list = [group for name, group in mount.groupby('camnum')]
#     # fig,ax = plt.subplots(figsize = (12,8))
#     # fig.suptitle(f'Mount {mountnum}',fontsize = 18)
#     cropfwhm =[]
#     cropsfwhm =[]
#     axratio =[]
#     telfwhm =[]
#     telairmass =[]
#     #TODO filter here for airmass
#
#     try:
#         for tel in tel_list: # loop on the telescopes of each mount
#             telnum = int(tel['camnum'].mean())
#             print(f'Analysing tel{telnum}')
#             # TODO filter here for 5% percentile of fwhm and airmass
#             # fill in miising croipids with nans
#             tel['cropid'] =pd.Categorical(tel['cropid'],categories=range(1,25))
#             # generate a data frame of the different observations
#             # generate a dataframe with FWHM mean and std values per crop id
#             cropfwhm_df = tel.groupby('cropid',observed = False)['fwhm'].agg(['mean','std'])
#             # genrate a list of the 24 cropid data per this telescope
#             cropid_list = [group for name, group in tel.groupby('cropid',observed = False)]
#             # plt.scatter(cropid_list[8]['airmass'],cropid_list[8]['fwhm'])
#             grouped_by_obsdate = tel.groupby('dateobs',observed = False)
#             tel_obs_df = grouped_by_obsdate.agg({
#                                                 'fwhm':['mean','std'],
#                                                 'airmass':['mean']})
#             telfwhm.append(tel_obs_df['fwhm']['mean'])
#             telairmass.append(tel_obs_df['airmass']['mean'])
#             tel_labels.append(f'{mountnum}.{telnum}')
#             telmean.append(cropfwhm_df['mean'].agg('mean'))
#             telstd.append(cropfwhm_df['std'].agg('mean'))
#             # teldelta.append(cropfwhm_df['mean'].max()-cropfwhm_df['mean'].min())
#             cropxvals = cropfwhm_df.index.values
#             cropfwhm.append(cropfwhm_df['mean'].values) # mean values of cropids per telescope
#             cropsfwhm.append(cropfwhm_df['std'].values)
#         #plots per mount
#         # plot FWHM errorbar plot vs cropid per telescope for a mount use telescope colors
#         if plots.vimg_FWHM_vs_cropid:
#             pla.plot_property_vs_cropid_per_mount(
#                     vals=cropfwhm,
#                     val_stds =cropsfwhm,
#                     mountnum = mountnum,
#                     property_name=('FWHM','[arcsec]'),
#                     time_span_stamp= time_span_stamp,
#                     outdir= outdir)
#         # plot FWHM vs airmass per telescope
#         if plots.vimg_FWHM_vs_airmass:
#             pla.telescope_scatterplot_per_mount(mountnum= mountnum,
#                                           xvals = telairmass,
#                                           yvals=telfwhm,
#                                           xproperty_name = ('airmass',''),
#                                           yproperty_name = ('mean FWHM','[arcsec]'),
#                                           time_span_stamp = time_span_stamp,
#                                           outdir= outdir)
#
#         #                                               )
#         # Plot fwhm maps for four telescopes on a mount
#         if plots.vimg_FWHM_tel_maps:
#            pla.plot_mount_telescope_maps(mountnum= mountnum,
#                                          vals=cropfwhm,
#                                          property_name = ('mean FWHM','[arcsec]'),
#                                          time_span_stamp = time_span_stamp,
#                                          outdir= outdir)
#         # calculate mean axis ratio for each telescope
#         cropaxes_df = tel.groupby('cropid',observed =
#                                   False)[['med_a','med_b']].agg('mean')
#         cropaxes_df['axratio']=cropaxes_df['med_a']/cropaxes_df['med_b']
#         axratio.append(cropaxes_df['axratio'].values)
#         # Plot axis ratio maps for four telescopes on a mount
#         if plots.vimg_axis_ratio_tel_maps:
#            pla.plot_mount_telescope_maps(mountnum= mountnum,
#                                          vals=axratio,
#                                          property_name = ('mean axis ratio',''),
#                                          time_span_stamp = time_span_stamp,
#                                          outdir= outdir)
#
#     except Exception as e:
#         print(e)
#         print(f'Problem with mapping mount {mountnum} - {e}')
#         continue
#
# if plots.vimg_FWHM_tel_stats:
#     pla.plot_tel_stats(vals=telmean,
#                            val_stds= telstd,
#                            tel_labels= tel_labels,
#                            property_name = ('vimgFWHM','[arcsec]'),
#                            time_span_stamp = time_span_stamp,
#                            color = 'orange',
#                            outdir= outdir)

    
    #     yerrs = cropfwhm_df['std'].values
    #     ax.errorbar(xvals,yvals,yerrs,
    #                  color = pla.colors[telnum-1],
    #                  label = f'tel{telnum}',
    #                  marker ='o',
    #                  markersize = 6)
    #     ax.set_xticks(np.arange(1,25,1))
    #     ax.tick_params(axis='both', labelsize = 14)
        
    # # ax0.set_ylim([0,5])
    # ax.set_ylabel('FWHM [pix]',fontsize =14)
    # ax.set_xlabel('cropnum',fontsize = 14)
    # ax.grid(True)
    # ax.legend()
    # plt.show()
    
    # telgroups = mount.groupby('camnum')
    # telstat = telgroups['fwhm'].agg(['mean','std'])
    # teldflist = [group for name,group in telgroups] 
    # for teldf in teldflist:
    #     telnum = int(teldf['camnum'].mean())
    #     for crop in range(1,25):
    #         newrows.append({'mount':mountnum,
    #                   'tel':telnum,
    #                   'mount':mountnum,
    #                   'crop':crop,
    #                 'meanfwhm':float(teldf[teldf['cropid']==crop]['fwhm'].mean()),
    #                 'stdfwhm': teldf[teldf['cropid']==crop]['fwhm'].std()
    #                 })
            # telnum = telgroups.index.values
            # numoftels = len(telnum)
            # telmeans = telstat['mean'].values
            # telstds = telstat['std'].values
      
# imq = pd.concat([imq,pd.DataFrame(newrows)],ignore_index=True)
# imq_df_csv_file_name = pla.save_df_to_csv(df = imq,
#                                        dfname='imq',
#                                        outdir= db_out_path,
#                                        time_span_stamp = time_span_stamp)       
 
    
    # filter by fov keys
    # for  fovkey in fov.keys():
        
    #%% plot for each mount mean and std FWHM 
# fig, axes = plt.subplots(5, 2, figsize=(14, 20), sharex=True)
# for mountnum in range (1,nmounts+1):
#     for telnum in range(1,5):
#         average= imq[(imq['mount']==mountnum) & (imq['tel']==telnum)]['meanfwhm']
#         std = imq[(imq['mount']==mountnum) & (imq['tel']==telnum)]['stdfwhm']
           
    # if numoftels == 4:
    #     fig0,ax0 = plt.subplots(figsize=(12,8))
    #     tel_labels = ['1','2','3','4']
    #     tel_colors = pla.colors
    #     ax0.bar(telnum,telmeans, 
    #             color = tel_colors, 
    #             capsize =4,
    #             width = 0.1,
    #             label = tel_labels)
    #     ax0.errorbar(telnum,telmeans,
    #                 yerr = telstds,
    #                 fmt='none', 
    #                 capsize =0, 
    #                 color='red')
    #     ax0.grid(axis='y')
    #     ax0.tick_params(axis='x',labelsize=10)
    #     ax0.tick_params(axis='y',labelsize=12)
    #     ax0.set_ylabel('FWHM Minor Axis [pix]',fontsize=14)
    #     ax0.set_xlabel('Telescope #',fontsize =14)
    #     ax0.set_title(f'Mount {mountnum} \n Telescope Stats {spanstr[0]} - {spanstr[1]}', fontsize = 18)
    #     plt.savefig(os.path.join(outdir,'FWHM_telescope_stats.png'))
    #     plt.show()
    # else:
    #     print(f'Number of elescopes in mount {mountnum} is {numoftels}')
    #     continue
    

