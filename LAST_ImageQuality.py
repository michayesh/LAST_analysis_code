#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 23:14:46 2025
Get data from the  last.visit.images and plot image quality metrics for
 telescopes
@author: micha
"""

import tomllib as tom
import sys
import pyLAST as pla
import pandas as pd
# import time
import numpy as np
from datetime import datetime
import os
from matplotlib import pyplot as plt
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
#TODO generate a function for thsi section
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


mountlist = [group for name, group in vimg_df.groupby('mountnum')]

imq =pd.DataFrame(columns =['mount','tel','crop','meanfwhm','stdfwhm'])
# mountdict = {group_key: group_df for group_key, group_df in vimg_df.groupby('mountnum')}
# mountlist =[mountdict[key] for key in mountdict.keys()]
#%% loop on mounts and generate sub groups by telescopes and by fov regions (keys)
newrows=[]

for mount in mountlist:
    mountnum = int(mount['mountnum'].mean())
    print(f'mapping mount {mountnum}')
    tel_list = [group for name, group in mount.groupby('camnum')]
    # fig,ax = plt.subplots(figsize = (12,8))
    # fig.suptitle(f'Mount {mountnum}',fontsize = 18)
    fwhm =[]
    axratio =[]
    telmean=[]
    telstd=[]
    # sfwhm =[] 
    try:
        for tel in tel_list:
            telnum = int(tel['camnum'].mean())
            print(f'Mapping tel{telnum}')
            # fill in miising croipids with nans
            tel['cropid'] =pd.Categorical(tel['cropid'],categories=range(1,25))
            cropfwhm_df = tel.groupby('cropid',observed = False)['fwhm'].agg(['mean','std'])
            telmean.append(cropfwhm_df['mean'].agg('mean'))
            telstd.append(cropfwhm_df['std'].agg('mean'))
            xvals = cropfwhm_df.index.values
            fwhm.append(cropfwhm_df['mean'].values)
            # calculate mean axis ratio for each telescope
            cropaxes_df = tel.groupby('cropid',observed = False)[['med_a','med_b']].agg('mean')
            cropaxes_df['axratio']=cropaxes_df['med_a']/cropaxes_df['med_b']
            axratio.append(cropaxes_df['axratio'].values)
            # sfwhm.append(cropfwhm_df['std'].values)
        f1= pla.plot_mount_telescope_maps(mountnum= mountnum,
                                         vals=fwhm, 
                                         property_name = ('mean FWHM','[pix]'),
                                         time_span_stamp = time_span_stamp,
                                         outdir= outdir)
        f2= pla.plot_mount_telescope_maps(mountnum= mountnum,
                                         vals=axratio, 
                                         property_name = ('mean axis ratio',''),
                                         time_span_stamp = time_span_stamp,
                                         outdir= outdir)
        
    except Exception as e:
        print(e)
        print(f'Problem with mapping mount {mountnum} - {e}')
        continue
            
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
    

print('Finished')