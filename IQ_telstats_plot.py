import sys
import pyLAST as pla
import pandas as pd
import glob
from datetime import datetime
import os
import matplotlib
matplotlib.use('Qt5Agg') # or 'TkAgg'
from matplotlib import pyplot as plt
csv_folder = sys.argv[1]
telstat_csv_file_list = glob.glob(os.path.join(csv_folder,'*.csv'))
if not telstat_csv_file_list:
    print('No csv files found. Exiting.')
    sys.exit()
telstat_df_list = []
for csv_file in telstat_csv_file_list:
    telstat_df_list.append(pd.read_csv(csv_file))
fig,ax = plt.subplots(figsize=(20,8))
property_name = ('FWHM','arcsec')
for index,telstat_df in enumerate(telstat_df_list):
    tel_labels = telstat_df['tel_labels'].astype(str)
    vals =telstat_df['telmean'].values
    val_stds = telstat_df['telstd'].values
    ax.bar(tel_labels, vals, color=pla.tabcolors[index], capsize=4)
    ax.errorbar(tel_labels, vals,
                yerr=val_stds,
                fmt='none',
                capsize=0,
                color='black')
ax.grid(axis='y')
ax.tick_params(axis='x',labelsize=10)
ax.tick_params(axis='y',labelsize=12)
ax.set_ylabel('%s %s'%(property_name[0],property_name[1]),fontsize=14)
ax.set_xlabel('Telescope Label',fontsize =14)
# ax.set_title(f'{property_name[0]} telescope stats {time_span_stamp[1]} - {time_span_stamp[2]}', fontsize = 18)
# plt.savefig(os.path.join(outdir,property_name[0]+'_telescope_stats.png'))
print('Finished')