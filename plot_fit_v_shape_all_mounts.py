#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 13:55:47 2025

@author: ocs
"""

import pandas as pd
import sqlalchemy
import json
import os
import csv
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FuncFormatter
import numpy as np
from collections import defaultdict
import clickhouse_connect
from clickhouse_driver import Client
import math

#from load_mat_analyze_fit5_07_09 import basic_processing_FWHM, keep_2_elements_only
# select the computer from which to run: LAST_0 or euclid
database = 'euclid' #'LAST_0'
euclidhost = '10.150.28.18'
euclidport = 9000
eucliduser ='default'
euclidpw = 'PassRoot'

if database == 'LAST_0':
    # LAST_0 as client using clickhouse_connect
    client = clickhouse_connect.get_client(host='10.23.1.25', port=8123, \
             username='last_user', password='physics', database='observatory_operation')        
elif database == 'euclid':
    # euclid as client using clickhouse_driver
    client = Client(host=euclidhost, port=euclidport, \
             user=eucliduser, password=euclidpw, database='observatory_operation')        


def read_DB(N_days, N_read, dB_name, rediskey_prefix, extra=None):
    '''A general function for reading a given range of days from dB_name 
    looking for rediskey prefix. Note, the extra parameter is not mandatory and
    is used to add another condition 
    N_days is the number of days before present to search for
    N_read limits the search to N_read days after the initial day, set to -1 for all
    dB_name is either operation_strings or operation_numbers
    examples of rediskey_prefix are: 'unitCS.set.FocusData:', 'XerxesMountBinary.get.Dec', 
    'XerxesMountBinary.get.Status'. An example of extra is: "value LIKE 'tracking'",
    "value like '%LoopCompleted%:%true%' "   '''
    if N_read == -1: N_read=N_days
    query_string = build_range_query(N_days, N_read, dB_name, rediskey_prefix,extra)
    print(query_string)

    if database == 'LAST_0':    
        result = client.query(query_string)
        df = pd.DataFrame(result.result_rows, columns=['rediskey', 'time', 'value']).set_index('rediskey')
    elif database == 'euclid':
        result = client.execute(query_string)
        df = pd.DataFrame(result, columns=["rediskey", "time", "value"]).set_index("rediskey")
        
    # Convert to DataFrame
    
    print('loaded ', len(df), 'items')
    print('Here is the first line:\n',df.tail(1))
    print('\n')
    return df


def build_range_query(N_days, N_read, table: str, rediskey_prefix: str, extra_condition=None) -> str:
    """ Build a ClickHouse SQL query for a given time range.

    N_days : int       Number of days ago to start (start at 12:00:00 that day).
    N_read : int       Number of days to include in the query range.
    table : str        Name of the ClickHouse table (e.g., 'operation_strings').
    rediskey_prefix : str  The prefix string for startsWith(rediskey, ...).
    extra_condition (str, optional): Additional SQL condition to append to WHERE
    Returns  query : str    SQL query string.  """
    # Compute start date (N days ago at noon)
    start_date = datetime.now() - timedelta(days=N_days)
    start_date = start_date.replace(hour=12, minute=0, second=0, microsecond=0)
    # Compute end date (start_date + N_read days)
    #end_date = start_date + timedelta(days=N_read)
    end_date = start_date + timedelta(hours=24)
    # Format for ClickHouse (YYYY-MM-DD HH:MM:SS)
    start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
    end_str   = end_date.strftime('%Y-%m-%d %H:%M:%S')

    # Build query
    query = f"""
        SELECT * FROM {table}
        WHERE startsWith(rediskey, '{rediskey_prefix}')
          AND time > '{start_str}'
          AND time < '{end_str}'    """
    # add extra condition if provided
    if extra_condition:
        query += f" AND {extra_condition}"   
    # always order by time descding
    query += " ORDER BY time DESC "         
    return query.strip()    

def keep_2_elements_only(s):
    '''removes centeral number only if it is = 1
    This function is used to remove redundant '1' in the mount.telescope numbers '''

    parts = s.split('.')
    if len(parts) == 3 and parts[1]=='1':
        return f"{parts[0]}.{parts[2]}"
    return s 

def basic_processing_FWHM(df):
    '''Basic processing includes preparing index col to only include mount.scope, replace _ by .
    THen, open the json and Create a col called HH-MM-DD-mm and (if missing) add a column TimeStarted
    returns the input df sorted by TimeStarted. Also, the minor and major columns are mult. by 2.355.
    Finally, empty fit rows are removed and only succesful fits maintained'''
    
    #df['value'] = df['value'].apply(json.loads)
    df['value'] = df['value'].apply(json.loads)
    df.index = df.index.str.split(':').str[-1]
    #repair some of the errors in the index
    df.index = df.index.astype(str).str.replace('_', '.', regex=False)
    #sometimes the mount.telescope numbers have a '.1' between them, this removes it
    df.index = df.index.map(keep_2_elements_only)
    
    
    # Step 2: Normalize top-level fields (excluding ResTable)
    top_level = pd.json_normalize(df['value'])

    top_level.index = df.index
    top_level_no_restable = top_level.drop(columns='ResTable', errors='ignore')
    #print('original size', top_level_no_restable.shape)
    # Step 3: Extract and flatten ResTable into wide format
    restables = []
    
    # Loop over each row to flatten ResTable into columns
    '''Currently, Here I need the restable so am adding it'''
    require_also_ResTable = True
    if require_also_ResTable:
        for idx, row in top_level.iterrows():
            row_data = {}
            restable = row.get('ResTable', None)
        
            # Make sure ResTable is a list of dicts
            if isinstance(restable, list):
                for i, item in enumerate(restable):
                    if isinstance(item, dict):
                        for key, value in item.items():
                            col_name = f'ResTable_{i}.{key}'
                            row_data[col_name] = value
            # Build a Series for the row
            row_series = pd.Series(row_data, name=idx)
            restables.append(row_series)
        
        # Combine ResTable expansions into a DataFrame
        restable_df = pd.DataFrame(restables)
    
    # Step 4: Combine everything and add a time column
    if require_also_ResTable:
        final_df = pd.concat([top_level_no_restable, restable_df], axis=1)
    else:
        final_df = pd.concat([df['time'], top_level_no_restable], axis=1)
    #Create a column in HH-dd-mm notation for the start time
    if 'TimeEnded' in final_df.columns:
        final_df["HH-MM-DD-mm"] = final_df["TimeEnded"].apply(lambda jd: julian_to_ddmm(jd, fmt='%H_%M_%d_%m'))
    # if TimeStarted coloumn is missing, generate it from time    
    elif 'time' in final_df.columns and 'TimeStarted' not in final_df.columns:
        final_df["HH-MM-DD-mm"] = pd.to_datetime(final_df['time']).dt.strftime('%H_%M_%d_%m')
        final_df['time'] = pd.to_datetime(final_df['time'], errors='coerce')   
        # subtracting t0 (1970) to make a time-diff and converting to days and adding (1970-jd) to convert to jd
        final_df['TimeStarted'] = (final_df['time'] - t0).dt.total_seconds() / (24 * 3600) + 2440587.5

    '''for FWHM data need to convert minor and major columns by multiplying by 2*sqrt(ln(4)) = 2.355'''
    if 'minor' in final_df.columns:
        final_df['minor'] = final_df['minor'] * 2.355 
    if 'major' in final_df.columns:
        final_df['major'] = final_df['major'] * 2.355     
        
    #Now we filter:
    #final_df = final_df[final_df["Status"] != "Fault"]
    # if 'LoopCompleted' in final_df.columns:
    #     final_df = final_df[final_df["LoopCompleted"] == True]
    # '''for focus we want to filter out bad fit attempts'''
    # if 'FitParam' in df:
    #     df = [df["FitParam"] != [None, None, None]] # remove focus sesstions that did not produce a good Fit
    #     #df = [df['Status'] == ['Found.']]  keep only good "Found" focuses
    #final_df = final_df.sort_values(by="Counter").sort_index()
    #return final_df
    return final_df.sort_values(by='TimeStarted', ascending=True)

def julian_to_ddmm(jd, fmt='%d-%m'):
    """ Convert Julian Day number(s) to formatted date string(s).
    Parameters:
        jd (float or array-like): Julian Day(s)
        fmt (str): datetime.strftime format string (default '%d-%m')
    Returns: str or pd.Series: formatted date string(s)   """    
    # Convert input to numpy array for uniformity
    jd_arr = np.atleast_1d(jd)
    ts = pd.to_datetime((jd_arr - 2440587.5) * 86400, unit='s', utc=True)
    # Format dates as strings
    formatted = ts.strftime(fmt)
    # If input was scalar, return single string, else return Series
    if np.isscalar(jd):
        return formatted[0]
    else:
        return pd.Series(formatted, index=None)    

def make_focpos_fwhm_dict(row, n_max=20):
    ''''For each of the focusing positions open column ResTable_{n}
    and extract FocPos and FWHM and create return as a dictionary called pairs'''
    pairs = {}
    for n in range(n_max):
        foc_col = f"ResTable_{n}.FocPos"
        fwhm_col = f"ResTable_{n}.FWHM"
        if foc_col in row and fwhm_col in row:
            # Only add if both are not null
            if pd.notna(row[foc_col]) and pd.notna(row[fwhm_col]):
                pairs[row[foc_col]] = row[fwhm_col]
    return pairs

def _safe_float(val):
    """Return a float when possible. Handles list/tuple/scalars and None/NaN.
       Returns None if conversion not possible."""
    if val is None:
        return None
    # if it's a list/tuple/np.ndarray, take first element (MATLAB-style)
    if isinstance(val, (list, tuple, np.ndarray)):
        if len(val) == 0:
            return None
        val = val[0]
    try:
        # use pandas isna to catch NaN-like
        if pd.isna(val):
            return None
    except Exception:
        pass
    try:
        return float(val)
    except Exception:
        return None


def plot_focus_by_index(df, zoom=True, inset_width="40%", inset_height="40%", inset_loc="upper center"):
    """
    Plot focus data grouped by unique index values.
    - Blue dots: measurement points
    - Red + : BestPos/BestFWHM (numbered only in inset)
    - Black line: vshape fit
    """

    # Keep only completed loops, this is redundant, since we only read completed from dB
    if "LoopCompleted" in df.columns:
        df = df[df["LoopCompleted"] == True]

    unique_indices = df.index.unique()
    n_plots = len(unique_indices)
    if n_plots == 0:
        print("No data (no unique indices after filtering).")
        return

    fig, axes = plt.subplots(n_plots, 1, figsize=(6, 4 * n_plots), squeeze=False)

    for ax, idx in zip(axes.flat, unique_indices):
        rows = df.loc[[idx]]
        mount_str, scope_str = idx.split(".")
        mount_num = int(mount_str.lstrip("0"))  # convert "09" -> 9, "10" -> 10
        scope_num = int(scope_str)              # scope is usually 1-4
        
        # lookup in df_medians
        row_median = df_medians[ (df_medians['mount'] == mount_num) & (df_medians['scope'] == scope_num) ]
    
        # inset created once per subplot
        axins = inset_axes(ax, width=inset_width, height=inset_height, loc=inset_loc) if zoom else None
        
        #plot the median value + std
        if not row_median.empty:
            median_val = row_median['median'].values[0]
            std_val = row_median['std'].values[0]
        
            if pd.notna(median_val) and pd.notna(std_val):
                if axins is not None:
                    axins.axvline(x=median_val, color='green', linestyle='-', linewidth=1.5, zorder=5)
                    axins.hlines(y=4, xmin=median_val - std_val, xmax=median_val + std_val,
                                 color='green', linestyle='-', linewidth=1.5, zorder=5)
        x_mins, x_maxs, fit_param_list, centers = [], [], [], []

        for j, (_, row) in enumerate(rows.iterrows(), start=1):
            points = row.get("Points", {})
            if isinstance(points, dict) and len(points) > 0:
                valid_pairs = [( _safe_float(k), _safe_float(v) )
                               for k, v in points.items()
                               if _safe_float(k) is not None and _safe_float(v) is not None]
                if not valid_pairs:
                    continue

                x_vals = np.array([p[0] for p in valid_pairs])
                y_vals = np.array([p[1] for p in valid_pairs])

                x_mins.append(np.nanmin(x_vals))
                x_maxs.append(np.nanmax(x_vals))

                # main plot points
                ###ax.scatter(x_vals, y_vals, color="blue", s=30, zorder=2)
                ###if axins is not None:
                ###    axins.scatter(x_vals, y_vals, color="blue", s=12, zorder=2)
                # check status
                status = row.get("Status", "")
        
                if status == "Found.":
                    # filled blue dots
                    ax.scatter(x_vals, y_vals, color="blue", s=30, zorder=2)
                    if axins is not None:
                        axins.scatter(x_vals, y_vals, color="blue", s=12, zorder=2)
                else:
                    # hollow blue circles
                    ax.scatter(x_vals, y_vals, facecolors="none", edgecolors="blue", s=30, zorder=2)
                    if axins is not None:
                        axins.scatter(x_vals, y_vals, facecolors="none", edgecolors="blue", s=12, zorder=2)

                # Best point
                bestpos = _safe_float(row.get("BestPos"))
                bestfwhm = _safe_float(row.get("BestFWHM"))
                if bestpos is not None and bestfwhm is not None:
                    if status == "Found.":
                        main_size, inset_size = 300, 150  # larger marker
                    else:
                        main_size, inset_size = 80, 40   # smaller marker
                
                    ax.scatter(bestpos, bestfwhm, color="red", marker="+", s=main_size, zorder=3)
                
                    if axins is not None:
                        axins.scatter(bestpos, bestfwhm, color="red", marker="+", s=inset_size, zorder=3)
                        # numbering only in inset
                        axins.text(bestpos - 10, bestfwhm + 0.1, str(j),
                                   color="red", fontsize=8, ha="center", va="bottom", zorder=4)
                # Fit params
                fit_param = row.get("FitParam")
                if isinstance(fit_param, (list, tuple)) and len(fit_param) == 3:
                    center = _safe_float(fit_param[0])
                    offset = _safe_float(fit_param[1])
                    slope = _safe_float(fit_param[2])
                    if None not in (center, offset, slope):
                        fit_param_list.append((slope, center, offset))
                        centers.append(center)

        # draw fits
        if x_mins and x_maxs:
            x_fit = np.linspace(min(x_mins), max(x_maxs), 400)
            for (slope, center, offset) in fit_param_list:
                y_fit = vshape(x_fit, slope, center, offset)
                ax.plot(x_fit, y_fit, "k-", linewidth=1.0, zorder=1)
                if axins is not None:
                    axins.plot(x_fit, y_fit, "k-", linewidth=0.8, zorder=1)

        # configure inset
        if axins is not None:
            if centers:
                b_center = float(np.median(centers))
                axins.set_xlim(b_center - 200, b_center + 150)
            elif x_mins and x_maxs:
                mid = 0.5 * (min(x_mins) + max(x_maxs))
                axins.set_xlim(mid - 200, mid + 150)
            axins.set_ylim(-2, 10)
            formatter = FuncFormatter(lambda x, _: f"{int(round(x)) % 1000:03d}")
            axins.xaxis.set_major_formatter(formatter)
            axins.grid(True)
            axins.tick_params(labelsize=8)

        # main axes cosmetics
        ax.set_title(f"Index: {idx}")
        ax.set_ylim(-1, 40)
        ax.set_xlabel("FocPos")
        ax.set_ylabel("FWHM")
        ax.grid(True)

    plt.tight_layout()
    label = 'fit'
    filename = f'{mount_num}.{scope_num}'
    plot_saving(output_directory, filename, label)
    plt.show()
    dx = [-100, -80, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 80, 100, 120]
    y_eran = []
    for delta in dx:
        y_eran.append(np.round(float(vshape(center+delta, slope, center, offset)),3))
    with open("output.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([mount_num, scope_num, *y_eran])


def plot_focus_pages(df, zoom=True, inset_width="40%", inset_height="40%", inset_loc="upper center"):
    """
    Plot focus traces with one trace per subplot.
    - 12 subplots per page (3x4 grid)
    - 4 pages total (one per scope)
    - Each subplot includes an inset zoom
    Uses global df_medians and output_directory.
    """

    global df_medians, output_directory

    if "LoopCompleted" in df.columns:
        df = df[df["LoopCompleted"] == True]

    # Parse mount/scope from index "XX.Y"

    df[['mount_str', 'scope_str']] = df['level_0'].str.split('.', expand=True)
    df['mount'] = df['mount_str'].astype(int)
    df['scope'] = df['scope_str'].astype(int)
    mount_num = df['mount'][0]
    for scope_num in sorted(df['scope'].unique()):
        df_scope = df[df['scope'] == scope_num]
        if df_scope.empty:
            continue

        # Sort to make plots consistent
        df_scope = df_scope.sort_values(['TimeStarted'])
        traces_per_page = 12
        unique_indices = df_scope.index.unique()
        n_pages = math.ceil(len(unique_indices) / traces_per_page)
        
        for page in range(n_pages):
            start = page * traces_per_page
            end = start + traces_per_page
            df_page = df_scope.loc[unique_indices[start:end]]
        
            fig, axes = plt.subplots(3, 4, figsize=(16, 9), squeeze=False)
            fig.suptitle(f"Mount {mount_num} Scope {scope_num} – Page {page + 1}", fontsize=14)
            axes = axes.flatten()
            i=0
            for ax, (_, row) in zip(axes, df_page.iterrows()):
                i+=1
                points = row.get("Points", {})
                if not isinstance(points, dict) or len(points) == 0:
                    ax.axis("off")
                    continue
        
                valid_pairs = [
                    (_safe_float(k), _safe_float(v))
                    for k, v in points.items()
                    if _safe_float(k) is not None and _safe_float(v) is not None
                ]
                if not valid_pairs:
                    ax.axis("off")
                    continue
                # add inset
                axins = inset_axes(ax, width=inset_width, height=inset_height, loc=inset_loc)
                
                # plot median line + std range
                row_median = df_medians[(df_medians['mount'] == mount_num) & (df_medians['scope'] == scope_num)]
                if not row_median.empty:
                    median_val = row_median['median'].values[0]
                    std_val = row_median['std'].values[0]
                    if pd.notna(median_val) and pd.notna(std_val) and axins is not None:
                        axins.axvline(x=median_val, color='green', linestyle='-', linewidth=1.5)
                        axins.hlines(y=4, xmin=median_val - std_val, xmax=median_val + std_val, color='green')
                
        
                x_vals = np.array([p[0] for p in valid_pairs])
                y_vals = np.array([p[1] for p in valid_pairs])
                #ax.scatter(x_vals, y_vals, color="blue", s=25)
                
                # check status
                status = row.get("Status", "")
        
                if status == "Found.":
                    # filled blue dots
                    ax.scatter(x_vals, y_vals, color="blue", s=30, zorder=2)
                    if axins is not None:
                        axins.scatter(x_vals, y_vals, color="blue", s=12, zorder=2)
                else:
                    # hollow blue circles
                    ax.scatter(x_vals, y_vals, facecolors="none", edgecolors="blue", s=30, zorder=2)
                    if axins is not None:
                        axins.scatter(x_vals, y_vals, facecolors="none", edgecolors="blue", s=12, zorder=2)
                if axins is not None:
                   axins.scatter(x_vals, y_vals, color="blue", s=10)
                
                
                # Best point
                bestpos = _safe_float(row.get("BestPos"))
                bestfwhm = _safe_float(row.get("BestFWHM"))
                if bestpos is not None and bestfwhm is not None:
                    if status == "Found.":
                        main_size, inset_size = 300, 150  # larger marker
                    else:
                        main_size, inset_size = 80, 40   # smaller marker
                
                    ax.scatter(bestpos, bestfwhm, color="red", marker="+", s=main_size, zorder=3)
                
                    if axins is not None:
                        axins.scatter(bestpos, bestfwhm, color="red", marker="+", s=inset_size, zorder=3)
                        
                        
                # fit curve
                fit_param = row.get("FitParam")
                if isinstance(fit_param, (list, tuple)) and len(fit_param) == 3:
                    #slope, center, offset = fit_param
                    center = _safe_float(fit_param[0])
                    offset = _safe_float(fit_param[1])
                    slope = _safe_float(fit_param[2])
                    if None not in (slope, center, offset):
                        x_fit = np.linspace(min(x_vals), max(x_vals), 400)
                        y_fit = vshape(x_fit, slope, center, offset)
                        ax.plot(x_fit, y_fit, "k-", linewidth=1)
                        if axins is not None:
                            axins.plot(x_fit, y_fit, "k-", linewidth=0.8)
                            axins.set_xlim(center - 150, center + 150)
                            axins.set_ylim(0, 6)
                            axins.grid(True)
                            axins.tick_params(labelsize=7)
                            axins.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(round(x)) % 1000:03d}"))
                
                text_str = f"{center:5.0f}" if pd.notna(center) else "  NaN "
                ax.text(
                    0.95, 0.05, text_str,
                    transform=ax.transAxes,
                    fontsize=12,
                    color="red",
                    ha="right",
                    va="bottom",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="gray", boxstyle="round,pad=0.3")
                )
                            
                
                
                hour = "_".join(row.get("HH-MM-DD-mm", "").split('_')[0:2])
                Temperature = row.get("Temperature", "")
                # aesthetics
                ax.set_title(f"Focus {i} at {hour} T:{Temperature}   {status}", fontsize=11)
                ax.set_xlabel("FocPos")
                ax.set_ylabel("FWHM")
                ax.set_ylim(-1, 40)
                ax.grid(True)
            # hide extra axes if fewer than 12 rows
            for ax in axes[len(df_page):]:
                ax.axis("off")
        
            plt.tight_layout()
            label = 'fit'
            filename = f'separated_x12_{mount_num}.{scope_num}'
            plot_saving(output_directory, filename, label)
            plt.show()
        
    return df

# -----------------------------
# Helpers
# -----------------------------
def vshape(x, a, b, c, d=None):
    """Simple V-shape function to mimic MATLAB's tools.math.fun.vShape"""
    #return np.abs(a * (x - b)) + c  # crude approximation
    return np.abs(np.sqrt( (a * (x - b))**2 + c**2 ) ) # with the parabolic bottom

def plot_saving(output_directory, file_name, label):
    '''A general function for saving the plotted images to a folder called label.
    the label tell the program what group of graphs this belongs to and creates a 
    subfolder in the main output directory. The filename (without the extension)
    is then used to generate a full filename and save it'''
    
    folder_name = f"{label}"     
    folder_path = os.path.join(output_directory, folder_name)
    os.makedirs(folder_path, exist_ok=True)   # Create the output directory if it doesn't exist
    filtered_file_name = f"{file_name}.jpg"
    full_file_path = os.path.join(folder_path, filtered_file_name) 
    plt.savefig(full_file_path, format='jpg', dpi=300, bbox_inches='tight')
    return full_file_path
# -----------------------------
# Main routine
# -----------------------------

# read the CSV in which the median results are found
medians_file = "/home/micha/Dropbox/RonArad_Share_with_Micha/medians_BestPos_30-09 --- 20-10.csv"
df_medians = pd.read_csv(medians_file)  # must have columns: mount, scope, median, std
print('\nLoaded Medians from file', medians_file,'\n')

t0 = pd.Timestamp("1970-01-01T00:00:00")    
N_days = 3  #This is the total number of days to analyze
N_show = -1   #-1 for all, N_show smaller than N_days allows to see older data
'''example 3,1 will show only 1 day that occured 3 days ago
           2,1 same but for the day before yesterday
           1,1 is yesterday'''
base_path = '/home/ocs/Documents/output'
# output_directory = os.path.join(base_path,('v_shape_' + 
#                str(N_days) + '_' + str(N_show) + 
#                '_' + datetime.now().strftime("%H-%M")))
output_directory = r'/home/micha/Dropbox/WAO/LAST_analysis'

for mount in range(1,11):
    mount_str = f"{mount:02d}"  # 01, 02, ..., 10
    df_focus = read_DB(N_days, N_show, "operation_strings",
        f"unitCS.set.FocusData:{mount_str}", "value like '%LoopCompleted%:%true%' "       )

    # check for empty df or None
    if df_focus is None or df_focus.empty:
        print(f"No data for mount {mount_str}, skipping.")
        continue

    df_focus = basic_processing_FWHM(df_focus)
    df_focus["index"] = df_focus.index
    df_focus = df_focus[df_focus["LoopCompleted"] == True]
    df_focus['hour'] = df_focus['HH-MM-DD-mm'].str.split('_').str[0].astype(float) + df_focus['HH-MM-DD-mm'].str.split('_').str[1].astype(float)/60.
    
    df_sorted = df_focus.sort_values(by=["index", "TimeStarted", "Counter"]).reset_index()
    
    # Compute time differences within each index group
    epsilon = 300 / 86400  # 5 minutes in days
    groups = []
    for idx, group in df_sorted.groupby("index"):
        times = group["TimeStarted"].values
        # compute differences between consecutive times
        dt = np.diff(times, prepend=times[0] - epsilon*2)  # make first diff > epsilon
        # mark start of new group if jump > epsilon
        group_id = (dt > epsilon).cumsum()
        group = group.copy()
        group["Group"] = group_id  # optional: store group number
        groups.append(group)
    
    df_grouped = pd.concat(groups).set_index("index")
    df_grouped = df_grouped[df_grouped["Status"] != ""].sort_index()
    df_grouped['Points'] = df_grouped.apply(make_focpos_fwhm_dict, axis=1)
    
    #plot_focus_by_index(df_grouped)
    #print('finished first type of plot')
    df_out = plot_focus_pages(df_grouped)
