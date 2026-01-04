#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 23:34:54 2025

@author: micha
"""
import pandas as pd
import numpy as np


def plot_bestpos_vs_temp_by_mount(df: pd.DataFrame, output_directory: str,
                                  plots: dict, regular_fit: bool,
                                  x_axis: str, y_axis: str) -> pd.DataFrame:
    """ Plots y-axis data vs. x-axis data.
    It splits the data by mount number (int(indexes)) and plots nmounts (10) 
    subplots.
    Each subplot includes 4 scopes (based on fractional part of indexes).
    The y-limits are set by the 99th percentile of y_axis for that mount.
    Then, a linear fit is performed for each trace and plotted and saved.
    Specifically for BestPos the tick numbers are shifted to place assure all 4 
    scopes are in the same range with 150 tick spacing. The linear fit is 
    either simple,     or with constraints + center + weight to favor clusters,
    see the definition in Main.     Different symbols are given for the 
    first 12 days in a run. The first 2 hrs of observations are excluded
    (to reduce noise)
    parameters:
        df =
        output_directory = folder for saving the plots
        plots = a dictionary with boolean flags to determine which plots to 
        plot
    """
    data_for_distribution_plot = []
    if y_axis == 'minor':
        dt = np.diff(df[x_axis])
        gap_indices = np.where(dt >= 0.05)[0] + 1
        start_indexes = np.insert(gap_indices, 0, 0)
        print(len(start_indexes), start_indexes)
        if len(start_indexes):
            col_idx = df.columns.get_loc('HH-MM-DD-mm')
            print(df.iloc[start_indexes, col_idx])
            df = filter_1hr_after_start(df, start_indexes, 0.083)
            # filter out 2 hrs
    if y_axis == 'BestPos':
        df = df[df['Status'] == 'Found.'].copy()
        # keep only focus data that is good
        print()
    medians_list = []   # this is for exporting the median values
    indexes = df.index.to_numpy(dtype=float)
    df = df.copy()
    int_part, frac_part = np.divmod(indexes, 1)
    # Assign mount (0–9) and scope (0–3)
    df['mount'] = int_part.astype(int)
    df['scope'] = np.round(frac_part * 10).astype(int)
    # Setup 10 mounts (0–9)
    fig, axes = plt.subplots(5, 2, figsize=(14, 20), sharex=True)
    axes = axes.flatten()

    if 'TimeStarted' in df.columns:
        latest_time = julian_to_ddmm(df['TimeStarted'].max())
        earliest_time = julian_to_ddmm(df['TimeStarted'].min())
    elif 'time' in df.columns:
        latest_time = df['time'].max()
        earliest_time = df['time'].min()

    title = '' + y_axis + ' vs. ' + x_axis + ' for: ' + \
        str(earliest_time) + ' --- ' + str(latest_time)

    # Prepare to collect fit results
    fit_results = []
    high_values = []
    for mount_num in sorted(df['mount'].unique()):
        ax = axes[mount_num-1]  # axes run from 0-9 and mount numbers are 1-10
        # Filter for this mount
        df_mount = df[df['mount'] == mount_num]

        # Compute initial y-axis limits for each mount
        median = df_mount[y_axis].median()
        lower_limit_mount = median
        upper_limit_mount = median

        # Plot each scope
        # this places the traces one above the other spaced by 150
        y_goal = [0, 34850, 34700, 34550, 34400]
        for scope_num in range(1, 5):  # keeps scope names 1-4
            df_scope = df_mount[df_mount['scope'] == scope_num]
            if df_scope.empty:
                print(f'mount {mount_num} scope {scope_num} is empty')
                std_int = 0
                continue
            # Calculate scope-specific thresholds
            median = df_scope[y_axis].median()
            std = df_scope[y_axis].std()
            q99 = df_scope[y_axis].quantile(0.99)
            range99 = abs(q99 - median)

            # Define lower and upper limits for filtering scope data and y-limit
            # taking 3x 99 percentile makes sure all points are kept except for
            #a crazy outlier
            lower_limit = median - 3*range99
            upper_limit = median + 3*range99

            # append a row
            if pd.isna(median):
                median_int = None  # or keep as np.nan
            else:
                median_int = int(median)
            if pd.isna(std):
                std_int = None  # or keep as np.nan
            else:
                std_int = int(std)
            medians_list.append({
                'mount': mount_num,
                'scope': scope_num,
                'median': median_int,
                'std': std_int})
            date_range = str(earliest_time) + ' --- ' + str(latest_time)
            try:
                if std_int > 10:
                    data_for_distribution_plot.append((list(df_scope[y_axis]),
                                                       median_int, std_int, 
                                                       int(mount_num), 
                                                       scope_num, date_range))
            except (NameError, TypeError): 
                # std_int is not defined or invalid (e.g., None)
                pass
            # Filter BestPos to within allowed range
            df_scope = df_scope[(df_scope[y_axis] >= lower_limit) &
                                (df_scope[y_axis] <= upper_limit)]
            '''Here can still split df_scope into separate days'''
            indexes = find_noon_rollover_indexes(df_scope)
            # print(f'\n mount {mount_num} scope{scope_num} has {indexes}')
            markers = ['o', '+', '^', 'D', 'P',
                       '*', 'x', 'v', '<', '>', '.', 's']
            x = df_scope[x_axis].to_numpy()
            # y = df_scope[y_axis].to_numpy() #absolute value of focus
            y = df_scope[y_axis].to_numpy()-median+y_goal[scope_num]

            # Remove NaNs
            valid = ~np.isnan(x).flatten() & ~np.isnan(y)
            x = x[valid].reshape(-1, 1)  # <== FIX: reshape here again
            y = y[valid]
            # print('valid pts:',len(y))
            indexes = indexes + [len(y)]

            # Linear fit
            if len(x) >= 2:
                if regular_fit:
                    '''For regular linear fit'''
                    y_pred, R2, m, b, text = linear_fit(x, y, 'col', True)
                elif not constraints:
                    '''For fixed slope and calculate only intercept use:'''
                    slope_fixed = 18
                    y_pred, R2, m, b, text = linear_fit_intercept_only(
                        x, y, 'col', slope_fixed)
                else:
                    y_pred, R2, m, b, text = fit_constrained_linear(x, y,'col',
                                           [constraints[0], constraints[1]], 
                                           constraints[2], constraints[3], 
                                           bandwidth=0.2)
                # print(f"mount {mount_num} scope{scope_num}, 
                # b: {num_round_2(b)}, R²: {num_round_2(R2)}")
                # Plot the fit line
                if y_axis != 'minor':
                    ax.plot(x, y_pred, color=colors[scope_num-1], linewidth=1)

                # Store fit results
                fit_results.append({
                    'mount': mount_num,
                    'scope': scope_num,
                    'slope': num_round_2(m),
                    'intercept': num_round_2(b),
                    'R2_score': num_round_2(R2),
                    'n_points': len(x)
                })
            # Scatter plot
            else:
                m = 0
                # For plotting all the points in the same symbol
                # ax.scatter(x, y, s=3, alpha=0.5, label=f'{scope_num} 
                # {num_round_2(m)} {num_round_2(R2,2)}')

            # for separate symbols for different days:
            for i in range(len(indexes) - 1):
                start = indexes[i]
                end = indexes[i+1]
                if 'R2' not in locals():
                    R2 = -1  # Make sure there is a value,to avoid errors 
                    # if missing
                ax.scatter(x[start:end], y[start:end],
                           marker=markers[i % len(markers)], s=15,
                           color=colors[scope_num-1],
                           # label only for first slice
                           label=
                           f'{scope_num} {num_round_2(m)} {num_round_2(R2, 2)}'
                           if i == 0 else None)

            lower_limit_mount = min(lower_limit_mount, lower_limit)
            upper_limit_mount = max(upper_limit_mount, upper_limit)
            if y_axis == 'minor':
                threshold = 3
                if len(x) > 0:
                    ax.plot([x.min(), x.max()], [threshold, threshold],
                            color='black', linewidth=2)
                lower_limit_mount = 1
                upper_limit_mount = 4
                temp = (mount_num, scope_num, len(x[y > threshold].tolist()))
                high_values.append(temp)

        # ax.set_ylim([lower_limit_mount-extra_y, upper_limit_mount+extra_y]) 
        #this is for absolute value of focus
        ax.set_ylim([34250, 35000])
        ax.set_title(f'Mount {mount_num}')
        ax.set_ylabel(y_axis)
        ax.grid(True, linestyle='--', alpha=0.3)
        if mount_num >= 8:
            ax.set_xlabel(x_axis)
        ax.legend(loc='upper left', fontsize=12)
    if not regular_fit and not constraints:
        title = title + ' slope=' + str(slope_fixed)

    fig.suptitle(title, fontsize=18, y=1.02)
    plt.tight_layout()
    filename = f'BestPos_vs_Temp_fits {earliest_time}-{latest_time}'
    label = str('focus')
    plot_saving(output_directory, filename, label)
    plt.show()

    # Now we export the Fit_results to an excel file and color the 
    #"bad fit columns
    write_append_cols_excel(fit_results)

    df_high = pd.DataFrame(high_values, columns=['mount', 'scope', 'times'])
    # out = os.path.join(output_directory,(file_short + '_high.xlsx'))
    out = os.path.join(output_directory, (f'{date_range}' + '_high.xlsx'))
    df_high.to_excel(out, index=False)
    df_medians = pd.DataFrame(medians_list)
    file_name = os.path.join(
        output_directory, f'medians_BestPos_{date_range}.csv')
    df_medians.to_csv(file_name, index=False)
    print(f'saved medians to file:  medians_BestPos_{date_range}.csv')
    if plots['Focus_distribution_and_median']:
        for row in data_for_distribution_plot:
            plot_distribution_if_outliers(row)

    return df_medians