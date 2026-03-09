#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot FWHM vs. observation date for selected crop IDs across all telescopes
in each mount.

For each mount, generates a 2x2 figure with one subplot per telescope.
Each subplot shows a scatter plot of FWHM vs. dateobs for crop IDs
1, 6, 19, and 24, each in a distinct color.

Usage:
    python plot_fwhm_vs_time.py <startdate> <enddate> <local_run> [plot_start] [plot_end]

Arguments:
    startdate  : str, format dd/mm/yy  — start of the data query range
    enddate    : str, format dd/mm/yy  — end of the data query range
    local_run  : '1'/'true'  = read from local CSV cache
                 '0'/'false' = read from database (requires VPN)
    plot_start : str, format dd/mm/yy  (optional) — start of the plotted sub-range
    plot_end   : str, format dd/mm/yy  (optional) — end of the plotted sub-range
                 If omitted, the full query range is plotted.
"""

import sys
import os
from datetime import datetime, timezone

import pyLAST as pla
import pandas as pd
import matplotlib.dates as mdates
from matplotlib import pyplot as plt


# --- Constants ---
FIGSIZE = (18, 10)
FONT_SIZE_TITLE = 13
FONT_SIZE_LABEL = 11
FONT_SIZE_TICK = 9
FONT_SIZE_LEGEND = 9
FONT_SIZE_SUPTITLE = 15

CROP_IDS_OF_INTEREST = [1, 6,10, 19, 24]
# One color per crop ID — using a colorblind-friendly palette
CROP_COLORS = {1: "#1f77b4", 6: "#ff7f0e", 10: "#000000", 19: "#2ca02c", 24: "#d62728"}
MARKER_SIZE = 8
MARKER_ALPHA = 0.6
FWHM_YLIM = (2, 8)  # shared y-axis range for all subplots [arcsec]


def parse_date(date_str: str, label: str) -> datetime:
    """Parse a dd/mm/yy date string, exiting with a clear error on failure."""
    try:
        return datetime.strptime(date_str.strip(), "%d/%m/%y").replace(tzinfo=timezone.utc,hour=12)
    except ValueError:
        print(f"Error: {label} '{date_str}' is not a valid dd/mm/yy date.")
        sys.exit(1)


def parse_args() -> tuple[str, str, bool, datetime | None, datetime | None]:
    """Parse and validate command-line arguments.

    Returns (startdate, enddate, local_run, plot_start, plot_end).
    plot_start / plot_end are datetime objects or None if not provided.
    """
    if len(sys.argv) not in (4, 6):
        print("Usage: python plot_fwhm_vs_time.py <startdate> <enddate> <local_run> [plot_start] [plot_end]")
        print("  startdate / enddate         : dd/mm/yy  (data query range)")
        print("  local_run                   : 1/true or 0/false")
        print("  plot_start / plot_end       : dd/mm/yy  (optional plot sub-range)")
        sys.exit(1)

    startdate = sys.argv[1]
    enddate = sys.argv[2]
    local_run_arg = sys.argv[3].strip().lower()

    if local_run_arg in ("1", "true"):
        local_run = True
    elif local_run_arg in ("0", "false"):
        local_run = False
    else:
        print(f"Error: local_run must be 1/true or 0/false, got '{sys.argv[3]}'")
        sys.exit(1)

    plot_start = parse_date(sys.argv[4], "plot_start") if len(sys.argv) == 6 else None
    plot_end   = parse_date(sys.argv[5], "plot_end")   if len(sys.argv) == 6 else None

    if plot_start and plot_end and plot_start >= plot_end:
        print("Error: plot_start must be earlier than plot_end.")
        sys.exit(1)

    return startdate, enddate, local_run, plot_start, plot_end


def setup_output_dirs(timestamp: str, time_span_stamp: tuple) -> tuple[str, str]:
    """Create and return the output and database directories for this run."""
    outdir = os.path.join(pla.OUTPUT_PATH, timestamp + "_output")
    os.makedirs(outdir, exist_ok=True)

    db_out_path = os.path.join(pla.DATABASE_PATH, time_span_stamp[0])
    os.makedirs(db_out_path, exist_ok=True)

    return outdir, db_out_path


def plot_fwhm_vs_time_per_mount(
    mount: pd.DataFrame,
    mountnum: int,
    time_span_stamp: tuple,
    outdir: str,
    plot_start: datetime | None = None,
    plot_end: datetime | None = None,
) -> None:
    """Plot a 2x2 figure of FWHM vs. dateobs for each telescope in a mount.

    Each subplot corresponds to one telescope. Within each subplot, crop IDs
    defined in CROP_IDS_OF_INTEREST are shown as scatter plots in distinct colors.
    If plot_start / plot_end are given, only data within that sub-range is shown
    and the x-axis is clamped accordingly.
    The figure is saved as a PNG in outdir.
    """
    tel_list = [group.copy() for _, group in mount.groupby("camnum")]
    n_tels = len(tel_list)

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=False, sharey=False)
    axes_flat = axes.flatten()

    # Hide any unused subplots (in case a mount has fewer than 4 telescopes)
    for ax in axes_flat[n_tels:]:
        ax.set_visible(False)

    for idx, tel in enumerate(tel_list):
        ax = axes_flat[idx]
        telnum = int(tel["camnum"].mean())

        # Parse dateobs to datetime for proper time-axis formatting
        tel["dateobs"] = pd.to_datetime(tel["dateobs"],format='mixed')

        # Filter to the requested plot sub-range if provided
        if plot_start is not None:
            tel = tel[tel["dateobs"] >= plot_start]
        if plot_end is not None:
            tel = tel[tel["dateobs"] <= plot_end]

        for crop_id in CROP_IDS_OF_INTEREST:
            crop_data = tel[tel["cropid"] == crop_id]
            if crop_data.empty:
                continue
            ax.scatter(
                crop_data["dateobs"],
                crop_data["fwhm"],
                label=f"crop {crop_id}",
                color=CROP_COLORS[crop_id],
                s=MARKER_SIZE,
                alpha=MARKER_ALPHA,
            )

        ax.set_title(f"Tel {mountnum}.{telnum}", fontsize=FONT_SIZE_TITLE)
        ax.set_xlabel("Observation Date [UTC]", fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel("FWHM [arcsec]", fontsize=FONT_SIZE_LABEL)
        ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
        # Use hourly ticks if the plot sub-range is less than 4 days,
        # otherwise fall back to daily ticks
        if plot_start is not None and plot_end is not None:
            ndays = (plot_end - plot_start).days
        # else:
        #     ndays = (parse_date(time_span_stamp[2],"")- parse_date(time_span_stamp[1],"")).days

        short_range = (
                plot_start is not None
                and plot_end is not None
                and ndays < 4
        )
        if short_range:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval= int(ndays)))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
        else:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        if plot_start is not None and plot_end is not None:
            ax.set_xlim(plot_start, plot_end)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.set_ylim(FWHM_YLIM)
        ax.grid(axis="both", linestyle="--", alpha=0.5)
        ax.legend(fontsize=FONT_SIZE_LEGEND, markerscale=2)

    plot_start_str = plot_start.strftime("%d/%m/%y") if plot_start else time_span_stamp[1]
    plot_end_str   = plot_end.strftime("%d/%m/%y")   if plot_end   else time_span_stamp[2]
    fig.suptitle(
        f"Mount {mountnum} — FWHM vs. Observation Date\n"
        f"{plot_start_str} to {plot_end_str}",
        fontsize=FONT_SIZE_SUPTITLE,
    )
    plt.tight_layout()

    filename = f"{time_span_stamp[0]}_mount{mountnum}_fwhm_vs_time.png"
    filepath = os.path.join(outdir, filename)
    plt.savefig(filepath, dpi=150)
    print(f"  Saved: {filepath}")
    plt.close(fig)


def main() -> None:
    startdate, enddate, local_run, plot_start, plot_end = parse_args()

    timestamp = datetime.now().strftime("%y%m%d%H%M")
    in_time_span_stamp = pla.generate_time_span_str(startdate=startdate, enddate=enddate)

    outdir, db_out_path = setup_output_dirs(timestamp, in_time_span_stamp)
    print(f"Output directory : {outdir}")
    print(f"Database path    : {db_out_path}")

    vimg_df = pla.get_vimg_df(
        localrun=local_run,
        dbpath=pla.DATABASE_PATH,
        startdate=startdate,
        enddate=enddate,
    )

    mount_list = [group for _, group in vimg_df.groupby("mountnum")]

    for mount in mount_list:
        mountnum = int(mount["mountnum"].mean())
        print(f"Plotting mount {mountnum}")
        try:
            plot_fwhm_vs_time_per_mount(mount, mountnum, in_time_span_stamp, outdir, plot_start, plot_end)
        except Exception as e:
            print(f"Warning: Could not plot mount {mountnum} - {e}")
            plt.close("all")

    print("Finished.")


if __name__ == "__main__":
    main()