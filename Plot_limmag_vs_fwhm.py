#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Get data from last.visit.images and plot limit magnitude vs FWHM.

Generates the following plots in an output directory based on
pla.OUTPUT_PATH + a timestamp of the current run:
  1. Scatter plot of limmag vs fwhm for the full dataset.
  2. 2-D histogram heatmap for the full dataset.
  3. Per-telescope 2-D histogram heatmap for every (mountnum, camnum) pair,
     saved in per-mount subdirectories.

Usage:
    python plot_limmag_vs_fwhm.py <startdate> <enddate> <local_run> <split_crops>

Positional arguments:
    startdate   : str, format dd/mm/yy
    enddate     : str, format dd/mm/yy
    local_run   : bool, '1'/'true' = read from local CSV cache,
                        '0'/'false' = read from database (requires VPN)
    split_crops : bool, '1'/'true'  = one series per crop in CROP_IDS_OF_INTEREST
                        '0'/'false' = all crops combined in a single series
"""

import sys
import os
import argparse
from datetime import datetime

import numpy as np
import pyLAST as pla
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

CROP_IDS_OF_INTEREST = [1, 6, 10, 19, 24]

# Pixel scale
ARCSEC_PER_PIXEL = 1.25  # arcsec / pixel

# 2-D histogram bin definitions
FWHM_RANGE   = (2, 8)       # arcsec
LIMMAG_RANGE = (16, 24)     # mag
FWHM_BINS    = 200
LIMMAG_BINS  = 200


def _parse_bool(value: str, name: str) -> bool:
    """Parse a '1'/'true'/'0'/'false' string into a bool."""
    v = value.strip().lower()
    if v in ("1", "true"):
        return True
    if v in ("0", "false"):
        return False
    print(f"Error: {name} must be 1/true or 0/false, got '{value}'")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse positional command-line arguments."""
    parser = argparse.ArgumentParser(
        usage="%(prog)s <startdate> <enddate> <local_run> <split_crops>",
        add_help=True,
    )
    parser.add_argument("startdate",   help="Start date dd/mm/yy")
    parser.add_argument("enddate",     help="End date dd/mm/yy")
    parser.add_argument("local_run",   help="1/true = local CSV, 0/false = database")
    parser.add_argument("split_crops", help="1/true = per-crop series, 0/false = single series")

    args = parser.parse_args()
    args.local_run   = _parse_bool(args.local_run,   "local_run")
    args.split_crops = _parse_bool(args.split_crops, "split_crops")

    return args


def _subset_label(mountnum: int | None, camnum: int | None) -> str:
    """Human-readable label describing the active mount/telescope filter."""
    if mountnum is None:
        return "all mounts"
    if camnum is None:
        return f"mount {mountnum}"
    return f"mount {mountnum} / telescope {camnum}"


def _filename_tag(mountnum: int | None, camnum: int | None) -> str:
    """Short filename tag for the active mount/telescope filter."""
    if mountnum is None:
        return ""
    if camnum is None:
        return f"_m{mountnum}"
    return f"_m{mountnum}_t{camnum}"


def plot_scatter(vimg_df, split_crops: bool, time_span_stamp: tuple,
                 outdir: str, subset_label: str, filename_tag: str) -> None:
    """Save scatter plot of limmag vs fwhm."""
    fig, ax = plt.subplots(figsize=(9, 6))

    if split_crops:
        for cropid in CROP_IDS_OF_INTEREST:
            subset = vimg_df[vimg_df["cropid"] == cropid]
            ax.scatter(subset["fwhm"], subset["limmag"], s=5, alpha=0.3, label=f"crop {cropid}")
        ax.legend(title="Crop ID", markerscale=3)
    else:
        ax.scatter(vimg_df["fwhm"], vimg_df["limmag"], s=5, alpha=0.3)

    ax.set_xlabel("FWHM [arcsec]")
    ax.set_ylabel("Limiting Magnitude")
    ax.set_title(f"Limiting Magnitude vs FWHM  —  {subset_label}\n{time_span_stamp[0]}")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    outfile = os.path.join(outdir, f"{time_span_stamp[0]}{filename_tag}_limmag_vs_fwhm_scatter.png")
    fig.savefig(outfile, dpi=150)
    print(f"  Saved scatter    : {outfile}")
    plt.close(fig)


def plot_2d_histogram(vimg_df, time_span_stamp: tuple, outdir: str,
                      subset_label: str, filename_tag: str) -> None:
    """Save 2-D histogram heatmap of observation counts per (fwhm, limmag) bin."""
    fwhm_bins   = np.linspace(*FWHM_RANGE,   FWHM_BINS   + 1)
    limmag_bins = np.linspace(*LIMMAG_RANGE, LIMMAG_BINS + 1)

    counts, _, _ = np.histogram2d(
        vimg_df["fwhm"],
        vimg_df["limmag"],
        bins=[fwhm_bins, limmag_bins],
    )
    # Mask empty bins so they render as background rather than the low-count colour
    counts = np.ma.masked_equal(counts, 0)

    fig, ax = plt.subplots(figsize=(10, 7))
    img = ax.pcolormesh(
        fwhm_bins, limmag_bins, counts.T,
        norm=LogNorm(vmin=1),
        cmap="viridis",
    )
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("Number of observations")

    ax.set_xlabel("FWHM [arcsec]")
    ax.set_ylabel("Limiting Magnitude")
    ax.grid(True, linestyle="--", alpha=0.4, color="black")

    # Top x-axis: FWHM in pixels
    ax_top = ax.twiny()
    ax_top.set_xlim(np.array(ax.get_xlim()) / ARCSEC_PER_PIXEL)
    ax_top.set_xlabel("FWHM [pixels]")

    ax.set_title(
        f"Observation count per (FWHM, limmag) bin  —  {subset_label}\n{time_span_stamp[0]}",
        pad=30,
    )
    fig.tight_layout()

    outfile = os.path.join(outdir, f"{time_span_stamp[0]}{filename_tag}_limmag_vs_fwhm_heatmap.png")
    fig.savefig(outfile, dpi=150)
    print(f"  Saved heatmap    : {outfile}")
    plt.close(fig)


def plot_per_telescope_histograms(vimg_df, split_crops: bool,
                                  time_span_stamp: tuple, outdir: str) -> None:
    """Generate scatter + heatmap for every (mountnum, camnum) pair."""
    mount_groups = vimg_df.groupby("mountnum")

    for mountnum, mount_df in mount_groups:
        mount_subdir = os.path.join(outdir, f"mount_{mountnum}")
        os.makedirs(mount_subdir, exist_ok=True)
        print(f"Mount {mountnum}")

        tel_groups = mount_df.groupby("camnum")
        for camnum, tel_df in tel_groups:
            if tel_df.empty:
                print(f"  Telescope {camnum}: no data, skipping.")
                continue
            print(f"  Telescope {camnum}  ({len(tel_df)} rows)")
            label = _subset_label(mountnum, camnum)
            tag   = _filename_tag(mountnum, camnum)
            try:
                plot_scatter(tel_df, split_crops, time_span_stamp, mount_subdir, label, tag)
                plot_2d_histogram(tel_df, time_span_stamp, mount_subdir, label, tag)
            except Exception as e:
                print(f"  Warning: failed for mount {mountnum} / telescope {camnum}: {e}")
                plt.close("all")


def main() -> None:
    args = parse_args()

    timestamp = datetime.now().strftime("%y%m%d%H%M")
    time_span_stamp = pla.generate_time_span_str(startdate=args.startdate, enddate=args.enddate)

    outdir = os.path.join(pla.OUTPUT_PATH, timestamp + "_output")
    os.makedirs(outdir, exist_ok=True)
    print(f"Output directory : {outdir}")

    vimg_df = pla.get_vimg_df(
        localrun=args.local_run,
        dbpath=pla.DATABASE_PATH,
        startdate=args.startdate,
        enddate=args.enddate,
    )

    # --- Full-dataset plots ---
    print("Plotting full dataset...")
    plot_scatter(vimg_df, args.split_crops, time_span_stamp, outdir,
                 subset_label="all mounts", filename_tag="")
    plot_2d_histogram(vimg_df, time_span_stamp, outdir,
                      subset_label="all mounts", filename_tag="")

    # --- Per-telescope plots ---
    print("Plotting per-telescope histograms...")
    plot_per_telescope_histograms(vimg_df, args.split_crops, time_span_stamp, outdir)

    plt.close("all")
    print("Finished.")


if __name__ == "__main__":
    main()