#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Get data from last.visit.images and plot image quality metrics for telescopes.

Generates the full set of plots and telescope maps in an output directory
based on pla.OUTPUT_PATH + a timestamp of the current run.

Usage:
    python get_vimg_stats.py <startdate> <enddate> <local_run>

Arguments:
    startdate  : str, format dd/mm/yy
    enddate    : str, format dd/mm/yy
    local_run  : bool, '1'/'true' = read from local CSV cache,
                       '0'/'false' = read from database (requires VPN)

Adapted from LAST_imageQuality.py by Claude Sonnet 4.6
"""

import sys
import os
from datetime import datetime

import pyLAST as pla
import pandas as pd
from matplotlib import pyplot as plt


# --- Constants ---
FWHM_PROPERTY = ("FWHM", "[arcsec]")
VIMG_FWHM_PROPERTY = ("vimgFWHM", "[arcsec]")
AXIS_RATIO_PROPERTY = ("mean axis ratio", "")
MEAN_FWHM_PROPERTY = ("mean FWHM", "[arcsec]")
TELSTATS_SUFFIX = "_telstats.csv"


def parse_args() -> tuple[str, str, bool]:
    """Parse and validate command-line arguments."""
    if len(sys.argv) != 4:
        print("Usage: python get_vimg_stats.py <startdate> <enddate> <local_run>")
        print("  startdate / enddate : dd/mm/yy")
        print("  local_run           : 1/true or 0/false")
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

    return startdate, enddate, local_run


def setup_output_dirs(timestamp: str, time_span_stamp: tuple) -> tuple[str, str]:
    """Create and return the output and database directories for this run."""
    outdir = os.path.join(pla.OUTPUT_PATH, timestamp + "_output")
    os.makedirs(outdir, exist_ok=True)

    db_out_path = os.path.join(pla.DATABASE_PATH, time_span_stamp[0])
    os.makedirs(db_out_path, exist_ok=True)

    return outdir, db_out_path


def analyse_telescope(tel: pd.DataFrame, mountnum: int) -> dict:
    """Extract FWHM and axis ratio metrics from a single telescope's data.

    Returns a dict with keys: telnum, tel_label, telmean, telstd,
    cropfwhm, cropsfwhm, axratio, telfwhm, telairmass.
    """
    tel = tel.copy()  # avoid SettingWithCopyWarning
    telnum = int(tel["camnum"].mean())
    print(f"  Analysing tel {telnum}")

    tel["cropid"] = pd.Categorical(tel["cropid"], categories=range(1, 25))

    cropfwhm_df = tel.groupby("cropid", observed=False)["fwhm"].agg(["mean", "std"])

    grouped_by_obsdate = tel.groupby("dateobs", observed=False)
    tel_obs_df = grouped_by_obsdate.agg({"fwhm": ["mean", "std"], "airmass": ["mean"]})

    cropaxes_df = tel.groupby("cropid", observed=False)[["med_a", "med_b"]].agg("mean")
    cropaxes_df["axratio"] = cropaxes_df["med_a"] / cropaxes_df["med_b"]

    return {
        "telnum": telnum,
        "tel_label": f"{mountnum}.{telnum}",
        "telmean": cropfwhm_df["mean"].mean(),
        "telstd": cropfwhm_df["std"].mean(),
        "cropfwhm": cropfwhm_df["mean"].values,
        "cropsfwhm": cropfwhm_df["std"].values,
        "axratio": cropaxes_df["axratio"].values,
        "telfwhm": tel_obs_df["fwhm"]["mean"],
        "telairmass": tel_obs_df["airmass"]["mean"],
    }


def analyse_mount(mount: pd.DataFrame, outdir: str, time_span_stamp: tuple) -> list[dict]:
    """Analyse all telescopes in a mount and generate per-mount plots.

    Returns a list of per-telescope metric dicts (from analyse_telescope).
    Skips the mount with a printed warning if any error occurs.
    """
    mountnum = int(mount["mountnum"].mean())
    print(f"Analysing mount {mountnum}")

    tel_list = [group for _, group in mount.groupby("camnum")]
    tel_results = []

    try:
        for tel in tel_list:
            tel_results.append(analyse_telescope(tel, mountnum))

        cropfwhm  = [r["cropfwhm"]  for r in tel_results]
        cropsfwhm = [r["cropsfwhm"] for r in tel_results]
        axratio   = [r["axratio"]   for r in tel_results]

        pla.plot_property_vs_cropid_per_mount(
            vals=cropfwhm,
            val_stds=cropsfwhm,
            mountnum=mountnum,
            property_name=FWHM_PROPERTY,
            time_span_stamp=time_span_stamp,
            outdir=outdir,
        )

        pla.plot_mount_telescope_maps(
            mountnum=mountnum,
            vals=cropfwhm,
            property_name=MEAN_FWHM_PROPERTY,
            time_span_stamp=time_span_stamp,
            outdir=outdir,
        )

        pla.plot_mount_telescope_maps(
            mountnum=mountnum,
            vals=axratio,
            property_name=AXIS_RATIO_PROPERTY,
            time_span_stamp=time_span_stamp,
            outdir=outdir,
        )

        plt.close("all")

    except Exception as e:
        print(f"Warning: Problem analysing mount {mountnum} - {e}")
        plt.close("all")

    return tel_results


def save_telstats(tel_results: list[dict], time_span_stamp: tuple, outdir: str) -> None:
    """Save aggregated telescope statistics to a CSV file."""
    telstats_df = pd.DataFrame({
        "tel_labels": [r["tel_label"] for r in tel_results],
        "telmean":    [r["telmean"]   for r in tel_results],
        "telstd":     [r["telstd"]    for r in tel_results],
    })
    telstats_df.attrs["time_span_stamp"] = time_span_stamp[0]
    telstats_df.attrs["start_date"] = time_span_stamp[1]
    telstats_df.attrs["end_date"] = time_span_stamp[2]

    filename = time_span_stamp[0] + TELSTATS_SUFFIX
    filepath = os.path.join(outdir, filename)
    telstats_df.to_csv(filepath)
    print(f"Saved telstats: {filepath}")

    return telstats_df


def main() -> None:
    startdate, enddate, local_run = parse_args()

    timestamp = datetime.now().strftime("%y%m%d%H%M")
    time_span_stamp = pla.generate_time_span_str(startdate=startdate, enddate=enddate)

    outdir, db_out_path = setup_output_dirs(timestamp, time_span_stamp)
    print(f"Output directory : {outdir}")
    print(f"Database path    : {db_out_path}")

    vimg_df = pla.get_vimg_df(
        localrun=local_run,
        dbpath=pla.DATABASE_PATH,
        startdate=startdate,
        enddate=enddate,
    )

    mount_list = [group for _, group in vimg_df.groupby("mountnum")]

    all_tel_results = []
    for mount in mount_list:
        results = analyse_mount(mount, outdir, time_span_stamp)
        all_tel_results.extend(results)

    if not all_tel_results:
        print("No telescope data was successfully analysed. Exiting.")
        sys.exit(1)

    pla.plot_tel_stats(
        vals=[r["telmean"] for r in all_tel_results],
        val_stds=[r["telstd"] for r in all_tel_results],
        tel_labels=[r["tel_label"] for r in all_tel_results],
        property_name=VIMG_FWHM_PROPERTY,
        time_span_stamp=time_span_stamp,
        outdir=outdir,
        colorindex=None,
    )

    save_telstats(all_tel_results, time_span_stamp, outdir)
    plt.close("all")
    print("Finished.")


if __name__ == "__main__":
    main()