"""
Astronomical Star Field Generator
Generates a simulated star field as a 16-bit FITS file (and optionally a PNG preview).

Two PSF modes:
  - Gaussian  : default in-focus PSF, controlled by --max-size
  - Annular   : out-of-focus PSF (donut), enabled with --defocus
                Parameters: --psf-outer-r, --psf-inner-ratio,
                            --psf-offset-frac, --psf-azimuth

Dependencies:
    pip install Pillow astropy numpy
"""

import random
import argparse
import datetime
import numpy as np
from PIL import Image
from astropy.io import fits


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UINT16_MAX = 65535  # 2^16 - 1  —  full-well depth of a 16-bit detector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rand(lo: float, hi: float) -> float:
    return random.uniform(lo, hi)


def _apply_stamp(data: np.ndarray, cx: float, cy: float,
                 kernel: np.ndarray, peak_adu: float) -> None:
    """
    Composite a PSF kernel stamp onto the data array at (cx, cy).

    The kernel is assumed to be already normalised (sum = 1).  Each pixel
    receives  kernel[i,j] * peak_adu  flux, clipped to uint16 range.
    """
    H, W    = data.shape
    stamp_r = (kernel.shape[0] - 1) // 2
    xi, yi  = int(round(cx)), int(round(cy))

    x0, x1 = xi - stamp_r, xi + stamp_r + 1
    y0, y1 = yi - stamp_r, yi + stamp_r + 1

    kx0 = max(0, -x0);  ky0 = max(0, -y0)
    x0  = max(x0, 0);   y0  = max(y0, 0)
    x1  = min(x1, W);   y1  = min(y1, H)
    kx1 = kx0 + (x1 - x0)
    ky1 = ky0 + (y1 - y0)

    if x1 <= x0 or y1 <= y0:
        return

    patch    = kernel[ky0:ky1, kx0:kx1] * peak_adu
    combined = data[y0:y1, x0:x1].astype(np.int32) + patch.astype(np.int32)
    data[y0:y1, x0:x1] = np.clip(combined, 0, UINT16_MAX).astype(np.uint16)


# ---------------------------------------------------------------------------
# Gaussian PSF
# ---------------------------------------------------------------------------

def gaussian_psf(sigma: float, stamp_r: int) -> np.ndarray:
    """Return a 2-D Gaussian kernel normalised to sum = 1."""
    y, x    = np.ogrid[-stamp_r:stamp_r + 1, -stamp_r:stamp_r + 1]
    kernel  = np.exp(-(x**2 + y**2) / (2 * sigma**2)).astype(np.float32)
    return kernel / kernel.sum()


def stamp_star_gaussian(data: np.ndarray, cx: float, cy: float,
                        peak_adu: float, psf_radius: float) -> None:
    """
    Stamp a Gaussian PSF onto the data array.

    Parameters
    ----------
    peak_adu   : peak pixel value in ADU for an unresolved point source
    psf_radius : Gaussian sigma in pixels
    """
    sigma   = max(psf_radius * 0.5, 0.4)
    stamp_r = max(int(psf_radius * 3), 2)
    kernel  = gaussian_psf(sigma, stamp_r)
    # Re-scale so the peak pixel (centre) equals peak_adu
    center_val = kernel[stamp_r, stamp_r]
    if center_val > 0:
        kernel = kernel / center_val  # peak = 1
    _apply_stamp(data, cx, cy, kernel, peak_adu)


# ---------------------------------------------------------------------------
# Annular (out-of-focus) PSF
# ---------------------------------------------------------------------------

def annular_psf(outer_r: float,
                inner_ratio: float,
                offset_fraction: float,
                azimuth_deg: float,
                edge_width: float = 0.7) -> np.ndarray:
    """
    Build a 2-D out-of-focus PSF shaped as a filled annulus whose inner
    circle is offset from the outer circle's centre.

    Parameters
    ----------
    outer_r          : outer radius of the annulus in pixels
    inner_ratio      : inner_radius / outer_r  (0 = filled disc, <1 = annulus)
    offset_fraction  : displacement of the inner circle centre as a fraction
                       of outer_r  (0 = concentric)
    azimuth_deg      : direction of the offset in degrees; convention is
                       astronomical azimuth: 0° = up (−Y), 90° = right (+X)
    edge_width       : soft-edge half-width in pixels for anti-aliasing

    Returns
    -------
    2-D float32 ndarray of shape (2*stamp_r+1, 2*stamp_r+1),
    normalised so that the array sums to 1.
    """
    outer_r  = max(outer_r, 1.0)
    inner_r  = outer_r * float(np.clip(inner_ratio, 0.0, 0.99))
    # Cap offset so the inner disc stays entirely within the outer disc
    max_offset = max(outer_r - inner_r, 0.0)
    offset   = min(float(np.clip(offset_fraction, 0.0, 1.0)) * outer_r, max_offset)

    az_rad   = np.radians(azimuth_deg)
    ox       =  offset * np.sin(az_rad)   # column shift (+X = right)
    oy       = -offset * np.cos(az_rad)   # row shift    (−Y = up)

    stamp_r  = int(np.ceil(outer_r + edge_width)) + 1
    y, x     = np.ogrid[-stamp_r:stamp_r + 1, -stamp_r:stamp_r + 1]

    dist_outer = np.sqrt(x**2       + y**2)
    dist_inner = np.sqrt((x - ox)**2 + (y - oy)**2)

    def soft_step(dist, radius):
        """Smooth 0→1 transition: 0 outside radius, 1 well inside."""
        return np.clip((radius + edge_width - dist) / (2.0 * edge_width), 0.0, 1.0)

    inside_outer  = soft_step(dist_outer, outer_r)
    outside_inner = 1.0 - soft_step(dist_inner, inner_r)

    kernel = (inside_outer * outside_inner).astype(np.float32)

    total = kernel.sum()
    if total > 0:
        kernel /= total
    return kernel


def stamp_star_annular(data: np.ndarray, cx: float, cy: float,
                       peak_adu: float,
                       outer_r: float,
                       inner_ratio: float,
                       offset_fraction: float,
                       azimuth_deg: float) -> None:
    """
    Stamp an annular out-of-focus PSF onto the data array.

    The kernel is flux-normalised (sum = 1).  peak_adu sets the total
    integrated flux of the star; individual pixel values will be much lower
    than peak_adu because the light is spread over the annular ring.

    Parameters
    ----------
    outer_r          : outer radius of the annulus in pixels
    inner_ratio      : inner_radius / outer_r
    offset_fraction  : inner-circle centre offset as a fraction of outer_r
    azimuth_deg      : offset direction (0° = up / −Y, 90° = right / +X)
    """
    kernel = annular_psf(outer_r, inner_ratio, offset_fraction, azimuth_deg)
    _apply_stamp(data, cx, cy, kernel, peak_adu)


# ---------------------------------------------------------------------------
# Sky background (Poisson photon noise + Gaussian read noise)
# ---------------------------------------------------------------------------

def make_sky_background(width: int, height: int,
                        sky_level: float = 0.015,
                        read_noise: float = 0.003) -> np.ndarray:
    """
    Simulate a realistic CCD sky background frame.

    sky_level  : mean sky flux as a fraction of UINT16_MAX (e.g. 0.015 ≈ 1000 ADU)
    read_noise : Gaussian read-noise sigma as a fraction of UINT16_MAX
    """
    mean_adu    = sky_level  * UINT16_MAX
    noise_sigma = read_noise * UINT16_MAX
    sky = np.random.poisson(mean_adu, size=(height, width)).astype(np.float32)
    sky += np.random.normal(0, noise_sigma, size=(height, width))
    return np.clip(sky, 0, UINT16_MAX).astype(np.uint16)


# ---------------------------------------------------------------------------
# FITS header
# ---------------------------------------------------------------------------

def build_header(width: int, height: int, num_stars: int,
                 seed: int | None, defocus: bool,
                 outer_r: float, inner_ratio: float,
                 offset_fraction: float, azimuth_deg: float,
                 exposure_factor: float = 1.0) -> fits.Header:
    hdr = fits.Header()
    hdr["SIMPLE"]  = (True,   "File conforms to FITS standard")
    hdr["BITPIX"]  = (16,     "16-bit unsigned integer pixels")
    hdr["NAXIS"]   = (2,      "Number of data axes")
    hdr["NAXIS1"]  = (width,  "Image width [pixels]")
    hdr["NAXIS2"]  = (height, "Image height [pixels]")
    hdr["BZERO"]   = (0,      "No offset: data stored as native uint16")
    hdr["BSCALE"]  = (1,      "Default scaling factor")
    hdr["BUNIT"]   = ("ADU",  "Pixel flux unit")
    hdr["INSTRUME"]= ("SimCam",    "Simulated CCD camera")
    hdr["TELESCOP"]= ("StarSim",   "Simulated telescope")
    hdr["OBJECT"]  = ("STARFIELD", "Simulated star field")
    hdr["EXPTIME"] = (300.0 * exposure_factor, "[s] Simulated exposure time")
    hdr["EXPFACT"] = (exposure_factor, "Exposure time scale factor for defocused mode")
    hdr["GAIN"]    = (1.5,    "[e-/ADU] CCD gain")
    hdr["RDNOISE"] = (5.0,    "[e-] Read noise RMS")
    hdr["SATURATE"]= (UINT16_MAX, "[ADU] Saturation level")
    hdr["PIXSCALE"]= (0.5,    "[arcsec/pixel] Plate scale")
    hdr["NSTARS"]  = (num_stars, "Number of simulated stars")
    hdr["RANDSEED"]= (seed if seed is not None else -1, "RNG seed (-1 = random)")
    hdr["PSFMODE"] = ("ANNULAR" if defocus else "GAUSSIAN", "PSF model used")
    if defocus:
        hdr["PSFOUTR"] = (outer_r,        "[px] Annular PSF outer radius")
        hdr["PSFINRAT"]= (inner_ratio,     "Annular PSF inner/outer radius ratio")
        hdr["PSFOFFST"]= (offset_fraction, "Inner circle offset fraction of outer_r")
        hdr["PSFAZ"]   = (azimuth_deg,     "[deg] Offset azimuth (0=up, 90=right)")
    hdr["DATE-OBS"]= (datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S"),
                      "UTC date of simulated observation")
    hdr["COMMENT"]  = "Simulated 16-bit astronomical image - generated by starfield.py"
    return hdr


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_starfield(
    width: int = 6422,
    height: int = 9600,
    num_stars: int = 500,
    max_star_size: float = 3.0,
    sky_level: float = 0.015,
    defocus: bool = False,
    psf_outer_r: float = 10.0,
    psf_inner_ratio: float = 0.5,
    psf_offset_frac: float = 0.0,
    psf_azimuth: float = 0.0,
    exposure_factor: float = 1.0,
    seed: int | None = None,
    output_fits: str = "starfield.fits",
    output_png: str | None = None,
) -> np.ndarray:
    """
    Generate a simulated 16-bit astronomical star field and write a FITS file.

    Parameters
    ----------
    width, height    : image size in pixels
    num_stars        : number of stars to simulate
    max_star_size    : max Gaussian PSF sigma in pixels (ignored when defocus=True)
    sky_level        : mean sky background as fraction of 65535
    defocus          : if True, use annular PSF instead of Gaussian
    psf_outer_r      : [annular] outer radius in pixels
    psf_inner_ratio  : [annular] inner_radius / outer_r  (0 = solid disc)
    psf_offset_frac  : [annular] inner circle centre offset as fraction of outer_r
    psf_azimuth      : [annular] offset direction in degrees (0=up, 90=right)
    exposure_factor  : multiplier applied to defocused star flux (default 1.0)
    seed             : RNG seed for reproducibility (None = random)
    output_fits      : destination path for the .fits file
    output_png       : optional path for a log-stretched PNG preview

    Returns
    -------
    np.ndarray  uint16 array of shape (height, width) in ADU
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 1. Sky background with detector noise
    data = make_sky_background(width, height, sky_level=sky_level)

    # 2. Pre-build the annular kernel once (it is identical for all stars)
    if defocus:
        _annular_kernel = annular_psf(
            psf_outer_r, psf_inner_ratio, psf_offset_frac, psf_azimuth
        )

    # 3. Stars
    for _ in range(num_stars):
        x          = rand(0, width)
        y          = rand(0, height)
        brightness = random.random()
        peak_adu   = (0.3 + 0.7 * brightness ** 2) * UINT16_MAX

        if defocus:
            _apply_stamp(data, x, y, _annular_kernel, peak_adu * exposure_factor)
        else:
            psf_r = rand(0.4, max_star_size) * (0.4 + 0.6 * brightness)
            stamp_star_gaussian(data, x, y, peak_adu, psf_r)

    # 4. Write FITS — uint16 stored directly (BZERO=0, no signed conversion)
    hdr = build_header(width, height, num_stars, seed,
                       defocus, psf_outer_r, psf_inner_ratio,
                       psf_offset_frac, psf_azimuth, exposure_factor)
    hdu = fits.PrimaryHDU(data=data, header=hdr)
    hdu.writeto(output_fits, overwrite=True)
    psf_desc = (f"annular (outer_r={psf_outer_r}, ratio={psf_inner_ratio}, "
                f"offset={psf_offset_frac}, az={psf_azimuth} deg)"
                if defocus else "gaussian")
    print(f"FITS saved : {output_fits}  ({width}x{height} px, 16-bit, "
          f"{num_stars} stars, PSF={psf_desc})")

    # 5. Optional PNG preview (log stretch for visual clarity)
    if output_png:
        arr = data.astype(np.float32)
        lo  = np.percentile(arr, 0.5)
        hi  = np.percentile(arr, 99.8)
        arr = np.clip(arr, lo, hi)
        arr = np.log1p(arr - lo)
        arr = (arr / arr.max() * 255).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(output_png)
        print(f"PNG preview: {output_png}")

    return data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Generate a 16-bit astronomical FITS star field.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # General
    p.add_argument("--width",      type=int,   default=6422,
                   help="Image width in pixels")
    p.add_argument("--height",     type=int,   default=9600,
                   help="Image height in pixels")
    p.add_argument("--stars",      type=int,   default=500,
                   help="Number of stars")
    p.add_argument("--sky-level",  type=float, default=0.015,
                   help="Sky background level [0–1 fraction of 65535]")
    p.add_argument("--seed",       type=int,   default=None,
                   help="Random seed for reproducibility")
    p.add_argument("--output",     type=str,   default="starfield.fits",
                   help="Output FITS file path")
    p.add_argument("--png",        type=str,   default=None,
                   help="Optional PNG preview output path")

    # Gaussian PSF (default mode)
    p.add_argument("--max-size",   type=float, default=3.0,
                   help="[Gaussian] Max PSF sigma in pixels")

    # Annular / out-of-focus PSF
    p.add_argument("--defocus",         action="store_true",
                   help="Use annular out-of-focus PSF instead of Gaussian")
    p.add_argument("--exposure-factor", type=float, default=600.0,
                   help="[Annular] Flux multiplier for defocused stars (>1 = brighter)")
    p.add_argument("--psf-outer-r",     type=float, default=10.0,
                   help="[Annular] Outer radius in pixels")
    p.add_argument("--psf-inner-ratio", type=float, default=0.4,
                   help="[Annular] Inner/outer radius ratio (0=solid disc, <1=annulus)")
    p.add_argument("--psf-offset-frac", type=float, default=0.0,
                   help="[Annular] Inner circle centre offset as fraction of outer_r")
    p.add_argument("--psf-azimuth",     type=float, default=0.0,
                   help="[Annular] Offset direction in degrees (0=up / -Y, 90=right / +X)")

    args = p.parse_args()

    generate_starfield(
        width=args.width,
        height=args.height,
        num_stars=args.stars,
        max_star_size=args.max_size,
        sky_level=args.sky_level,
        defocus=args.defocus,
        exposure_factor=args.exposure_factor,
        psf_outer_r=args.psf_outer_r,
        psf_inner_ratio=args.psf_inner_ratio,
        psf_offset_frac=args.psf_offset_frac,
        psf_azimuth=args.psf_azimuth,
        seed=args.seed,
        output_fits=args.output,
        output_png=args.png,
    )


if __name__ == "__main__":
    main()