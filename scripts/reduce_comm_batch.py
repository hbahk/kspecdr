"""
Batch reduction for KSPEC commissioning nights.

Phase 0 — calibration (always regenerated):
    For each calib directory: convert Flat_*/Arc_* raw frames to isoplane
    format, run make_im, make_tlm (from flats) and reduce_arc (wavelength
    calibration). Check images are saved under calib/chkimg/.

Phase 1 — object frames:
    Reduces all tile_*.fits object frames for:
        20260124, 20260125, 20260127, 20260128, 20260129, 20260204, 20260205
    Check images are saved under processed/chkimg/.

Phase 2 — skyline sky recalculation:
    Recalculates PRIMARY and VARIANCE for each output file using a per-fiber
    Gaussian fit to the O I 5578 Å sky line in the RWSS and SKY extensions.

Usage:
    python reduce_comm_batch.py [--skip-existing] [--no-skyline]

    --skip-existing  Skip tiles whose output _red.fits already exists.
    --no-skyline     Skip the skyline sky recalculation (Phase 2).
"""
import argparse
import logging
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import curve_fit

from kspecdr.inst.isoplane import write_isoplane_converted_image
from kspecdr.io.image import ImageFile
from kspecdr.preproc.make_im import make_im
from kspecdr.preproc.preproc import reduce_bias, reduce_dark
from kspecdr.reduce_object import reduce_object
from kspecdr.tlm.make_tlm import make_tlm
from kspecdr.extract.make_ex import make_ex
from kspecdr.extract.reduce_arc import reduce_arc
from kspecdr.reduce_fflat import reduce_fflat
from kspecdr.wavecal.arc_io import read_arc_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WD = Path.home() / "Research/kspec/kspecdr"
COMM = WD / "resources" / "comm"
ASSIGNDIR = COMM / "assign"
ARC_TABLES_DIR = WD / "data" / "arc_tables"

DATES = [
    "20260124",
    "20260125",
    "20260127",
    "20260128",
    "20260129",
    "20260204",
    "20260205",
]

# Number of KSPEC science fibers (used for calibration frame conversion)
KSPEC_NFIBERS = 14

# Master dark (20260130 long dark; used when a date has no dark of its own)
MASTER_DARK_GLOBAL = COMM / "20260130" / "calib" / "mdark_2400s.fits"

# PCA glow templates
GLOW_TEMPLATE_DIR = COMM / "glow_templates"
GLOW_PCA_CUBE = GLOW_TEMPLATE_DIR / "glow_pca_cube.fits"
# Per-pixel dark-current rate map (optional; set to None if not available)
DC_RATE_FILE: Path | None = GLOW_TEMPLATE_DIR / "dc_rate.fits"
if DC_RATE_FILE is not None and not DC_RATE_FILE.exists():
    DC_RATE_FILE = None

# Row ranges (axis=0) used when fitting the glow model.
# Include only detector regions that are not illuminated by fibers.
GLOW_FIT_ROWS = [(0, 500), (800, 1300)]

# Number of PCA components to use in the glow fit
GLOW_N_COMPONENTS = 5

# ---------------------------------------------------------------------------
# Per-date calibration configuration
# ---------------------------------------------------------------------------
# For each date: which calib dir supplies mbias/tlm/fflat,
# and which dark file to use (None = no dark subtraction).
DATE_CONFIG = {
    "20260124": {
        "calib_dir": COMM / "20260124" / "calib",
        "dark": None,
        "mbias_override": COMM / "20260124" / "calib" / "mbias.fits",
        "overscan_region": (0, 1, 0, 1340),
    },
    "20260125": {
        "calib_dir": COMM / "20260125" / "calib",
        "dark": None,
        "mbias_override": COMM / "20260125" / "calib" / "mbias.fits",
        "overscan_region": (0, 1, 0, 1340),
    },
    "20260127": {
        "calib_dir": COMM / "20260127" / "calib",
        "dark": None,
        "mbias_override": COMM / "20260127" / "calib" / "mbias.fits",
        "overscan_region": (0, 1, 0, 1340),
    },
    "20260128": {
        "calib_dir": COMM / "20260128" / "calib",
        "dark": None,
        "mbias_override": COMM / "20260128" / "calib" / "mbias.fits",
        "overscan_region": (0, 1, 0, 1340),
    },
    "20260129": {
        "calib_dir": COMM / "20260129" / "calib",
        "dark": None,
        "mbias_override": COMM / "20260129" / "calib" / "mbias.fits",
        "overscan_region": (0, 1, 0, 1340),
    },
    "20260204": {
        # Own mbias/dark present; TLM from 20260129 (nearest night with TLM)
        "calib_dir": COMM / "20260129" / "calib",
        "mbias_override": COMM / "20260204" / "calib" / "mbias.fits",
        "dark": COMM / "20260204" / "calib" / "mdark.fits",
        "overscan_region": (0, 31, 0, 1340),
    },
    "20260205": {
        # No tile files — included for completeness, will be skipped
        "calib_dir": COMM / "20260205" / "calib",
        "dark": COMM / "20260205" / "calib" / "mdark.fits",
        "mbias_override": COMM / "20260205" / "calib" / "mbias.fits",
        "overscan_region": (0, 31, 0, 1340),
    },
}

# Arc RED fallback order per spec_set (highest priority first)
ARC_RED_SEARCH_ORDER = [
    COMM / "20260129" / "calib" / "converted",
    COMM / "20260127" / "calib" / "converted",
    COMM / "20260125" / "calib" / "converted",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_global_arc_red_lookup() -> dict:
    """Scan all converted/ dirs (priority order) and build spec_set → Path."""
    lookup: dict[str, Path] = {}
    for conv_dir in reversed(ARC_RED_SEARCH_ORDER):  # lowest priority first
        if not conv_dir.exists():
            continue
        for p in conv_dir.glob("cArc_*_red.fits"):
            # Extract spec_set from filename, e.g. "cArc_300_490_240s ..."
            m = re.match(r"cArc_(\d+_\d+)_", p.name)
            if m:
                lookup[m.group(1)] = p
    return lookup


ARC_RED_LOOKUP: dict[str, Path] = _build_global_arc_red_lookup()


def extract_tile_base(filename: str) -> str:
    """
    Extract the canonical tile base name (tile_NNN) from a raw filename.

    Examples
    --------
    'tile_046_150_620 2026 January 27 14_53_58 1.fits' → 'tile_046'
    'tile_060_MTL 2026 February 04 12_28_50 1.fits'   → 'tile_060'
    'tile_032 2026 January 25 1.fits'                  → 'tile_032'
    """
    stem = Path(filename).stem.split()[0]  # first token before any space
    m = re.match(r"^(tile_\d+)", stem)
    if m:
        return m.group(1)
    return stem  # fallback (e.g. tilex_034)


def build_assign_table(tile_base: str) -> Table:
    """
    Load the fiber assignment table for *tile_base* and attach TYPE / NAME.
    """
    colnames = [
        "fiber_ID",
        "target_x",
        "target_y",
        "target_ra",
        "target_dec",
        "target_rank",
        "target_class",
    ]
    path = ASSIGNDIR / f"{tile_base}.assign.txt"
    if not path.exists():
        raise FileNotFoundError(f"Assign table not found: {path}")

    tbl = Table.read(path, names=colnames, format="ascii")
    
    tbl["TYPE"] = "N"
    tbl["TYPE"][np.isin(tbl["target_class"], [9])] = "S"
    tbl["TYPE"][np.isin(tbl["target_class"], [8])] = "C"
    tile_num = int(tile_base[-3:])
    if (tile_num < 50) or (tile_num > 100):
        tbl["TYPE"][np.isin(tbl["target_class"], [1, 2])] = "P"
    else:
        tbl["TYPE"][np.isin(tbl["target_class"], [0, 1, 2, 3])] = "P"
    tbl["NAME"] = np.array([f"FIB{i}" for i in tbl["fiber_ID"]])
    return tbl


# ---------------------------------------------------------------------------
# Calibration filename helpers
# ---------------------------------------------------------------------------

def extract_spec_set_from_calib_name(filename: str) -> str | None:
    """
    Extract spec_set (e.g. '300_490') from an Arc_/Flat_ raw filename.

    Examples
    --------
    'Arc_300_490_240s 2026 January 25.fits' → '300_490'
    'Flat_150_620 2026 January 29.fits'     → '150_620'
    """
    m = re.match(r"^(?:Arc|Flat)_(\d+)_(\d+)", Path(filename).name)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return None


def extract_exptime_from_calib_name(filename: str) -> int:
    """Return exposure time in seconds from a calib filename, or 0 if absent."""
    m = re.search(r"_(\d+)s\b", Path(filename).stem)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# Check image helpers
# ---------------------------------------------------------------------------

def _pct_norm(data: np.ndarray, lo: float = 1.0, hi: float = 99.0):
    """Return (vmin, vmax) from percentile clipping, ignoring NaN."""
    return float(np.nanpercentile(data, lo)), float(np.nanpercentile(data, hi))


def save_chkimg_2d(data: np.ndarray, out_path: Path, title: str = "") -> None:
    """Save a percentile-stretched 2-D image as a PNG check image."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vmin, vmax = _pct_norm(data)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="gray")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Spectral pixel")
    ax.set_ylabel("Spatial pixel")
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("  chkimg: %s", out_path.name)


def save_chkimg_tlm(im_path: Path, tlm_path: Path, out_path: Path, title: str = "") -> None:
    """Save a 2-D image with fiber tramline traces overlaid as a PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with fits.open(im_path) as h:
        data = h[0].data.astype(float)
    with fits.open(tlm_path) as h:
        tlm = h[0].data  # shape (nfib, nspec)
    vmin, vmax = _pct_norm(data)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(data, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="gray")
    nfib, nspec = tlm.shape
    colors = plt.cm.rainbow(np.linspace(0, 1, nfib))
    xs = np.arange(nspec)
    for i in range(nfib):
        ax.plot(xs, tlm[i], color=colors[i], lw=0.6, alpha=0.85)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Spectral pixel")
    ax.set_ylabel("Spatial pixel")
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("  chkimg (TLM): %s", out_path.name)


def save_chkimg_spectra(red_path: Path, out_path: Path, title: str = "",
                        n_show: int = 5) -> None:
    """Save a sample of extracted spectra as a PNG check image."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with fits.open(red_path) as h:
        data = h[0].data.astype(float)      # (nfib, nspec)
        extnames = [hdu.name for hdu in h]
        if "WAVELA" in extnames:
            wave = h["WAVELA"].data
            wave = wave[0] if wave.ndim == 2 else wave
            xlabel = r"Wavelength ($\AA$)"
        else:
            wave = np.arange(data.shape[1])
            xlabel = "Pixel"
    nfib = data.shape[0]
    indices = np.linspace(0, nfib - 1, min(n_show, nfib), dtype=int)
    fig, ax = plt.subplots(figsize=(12, 4))
    for i in indices:
        ax.plot(wave, data[i], lw=0.8, label=f"Fib {i + 1}", alpha=0.85)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Counts")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("  chkimg (spectra): %s", out_path.name)


def save_chkimg_wavecal(conv_dir: Path, chkimg_dir: Path, spec_set: str,
                        title: str = "") -> None:
    """Save a 3-panel wavecal diagnostic check image as a PNG."""
    diagnostic_dir = conv_dir / "diagnostic"
    if not diagnostic_dir.exists():
        logger.warning("  wavecal diagnostic dir not found: %s", diagnostic_dir)
        return

    try:
        wavecalid = Table.read(diagnostic_dir / "identified_arcs.dat",
                               format="ascii.fixed_width_two_line")
        wavecalcoeff = Table.read(diagnostic_dir / "global_fit_coefficients.dat",
                                  format="ascii.fixed_width_two_line")
        calispec = Table.read(diagnostic_dir / "CALIBRATED_SPECTRA.dat",
                              format="ascii", names=["wave", "flux"])
    except Exception as exc:
        logger.warning("  wavecal diagnostic files missing for spec_set=%s: %s", spec_set, exc)
        return

    x_pts = wavecalid["x_pts"]
    y_pts = wavecalid["y_pts"]
    residuals = wavecalid["residuals"]
    outliers = wavecalid["outliers"].data == "True"
    coeffs = wavecalcoeff["coeffs"]

    xlim = [np.min(calispec["wave"]), np.max(calispec["wave"])]

    try:
        wlist, _ilist, _labels, _nlist = read_arc_file(
            2, np.array(xlim), "HgArNeKrCd", arc_dir=ARC_TABLES_DIR
        )
    except Exception as exc:
        logger.warning("  read_arc_file failed for spec_set=%s: %s", spec_set, exc)
        wlist = []

    out_path = chkimg_dir / f"arc_{spec_set}_wavecal.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(15, 5), sharex=True)

    ax = axes[0]
    ax.plot(calispec["wave"], calispec["flux"], label="Calibrated Spectrum", c="tab:red")
    for w in wlist:
        ax.axvline(w, color="skyblue", lw=0.5, zorder=0)
    y_pts_inv = np.polyval(coeffs, x_pts)
    for v in y_pts_inv:
        ax.axvline(v, color="tab:red", lw=0.5, zorder=0)
    ax.legend()
    ax.set_yscale("log")
    ax.set_xlim(xlim)
    ax.set_ylabel("Count")
    if title:
        ax.set_title(title, fontsize=9)

    ax = axes[1]
    ax.plot(y_pts, residuals, c="tab:red", ls="", marker="+")
    ax.plot(y_pts[~outliers], residuals[~outliers], c="k", ls="", marker="d", ms=2,
            label="Matched")
    ax.legend()
    ax.axhline(0, c="gray", lw=0.5, zorder=0)
    ax.set_ylabel(r"Residual ($\AA$)")
    rms = np.sqrt(np.mean(residuals[~outliers] ** 2))
    ax.text(0.05, 0.95, f"RMS: {rms:.2f} $\\AA$", transform=ax.transAxes,
            ha="left", va="top")

    ax = axes[2]
    xx = np.arange(1340)
    yy = np.polyval(coeffs, xx)
    ax.plot(yy, xx, label="Solution", c="green")
    ax.plot(y_pts, x_pts, c="tab:red", ls="", marker="+")
    ax.plot(y_pts[~outliers], x_pts[~outliers], c="k", ls="", marker="d", ms=2,
            label="Matched")
    ax.legend()
    ax.set_xlabel(r"Wavelength ($\AA$)")
    ax.set_ylabel("Pixel")
    ax.set_xlim(xlim)

    fig.subplots_adjust(hspace=0.0)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("  chkimg (wavecal): %s", out_path.name)


# ---------------------------------------------------------------------------
# Skyline recalculation
# ---------------------------------------------------------------------------

def gaussian_linear(x, a, mu, sigma, slope, intercept):
    """Gaussian + linear continuum model."""
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + slope * x + intercept


def recalculate_sky_skyline(red_path: Path) -> None:
    """
    Recalculate PRIMARY and VARIANCE extensions using a per-fiber Gaussian
    fit to the O I 5578 Å sky line in RWSS and SKY.

    sky_scale = amplitude_rwss / amplitude_sky
    new_primary  = RWSS  − sky_scale * SKY
    new_variance = RWSSVAR + sky_scale² * SKYVAR

    If sky_scale ≤ 0 a warning is logged but the value is used as-is.
    If the fiber fit fails, sky_scale falls back to 1.0 (original result).
    """
    logger.info("  Skyline recalculation: %s", red_path.name)

    with fits.open(red_path) as hdul:
        extnames = [h.name for h in hdul]

        if "RWSS" not in extnames or "SKY" not in extnames:
            logger.warning("  RWSS or SKY extension missing — skipping %s", red_path.name)
            return

        if "WAVELA" not in extnames:
            logger.warning("  WAVELA extension missing — skipping skyline fit for %s", red_path.name)
            return

        rwss = hdul["RWSS"].data.astype(float)
        rwssvar = hdul["RWSSVAR"].data.astype(float) if "RWSSVAR" in extnames else np.zeros_like(rwss)
        sky = hdul["SKY"].data.astype(float)
        skyvar = hdul["SKYVAR"].data.astype(float) if "SKYVAR" in extnames else np.zeros_like(sky)

        wave_data = hdul["WAVELA"].data
        wave = wave_data[0] if wave_data.ndim == 2 else wave_data

        nfib = rwss.shape[0]

    # Sky-line window (O I 5578 Å)
    skyline_idx = np.where((wave > 5540) & (wave < 5620))[0]
    if len(skyline_idx) < 5:
        logger.warning(
            "  Sky-line window (5540–5620 Å) has fewer than 5 pixels in %s — skipping",
            red_path.name,
        )
        return

    w_win = wave[skyline_idx]
    sky_win = sky[skyline_idx]

    # Fit sky spectrum once
    sky_amp_guess = float(np.nanmax(sky_win) - np.nanmin(sky_win))
    sky_cont_guess = float(np.nanmedian(sky_win))
    try:
        popt_sky, _ = curve_fit(
            gaussian_linear,
            w_win,
            sky_win,
            p0=[sky_amp_guess, 5578.0, 5.0, 0.0, sky_cont_guess],
            maxfev=5000,
        )
    except RuntimeError:
        logger.warning(
            "  SKY Gaussian fit failed for %s — skipping skyline recalculation",
            red_path.name,
        )
        return

    new_primary = np.empty_like(rwss)
    new_variance = np.empty_like(rwss)

    for i in range(nfib):
        rwss_win = rwss[i][skyline_idx]
        amp_guess = float(np.nanmax(rwss_win) - np.nanmin(rwss_win))
        cont_guess = float(np.nanmedian(rwss_win))

        try:
            popt_rwss, _ = curve_fit(
                gaussian_linear,
                w_win,
                rwss_win,
                p0=[amp_guess, 5578.0, 5.0, 0.0, cont_guess],
                maxfev=5000,
            )
            sky_scale = popt_rwss[0] / popt_sky[0]
        except RuntimeError:
            logger.warning(
                "  Fiber %d: RWSS Gaussian fit failed — using sky_scale=1.0", i
            )
            sky_scale = 1.0

        if sky_scale <= 0:
            logger.warning(
                "  Fiber %d: negative sky scale (%.4f) — applying as-is", i, sky_scale
            )

        new_primary[i] = rwss[i] - sky_scale * sky
        new_variance[i] = rwssvar[i] + sky_scale ** 2 * skyvar

    # Write updated extensions back
    with fits.open(red_path, mode="update") as hdul:
        hdul["PRIMARY"].data = new_primary.astype(hdul["PRIMARY"].data.dtype)
        hdul["VARIANCE"].data = new_variance.astype(hdul["VARIANCE"].data.dtype)
        hdul["PRIMARY"].header.add_history(
            "Sky recalculated via skyline Gaussian fit (O I 5578 AA)"
        )
        hdul.flush()

    logger.info("  Updated PRIMARY and VARIANCE in %s", red_path.name)


# ---------------------------------------------------------------------------
# Per-date calibration reduction (Phase 0)
# ---------------------------------------------------------------------------

def reduce_global_dark() -> None:
    """
    Regenerate MASTER_DARK_GLOBAL (20260130 2400s dark) from raw frames.

    Also rebuilds the 20260130 master bias first so dark frames are
    bias-subtracted before combination.
    """
    calib_dir = COMM / "20260130" / "calib"
    if not calib_dir.exists():
        logger.warning("20260130 calib dir not found — skipping global dark reduction")
        return

    conv_dir = calib_dir / "converted"
    conv_dir.mkdir(exist_ok=True)

    # 1. Master bias for 20260130
    mbias_30 = calib_dir / "mbias.fits"
    bias_files = sorted(calib_dir.glob("Bias*.fits"))
    if bias_files:
        bias_conv = [conv_dir / f.name.replace("Bias", "cBias", 1) for f in bias_files]
        for braw, bconv in zip(bias_files, bias_conv):
            write_isoplane_converted_image(braw, bconv, "BIAS", n_fibers=KSPEC_NFIBERS)
        try:
            reduce_bias([p.as_posix() for p in bias_conv], output_file=mbias_30.as_posix())
            logger.info("20260130: mbias written")
        except Exception as exc:
            logger.error("20260130: reduce_bias failed: %s", exc)
            mbias_30 = None
    else:
        mbias_30 = None

    # 2. Master dark (2400s) for 20260130 — used as MASTER_DARK_GLOBAL
    dark_2400 = sorted(calib_dir.glob("Dark_2400s*.fits"))
    if not dark_2400:
        logger.warning("20260130: no Dark_2400s*.fits found — MASTER_DARK_GLOBAL not updated")
        return
    try:
        reduce_dark(
            [f.as_posix() for f in dark_2400],
            output_file=MASTER_DARK_GLOBAL.as_posix(),
            bias_filename=mbias_30.as_posix() if mbias_30 else None,
        )
        logger.info("20260130: MASTER_DARK_GLOBAL written: %s", MASTER_DARK_GLOBAL.name)
    except Exception as exc:
        logger.error("20260130: reduce_dark failed: %s", exc)


def reduce_calib_date(date: str) -> None:
    """
    Regenerate calibration products (TLM, arc wavelength solution, fflat) for *date*.

    Flat frames  → isoplane convert → make_im → make_tlm + make_ex
    Arc frames   → isoplane convert → make_im → reduce_arc → cArc_*_red.fits
    Fiber flat   → reduce_fflat (using flat_ex + arc_red) → fflat_{spec_set}_red.fits

    Check images are written to calib/chkimg/.
    Existing products are always overwritten.
    """
    config = DATE_CONFIG.get(date, {})
    calib_dir: Path = config.get("calib_dir", COMM / "20260125" / "calib")

    # Only process dates that own their calib_dir.
    if not calib_dir.is_relative_to(COMM / date):
        logger.info(
            "%s: calib_dir belongs to another date — skipping calib reduction", date
        )
        return

    if not calib_dir.exists():
        logger.warning("%s: calib_dir not found: %s — skipping", date, calib_dir)
        return

    mbias_path: Path | None = config.get("mbias_override", calib_dir / "mbias.fits")
    if mbias_path is not None and not mbias_path.exists():
        logger.warning("%s: mbias not found at %s — skipping bias subtraction", date, mbias_path)
        mbias_path = None

    # Use per-date dark if available, otherwise fall back to global master dark.
    dark_path: Path | None = config.get("dark", None)
    if dark_path is None or not dark_path.exists():
        dark_path = MASTER_DARK_GLOBAL if MASTER_DARK_GLOBAL.exists() else None
    use_dark = dark_path is not None
    overscan_region: tuple = config.get("overscan_region", (0, 1, 0, 1340))

    conv_dir = calib_dir / "converted"
    chkimg_dir = calib_dir / "chkimg"
    conv_dir.mkdir(exist_ok=True)
    chkimg_dir.mkdir(exist_ok=True)

    # Shared make_im call for calib frames (same as notebook: LACOSMIC + bias + dark).
    def _make_im_calib(raw_conv_path: Path, im_path: Path) -> None:
        # Guard against shape mismatch between the dark and this frame
        # (e.g. master dark is (1300,1340) but some dates read out (1330,1340)).
        _use_dark = use_dark
        if use_dark and dark_path is not None:
            with fits.open(raw_conv_path) as _hf, fits.open(dark_path) as _hd:
                if _hf[0].data.shape != _hd[0].data.shape:
                    logger.warning(
                        "  Dark shape %s ≠ frame shape %s — skipping dark for %s",
                        _hd[0].data.shape, _hf[0].data.shape, raw_conv_path.name,
                    )
                    _use_dark = False
        make_im(
            raw_conv_path.as_posix(),
            im_filename=im_path.as_posix(),
            use_bias=mbias_path is not None,
            bias_filename=mbias_path.as_posix() if mbias_path else None,
            overscan_region=overscan_region,
            use_dark=_use_dark,
            dark_filename=dark_path.as_posix() if _use_dark else None,
            cosmic_ray_method="LACOSMIC",
            verbose=False,
        )

    # -----------------------------------------------------------------------
    # Phase 0 pre: Bias → master bias
    # -----------------------------------------------------------------------
    bias_files = sorted(calib_dir.glob("Bias*.fits"))
    if bias_files:
        bias_conv = [conv_dir / f.name.replace("Bias", "cBias", 1) for f in bias_files]
        for braw, bconv in zip(bias_files, bias_conv):
            write_isoplane_converted_image(braw, bconv, "BIAS", n_fibers=KSPEC_NFIBERS)
        mbias_out = calib_dir / "mbias.fits"
        try:
            reduce_bias([p.as_posix() for p in bias_conv], output_file=mbias_out.as_posix())
            mbias_path = mbias_out
            logger.info("%s: mbias written: %s", date, mbias_out.name)
        except Exception as exc:
            logger.error("%s: reduce_bias failed: %s", date, exc)
    else:
        logger.warning("%s: no Bias*.fits found — mbias unchanged", date)

    # -----------------------------------------------------------------------
    # Phase 0 pre: Dark → master dark
    # -----------------------------------------------------------------------
    dark_files = sorted(calib_dir.glob("Dark_*.fits"))
    if dark_files:
        mdark_out: Path = config.get("dark") or (calib_dir / "mdark.fits")
        try:
            reduce_dark(
                [f.as_posix() for f in dark_files],
                output_file=mdark_out.as_posix(),
                bias_filename=mbias_path.as_posix() if mbias_path else None,
            )
            dark_path = mdark_out
            use_dark = True
            logger.info("%s: mdark written: %s", date, mdark_out.name)
        except Exception as exc:
            logger.error("%s: reduce_dark failed: %s", date, exc)
    else:
        logger.info("%s: no Dark_*.fits found — dark unchanged", date)

    # Track per-spec_set paths needed for fflat reduction.
    # calib_by_spec[spec_set] = {"flat_ex": Path, "arc_red": Path}
    calib_by_spec: dict[str, dict] = {}

    # -----------------------------------------------------------------------
    # Phase 0a: Flat → TLM + make_ex
    # -----------------------------------------------------------------------
    flat_files = sorted(calib_dir.glob("Flat_*.fits"))

    # Group by spec_set; keep the flat with the longest exposure.
    flat_by_spec: dict[str, Path] = {}
    for f in flat_files:
        ss = extract_spec_set_from_calib_name(f.name)
        if ss is None:
            continue
        if ss not in flat_by_spec or (
            extract_exptime_from_calib_name(f.name)
            > extract_exptime_from_calib_name(flat_by_spec[ss].name)
        ):
            flat_by_spec[ss] = f

    for spec_set, flat_raw in sorted(flat_by_spec.items()):
        logger.info("%s: flat → TLM  spec_set=%s  (%s)", date, spec_set, flat_raw.name)

        conv_name = flat_raw.name.replace("Flat_", "cFlat_", 1)
        conv_path = conv_dir / conv_name
        im_path = conv_dir / (conv_path.stem + "_im.fits")
        ex_path = conv_dir / (conv_path.stem + "_ex.fits")
        tlm_out = calib_dir / f"tlm_{spec_set}.fits"

        # 1. Isoplane convert
        write_isoplane_converted_image(flat_raw, conv_path, "MFFFF",
                                       n_fibers=KSPEC_NFIBERS)
        with fits.open(conv_path) as h:
            save_chkimg_2d(h[0].data.astype(float),
                           chkimg_dir / f"flat_{spec_set}_conv.png",
                           title=f"{date}  flat {spec_set} — isoplane convert")

        # 2. make_im
        _make_im_calib(conv_path, im_path)
        with fits.open(im_path) as h:
            save_chkimg_2d(h[0].data.astype(float),
                           chkimg_dir / f"flat_{spec_set}_im.png",
                           title=f"{date}  flat {spec_set} — make_im")

        # 3. make_tlm + make_ex
        try:
            tlm_args = {
                "IMAGE_FILENAME": im_path.as_posix(),
                "TLMAP_FILENAME": tlm_out.as_posix(),
                "EXTRAC_FILENAME": ex_path.as_posix(),
            }
            make_tlm(tlm_args)
            make_ex(tlm_args)
            save_chkimg_tlm(im_path, tlm_out,
                            chkimg_dir / f"flat_{spec_set}_tlm.png",
                            title=f"{date}  flat {spec_set} — tramlines")
            calib_by_spec[spec_set] = {"flat_ex": ex_path, "flat_conv": conv_path}
            logger.info("%s: TLM + flat_ex written for spec_set=%s", date, spec_set)
        except Exception as exc:
            logger.error("%s: flat TLM/ex failed for %s: %s", date, flat_raw.name, exc)

    # -----------------------------------------------------------------------
    # Phase 0b: Arc → wavelength calibration
    # -----------------------------------------------------------------------
    arc_files = sorted(calib_dir.glob("Arc_*.fits"))

    # Group by spec_set; keep the arc with the longest exposure.
    arc_by_spec: dict[str, Path] = {}
    for f in arc_files:
        ss = extract_spec_set_from_calib_name(f.name)
        if ss is None:
            continue
        if ss not in arc_by_spec or (
            extract_exptime_from_calib_name(f.name)
            > extract_exptime_from_calib_name(arc_by_spec[ss].name)
        ):
            arc_by_spec[ss] = f

    # Arc spec_sets to skip (no usable wavelength solution available)
    ARC_SKIP = {"600_430"}

    for spec_set, arc_raw in sorted(arc_by_spec.items()):
        if spec_set in ARC_SKIP:
            logger.info("%s: arc spec_set=%s in skip list — skipping", date, spec_set)
            continue

        # Find TLM: exact match first, then fall back to any available spec_set.
        # TLM encodes only spatial fiber positions (unchanged across grating settings),
        # so any TLM from the same night is valid for arc extraction.
        if spec_set in calib_by_spec:
            tlm_spec = spec_set
        elif calib_by_spec:
            tlm_spec = sorted(calib_by_spec.keys())[0]
            logger.warning(
                "%s: no TLM for spec_set=%s — using %s TLM as fallback",
                date, spec_set, tlm_spec,
            )
        else:
            logger.warning(
                "%s: no TLM available at all — skipping arc reduction for spec_set=%s",
                date, spec_set,
            )
            continue

        tlm_path = calib_dir / f"tlm_{tlm_spec}.fits"
        conv_name = arc_raw.name.replace("Arc_", "cArc_", 1)
        conv_path = conv_dir / conv_name
        im_path = conv_dir / (conv_path.stem + "_im.fits")
        ex_path = conv_dir / (conv_path.stem + "_ex.fits")
        red_path = conv_dir / (conv_path.stem + "_red.fits")

        logger.info("%s: arc → wavecal  spec_set=%s  (%s)", date, spec_set, arc_raw.name)

        # 1. Isoplane convert
        write_isoplane_converted_image(arc_raw, conv_path, "MFARC",
                                       n_fibers=KSPEC_NFIBERS)
        with fits.open(conv_path) as h:
            save_chkimg_2d(h[0].data.astype(float),
                           chkimg_dir / f"arc_{spec_set}_conv.png",
                           title=f"{date}  arc {spec_set} — isoplane convert")

        # 2. make_im
        _make_im_calib(conv_path, im_path)
        with fits.open(im_path) as h:
            save_chkimg_2d(h[0].data.astype(float),
                           chkimg_dir / f"arc_{spec_set}_im.png",
                           title=f"{date}  arc {spec_set} — make_im")

        # 3. reduce_arc  (IMAGE_FILENAME already exists → skips internal make_im)
        arc_args = {
            "IMAGE_FILENAME": im_path.as_posix(),
            "EXTRAC_FILENAME": ex_path.as_posix(),
            "OUTPUT_FILENAME": red_path.as_posix(),
            "TLMAP_FILENAME": tlm_path.as_posix(),
            "ARCDIR": ARC_TABLES_DIR.as_posix(),
            "LAMPNAME": "HgArNeKrCd",
            "USE_GENCAL": True,
        }
        try:
            reduce_arc(arc_args, get_diagnostic=True, diagnostic_dir=conv_dir/"diagnostic")
            logger.info("%s: arc RED written: %s", date, red_path.name)
            calib_by_spec.setdefault(spec_set, {})["arc_red"] = red_path
            if red_path.exists():
                save_chkimg_spectra(red_path,
                                    chkimg_dir / f"arc_{spec_set}_red.png",
                                    title=f"{date}  arc {spec_set} — wavelength solution")
                save_chkimg_wavecal(conv_dir, chkimg_dir, spec_set,
                                    title=f"{date}  arc {spec_set} — wavecal diagnostic")
        except Exception as exc:
            logger.error("%s: reduce_arc failed for %s: %s", date, arc_raw.name, exc)

    # -----------------------------------------------------------------------
    # Phase 0c: Fiber flat-field reduction
    # -----------------------------------------------------------------------
    for spec_set, products in sorted(calib_by_spec.items()):
        flat_ex = products.get("flat_ex")
        flat_conv = products.get("flat_conv")
        arc_red = products.get("arc_red")

        if flat_ex is None or arc_red is None:
            logger.warning(
                "%s: missing flat_ex or arc_red for spec_set=%s — skipping fflat",
                date, spec_set,
            )
            continue

        fflat_out = calib_dir / f"fflat_{spec_set}_red.fits"
        logger.info("%s: fflat  spec_set=%s", date, spec_set)
        try:
            reduce_fflat({
                "RAW_FILENAME": flat_conv.as_posix(),
                "EXTRAC_FILENAME": flat_ex.as_posix(),
                "OUTPUT_FILENAME": fflat_out.as_posix(),
                "WAVEL_FILENAME": arc_red.as_posix(),
                "DO_TLMAP": False,
                "DO_EXTRA": False,
                "DO_REDFL": True,
                "TRUNCFLAT": True,
                "USEFLATSTART": 30,
                "USEFLATEND": 1310,
                "LAF_FLAG": False,
                "BSSMOOTH": False,
                "VERBOSE": False,
            })
            logger.info("%s: fflat written: %s", date, fflat_out.name)
        except Exception as exc:
            logger.error("%s: reduce_fflat failed for spec_set=%s: %s", date, spec_set, exc)


# ---------------------------------------------------------------------------
# Per-date reduction
# ---------------------------------------------------------------------------

def reduce_date(date: str, force: bool = False) -> list[Path]:
    """Reduce all tile files for *date*. Returns list of produced _red.fits paths."""
    obsdir = COMM / date
    if not obsdir.exists():
        logger.warning("Obs dir not found: %s — skipping", obsdir)
        return []

    raw_files = sorted(obsdir.glob("tile_*.fits"))
    if not raw_files:
        logger.info("%s: no tile_*.fits files found — skipping", date)
        return []

    config = DATE_CONFIG.get(date, {})
    calib_dir: Path = config.get("calib_dir", COMM / "20260125" / "calib")
    dark_path: Path | None = config.get("dark", None)
    overscan_region: tuple = config.get("overscan_region", (0, 1, 0, 1340))
    # Per-date mbias override (e.g. 20260204 has its own mbias)
    mbias_path: Path = config.get("mbias_override", calib_dir / "mbias.fits")

    if not mbias_path.exists():
        logger.warning("%s: mbias not found at %s — make_im will run without bias", date, mbias_path)
        mbias_path = None

    use_dark = dark_path is not None and dark_path.exists()
    if config.get("dark") and not use_dark:
        logger.warning("%s: dark file %s not found — skipping dark subtraction", date, dark_path)

    conv_dir = obsdir / "converted"
    proc_dir = obsdir / "processed"
    chkimg_dir = proc_dir / "chkimg"
    conv_dir.mkdir(exist_ok=True)
    proc_dir.mkdir(exist_ok=True)
    chkimg_dir.mkdir(exist_ok=True)

    produced: list[Path] = []

    for raw in raw_files:
        tile_base = extract_tile_base(raw.name)

        # --- Early skip (pure string ops — no file I/O needed) ---
        _conv_name = raw.name.replace(tile_base, "c" + tile_base, 1)
        _red_path = proc_dir / (Path(_conv_name).stem + "_red.fits")
        if _red_path.exists() and force is False:
            logger.info("%s: output already exists, skipping: %s", date, _red_path.name)
            produced.append(_red_path)
            continue

        logger.info("%s: processing %s (tile_base=%s)", date, raw.name, tile_base)

        # --- Assign table ---
        try:
            assign_table = build_assign_table(tile_base)
        except FileNotFoundError as exc:
            logger.warning("  %s — skipping", exc)
            continue

        # --- Convert raw → isoplane format ---
        conv_name = raw.name.replace(tile_base, "c" + tile_base, 1)
        conv_path = conv_dir / conv_name
        write_isoplane_converted_image(
            raw, conv_path, "OBJECT", fiber_table=assign_table
        )
        with fits.open(conv_path) as h:
            save_chkimg_2d(h[0].data.astype(float),
                           chkimg_dir / f"{conv_path.stem}_conv.png",
                           title=f"{date}  {tile_base} — isoplane convert")

        # --- make_im ---
        im_path = conv_dir / (conv_path.stem + "_im.fits")
        # Guard against shape mismatch between the dark and this frame.
        _use_dark = use_dark
        if use_dark and dark_path is not None:
            with fits.open(conv_path) as _hf, fits.open(dark_path) as _hd:
                if _hf[0].data.shape != _hd[0].data.shape:
                    logger.warning(
                        "  Dark shape %s ≠ frame shape %s — skipping dark for %s",
                        _hd[0].data.shape, _hf[0].data.shape, conv_path.name,
                    )
                    _use_dark = False
        use_glow = GLOW_PCA_CUBE.exists()
        make_im(
            conv_path.as_posix(),
            im_filename=im_path.as_posix(),
            cosmic_ray_method="LACOSMIC",
            bias_filename=mbias_path.as_posix() if mbias_path else None,
            use_bias=mbias_path is not None,
            overscan_region=overscan_region,
            use_dark=_use_dark,
            dark_filename=dark_path.as_posix() if _use_dark else None,
            use_glow_pca=use_glow,
            glow_pca_filename=GLOW_PCA_CUBE.as_posix() if use_glow else None,
            dc_rate_filename=DC_RATE_FILE.as_posix() if DC_RATE_FILE else None,
            glow_n_components=GLOW_N_COMPONENTS,
            glow_fit_rows=GLOW_FIT_ROWS,
            verbose=False,
        )
        with fits.open(im_path) as h:
            save_chkimg_2d(h[0].data.astype(float),
                           chkimg_dir / f"{conv_path.stem}_im.png",
                           title=f"{date}  {tile_base} — make_im")

        # --- Determine spec_set from IM header ---
        try:
            with ImageFile(im_path.as_posix()) as img:
                grating = img.get_header_value("GRATID")
                lambdac = float(img.get_header_value("LAMBDAC"))
            spec_set = f"{grating}_{lambdac / 10:.0f}"
        except Exception as exc:
            logger.warning("  Could not read GRATID/LAMBDAC from %s: %s — skipping", im_path.name, exc)
            continue

        # --- Calibration file paths ---
        tlm_path = calib_dir / f"tlm_{spec_set}.fits"
        fflat_path = calib_dir / f"fflat_{spec_set}_red.fits"
        arc_red_path = ARC_RED_LOOKUP.get(spec_set)

        if not tlm_path.exists():
            logger.warning(
                "  TLM not found for spec_set=%s at %s — skipping %s",
                spec_set, tlm_path, raw.name,
            )
            continue

        if arc_red_path is None or not arc_red_path.exists():
            logger.warning(
                "  No arc RED found for spec_set=%s — wavelength calibration will be skipped",
                spec_set,
            )

        ex_path = conv_dir / (conv_path.stem + "_ex.fits")
        red_path = proc_dir / (conv_path.stem + "_red.fits")

        # --- reduce_object ---
        args = {
            "IMAGE_FILENAME": im_path.as_posix(),
            "EXTRAC_FILENAME": ex_path.as_posix(),
            "OUTPUT_FILENAME": red_path.as_posix(),
            "TLMAP_FILENAME": tlm_path.as_posix(),
            "FFLAT_FILENAME": fflat_path.as_posix() if fflat_path.exists() else "",
            "USEFFLAT": False,
            "WAVEL_FILENAME": arc_red_path.as_posix() if arc_red_path else "",
            "TPMETH": "OFF",
            "INC_RWSS": True,
            "SKYSUB": True,
            "SKYCOMBINE": "SIGCLIP",
            "SKYCOMBINE_SIGMA": 3.0,
            "SKYCOMBINE_ITERS": 15,
            "CALIBFLUX": False,
            "VERBOSE": False,
        }

        try:
            reduce_object(args)
            logger.info("  Written: %s", red_path.name)
            produced.append(red_path)
            save_chkimg_spectra(red_path,
                                chkimg_dir / f"{conv_path.stem}_red.png",
                                title=f"{date}  {tile_base} — reduced spectra")
        except Exception as exc:
            logger.error("  reduce_object failed for %s: %s", raw.name, exc)

    return produced


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tiles whose output _red.fits already exists.",
    )
    parser.add_argument(
        "--no-skyline",
        action="store_true",
        help="Skip the skyline sky recalculation (Phase 2).",
    )
    args = parser.parse_args()

    all_reds: list[Path] = []

    logger.info("=== Phase 0 pre: global master dark (20260130) ===")
    reduce_global_dark()

    logger.info("=== Phase 0: calibration (TLM + wavelength calibration) ===")
    for date in DATES:
        logger.info("--- Date: %s ---", date)
        reduce_calib_date(date)

    # Rebuild arc RED lookup so Phase 1 finds files generated by Phase 0.
    global ARC_RED_LOOKUP
    ARC_RED_LOOKUP = _build_global_arc_red_lookup()
    logger.info("Arc RED lookup rebuilt: %d spec_set(s) found", len(ARC_RED_LOOKUP))

    logger.info("=== Phase 1: reduce_object ===")
    for date in DATES:
        logger.info("--- Date: %s ---", date)
        reds = reduce_date(date, force=not args.skip_existing)
        all_reds.extend(reds)

    if args.no_skyline:
        logger.info("=== Phase 2: skyline sky recalculation skipped (--no-skyline) ===")
    else:
        logger.info("=== Phase 2: skyline sky recalculation ===")
        for red_path in all_reds:
            try:
                recalculate_sky_skyline(red_path)
            except Exception as exc:
                logger.error("  Skyline recalculation failed for %s: %s", red_path.name, exc)

    logger.info("Done. Processed %d file(s).", len(all_reds))


if __name__ == "__main__":
    main()
