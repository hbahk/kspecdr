"""
Batch object reduction for KSPEC commissioning nights.

Reduces all tile_*.fits object frames for the following dates:
    20260124, 20260125, 20260127, 20260128, 20260129, 20260204, 20260205

After reduce_object completes, recalculates PRIMARY and VARIANCE for each
output file using a per-fiber Gaussian fit to the O I 5578 Å sky line in
the RWSS and SKY extensions.

Usage:
    python reduce_comm_batch.py [--force]

    --force   Re-reduce even if the output _red.fits already exists.
"""
import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import curve_fit

from kspecdr.inst.isoplane import write_isoplane_converted_image
from kspecdr.io.image import ImageFile
from kspecdr.preproc.make_im import make_im
from kspecdr.reduce_object import reduce_object

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

DATES = [
    "20260124",
    "20260125",
    "20260127",
    "20260128",
    "20260129",
    "20260204",
    "20260205",
]

# Master dark (20260130 long dark; used when a date has no dark of its own)
MASTER_DARK_GLOBAL = COMM / "20260130" / "calib" / "mdark_2400s.fits"

# ---------------------------------------------------------------------------
# Per-date calibration configuration
# ---------------------------------------------------------------------------
# For each date: which calib dir supplies mbias/tlm/fflat,
# and which dark file to use (None = no dark subtraction).
DATE_CONFIG = {
    "20260124": {
        "calib_dir": COMM / "20260125" / "calib",
        "dark": COMM / "20260205" / "calib" / "mdark_cut.fits",
    },
    "20260125": {
        "calib_dir": COMM / "20260125" / "calib",
        "dark": None,
    },
    "20260127": {
        "calib_dir": COMM / "20260127" / "calib",
        "dark": None,
    },
    "20260128": {
        # No own TLM/mbias — fall back to the nearest night with calibs
        "calib_dir": COMM / "20260127" / "calib",
        "dark": None,
    },
    "20260129": {
        "calib_dir": COMM / "20260129" / "calib",
        "dark": None,
    },
    "20260204": {
        # Own mbias/dark present; TLM from 20260129 (nearest night with TLM)
        "calib_dir": COMM / "20260129" / "calib",
        "mbias_override": COMM / "20260204" / "calib" / "mbias.fits",
        "dark": COMM / "20260204" / "calib" / "mdark.fits",
    },
    "20260205": {
        # No tile files — included for completeness, will be skipped
        "calib_dir": COMM / "20260205" / "calib",
        "dark": None,
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
    tbl["TYPE"][np.isin(tbl["target_class"], [1, 2])] = "P"
    tbl["TYPE"][np.isin(tbl["target_class"], [9])] = "S"
    tbl["TYPE"][np.isin(tbl["target_class"], [8])] = "C"
    tbl["NAME"] = np.array([f"FIB{i}" for i in tbl["fiber_ID"]])
    return tbl


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
    conv_dir.mkdir(exist_ok=True)
    proc_dir.mkdir(exist_ok=True)

    produced: list[Path] = []

    for raw in raw_files:
        tile_base = extract_tile_base(raw.name)
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

        # --- make_im ---
        im_path = conv_dir / (conv_path.stem + "_im.fits")
        make_im(
            conv_path.as_posix(),
            im_filename=im_path.as_posix(),
            cosmic_ray_method="LACOSMIC",
            bias_filename=mbias_path.as_posix() if mbias_path else None,
            use_bias=mbias_path is not None,
            use_dark=use_dark,
            dark_filename=dark_path.as_posix() if use_dark else None,
            verbose=False,
        )

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

        if red_path.exists() and not force:
            logger.info("  Output already exists, skipping: %s", red_path.name)
            produced.append(red_path)
            continue

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
        except Exception as exc:
            logger.error("  reduce_object failed for %s: %s", raw.name, exc)

    return produced


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-reduce even if output _red.fits already exists.",
    )
    args = parser.parse_args()

    all_reds: list[Path] = []

    logger.info("=== Phase 1: reduce_object ===")
    for date in DATES:
        logger.info("--- Date: %s ---", date)
        reds = reduce_date(date, force=args.force)
        all_reds.extend(reds)

    logger.info("=== Phase 2: skyline sky recalculation ===")
    for red_path in all_reds:
        try:
            recalculate_sky_skyline(red_path)
        except Exception as exc:
            logger.error("  Skyline recalculation failed for %s: %s", red_path.name, exc)

    logger.info("Done. Processed %d file(s).", len(all_reds))


if __name__ == "__main__":
    main()
