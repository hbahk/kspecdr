import logging
import shutil
import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from scipy.interpolate import LSQUnivariateSpline

from .io.image import ImageFile
from .preproc.make_im import make_im
from .extract.make_ex import make_ex
from .tlm.make_tlm import make_tlm
from .wavecal.scrunch import scrunch_from_arc_id

logger = logging.getLogger(__name__)

# Constants
MAX_NFIBRES = 1000
VAL__BADR = np.nan

def reduce_fflat(args: Dict[str, Any]):
    """
    Reduces a raw multi-fibre flat file to produce im(age), ex(tracted) and
    red(uced) fflat files.
    """
    # 1. Initialization
    raw_fname = args.get('RAW_FILENAME')
    im_fname = args.get('IMAGE_FILENAME')
    ex_fname = args.get('EXTRAC_FILENAME')
    red_fname = args.get('OUTPUT_FILENAME')
    tlm_fname = args.get('TLMAP_FILENAME')

    # 2. Make IM
    if args.get('DO_TLMAP', True):
        if raw_fname and not (im_fname and Path(im_fname).exists()):
             im_fname = make_im(raw_fname, im_filename=im_fname, verbose=args.get('VERBOSE', False))
             args['IMAGE_FILENAME'] = im_fname

    # 3. Make TLM
    if args.get('DO_TLMAP', True):
        if im_fname and not (tlm_fname and Path(tlm_fname).exists()):
             make_tlm(args)
        if args.get('MAKE_TLM_ONLY', False):
             logger.info("Only creating TLM this pass.")
             return

    # 4. Make EX
    if args.get('DO_EXTRA', True):
        if not (ex_fname and Path(ex_fname).exists()):
             make_ex(args)
             logger.info("Fibre Flat Frame Extracted")

    if not args.get('DO_REDFL', True):
        return

    # 5. Create RED
    if not ex_fname or not red_fname:
         logger.error("EXTRAC_FILENAME or OUTPUT_FILENAME not specified.")
         return

    logger.info(f"Reducing flat spectra data from {ex_fname}")
    shutil.copy2(ex_fname, red_fname)

    # 6. Scrunch Flat Frame logic
    tmp_fname = "tmp_mfsfff.fits"
    shutil.copy2(ex_fname, tmp_fname)

    try:
        arc_fname = args.get('WAVEL_FILENAME')

        # 6.1 Scrunch TMP to Arc
        scrunch_from_arc_id(tmp_fname, arc_fname, args, reverse=False)

        # 6.2 Average and Normalize TMP (DO_AFF)
        do_aff(tmp_fname, args)

        # 6.3 Un-Scrunch TMP (Reverse)
        scrunch_from_arc_id(tmp_fname, arc_fname, args, reverse=True)

        # 6.4 Extrapolate (unless B-Spline smooth requested)
        bs_smooth = args.get('BSSMOOTH', False)

        # Read TMP for manipulation
        with ImageFile(tmp_fname, mode='UPDATE') as tmp_file:
             data = tmp_file.read_image_data().T # (NSPEC, NFIB)

             if not bs_smooth:
                 cmfff_extrap(data)
                 tmp_file.write_image_data(data.T)
             else:
                 logger.info("Skipping extrapolation for B-Spline smoothing.")

        # 6.5 Normalize RED by TMP (CMFFF_NORM)
        with ImageFile(red_fname, mode='UPDATE') as red_file:
             red_data = red_file.read_image_data().T
             red_var = red_file.read_variance_data().T

             with ImageFile(tmp_fname, mode='READ') as tmp_file:
                 aff_data = tmp_file.read_image_data().T

             trunc_flat = args.get('TRUNCFLAT', False)
             start_pix = args.get('USEFLATSTART', 1)
             end_pix = args.get('USEFLATEND', 2048)

             cmfff_norm(red_data, red_var, aff_data, trunc_flat, start_pix, end_pix)

             red_file.write_image_data(red_data.T)
             red_file.write_variance_data(red_var.T)
             red_file.set_header_value("SCRUNCH", True)

        # 7. B-Spline Smoothing (if requested)
        if bs_smooth:
             bs_smooth_redflat(red_fname, args)

        # Set class
        with ImageFile(red_fname, mode='UPDATE') as red_file:
             red_file.set_class("REDUCED")

    finally:
        if os.path.exists(tmp_fname):
            os.remove(tmp_fname)

    logger.info(f"Fibre Flat Frame Reduced: {red_fname}")


def do_aff(filename, args):
    """
    Normalise and average the spectra.
    """
    laf_flag = args.get('LAF_FLAG', False)
    laf_par = args.get('LAF_PAR', 10)
    trunc_flat = args.get('TRUNCFLAT', False)
    start_pix = args.get('USEFLATSTART', 1)
    end_pix = args.get('USEFLATEND', 2048)
    plot = args.get('AFFPLOT', False)

    with ImageFile(filename, mode='UPDATE') as f:
        data = f.read_image_data().T # (NSPEC, NFIB)
        nx, nf = data.shape
        fiber_types, _ = f.read_fiber_types(nf)

        goodfib = np.array([ft in ['P', 'S'] for ft in fiber_types[:nf]])
        logger.info(f"{np.sum(goodfib)} good fibres used to make flat field")

        cmfff_aver(data, goodfib, laf_flag, laf_par, trunc_flat, start_pix, end_pix)

        f.write_image_data(data.T)

def cmfff_aver(data, goodfib, laf_flag, laf_par, trunc_flat, start_pix, end_pix):
    """
    Normalizes fibers by median, then averages them.
    """
    nx, nf = data.shape

    # 1. Normalize each fiber by median
    for fib in range(nf):
        if not goodfib[fib]:
            data[:, fib] = np.nan
            continue

        spec = data[:, fib]
        valid = np.isfinite(spec)

        if trunc_flat:
             mask = np.ones(nx, dtype=bool)
             mask[:start_pix-1] = False
             mask[end_pix:] = False
             valid = valid & mask

        if np.sum(valid) == 0:
            goodfib[fib] = False
            data[:, fib] = np.nan
            continue

        median_val = np.median(spec[valid])
        if median_val == 0:
            goodfib[fib] = False
            data[:, fib] = np.nan
            continue

        data[:, fib] /= median_val

    # 2. Average (Do All or Do Local)
    ypix = np.zeros(nx)
    if laf_flag:
        do_local(data, goodfib, laf_par, ypix)
    else:
        do_all(data, goodfib, ypix)

    if not laf_flag:
        for fib in range(nf):
            data[:, fib] = ypix

def do_all(data, goodfib, ypix):
    """
    Average across all fibers with sigma clipping.
    """
    nx, nf = data.shape
    clip = 5.0

    good_data = data[:, goodfib] # (NX, N_GOOD)
    if good_data.shape[1] == 0:
        ypix[:] = np.nan
        return

    masked_data = np.ma.masked_invalid(good_data)

    # Pass 1
    mean_spec = np.ma.mean(masked_data, axis=1)
    std_spec = np.ma.std(masked_data, axis=1)

    # Pass 2: Clip
    mean_broad = mean_spec[:, np.newaxis]
    std_broad = std_spec[:, np.newaxis]

    mask = np.abs(masked_data - mean_broad) > clip * std_broad
    masked_data.mask = masked_data.mask | mask

    final_mean = np.ma.mean(masked_data, axis=1).filled(np.nan)
    ypix[:] = final_mean

def do_local(data, goodfib, laf_par, ypix):
    """
    Local averaging using window +/- laf_par.
    Updates data in place.
    """
    nx, nf = data.shape
    clip = 5.0

    masked_data = np.ma.masked_invalid(data)
    masked_data[:, ~goodfib] = np.ma.masked

    # Iterate fibers
    for f in range(nf):
        low = max(0, f - laf_par)
        high = min(nf, f + laf_par + 1)

        window_data = masked_data[:, low:high] # (NX, WindowSize)

        # Mean & Std
        mean_win = np.ma.mean(window_data, axis=1)
        std_win = np.ma.std(window_data, axis=1)

        # Sigma Clip
        clip_mask = np.abs(window_data - mean_win[:, np.newaxis]) > clip * std_win[:, np.newaxis]

        temp_win = np.ma.array(window_data, mask=window_data.mask | clip_mask)
        final_mean = np.ma.mean(temp_win, axis=1).filled(np.nan)

        data[:, f] = final_mean
        if f == nf // 2:
             ypix[:] = final_mean

def cmfff_extrap(data):
    """
    Linearly extrapolate ends.
    """
    nx, nf = data.shape
    nf_fit = 5
    x_idx = np.arange(nx)

    for f in range(nf):
        spec = data[:, f]
        valid = np.isfinite(spec)
        valid_idx = np.where(valid)[0]

        if len(valid_idx) == 0:
            continue

        first = valid_idx[0]
        last = valid_idx[-1]

        if first > 0:
            fit_idx = valid_idx[:nf_fit]
            if len(fit_idx) >= 2:
                p = np.polyfit(fit_idx, spec[fit_idx], 1)
                data[:first, f] = np.polyval(p, x_idx[:first])

        if last < nx - 1:
            fit_idx = valid_idx[-nf_fit:]
            if len(fit_idx) >= 2:
                p = np.polyfit(fit_idx, spec[fit_idx], 1)
                data[last+1:, f] = np.polyval(p, x_idx[last+1:])

def cmfff_norm(red_data, red_var, aff_data, trunc_flat, start_pix, end_pix):
    """
    Normalize RED data by AFF data.
    """
    nx, nf = red_data.shape

    # 1. Divide by AFF
    valid = np.isfinite(red_data) & np.isfinite(aff_data) & (aff_data != 0)

    # Create output arrays
    # red_data is in-place
    red_data[valid] /= aff_data[valid]
    red_var[valid] /= (aff_data[valid]**2)

    red_data[~valid] = np.nan
    red_var[~valid] = np.nan

    # 2. Normalize by Median
    for f in range(nf):
        spec = red_data[:, f]

        mask = np.ones(nx, dtype=bool)
        if trunc_flat:
             mask[:start_pix-1] = False
             mask[end_pix:] = False

        valid_mask = np.isfinite(spec) & mask
        if np.sum(valid_mask) > 0:
            median = np.median(spec[valid_mask])
            if median != 0:
                red_data[:, f] /= median
                red_var[:, f] /= (median**2)
            else:
                red_data[:, f] = np.nan
        else:
            red_data[:, f] = np.nan

def bs_smooth_redflat(filename, args):
    """
    B-Spline Smoothing.
    """
    npars = args.get('BSSNPARS', 16)
    nsig = args.get('BSSNSIGM', 6.0)

    with ImageFile(filename, mode='UPDATE') as f:
        data = f.read_image_data().T
        var = f.read_variance_data().T
        nx, nf = data.shape

        for fidx in range(nf):
            spec = data[:, fidx]
            v = var[:, fidx]

            valid = np.isfinite(spec) & np.isfinite(v) & (v > 0)
            if np.sum(valid) < 2 * npars:
                data[:, fidx] = np.nan
                continue

            x = np.arange(nx)[valid]
            y = spec[valid]
            w = 1.0 / np.sqrt(v[valid])

            # Use simple LSQ spline with sigma clipping
            mask = np.ones(len(y), dtype=bool)

            for i in range(2):
                if np.sum(mask) < npars + 2:
                    break

                x_fit = x[mask]
                y_fit = y[mask]
                w_fit = w[mask]

                # Internal knots (uniform)
                if len(x_fit) > 0:
                    t = np.linspace(x_fit[0], x_fit[-1], npars - 2)
                else:
                    break

                try:
                    spl = LSQUnivariateSpline(x_fit, y_fit, t[1:-1], w=w_fit)
                    y_model = spl(x)

                    res = np.abs(y - y_model)
                    sigma = np.std(res[mask])
                    if sigma == 0:
                        break

                    new_mask = res < nsig * sigma
                    if np.all(new_mask == mask):
                        break
                    mask = new_mask
                except Exception:
                    break

            if 'spl' in locals():
                data[:, fidx] = spl(np.arange(nx))
            else:
                data[:, fidx] = np.nan

        f.write_image_data(data.T)
