"""
Reduce Arc Module

This module implements the reduction of arc frames to produce wavelength calibrated spectra.
"""

import sys
import logging
import shutil
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
from astropy.table import Table

from ..io.image import ImageFile
from ..preproc.make_im import make_im
from ..extract.make_ex import make_ex
from ..constants import *
from ..wavecal.calibrate import (
    calibrate_spectral_axes,
    extract_template_spectrum,
    find_arc_line_matches,
    fit_calibration_model,
    apply_calibration_model,
    find_reference_fiber,
    analyse_arc_signal
)
from ..wavecal.arc_io import read_arc_file

logger = logging.getLogger(__name__)


def reduce_arc(args: Dict[str, Any], get_diagnostic: Optional[bool] = False, diagnostic_dir: Optional[Path] = None) -> None:
    """
    Reduces a raw arc file to produce im(age), ex(tracted) and red(uced) arc files.
    """

    # 1. Initialize
    raw_filename = args.get("RAW_FILENAME")
    im_filename = args.get("IMAGE_FILENAME")
    ex_filename = args.get("EXTRAC_FILENAME")
    red_filename = args.get("OUTPUT_FILENAME")
    tlm_filename = args.get("TLMAP_FILENAME")

    if not raw_filename:
        logger.warning(
            "RAW_FILENAME not provided in args. Assuming previous steps completed or files exist."
        )

    # 1. Make IM
    if raw_filename and not (im_filename and Path(im_filename).exists()):
        im_filename = make_im(raw_filename, im_filename=im_filename, verbose=True)
        args["IMAGE_FILENAME"] = im_filename

    # 2. Make EX
    if im_filename and not (ex_filename and Path(ex_filename).exists()):
        if not tlm_filename:
            tlm_filename = im_filename.replace("_im.fits", "_tlm.fits")
            args["TLMAP_FILENAME"] = tlm_filename

        make_ex(args)
        if not ex_filename:
            ex_filename = im_filename.replace("_im.fits", "_ex.fits")
            args["EXTRAC_FILENAME"] = ex_filename

    # 3. Create RED file (Copy EX)
    if not red_filename:
        red_filename = ex_filename.replace("_ex.fits", "_red.fits")
        args["OUTPUT_FILENAME"] = red_filename

    shutil.copy2(ex_filename, red_filename)

    # 4. Wavelength Calibration
    with ImageFile(red_filename, mode="UPDATE") as red_file:
        instrument_code = red_file.get_instrument_code()
        use_generic_calibration = args.get("USE_GENCAL", False)

        if (
            instrument_code == INST_TAIPAN
            or instrument_code == INST_AAOMEGA_KOALA
            or instrument_code == INST_ISOPLANE
            or use_generic_calibration
        ):
            logger.info("Using Generic/TAIPAN Calibration Method")

            nx, nf = red_file.get_size()
            spectra = red_file.read_image_data().T
            variance = red_file.read_variance_data().T

            try:
                wave_hdu = red_file.hdul["WAVELA"]
                wave_data = wave_hdu.data.T
                ref_fib = nf // 2
                xptr = wave_data[:, ref_fib]
            except KeyError:
                logger.warning("WAVELA extension not found. Using indices.")
                xptr = np.arange(nx, dtype=float)

            wave_axis = np.zeros(nx + 1)
            wave_axis[1:nx] = 0.5 * (xptr[:-1] + xptr[1:])
            wave_axis[0] = wave_axis[1] - (wave_axis[2] - wave_axis[1])
            wave_axis[nx] = wave_axis[nx - 1] + (wave_axis[nx - 1] - wave_axis[nx - 2])

            fiber_types, _ = red_file.read_fiber_types(MAX__NFIBRES)
            goodfib = np.array([ft in ["P", "S"] for ft in fiber_types[:nf]])

            lamp = args.get("LAMPNAME", "")
            if not lamp:
                lamp = red_file.get_header_value("LAMPNAME", "")
                if instrument_code == INST_SPECTOR_HECTOR:
                    lamp += "_spector"

            arc_dir = args.get("ARCDIR", None)
            if not arc_dir:
                if raw_filename:
                    arc_dir = Path(raw_filename).parent
                else:
                    arc_dir = Path(".").resolve()

            wlist, ilist, _, listsize = read_arc_file(nx, xptr, lamp, arc_dir=arc_dir)

            if listsize == 0:
                logger.error("No arc lines found. Aborting calibration.")
                return

            maxshift = args.get("CRSCGMA_MS", 70)
            if instrument_code == INST_AAOMEGA_2DF:
                maxshift = 100
            if instrument_code == INST_AAOMEGA_SAMI:
                maxshift = 150

            pixcal_dp, status = calibrate_spectral_axes(
                nx, nf, spectra, variance, wave_axis, goodfib, wlist, ilist, listsize, maxshift,
                diagnostic=get_diagnostic, diagnostic_dir=diagnostic_dir,
            )

            if status == 0:
                new_wave = 0.5 * (pixcal_dp[:-1, :] + pixcal_dp[1:, :])
                red_file.write_wave_data(new_wave.T)
                shifts = np.zeros((nf, 4))
                shifts[:, 1] = 1.0
                red_file.write_shifts_data(shifts)
                logger.info("Wavelength calibration completed successfully.")

        else:
            logger.warning(
                f"Instrument {instrument_code} not supported for new calibration method."
            )

def reduce_arcs(args_list: List[Dict[str, Any]], get_diagnostic: Optional[bool] = False, diagnostic_dir: Optional[Path] = None) -> None:
    """
    Reduces multiple arc frames together by creating a combined landmark register
    and fitting a global wavelength solution.

    1. Preprocesses all frames (make_im, make_ex).
    2. Identifies a common reference fiber across all frames.
    3. Extracts landmarks from each frame using the common reference.
    4. Merges landmarks and templates to form a master dataset.
    5. Fits a global wavelength solution.
    6. Applies the global solution to all frames using the combined landmarks.
    """

    logger.info(f"Starting multi-arc reduction for {len(args_list)} frames.")

    # Container for frame-specific data
    frames_metadata = []

    # 1. Preprocessing and Common Reference Fiber Identification
    all_goodfibs = []

    for args in args_list:
        raw_filename = args.get("RAW_FILENAME")
        if not raw_filename:
            logger.error("RAW_FILENAME missing in reduce_arcs args.")
            continue

        # Ensure Make IM/EX
        im_filename = args.get("IMAGE_FILENAME")
        ex_filename = args.get("EXTRAC_FILENAME")
        tlm_filename = args.get("TLMAP_FILENAME")

        if not (im_filename and Path(im_filename).exists()):
             im_filename = make_im(raw_filename, im_filename=im_filename, verbose=True)
             args["IMAGE_FILENAME"] = im_filename

        if not (ex_filename and Path(ex_filename).exists()):
            if not tlm_filename:
                tlm_filename = im_filename.replace("_im.fits", "_tlm.fits")
                args["TLMAP_FILENAME"] = tlm_filename
            make_ex(args)
            if not ex_filename:
                ex_filename = im_filename.replace("_im.fits", "_ex.fits")
                args["EXTRAC_FILENAME"] = ex_filename

        # Store metadata
        meta = {
            "args": args,
            "raw_filename": raw_filename,
            "ex_filename": ex_filename,
        }

        # Read goodfibs for reference fiber logic
        with ImageFile(ex_filename, mode="READ") as ex_file:
            nx, nf = ex_file.get_size()
            fiber_types, _ = ex_file.read_fiber_types(MAX__NFIBRES)
            goodfib = np.array([ft in ["P", "S"] for ft in fiber_types[:nf]])

            meta["nx"] = nx
            meta["nf"] = nf
            meta["goodfib"] = goodfib
            meta["instrument_code"] = ex_file.get_instrument_code()

            # Lamp info
            lamp = args.get("LAMPNAME", "")
            if not lamp:
                lamp = ex_file.get_header_value("LAMPNAME", "")
            if meta["instrument_code"] == INST_SPECTOR_HECTOR:
                 lamp += "_spector"
            meta["lamp"] = lamp

            # Wavelength Prediction (needed for line ID)
            try:
                wave_hdu = ex_file.hdul["WAVELA"]
                wave_data = wave_hdu.data.T
                # We need the prediction for the reference fiber, but we don't know it yet.
                # Store full wave_data or read later? Reading later is safer but slower.
                # Let's store just the data we need or keep file closed.
                # Actually, we need to know the initial wavelength grid (cen_axis) for the template.
                # We can assume all frames have similar grating settings.
                # Let's just store the prediction for the middle fiber for now as a fallback?
                # No, we'll read it again in Pass 2 once we have ref_fib.
            except KeyError:
                pass

            all_goodfibs.append(goodfib)
            frames_metadata.append(meta)

    if not frames_metadata:
        logger.error("No valid frames to process.")
        return

    # Determine Common Reference Fiber
    # Intersection of goodfibs
    # Note: different frames might have slightly different NF? Assuming same setup.
    nf_min = min(f["nf"] for f in frames_metadata)
    common_goodfib = np.ones(nf_min, dtype=bool)

    for f in frames_metadata:
        common_goodfib &= f["goodfib"][:nf_min]

    master_ref_fib = find_reference_fiber(nf_min, common_goodfib)
    if master_ref_fib == -1:
        logger.error("No common reference fiber found across all frames.")
        return

    logger.info(f"Selected Master Reference Fiber: {master_ref_fib}")


    # 2. Extract Landmarks & Accumulate Data
    collected_lmrs = []
    collected_templates = []
    collected_muv = []
    collected_av = []
    collected_lamp_indices = []  # Track which lamp each muv belongs to
    lamp_list = []  # List of unique lamp names
    sum_sigma = 0.0

    # Store cen_axis from the first frame to use as the master axis
    master_cen_axis = None
    master_npix = frames_metadata[0]["nx"]

    for frame in frames_metadata:
        ex_filename = frame["ex_filename"]
        args = frame["args"]
        lamp = frame["lamp"]
        
        # Add to lamp_list (remove duplicates)
        if lamp not in lamp_list:
            lamp_list.append(lamp)
        lamp_idx = lamp_list.index(lamp)

        with ImageFile(ex_filename, mode="READ") as ex_file:
            nx, nf = ex_file.get_size()
            spectra = ex_file.read_image_data().T

            # Read prediction
            try:
                wave_hdu = ex_file.hdul["WAVELA"]
                wave_data = wave_hdu.data.T
                xptr = wave_data[:, master_ref_fib]
            except KeyError:
                xptr = np.arange(nx, dtype=float)

            # Axis setup
            wave_axis = np.zeros(nx + 1)
            wave_axis[1:nx] = 0.5 * (xptr[:-1] + xptr[1:])
            wave_axis[0] = wave_axis[1] - (wave_axis[2] - wave_axis[1])
            wave_axis[nx] = wave_axis[nx - 1] + (wave_axis[nx - 1] - wave_axis[nx - 2])
            cen_axis = 0.5 * (wave_axis[:-1] + wave_axis[1:])

            if master_cen_axis is None:
                master_cen_axis = cen_axis
                master_npix = nx

            # Read Arc Lines for this lamp
            arc_dir = args.get("ARCDIR", None)
            if not arc_dir:
                arc_dir = Path(frame["raw_filename"]).parent

            wlist, ilist, _, listsize = read_arc_file(nx, xptr, lamp, arc_dir=arc_dir)
            if listsize > 0:
                # Filter by range
                min_wave = min(wave_axis)
                max_wave = max(wave_axis)
                mask_tab = (wlist >= min_wave) & (wlist <= max_wave)
                collected_muv.append(wlist[mask_tab])
                collected_av.append(ilist[mask_tab])
                # Track which lamp each muv belongs to
                collected_lamp_indices.append(np.full(np.sum(mask_tab), lamp_idx))

            # Extract Template (using master_ref_fib)
            # This generates lmr relative to the master reference
            template_spectra, template_mask, lmr, sigma_inpix, nlm = extract_template_spectrum(
                spectra, nf, nx, frame["goodfib"], master_ref_fib, cen_axis, diagnostic=get_diagnostic, diagnostic_dir=diagnostic_dir,
            )

            # Accumulate
            collected_lmrs.append(lmr) # (NF, NLM)
            collected_templates.append(template_spectra)
            sum_sigma += sigma_inpix

    # 3. Combine Data
    # Combine LMRs horizontally: (NF, NLM1) + (NF, NLM2) -> (NF, NLM_Total)
    if not collected_lmrs:
        logger.error("No LMRs extracted.")
        return

    master_lmr = np.hstack(collected_lmrs)
    master_nlm = master_lmr.shape[1]
    logger.info(f"Combined LMR shape: {master_lmr.shape}, Total Landmarks: {master_nlm}")

    # Combine Templates (Simple Sum)
    # Assumes all templates are on the same pixel grid (aligned to master_ref_fib)
    master_template = np.sum(collected_templates, axis=0)
    master_template_mask = np.zeros_like(master_template, dtype=bool) # Re-calculate mask?
    # Ideally, mask is where count is low. For now, assume good coverage.

    # Combine Lamp Lines
    if collected_muv:
        all_muv = np.concatenate(collected_muv)
        all_av = np.concatenate(collected_av)
        all_lamp_indices = np.concatenate(collected_lamp_indices)  # Lamp index for each muv

        # Sort & Unique
        idx = np.argsort(all_muv)
        all_muv = all_muv[idx]
        all_av = all_av[idx]
        all_lamp_indices = all_lamp_indices[idx]  # Sorted lamp indices

        # Remove duplicates (same line from same lamp in different frames, or overlapping lamps)
        # We use a small tolerance or just unique values?
        unique_mu, unique_idx = np.unique(all_muv, return_index=True)
        master_muv = all_muv[unique_idx]
        master_av = all_av[unique_idx]
        master_lamp_indices = all_lamp_indices[unique_idx]  # Lamp indices for unique muv
    else:
        logger.error("No arc lines found in any frame.")
        return

    # Calculate blend mask for the master list
    # Use average sigma
    avg_sigma_inpix = sum_sigma / len(frames_metadata)
    disp = np.abs(master_cen_axis[-1] - master_cen_axis[0]) / (master_npix - 1)
    arcline_sigma = avg_sigma_inpix * disp

    m = len(master_muv)
    master_mask = np.zeros(m, dtype=bool)
    diffs = np.diff(master_muv)
    blend_indices = np.where(diffs < 3.0 * arcline_sigma)[0]
    for idx in blend_indices:
        if master_av[idx] < 10.0 * master_av[idx + 1] and master_av[idx + 1] < 10.0 * master_av[idx]:
            master_mask[idx] = True; master_mask[idx + 1] = True
        elif master_av[idx] >= 10.0 * master_av[idx + 1]:
            master_mask[idx + 1] = True
        else:
            master_mask[idx] = True

    # 4. Identify Lines & Global Fit
    maxshift = frames_metadata[0]["args"].get("CRSCGMA_MS", 70) # Use first frame's setting

    x_pts, y_pts, _ = find_arc_line_matches(
        master_template, master_template_mask, avg_sigma_inpix, master_cen_axis, master_npix,
        master_muv, master_av, master_mask, maxshift, diagnostic=get_diagnostic, diagnostic_dir=diagnostic_dir,
    )

    logger.info(f"Total points found on master template: {len(x_pts)}")

    if len(x_pts) < 4:
        logger.error("Not enough points found for global fit.")
        return

    # Find master_muv indices corresponding to each y_pts (wavelength) for lamp mapping
    # y_pts are a subset of master_muv, so find the closest index
    matched_indices = []
    for y in y_pts:
        idx = np.argmin(np.abs(master_muv - y))
        matched_indices.append(idx)
    
    # Use matched_indices to find the lamp for each x_pts
    lamps = [lamp_list[master_lamp_indices[i]] for i in matched_indices]

    coeffs, residuals, outliers = fit_calibration_model(
        np.array(x_pts), np.array(y_pts), poly_order=3
    )

    if len(residuals) > 0:
         rms_res = np.sqrt(np.mean((residuals**2)[~outliers]))
         logger.info(f"Global Fit RMS: {rms_res:.4f}")

    # 5. Apply Global Solution to All Frames
    # Using Master LMR and Master Ref Fib

    for frame in frames_metadata:
        args = frame["args"]
        ex_filename = frame["ex_filename"]
        red_filename = args.get("OUTPUT_FILENAME")
        if not red_filename:
            red_filename = ex_filename.replace("_ex.fits", "_red.fits")
            args["OUTPUT_FILENAME"] = red_filename

        shutil.copy2(ex_filename, red_filename)

        with ImageFile(red_filename, mode="UPDATE") as red_file:
            # Note: synchronise_calibration_last (called by apply) uses lmr to map
            # the global solution (at master_ref_fib) to each individual fiber.
            # By passing master_lmr, we use the combined set of landmarks for this mapping.
            pixcal_dp = apply_calibration_model(
                coeffs, frame["nx"], frame["nf"],
                frame["goodfib"], master_ref_fib,
                master_lmr, master_nlm
            )

            # Write results
            new_wave = 0.5 * (pixcal_dp[:-1, :] + pixcal_dp[1:, :])
            red_file.write_wave_data(new_wave.T)

            shifts = np.zeros((frame["nf"], 4))
            shifts[:, 1] = 1.0
            red_file.write_shifts_data(shifts)

            logger.info(f"Updated {red_filename} with global calibration.")
    
    # Write diagnostic file
    if get_diagnostic:
        if diagnostic_dir:
            if not diagnostic_dir.exists():
                diagnostic_dir.mkdir(parents=True, exist_ok=True)
        
            # identified arc lines in x_pts, y_pts, residuals, outliers, lamps
            diag = Table({
                "x_pts": x_pts, 
                "y_pts": y_pts, 
                "residuals": residuals, 
                "outliers": outliers,
                "lamps": lamps
            })
            diag.write(diagnostic_dir / "identified_arcs.dat", format="ascii.fixed_width_two_line", overwrite=True)
            logger.info(f"Diagnostic file written to {diagnostic_dir / 'identified_arcs.dat'}")
            
            # global fit coefficients
            diag = Table({"coeffs": coeffs})
            diag.write(diagnostic_dir / "global_fit_coefficients.dat", format="ascii.fixed_width_two_line", overwrite=True)
            logger.info(f"Diagnostic file written to {diagnostic_dir / 'global_fit_coefficients.dat'}")
        else:
            return {"x_pts": x_pts, "y_pts": y_pts, "residuals": residuals, "outliers": outliers, "coeffs": coeffs, "lamps": lamps}
        
    logger.info("Multi-arc reduction completed.")
