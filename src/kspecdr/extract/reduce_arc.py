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
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def reduce_arc(args: Dict[str, Any]) -> None:
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
            spectra = red_file.read_image_data(nx, nf).T
            variance = red_file.read_variance_data(nx, nf).T

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

def reduce_arcs(args_list: List[Dict[str, Any]], get_diagnostic: bool = False, diagnostic_dir: Optional[str] = None) -> None:
    """
    Reduces multiple arc frames together.

    1. Preprocesses all frames (make_im, make_ex).
    2. Extracts landmarks from all frames.
    3. Fits a global wavelength solution.
    4. Applies the solution to all frames.
    """

    logger.info(f"Starting multi-arc reduction for {len(args_list)} frames.")

    all_x_pts = []
    all_y_pts = []
    all_lamps = []

    # Store intermediate data to avoid re-reading
    frames_data = []

    # 1. Preprocess and Collect Points
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

        # Read Data
        with ImageFile(ex_filename, mode="READ") as ex_file:
            nx, nf = ex_file.get_size()
            spectra = ex_file.read_image_data(nx, nf).T

            fiber_types, _ = ex_file.read_fiber_types(MAX__NFIBRES)
            goodfib = np.array([ft in ["P", "S"] for ft in fiber_types[:nf]])

            # Find Reference Fiber
            ref_fib = find_reference_fiber(nf, goodfib)
            if ref_fib == -1:
                logger.error(f"No good fibers in {ex_filename}")
                continue

            # Read Wavelength/Axis info
            try:
                wave_hdu = ex_file.hdul["WAVELA"]
                wave_data = wave_hdu.data.T
                xptr = wave_data[:, ref_fib] # Use ref fiber's prediction
            except KeyError:
                xptr = np.arange(nx, dtype=float)

            # Edges
            wave_axis = np.zeros(nx + 1)
            wave_axis[1:nx] = 0.5 * (xptr[:-1] + xptr[1:])
            wave_axis[0] = wave_axis[1] - (wave_axis[2] - wave_axis[1])
            wave_axis[nx] = wave_axis[nx - 1] + (wave_axis[nx - 1] - wave_axis[nx - 2])
            cen_axis = 0.5 * (wave_axis[:-1] + wave_axis[1:])

            # Get Lamp info
            lamp = args.get("LAMPNAME", "")
            if not lamp:
                lamp = ex_file.get_header_value("LAMPNAME", "")

            instrument_code = ex_file.get_instrument_code()
            if instrument_code == INST_SPECTOR_HECTOR:
                 lamp += "_spector"

            arc_dir = args.get("ARCDIR", None)
            if not arc_dir:
                arc_dir = Path(raw_filename).parent

            wlist, ilist, _, listsize = read_arc_file(nx, xptr, lamp, arc_dir=arc_dir)
            if listsize == 0:
                logger.warning(f"No arc lines for {raw_filename} (Lamp: {lamp}). Skipping.")
                continue

            # Process Arc List (Filter by range)
            min_wave = min(wave_axis)
            max_wave = max(wave_axis)
            mask_tab = (wlist >= min_wave) & (wlist <= max_wave)
            muv = wlist[mask_tab]
            av = ilist[mask_tab]

            # Sort & Unique
            idx = np.argsort(muv)
            muv = muv[idx]
            av = av[idx]
            unique_mu, unique_idx = np.unique(muv, return_index=True)
            muv = muv[unique_idx]
            av = av[unique_idx]

            # Mask blends logic
            ref_signal = spectra[:, ref_fib]
            ref_signal = np.nan_to_num(ref_signal)
            _, _, sigma_inpix, _, _ = analyse_arc_signal(ref_signal)

            disp = np.abs(cen_axis[-1] - cen_axis[0]) / (nx - 1)
            arcline_sigma = sigma_inpix * disp

            m = len(muv)
            mask = np.zeros(m, dtype=bool)
            diffs = np.diff(muv)
            blend_indices = np.where(diffs < 3.0 * arcline_sigma)[0]
            for idx in blend_indices:
                if av[idx] < 10.0 * av[idx + 1] and av[idx + 1] < 10.0 * av[idx]:
                    mask[idx] = True; mask[idx + 1] = True
                elif av[idx] >= 10.0 * av[idx + 1]:
                    mask[idx + 1] = True
                else:
                    mask[idx] = True

            # Extract Template
            template_spectra, template_mask, lmr, sigma_inpix, nlm = extract_template_spectrum(
                spectra, nf, nx, goodfib, ref_fib, cen_axis, diagnostic=False
            )

            # Identify Lines
            maxshift = args.get("CRSCGMA_MS", 70)
            x_pts, y_pts, _ = find_arc_line_matches(
                template_spectra, template_mask, sigma_inpix, cen_axis, nx,
                muv, av, mask, maxshift, diagnostic=False
            )

            logger.info(f"Found {len(x_pts)} points in {raw_filename}")

            if len(x_pts) > 0:
                all_x_pts.extend(x_pts)
                all_y_pts.extend(y_pts)
                all_lamps.extend([lamp] * len(x_pts))

            # Store data needed for application
            frames_data.append({
                "args": args,
                "ex_filename": ex_filename,
                "nx": nx, "nf": nf,
                "goodfib": goodfib,
                "ref_fib": ref_fib,
                "lmr": lmr,
                "nlm": nlm
            })

    # 2. Global Fit
    logger.info(f"Total points collected: {len(all_x_pts)}")
    if len(all_x_pts) < 4:
        logger.error("Not enough points collected across all frames.")
        return

    coeffs, residuals, outliers = fit_calibration_model(
        np.array(all_x_pts), np.array(all_y_pts), poly_order=3
    )

    if len(residuals) > 0:
         rms_res = np.sqrt(np.mean(((residuals - np.median(residuals))**2)[~outliers]))
         logger.info(f"Global Fit RMS: {rms_res:.4f}")

    # 3. Apply to All Frames
    for frame in frames_data:
        args = frame["args"]
        ex_filename = frame["ex_filename"]
        red_filename = args.get("OUTPUT_FILENAME")
        if not red_filename:
            red_filename = ex_filename.replace("_ex.fits", "_red.fits")
            args["OUTPUT_FILENAME"] = red_filename

        shutil.copy2(ex_filename, red_filename)

        with ImageFile(red_filename, mode="UPDATE") as red_file:
            pixcal_dp = apply_calibration_model(
                coeffs, frame["nx"], frame["nf"],
                frame["goodfib"], frame["ref_fib"],
                frame["lmr"], frame["nlm"]
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
            if not Path(diagnostic_dir).exists():
                Path(diagnostic_dir).mkdir(parents=True, exist_ok=True)
        
            # identified arc lines in x_pts, y_pts, residuals, outliers, lamps as a table
            diag = Table({"x_pts": all_x_pts, "y_pts": all_y_pts, "residuals": residuals, "outliers": outliers, "lamps": all_lamps})
            diag.write(Path(diagnostic_dir) / "identified_arcs.dat", format="ascii.fixed_width_two_line", overwrite=True)
            logger.info(f"Diagnostic file written to {Path(diagnostic_dir) / 'identified_arcs.dat'}")
            
            # global fit coefficients
            diag = Table({"coeffs": coeffs})
            diag.write(Path(diagnostic_dir) / "global_fit_coefficients.dat", format="ascii.fixed_width_two_line", overwrite=True)
            logger.info(f"Diagnostic file written to {Path(diagnostic_dir) / 'global_fit_coefficients.dat'}")
        else:
            return {"x_pts": all_x_pts, "y_pts": all_y_pts, "residuals": residuals, "outliers": outliers, "lamps": all_lamps, "coeffs": coeffs}
        
    logger.info("Multi-arc reduction completed.")
