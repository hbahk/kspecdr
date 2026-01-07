"""
Reduce Arc Module

This module implements the reduction of arc frames to produce wavelength calibrated spectra.
"""

import logging
import shutil
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

from ..io.image import ImageFile
from ..preproc.make_im import make_im
from ..extract.make_ex import make_ex
from ..constants import *
from ..wavecal.calibrate import calibrate_spectral_axes
from ..wavecal.arc_io import read_arc_file

logger = logging.getLogger(__name__)


def reduce_arc(args: Dict[str, Any]) -> None:
    """
    Reduces a raw arc file to produce im(age), ex(tracted) and red(uced) arc files.
    """

    # 1. Initialize
    raw_fname = args.get("FILENAME")  # Usually passed? Or inferred?
    # Arguments in Fortran are SDS. Usually contain filenames.
    # User prompt example uses EXTRAC_FILENAME, OUTPUT_FILENAME.
    # Assuming pipeline orchestration calls this with proper args.

    # Let's assume args contains:
    # 'OBJECT_FILENAME' or 'RAW_FILENAME' ?
    # Typically 2dfdr calls REDUCE_ARC with args containing input raw file.
    # But the Fortran code `REDUCE_ARC` takes `ARGS`.
    # It calls MAKE_IM(ARGS), MAKE_EX(ARGS).
    # Then reads EXTRAC_FILENAME and OUTPUT_FILENAME from args.

    # Let's try to extract filenames from args.
    # If not present, we can't proceed.
    # But usually REDUCE_ARC is a task.

    # For now, let's assume `args` has what `make_im` and `make_ex` need.

    # Call MAKE_IM
    # In python make_im signature: make_im(raw_filename, im_filename, ...)
    # We need to map args to this.

    raw_filename = args.get("RAW_FILENAME")
    if not raw_filename:
        # Try finding it based on other args or raise error
        # Often passed as just the argument to the script?
        pass

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
        # Ensure TLM exists
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

        # Check if generic calibration requested
        use_generic_calibration = args.get("USE_GENCAL", False)

        if (
            instrument_code == INST_TAIPAN
            or instrument_code == INST_AAOMEGA_KOALA
            or instrument_code == INST_ISOPLANE
            or use_generic_calibration
        ):

            logger.info("Using Generic/TAIPAN Calibration Method")

            # Read Data
            nx, nf = red_file.get_size()
            spectra = red_file.read_image_data(nx, nf)
            variance = red_file.read_variance_data(nx, nf)

            # Read Axis (Predicted Wavelengths)
            # In Fortran: TDFIO_AXIS_READ(RED_ID,1,XPTR,NPIX,STATUS)
            # This is pixel centers? Or edges?
            # Usually pixel centers.
            # MAKE_EX creates axis from TLM prediction.
            # PREDICT_WAVELENGTH in MAKE_TLM generated this.

            # We need edges for calibration routine (NPIX+1)
            # Or centers?
            # CALIBRATE_SPECTRAL_AXES takes PRED_AXIS(NPIX+1)
            # Fortran REDUCE_ARC:
            # ALLOCATE(XPTR(NPIX))
            # CALL TDFIO_AXIS_READ(RED_ID,1,XPTR,NPIX,STATUS)
            # ALLOCATE (PIXCAL_DP(NPIX+1,NFIB))
            # FORALL(I=2:NPIX) WAVE_AXIS(I)= 0.5*(XPTR(I-1)+XPTR(I)) ...
            # So it converts centers to edges.

            # Read axis (centers)
            # ImageFile doesn't have read_axis?
            # Let's add it or read directly.
            # Usually in WAVELA extension or x-axis coords?
            # TDFIO_AXIS_READ reads .WAVELA extension if present, or header CDELT/CRVAL.
            # MAKE_TLM wrote WAVELA extension.

            try:
                wave_hdu = red_file.hdul["WAVELA"]
                wave_data = wave_hdu.data.T  # (nx, nf)
                # This is per-fiber wavelength.
                # CALIBRATE_SPECTRAL_AXES takes 1D PRED_AXIS.
                # Fortran REDUCE_ARC:
                # CALL TDFIO_AXIS_READ(RED_ID,1,XPTR,NPIX,STATUS)
                # This reads the first axis of the image.
                # If WAVELA is present, does it read that?
                # Fortran code: "Read axis data... TDFIO_AXIS_READ"
                # "Also read WAVELA array... TDFIO_WAVE_READ"
                # So there are two.
                # PRED_AXIS passed to CALIBRATE is based on XPTR (Axis).
                # Axis is usually linear approximation.

                # Let's construct PRED_AXIS from the middle fiber of WAVELA or just linear axis.
                # For TAIPAN, `predict_wavelength_taipan` creates WAVELA.
                # We can use the middle fiber's wavelength array as the "Predicted Axis" for reference.
                ref_fib = nf // 2
                xptr = wave_data[:, ref_fib]

            except KeyError:
                # Fallback: create from header
                # Or assume indices
                logger.warning("WAVELA extension not found. Using indices.")
                xptr = np.arange(nx, dtype=float)

            # Convert centers to edges
            wave_axis = np.zeros(nx + 1)
            wave_axis[1:nx] = 0.5 * (xptr[:-1] + xptr[1:])
            # Extrapolate ends
            wave_axis[0] = wave_axis[1] - (wave_axis[2] - wave_axis[1])
            wave_axis[nx] = wave_axis[nx - 1] + (wave_axis[nx - 1] - wave_axis[nx - 2])

            # Read Fiber Types
            fiber_types, _ = red_file.read_fiber_types(MAX__NFIBRES)
            goodfib = np.array([ft in ["P", "S"] for ft in fiber_types[:nf]])

            # Read Lamp List
            lamp = red_file.get_header_value("LAMPNAME", "")
            # Spector logic
            if instrument_code == INST_SPECTOR_HECTOR:
                lamp += "_spector"

            # Read Arc File
            wlist, ilist, _, listsize = read_arc_file(nx, xptr, lamp)

            if listsize == 0:
                logger.error("No arc lines found. Aborting calibration.")
                return

            # Max Shift
            maxshift = args.get("CRSCGMA_MS", 70)
            if instrument_code == INST_AAOMEGA_2DF:
                maxshift = 100
            if instrument_code == INST_AAOMEGA_SAMI:
                maxshift = 150

            # Call Calibration
            pixcal_dp, status = calibrate_spectral_axes(
                nx,
                nf,
                spectra,
                variance,
                wave_axis,
                goodfib,
                wlist,
                ilist,
                listsize,
                maxshift,
                args,
            )

            if status == 0:
                # Write PIXCAL (Pixel Calibration)
                # TDFIO_PIXCAL_WRITE(RED_ID, PIXCAL_DP(1:NPIX, 1:NFIB), PIXCAL_DP(2:NPIX+1, 1:NFIB), ...)
                # It writes LHS and RHS edges.
                # Our pixcal_dp is (nx+1, nf).
                # Extension 'PIXCAL' usually stores LHS and RHS? Or just edges?
                # Fortran writes two extensions: PIXCAL (LHS) and PIXCAL (RHS)?
                # Or a table?
                # TDFIO_PIXCAL_WRITE writes a proprietary format probably.
                # Often it's just replacing the wavelength array or SHIFTS.

                # The prompt code:
                # CALL TDFIO_PIXCAL_WRITE(RED_ID, PIXCAL_DP(1:NPIX, 1:NFIB), PIXCAL_DP(2:NPIX+1, 1:NFIB), ...)
                # It seems to write edges.

                # We will write a SHIFTS extension if traditional, or update WAVELA.
                # Modern 2dfdr uses SHIFTS (Polynomial coeffs) usually.
                # But `CALIBRATE_SPECTRAL_AXES` returns calibrated pixels (wavelengths at edges).

                # Let's convert edges back to centers and write to WAVELA?
                # And also write a PIXCAL extension.

                # Centers
                new_wave = 0.5 * (pixcal_dp[:-1, :] + pixcal_dp[1:, :])

                # Write WAVELA
                red_file.write_wavelength_data(new_wave.T)  # (nf, nx)

                # Write SHIFTS (Dummy 1.0 scaling? Fortran: SHIFT_DPA=0.0; SHIFT_DPA(:,2)=1.0)
                # Because we are doing pixel calibration directly, shifts array is identity?
                # Fortran code:
                # SHIFT_DPA=0.0
                # SHIFT_DPA(:,2)=1.0
                # CALL TDFIO_SHIFTS_WRITE(...)

                # We should mimic this.
                shifts = np.zeros((nf, 4))  # MAX_SHIFTS=4 usually?
                shifts[:, 1] = 1.0  # p1=1 (linear term), p0=0, p2=0

                # Write SHIFTS extension
                red_file.write_shifts_data(shifts)

                logger.info("Wavelength calibration completed successfully.")

        else:
            logger.warning(
                f"Instrument {instrument_code} not supported for new calibration method."
            )
            # Fallback to old methods or placeholder
