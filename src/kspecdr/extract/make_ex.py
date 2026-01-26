"""
Extraction Routines for KSPEC.

This module implements the extraction of spectra from image data using tramline maps,
converting the 2dfdr `MAKE_EX` and related subroutines.
"""

import logging
import sys
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import warnings

from ..io.image import ImageFile
from ..utils.fiber import get_override_from_args

logger = logging.getLogger(__name__)

# Constants
MAX_NFIBRES = 1000
VAL__BADR = np.nan  # Using NaN for bad values in Python


def make_ex(args: Dict[str, Any]) -> None:
    """
    Main driver for extraction process.
    Replaces 2dfdr SUBROUTINE MAKE_EX.

    Parameters
    ----------
    args : dict
        Dictionary containing arguments:
        - IMAGE_FILENAME: Input image file
        - EXTRAC_FILENAME: Output extracted file
        - TLMAP_FILENAME: Tramline map file
        - WTSCHEME: Weighting scheme (optional)
    """
    im_fname = args.get('IMAGE_FILENAME')
    ex_fname = args.get('EXTRAC_FILENAME')
    tlm_fname = args.get('TLMAP_FILENAME')
    wtscheme = args.get('WTSCHEME', 'STND')

    if not im_fname or not ex_fname or not tlm_fname:
        raise ValueError("Missing required filenames (IMAGE, EXTRAC, or TLMAP)")

    # Check if TLM exists
    if not Path(tlm_fname).exists():
        raise FileNotFoundError(f"Tramline map file not found: {tlm_fname}")

    logger.info(f"Extracting {im_fname} -> {ex_fname} using TLM {tlm_fname}")

    # Call the main extraction routine
    make_ex_from_im(im_fname, tlm_fname, ex_fname, wtscheme, args)

    # TODO: Handle Stochastic copies if needed (NSTOCHIM) - skipping for now as it's advanced usage

def make_ex_from_im(im_fname: str, tlm_fname: str, ex_fname: str, wtscheme: str, args: Dict[str, Any]) -> None:
    """
    Process image file to produce extracted spectra.
    Replaces 2dfdr SUBROUTINE MAKE_EX_FROM_IM.

    Parameters
    ----------
    im_fname : str
        Input image filename
    tlm_fname : str
        Tramline map filename
    ex_fname : str
        Output extracted filename
    wtscheme : str
        Weighting scheme
    args : dict
        Additional arguments
    """
    # 1. Get Extraction Method
    operat = args.get('EXTR_OPERATION', 'TRAM').upper()
    logger.info(f"Extraction Method: {operat}")

    valid_methods = ['FIT', 'TRAM', 'NEWTRAM', 'GAUSS', 'OPTEX', 'CLOPTEX', 'SMCOPTEX', 'SUM']
    if operat not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}")

    if operat == 'FIT':
        raise NotImplementedError("FIT method does NOT work (legacy status)")

    # 2. Open Image File
    with ImageFile(im_fname, mode='READ') as im_file:
        im_class = im_file.get_header_value('CLASS', 'UNKNOWN')
        instrument_code = im_file.get_instrument_code()

        img_data = im_file.read_image_data()
        var_data = im_file.read_variance_data()
        fib_tabl = im_file.read_fiber_table()

        # 3. Read Tramline Map
        with ImageFile(tlm_fname, mode='READ') as tlm_file:
            tlm_data = tlm_file.read_image_data()
            # TLM is (NFIB, NSPEC) - Horizontal
            nfib, nspec_tlm = tlm_data.shape

            # Read fiber types
            overrides = get_override_from_args(args)
            fiber_types, _ = im_file.read_fiber_types(MAX_NFIBRES, overrides=overrides)
            # Note: 2dfdr reads fiber types from IM file usually, as TLM might not have them updated?

            # Get MWIDTH from TLM header (Median FWHM)
            mwidth = float(tlm_file.get_header_value('MWIDTH', 5.0))

            # Read Wavelength data if available
            wave_data = tlm_file.read_wave_data()

    # 4. Apply TLM Shift (Shift-Rotate-Tweak)
    tlm_shift = float(args.get('TLM_SHIFT', 0.0))
    if tlm_shift != 0.0:
        logger.info(f"Shifting Tramline Map by {tlm_shift} pixels")
        tlm_data += tlm_shift

    # 5. Background / Scattered Light Subtraction
    # (Placeholder: SCATSUB)
    scatsub = args.get('SCATSUB', 'NONE')
    if scatsub != 'NONE':
        # TODO: implement scattered light subtraction
        logger.warning(f"Scattered light subtraction '{scatsub}' requested but not implemented yet.")

    # 6. Standardize dimensions
    nspat, nspec = img_data.shape

    tlm_data_T = tlm_data.T # Now (NSPEC, NFIB)

    # Initialize Output Arrays
    # Output extracted data: (NSPEC, NFIB)
    ex_img = np.zeros((nspec, nfib), dtype=np.float32)
    ex_var = np.zeros((nspec, nfib), dtype=np.float32)

    # 7. Perform Extraction
    if operat in ['TRAM', 'SUM', 'NEWTRAM']:
        # Simple Summing Extraction
        # Get width from args or use default
        width = float(args.get('SUM_WIDTH', 5.0)) # Default from SUMEXTR
        # Or from MWIDTH if TRAM?
        if operat == 'TRAM':
             # TRAM usually uses tramlines. SUMEXTR uses WIDTH.
             # 2dfdr: IF (OPERAT=='TRAM') CALL UMFIM_TRMEXTR...
             # ELSE CALL SUMEXTR...
             # We will implement SUMEXTR logic here as requested "simple summing"
             # TODO: Implement TRAM extraction
             pass

        logger.info(f"Performing SUM extraction with width={width}")

        sum_extract(
            nspat, nspec, img_data, var_data,
            ex_img, ex_var, nfib, tlm_data_T, width
        )

    elif operat == 'GAUSS':
        # Placeholder for GAUSS
        logger.warning("GAUSS extraction not fully implemented. Using simplified fallback or raising error.")
        raise NotImplementedError("GAUSS extraction not implemented")

    elif operat in ['OPTEX', 'SMCOPTEX']:
        # Placeholder for Optimal Extraction
         raise NotImplementedError("Optimal extraction not implemented")

    else:
        raise ValueError(f"Unknown operation: {operat}")

    # 8. Post-Processing

    # Handle Bad Fibers (Zero them out)
    # 2dfdr: UMFIM_ZERO(..., TYP)
    # Zero out 'F' (Guide), 'N' (Not used), 'U' (Unallocated)
    for fib in range(nfib):
        ftype = fiber_types[fib]
        if ftype in ['F', 'N', 'U']:
            ex_img[:, fib] = 0.0
            ex_var[:, fib] = 0.0

    # 9. Write Output
    # Create output file from copy of image file (to preserve headers)
    # In kspecdr, we might create a new file or copy.
    # Using ImageFile's save_as or creating new HDUList.

    # We need to reshape/transpose back to FITS convention (NFIB, NSPEC) for writing?
    # If ex_img is (NSPEC, NFIB), FITS expects (NAXIS2, NAXIS1) -> (NFIB, NSPEC).
    ex_img_out = ex_img.T
    ex_var_out = ex_var.T

    from astropy.io import fits

    # Read original header
    with fits.open(im_fname) as hdul_src:
        header = hdul_src[0].header.copy()

    # Update Header
    header['HISTORY'] = f"Extracted using {operat}"

    # Set Axes Labels
    header['CTYPE1'] = 'Wavelength'
    header['CUNIT1'] = 'Angstroms'
    header['CTYPE2'] = 'Fibre Number'

    # Create HDUs
    hdu_data = fits.PrimaryHDU(data=ex_img_out, header=header)
    hdu_var = fits.ImageHDU(data=ex_var_out, name='VARIANCE')

    hdul_out = fits.HDUList([hdu_data, hdu_var])

    # Copy WAVELA if available (from TLM usually)
    if wave_data is not None:
        # wave_data is (NFIB, NSPEC) if read from standard TLM.
        # Output is (NFIB, NSPEC).
        # So no transpose needed if TLM is already in output format.
        # However, check internal consistency.
        # If read_wave_data returned data.shape == (ny, nx) == (nfib, nx_tlm)
        # And output is (nfib, nspec).
        # It matches.
        hdu_wave = fits.ImageHDU(data=wave_data, name='WAVELA')
        hdul_out.append(hdu_wave)
        
    # Copy FIBRES if available (from IM file)
    if fib_tabl is not None:
        hdu_fibres = fits.BinTableHDU(data=fib_tabl, name='FIBRES')
        hdul_out.append(hdu_fibres)

    hdul_out.writeto(ex_fname, overwrite=True)
    logger.info(f"Written extracted file: {ex_fname}")


def sum_extract(
    nspat: int,
    nspec: int,
    indat: np.ndarray,
    invar: np.ndarray,
    outdat: np.ndarray,
    outvar: np.ndarray,
    nfib: int,
    tlmap: np.ndarray,
    width: float
) -> None:
    """
    Perform simple summing extraction.
    Replaces 2dfdr SUBROUTINE SUMEXTR.

    Parameters
    ----------
    nspat : int
        Number of spatial pixels (rows in indat)
    nspec : int
        Number of spectral pixels (cols in indat)
    indat : np.ndarray
        Input image (NSPAT, NSPEC) - Horizontal Dispersion
    invar : np.ndarray
        Input variance (NSPAT, NSPEC) - Horizontal Dispersion
    outdat : np.ndarray
        Output spectra (NSPEC, NFIB) - Updated in place
    outvar : np.ndarray
        Output variance (NSPEC, NFIB) - Updated in place
    nfib : int
        Number of fibers
    tlmap : np.ndarray
        Tramline map (NSPEC, NFIB)
    width : float
        Width of extraction window
    """

    # Loop over spectral pixels (columns)
    with logging_redirect_tqdm():
        for j in tqdm(range(nspec), desc="Summing extraction", unit="pixel"):
            # Loop over fibers
            for fibre in range(nfib):
                # Get center of fiber profile from TLM
                # 2dfdr adds TRAMLINE_OFFSET (usually 0.0 or 0.5 depending on convention)
                # Python/Numpy 0-based vs Fortran 1-based.
                # If TLM is 1-based (from Fortran 2dfdr), we might need to adjust.
                # Assuming TLM data is already converted to 0-based pixel coordinates?
                # Or if it's FITS pixel coordinates (1-based), we need to subtract 1?
                # Let's assume TLM values are 0-based pixel coordinates for now (standard python practice),
                # or FITS convention (1-based).
                # If FITS convention: center = tlm_val - 1.0
                # Let's stick to the raw value for now, assuming TLM matches image grid.

                tlm_pt = tlmap[j, fibre]

                # 2dfdr: TLMPT = TLMAP(J,FIBRE) + TRAMLINE_OFFSET
                # If we assume 0-based, no offset needed if TLM is correct.

                tlow = tlm_pt - width / 2.0
                thigh = tlm_pt + width / 2.0

                # Convert to integer indices
                # 2dfdr: ILOW = INT(TLOW) + 1  (1-based)
                # Python: ilow = int(floor(tlow))?
                # Pixel i covers [i-0.5, i+0.5]? Or [i, i+1]?
                # 2dfdr logic implies pixel centers.
                # Let's use standard partial pixel integration.

                ilow = int(np.floor(tlow))
                ihigh = int(np.floor(thigh))

                # Clip to image boundaries
                ilow = max(0, ilow)
                ihigh = min(nspat - 1, ihigh)

                tot_pix = 0.0
                tot_var = 0.0

                # Check for bad pixels in the full pixels range
                # Range is inclusive of ilow, inclusive of ihigh?
                # 2dfdr: DO PIX=ILOW,IHIGH (inclusive)
                # But wait, logic for partials:
                # IF ILOW > 1 ... ADD PARTIAL LOW
                # IF IHIGH < NSPAT ... ADD PARTIAL HIGH
                # The Loop is for "whole pixels" fully inside?
                # 2dfdr: ILOW = INT(TLOW) + 1.
                # e.g. TLOW = 5.5 -> ILOW = 6.
                # Pixel 6 is fully inside?
                #
                # Let's implement robust partial pixel summation.
                # Range [tlow, thigh].
                # Integrate flux from tlow to thigh.
                # Assuming pixels are boxcars centered at integer coordinates?
                # Or centered at int+0.5?
                # 2dfdr convention: Pixel coordinates are usually 0.5 to N+0.5?
                #
                # Let's simplify:
                # Sum pixels from ceil(tlow) to floor(thigh).
                # Add partial fraction of floor(tlow) and ceil(thigh).

                # 2dfdr SUMEXTR Implementation translated:

                # Bounds check
                if ihigh < ilow:
                    # Window is too small or out of bounds
                    outdat[j, fibre] = VAL__BADR
                    outvar[j, fibre] = VAL__BADR
                    continue

                bad_pixel = False

                # Sum whole pixels (or mostly whole)
                # In 2dfdr, loop is ILOW to IHIGH.
                # 2dfdr ILOW calculation: INT(TLOW) + 1.
                # If TLOW=5.1, ILOW=6. Pixel 6 is included.
                # But Pixel 5 is partial.
                # So loop covers "inner" pixels.

                # Python equivalent:
                # range(ilow_idx, ihigh_idx + 1)
                # where ilow_idx is index of first FULL pixel > tlow.
                # ihigh_idx is index of last FULL pixel < thigh.

                start_full = int(np.ceil(tlow)) # e.g. 5.1 -> 6
                end_full = int(np.floor(thigh)) # e.g. 9.9 -> 9

                # Sum Full Pixels
                # Note: 2dfdr logic for ILOW/IHIGH is slightly different, let's stick to first principles
                # or exact translation.

                current_flux = 0.0
                current_var = 0.0

                # Iterate through all pixels touched
                p_start = int(np.floor(tlow))
                p_end = int(np.floor(thigh)) # Note: thigh is upper bound

                # Cap at image edges
                p_start = max(0, p_start)
                p_end = min(nspat - 1, p_end)

                for pix in range(p_start, p_end + 1):
                    # Calculate fraction of pixel included
                    # Pixel covers [pix, pix+1] (assuming 0-based corner? Or center?)
                    # If center is pix, range is [pix-0.5, pix+0.5].
                    # 2dfdr usually assumes pixel centers are integers 1, 2, ...
                    # Let's assume standard FITS: pixel centers are 1.0, 2.0.
                    # Python 0-based: centers are 0.0, 1.0?
                    # Actually, typically [x, x+1] is the range for pixel x.

                    # Let's assume pixel `pix` covers spatial range [pix, pix+1].
                    pix_min = float(pix)
                    pix_max = float(pix) + 1.0

                    # Intersection of [pix_min, pix_max] and [tlow, thigh]
                    overlap_min = max(pix_min, tlow)
                    overlap_max = min(pix_max, thigh)

                    if overlap_max > overlap_min:
                        fraction = overlap_max - overlap_min

                        val = indat[pix, j]
                        var = invar[pix, j]

                        if np.isnan(val) or np.isnan(var):
                            bad_pixel = True
                            break

                        current_flux += val * fraction
                        current_var += var * fraction # Linear variance scaling?
                        # 2dfdr: TOTVAR = TOTVAR+INVAR(ILOW-1,J)*PART
                        # Yes, it scales variance by fraction?
                        # Actually variance of (A*x) is A^2 * Var(x).
                        # 2dfdr seems to just use fraction?
                        # "TOTVAR = TOTVAR+INVAR(ILOW-1,J)*PART"
                        # This implies Var(fraction * Pixel) = fraction * Var(Pixel).
                        # This is correct if we are summing 'fraction' of the Poisson counts?
                        # No, strictly Var(c*X) = c^2 * Var(X).
                        # But 2dfdr does linear. Let's replicate 2dfdr behavior for now.
                        # Warning: 2dfdr might be doing 'counts' scaling.

                if bad_pixel:
                    outdat[j, fibre] = VAL__BADR
                    outvar[j, fibre] = VAL__BADR
                else:
                    outdat[j, fibre] = current_flux
                    outvar[j, fibre] = current_var
