"""
Header conversion utilities for KSPEC data.

This module provides functions to convert raw headers from various instruments
(e.g., PIXIS/Isoplane backup CCD) to the standard format required by kspecdr.
"""

import logging
from astropy.io import fits
from astropy.io.fits.verify import VerifyError
from astropy.time import Time
from astropy.table import Table
import numpy as np
import re

logger = logging.getLogger(__name__)


def add_fiber_table(hdul: fits.HDUList, n_fibers: int, fiber_table: Table = None) -> None:
    """
    Add a dummy fiber table to the HDUList.

    Parameters
    ----------
    hdul : fits.HDUList
        The FITS HDUList to modify in place.
    n_fibers : int
        Number of fibers to include in the table.
    fiber_table : Table, optional
        Fiber table to add to the HDUList. If None, a dummy fiber table will be created.
    """
    logger.info(f"Adding fiber table with {n_fibers} fibers")

    if fiber_table is None:
        # Create columns
        # Standard 2dfdr fiber table columns:
        # TYPE (1A), NAME (80A), MAGNITUDE (E), RA (D), DEC (D), X (E), Y (E), etc.
        # For dummy/isoplane, we primarily need TYPE='P' (Program) or 'S' (Sky)
        # to ensure extraction works.

        # Create arrays
        # All Program fibers for now
        types = np.array(["P"] * n_fibers)
        names = np.array([f"Fiber {i+1}" for i in range(n_fibers)])

        # Create columns
        c1 = fits.Column(name="TYPE", format="1A", array=types)
        c2 = fits.Column(name="NAME", format="20A", array=names)

        # Create Binary Table HDU
        fib_hdu = fits.BinTableHDU.from_columns([c1, c2], name="FIBRES")

    else:
        if len(fiber_table) != n_fibers:
            logger.warning(f"Fiber table has {len(fiber_table)} rows, but n_fibers={n_fibers} were given. Ignoring n_fibers argument.")
            
        fibcols = []
        for col in fiber_table.columns:
            c = fits.Column(name=col.name, format=col.format, array=col.data)
            fibcols.append(c)
        fib_hdu = fits.BinTableHDU.from_columns(fibcols, name="FIBRES")

    # Append to HDUList
    hdul.append(fib_hdu)


def sanitize_header_drop_unparsable(
    hdr: fits.Header, max_passes: int = 50, verbose: bool = True
) -> fits.Header:
    """
    Return a cleaned copy of hdr by removing any cards that raise VerifyError (or ValueError)
    when converted to a FITS card image (str(card)). Also removes associated CONTINUE chains.

    This is meant for headers that contain broken OGIP long-string CONTINUE blocks.

    Parameters
    ----------
    hdr : fits.Header
    max_passes : int
        Safety limit to avoid infinite loops.
    verbose : bool
        Print which cards were dropped.

    Returns
    -------
    fits.Header
        Cleaned header.
    """
    new = hdr.copy()

    def safe_card_repr(card):
        # triggers the same logic as printing a header, but per-card
        return str(card)  # may raise VerifyError

    passes = 0
    while passes < max_passes:
        passes += 1
        changed = False

        i = 0
        while i < len(new.cards):
            card = new.cards[i]
            try:
                _ = safe_card_repr(card)
                i += 1
                continue
            except (VerifyError, ValueError) as e:
                # This card is problematic. Decide how much to delete.
                kw = card.keyword
                if verbose:
                    # card may not stringify, so print minimal info
                    print(
                        f"[sanitize] dropping card at idx={i} keyword={kw!r} due to {type(e).__name__}: {e}"
                    )

                # If we're in a CONTINUE chain, we should remove the whole chain.
                # Strategy:
                #  - find the start of the chain (go backwards while previous are CONTINUE)
                #  - if the immediate previous card is NOT CONTINUE, include that previous card too
                #    (because it's likely the long-string starter with '&')
                start = i
                while (
                    start > 0
                    and new.cards[start].keyword == "CONTINUE"
                    and new.cards[start - 1].keyword == "CONTINUE"
                ):
                    start -= 1

                # If the problematic card is CONTINUE and there is a preceding non-CONTINUE card,
                # that preceding card is very likely the parent long-string keyword.
                if (
                    new.cards[i].keyword == "CONTINUE"
                    and i > 0
                    and new.cards[i - 1].keyword != "CONTINUE"
                ):
                    start = i - 1

                # Now delete forward: start card + subsequent CONTINUEs
                end = start + 1
                while end < len(new.cards) and new.cards[end].keyword == "CONTINUE":
                    end += 1

                if verbose:
                    print(
                        f"[sanitize] removing cards [{start}:{end}) ({end-start} cards)"
                    )

                # Delete from end to start
                for j in range(end - 1, start - 1, -1):
                    del new[j]

                changed = True
                # After deletion, continue at same index (start), since list shrank
                i = start
            except Exception as e:
                raise e

        if not changed:
            break

    if passes >= max_passes and verbose:
        print("[sanitize] reached max_passes; header may still contain issues.")
    return new


# TODO: review this - time stamping per frame, add gain/rdnoise settings for different setups
def convert_isoplane_header(header: fits.Header, ndfclass: str) -> fits.Header:
    """
    Convert a PIXIS/Isoplane raw header to the standard kspecdr format.

    Parameters
    ----------
    header : fits.Header
        The original raw header.
    ndfclass : str
        The NDFCLASS of the data (e.g., "MFOBJECT", "MFARC", "MFFFF", "BIAS", "DARK").

    Returns
    -------
    fits.Header
        The converted header with standardized keywords.
    """
    # Create a copy to avoid modifying the original
    new_header = header.copy()
    
    # Drop non-parsable cards
    # TODO: add options for verbose and max_passes?
    new_header = sanitize_header_drop_unparsable(new_header, verbose=False)

    # 1. Instrument Name
    new_header["INSTRUME"] = ("ISOPLANE", "KSPEC Backup Spectrograph")

    # 2. Gain and Noise (Measured values for Low Noise / Low Gain)
    # RDN = 16.03 e- rms, Gain = 4.09 e-/ADU
    # TODO: change this to the dict-based calls depending on the CCD settings (IQ, Gain mode)
    new_header["RO_GAIN"] = (4.09, "Readout Amplifer gain (e-/ADU)")
    new_header["RO_NOISE"] = (16.03, "Readout noise (electrons)")

    # 3. Exposure Time
    # Raw header seems to have EXPOSURETIME in milliseconds in HIERARCH keywords
    # Try to find HIERARCH PI CAMERA SHUTTERTIMING EXPOSURETIME
    # Note: astropy handles HIERARCH keywords, but looking up by full string is safer
    exp_ms = None
    for card in header.cards:
        if "SHUTTERTIMING EXPOSURETIME" in card.keyword:
            try:
                exp_ms = float(card.value)
            except (ValueError, TypeError):
                pass
            break

    if exp_ms is not None:
        exposed = exp_ms / 1000.0
    else:
        # Fallback to EXPTIME if available, assuming it might be seconds or needs checking
        # The user provided example shows EXPTIME = '1000 ', which matched 1000ms
        exposed = header.get("EXPTIME", 0.0)
        # Heuristic: if > 100, assume it's ms? Or trust the HIERARCH one primarily.
        try:
            exposed = float(exposed)
            # If it matches the HIERARCH one (which was ms), it's probably ms.
            # But standard FITS is seconds.
            # Given the sample: EXPTIME='1000 ', HIERARCH...='1000' /milliseconds
            # It is safer to divide by 1000 if it's large and matches the ms value.
            if exposed > 100:
                exposed = exposed / 1000.0
        except:
            exposed = 0.0

    new_header["EXPOSED"] = (exposed, "Exposure Time (seconds)")
    new_header["TOTALEXP"] = (exposed, "Total Exposure (seconds)")
    new_header["ELAPSED"] = (exposed, "Elapsed Time (seconds)")

    # 4. Dates and Times
    # Input: DATE-OBS= '2024-08-24T13:47:39'
    date_obs = header.get("DATE-OBS", "")
    if date_obs:
        try:
            t = Time(date_obs, format="isot", scale="utc")
            new_header["UTDATE"] = (t.strftime("%Y:%m:%d"), "UT Date")
            new_header["UTSTART"] = (t.strftime("%H:%M:%S"), "UT Start")
            new_header["UTMJD"] = (t.mjd, "UT MJD at start of exposure")
            # UTEND would be UTSTART + EXPOSED
            t_end = t + (exposed / 86400.0)  # exposed is seconds
            new_header["UTEND"] = (t_end.strftime("%H:%M:%S"), "UT End")
            new_header["EPOCH"] = (t.jyear, "Current Epoch, Years A.D.")
        except Exception as e:
            logger.warning(f"Could not parse DATE-OBS: {e}")

    # 5. Detector Info
    # HIERARCH PI CAMERA SENSOR INFORMATION SENSORNAME
    sensor_name = "UNKNOWN"
    for card in header.cards:
        if "SENSOR INFORMATION SENSORNAME" in card.keyword:
            sensor_name = card.value
            break
    new_header["DETECTOR"] = (sensor_name, "Detector name")
    # Orientation of the detector readout
    ori_uncorr = header["PI ACQUISITION ORIENTATION LIGHTPATHORIENTATIONUNCORRECTED"]
    ori_result = header["PI CAMERA EXPERIMENT ACQUISITION ORIENTATION RESULT"]
    logger.debug(f"ori_uncorr: {ori_uncorr}")
    logger.debug(f"ori_result: {ori_result}")
    
    # Horizontal orientation of the detector readout
    i_ori_uncorr_horizontal = -1 if "FlippedHorizontally" in ori_uncorr else 1
    i_ori_result_horizontal = -1 if "FlippedHorizontally" in ori_result else 1
    flip_horizontal = i_ori_uncorr_horizontal * i_ori_result_horizontal
    new_header["FLIPHORI"] = (flip_horizontal, "Flip horizontal orientation")
    logger.debug(f"flip_horizontal: {flip_horizontal}")
    logger.debug(f"FLIPHORI: {new_header["FLIPHORI"]}")
    
    
    # Vertical orientation of the detector readout
    i_ori_uncorr_vertical = -1 if "FlippedVertically" in ori_uncorr else 1
    i_ori_result_vertical = -1 if "FlippedVertically" in ori_result else 1
    flip_vertical = i_ori_uncorr_vertical * i_ori_result_vertical
    new_header["FLIPVERT"] = (flip_vertical, "Flip vertical orientation")
    
    # 6. Grating Info
    # HIERARCH PI SPECTROMETER GRATING SELECTED = '[500nm,150][2][0]'
    # Need to parse: 150 lines/mm, 500nm center?
    grating_str = ""
    for card in header.cards:
        if "SPECTROMETER GRATING SELECTED" in card.keyword:
            grating_str = card.value
            break

    # Attempt to parse [Center, Lines]
    # Example: [500nm,150]
    match = re.search(r"\[(.*?),(\d+)\]", grating_str)
    if match:
        # center_val = match.group(1) # e.g. 500nm
        lines_per_mm = int(match.group(2))
        new_header["GRATLPMM"] = (lines_per_mm, "Grating Lines per mm")
        # Maybe use lines/mm as ID if no other ID?
        new_header["GRATID"] = (new_header.get("GRATLPMM"), "Grating ID")

    # 7. Central Wavelength
    # HIERARCH PI SPECTROMETER GRATING CENTERWAVELENGTH = '600' / nanometers
    center_wl_nm = None
    for card in header.cards:
        if "SPECTROMETER GRATING CENTERWAVELENGTH" in card.keyword:
            try:
                center_wl_nm = float(card.value)
            except:
                pass
            break

    if center_wl_nm is not None:
        lambdac_ang = center_wl_nm * 10.0
        new_header["LAMBDAC"] = (lambdac_ang, "Central wavelength in Angstrom")
        new_header["LAMBDAB"] = (lambdac_ang, "Compatibility keyword")

    # 8. Dispersion
    if match:
        reciprocal_linear_dispersion = 23.0 * (1200 / lines_per_mm) # Angstrom / mm, from the isoplane datasheet
        pixel_width = float(new_header["PI CAMERA SENSOR INFORMATION PIXEL WIDTH"]) * 1e-3 # micron to mm # TODO: check - height or width? they are the same for PIXIS1300BX
        dispersion = reciprocal_linear_dispersion * pixel_width # Angstrom / pixel
        new_header["DISPERS"] = (dispersion, "Central dispersion (Angstrom/pixel)") # negative because the dispersion is in the opposite direction of the wavelength
    # new_header["DISPERS"] = (-3.95, "Central dispersion (Angstrom/pixel)") # negative because the dispersion is in the opposite direction of the wavelength

    # 9. Spectrograph ID
    # Not strictly needed but good for completeness
    if "SPECTID" not in new_header:
        new_header["SPECTID"] = ("UNKNOWN", "Spectrograph ID")

    # 10. Observation Type
    # Try to derive from object name or other fields if possible
    # For now, default to OBJECT if unknown
    if "OBSTYPE" not in new_header:
        new_header["OBSTYPE"] = ("OBJECT", "Observation type")
        
    new_header["NDFCLASS"] = (ndfclass, "Data Reduction class name (NDFCLASS)")

    # 11. Dispersion Axis
    # 1 = Horizontal (L-R), 2 = Vertical (T-B)
    new_header["DISPAXIS"] = (1, "Dispersion axis (1=Horizontal, 2=Vertical)")

    return new_header


def write_isoplane_converted_image(fpath: str, output_fpath: str, ndfclass: str, n_fibers: int = None, fiber_table: Table = None) -> None:
    if n_fibers is None and fiber_table is None:
        raise ValueError("Either n_fibers or fiber_table must be provided")
    
    hdul = fits.open(fpath)
    hdr = hdul[0].header
    new_hdr = convert_isoplane_header(hdr, ndfclass=ndfclass)

    # add fiber table
    add_fiber_table(hdul, n_fibers=n_fibers, fiber_table=fiber_table)

    # just use the first frame for now
    if hdul[0].data.ndim == 3:
        if hdul[0].data.shape[0] > 1:
            raise ValueError("More than one frame in the input file. Use combine_image to combine frames.")
        else:
            hdul[0].data = hdul[0].data[0]
            # make new fits file with new header and fiber table
            new_hdr["NAXIS"] = 2
            new_hdr.remove("NAXIS3")
            
    elif hdul[0].data.ndim == 2:
        pass
    else:
        raise ValueError(f"Input data has {hdul[0].data.ndim} dimensions. Expected 2 or 3.")
    
    if new_hdr["FLIPHORI"] > 0:
        hdul[0].data = np.flip(hdul[0].data, axis=1)
        new_hdr["FLIPHORI"] = -1
        logger.info("Flipped horizontal orientation")
        
    if new_hdr["FLIPVERT"] > 0:
        hdul[0].data = np.flip(hdul[0].data, axis=0)
        new_hdr["FLIPVERT"] = -1
        logger.info("Flipped vertical orientation")
        
    hdul[0].header = new_hdr

    hdul.writeto(output_fpath, overwrite=True)