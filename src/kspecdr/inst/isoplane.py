"""
Header conversion utilities for KSPEC data.

This module provides functions to convert raw headers from various instruments
(e.g., PIXIS/Isoplane backup CCD) to the standard format required by kspecdr.
"""

import logging
from astropy.io import fits
from astropy.time import Time
import re

logger = logging.getLogger(__name__)

def convert_isoplane_header(header: fits.Header) -> fits.Header:
    """
    Convert a PIXIS/Isoplane raw header to the standard kspecdr format.

    Parameters
    ----------
    header : fits.Header
        The original raw header.

    Returns
    -------
    fits.Header
        The converted header with standardized keywords.
    """
    # Create a copy to avoid modifying the original
    new_header = header.copy()

    # 1. Instrument Name
    new_header['INSTRUME'] = ('ISOPLANE', 'KSPEC Backup CCD')

    # 2. Gain and Noise (Measured values for Low Noise / Low Gain)
    # RDN = 16.03 e- rms, Gain = 4.09 e-/ADU
    new_header['RO_GAIN'] = (4.09, 'Readout Amplifer gain (e-/ADU)')
    new_header['RO_NOISE'] = (16.03, 'Readout noise (electrons)')

    # 3. Exposure Time
    # Raw header seems to have EXPOSURETIME in milliseconds in HIERARCH keywords
    # Try to find HIERARCH PI CAMERA SHUTTERTIMING EXPOSURETIME
    # Note: astropy handles HIERARCH keywords, but looking up by full string is safer
    exp_ms = None
    for card in header.cards:
        if 'SHUTTERTIMING EXPOSURETIME' in card.keyword:
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
        exposed = header.get('EXPTIME', 0.0)
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

    new_header['EXPOSED'] = (exposed, 'Exposure Time (seconds)')
    new_header['TOTALEXP'] = (exposed, 'Total Exposure (seconds)')
    new_header['ELAPSED'] = (exposed, 'Elapsed Time (seconds)')

    # 4. Dates and Times
    # Input: DATE-OBS= '2024-08-24T13:47:39'
    date_obs = header.get('DATE-OBS', '')
    if date_obs:
        try:
            t = Time(date_obs, format='isot', scale='utc')
            new_header['UTDATE'] = (t.strftime('%Y:%m:%d'), 'UT Date')
            new_header['UTSTART'] = (t.strftime('%H:%M:%S'), 'UT Start')
            new_header['UTMJD'] = (t.mjd, 'UT MJD at start of exposure')
            # UTEND would be UTSTART + EXPOSED
            t_end = t + (exposed / 86400.0) # exposed is seconds
            new_header['UTEND'] = (t_end.strftime('%H:%M:%S'), 'UT End')
            new_header['EPOCH'] = (t.jyear, 'Current Epoch, Years A.D.')
        except Exception as e:
            logger.warning(f"Could not parse DATE-OBS: {e}")

    # 5. Detector Info
    # HIERARCH PI CAMERA SENSOR INFORMATION SENSORNAME
    sensor_name = 'UNKNOWN'
    for card in header.cards:
        if 'SENSOR INFORMATION SENSORNAME' in card.keyword:
             sensor_name = card.value
             break
    new_header['DETECTOR'] = (sensor_name, 'Detector name')

    # 6. Grating Info
    # HIERARCH PI SPECTROMETER GRATING SELECTED = '[500nm,150][2][0]'
    # Need to parse: 150 lines/mm, 500nm center?
    grating_str = ''
    for card in header.cards:
        if 'SPECTROMETER GRATING SELECTED' in card.keyword:
            grating_str = card.value
            break

    # Attempt to parse [Center, Lines]
    # Example: [500nm,150]
    match = re.search(r'\[(.*?),(\d+)\]', grating_str)
    if match:
        # center_val = match.group(1) # e.g. 500nm
        lines_per_mm = match.group(2)
        new_header['GRATLPMM'] = (int(lines_per_mm), 'Grating Lines per mm')
        # Maybe use lines/mm as ID if no other ID?
        new_header['GRATID'] = (new_header.get('GRATLPMM'), 'Grating ID')

    # 7. Central Wavelength
    # HIERARCH PI SPECTROMETER GRATING CENTERWAVELENGTH = '600' / nanometers
    center_wl_nm = None
    for card in header.cards:
        if 'SPECTROMETER GRATING CENTERWAVELENGTH' in card.keyword:
            try:
                center_wl_nm = float(card.value)
            except:
                pass
            break

    if center_wl_nm is not None:
        lambdac_ang = center_wl_nm * 10.0
        new_header['LAMBDAC'] = (lambdac_ang, 'Central wavelength in Angstrom')
        new_header['LAMBDAB'] = (lambdac_ang, 'Compatibility keyword')

    # 8. Dispersion
    # Requested to be 'not implemented' / placeholder
    # But usually a float. Let's set to a dummy value or leave it if existing (likely not)
    # Using 0.0 or 1.0 is safer than string 'not implemented' for float fields
    new_header['DISPERS'] = (0.0, 'Central dispersion (Angstrom/pixel) - NOT IMPL')

    # 9. Spectrograph ID
    # Not strictly needed but good for completeness
    if 'SPECTID' not in new_header:
        new_header['SPECTID'] = ('UNKNOWN', 'Spectrograph ID')

    # 10. Observation Type
    # Try to derive from object name or other fields if possible
    # For now, default to OBJECT if unknown
    if 'OBSTYPE' not in new_header:
        new_header['OBSTYPE'] = ('OBJECT', 'Observation type')

    return new_header
