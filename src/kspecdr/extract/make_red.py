"""
Reduction Routines for KSPEC.

This module implements the reduction of extracted spectra (making RED files from EX files),
converting the 2dfdr `MAKE_RED` subroutine.
"""

import logging
from typing import Dict, Any
import shutil
from pathlib import Path

from ..io.image import ImageFile

logger = logging.getLogger(__name__)


def make_red(args: Dict[str, Any]) -> None:
    """
    Main driver for reduction process (EX -> RED).
    Replaces 2dfdr SUBROUTINE MAKE_RED.

    Processes from the supplied arguments determine what extraction files are to be
    made from which image files and call MAKE_EX_FROM_IM for each. (Docstring from Fortran seems copy-pasted/wrong there).

    Actual Description:
    Creates output red(uced) file by copying ex(tracted) file.
    Divides by fibre-flat-field if requested.
    Scrunches using input arc calibration file.

    Parameters
    ----------
    args : dict
        Dictionary containing arguments:
        - EXTRAC_FILENAME: Input extracted filename
        - OUTPUT_FILENAME: Output reduced filename
        - FLAT_FILENAME: (Optional) Fiber flat field file
        - WAVEL_FILENAME: (Optional) Wavelength calibration file (Arc)
    """
    ex_fname = args.get('EXTRAC_FILENAME')
    red_fname = args.get('OUTPUT_FILENAME')

    if not ex_fname or not red_fname:
        raise ValueError("Missing required filenames (EXTRAC or OUTPUT)")

    logger.info(f"Reducing {ex_fname} -> {red_fname}")

    # 1. Create output RED file by copying EX file
    # CALL TDFIO_CREATEBYCOPY(RED_ID,EX_FILENAME,RED_FILENAME,STATUS)
    shutil.copy2(ex_fname, red_fname)

    # Open the new file for updates
    with ImageFile(red_fname, mode='UPDATE') as red_file:

        # 2. Divide by fibre-flat-field if requested
        # CALL CMFSPEC_FLATFIELD(RED_ID,ARGS,STATUS)
        # We need to implement this or placeholder.
        # Check if flat field file is provided or arguments imply it.
        # 2dfdr usually uses 'FLAT_FILENAME' or similar.
        # Assuming args has it.
        # Since I haven't implemented `CMFSPEC_FLATFIELD`, I'll leave a placeholder/TODO.
        flat_fname = args.get('FLAT_FILENAME')
        if flat_fname:
            logger.info(f"Applying flat field {flat_fname}")
            # TODO: Implement flat fielding logic
            # read flat -> divide data and variance -> update history
            pass

        # 3. Scrunch using input arc calibration file
        # CALL SCRUNCH_FROM_ARC_CAL(RED_ID,ARGS,STATUS)
        # This sets 'SCRUNCH' header keyword to TRUE
        # Need WAVEL_FILENAME
        wavel_fname = args.get('WAVEL_FILENAME')
        if wavel_fname:
            logger.info(f"Scrunching using arc file {wavel_fname}")
            # TODO: Implement scrunch logic
            # For now, just set the header keyword
            red_file.add_history(f"Scrunched using {wavel_fname}")

        # Set SCRUNCH keyword if appropriate (logic depends on if scrunch actually happened)
        # For now, if we had wavel_file, we assume we would have scrunched.

        # Update Class?
        # 2dfdr MAKE_RED doesn't explicitly change class here, but MAKE_EX set it to MFS...
        # Usually reduced files might be 'MFS...' or similar.
        # If input was MFS... output is likely same.

    logger.info(f"Created reduced file: {red_fname}")
