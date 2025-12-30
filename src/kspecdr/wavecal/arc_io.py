"""
Arc file I/O operations.
"""

import os
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

def read_arc_file(nx: int, xvec: np.ndarray, lamp: str, max_entries: int = 20000) -> tuple[np.ndarray, np.ndarray, list, int]:
    """
    Reads an ASCII text 'arc' file containing wavelengths and intensities.

    Parameters
    ----------
    nx : int
        Size of xvec
    xvec : np.ndarray
        Wavelength axis vector (predicted)
    lamp : str
        Lamp name
    max_entries : int
        Max entries to read

    Returns
    -------
    wlist : np.ndarray
        Wavelengths
    ilist : np.ndarray
        Intensities
    labels : list
        Labels
    nlist : int
        Number of entries
    """
    # 1. Determine filename
    # TODO: Implement searching in DRCONTROL_DIR or similar.
    # For now, assume it's in the current directory or 'arc_files' subdir

    fname = f"{lamp.lower()}.arc"

    # Check current dir
    if not os.path.exists(fname):
        # Try arc_files subdir
        possible_path = os.path.join("arc_files", fname)
        if os.path.exists(possible_path):
            fname = possible_path
        else:
            # Fallback or error
            logger.warning(f"Arc file {fname} not found.")
            return np.array([]), np.array([]), [], 0

    # 2. Get min/max from xvec
    rng = xvec[-1] - xvec[0]
    lmin = min(xvec[0], xvec[-1]) - 0.2 * rng
    lmax = max(xvec[0], xvec[-1]) + 0.2 * rng

    wlist = []
    ilist = []
    labels = []

    logger.info(f"Reading arc file {fname}")

    try:
        with open(fname, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('*') or line.startswith('#'):
                    # Check for #MAP (not implemented here yet)
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue

                try:
                    wave = float(parts[0])
                    inten = float(parts[1])
                except ValueError:
                    continue

                if "unreferenced" in line.lower():
                    inten = -inten

                if lmin <= wave <= lmax:
                    wlist.append(wave)
                    ilist.append(inten)

                    # Extract label if present (e.g. 'label=AC')
                    label = "??"
                    if "label=" in line.lower():
                        idx = line.lower().find("label=")
                        label = line[idx+6:idx+8]
                    labels.append(label)

                if len(wlist) >= max_entries:
                    break

    except Exception as e:
        logger.error(f"Error reading arc file: {e}")
        return np.array([]), np.array([]), [], 0

    return np.array(wlist), np.array(ilist), labels, len(wlist)
