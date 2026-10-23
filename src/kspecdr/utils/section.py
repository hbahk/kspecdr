"""FITS image sections (``BIASSEC``, ``DATASEC``) as numpy slices."""

import re
from typing import Tuple

Region = Tuple[int, int, int, int]

_SECTION_RE = re.compile(r"^\s*\[\s*(\d+)\s*:\s*(\d+)\s*,\s*(\d+)\s*:\s*(\d+)\s*\]\s*$")


def parse_fits_section(section: str) -> Region:
    """
    Convert a FITS section string into a numpy region.

    Parameters
    ----------
    section : str
        FITS section ``'[x1:x2,y1:y2]'``: 1-based, inclusive, x along NAXIS1 (columns) and y
        along NAXIS2 (rows). Reversed ranges (``x1 > x2``) are accepted.

    Returns
    -------
    tuple of int
        ``(r0, r1, c0, c1)``, 0-based and half-open, so the section is
        ``data[r0:r1, c0:c1]``. This is the ``overscan_region`` convention of
        ``kspecdr.preproc.make_im``.

    Raises
    ------
    ValueError
        If the string is not a 2-D FITS section.
    """
    m = _SECTION_RE.match(str(section))
    if m is None:
        raise ValueError(f"Not a 2-D FITS section: {section!r}")
    x1, x2, y1, y2 = (int(v) for v in m.groups())
    if min(x1, x2, y1, y2) < 1:
        raise ValueError(f"FITS section is 1-based: {section!r}")
    return min(y1, y2) - 1, max(y1, y2), min(x1, x2) - 1, max(x1, x2)


def format_fits_section(region: Region) -> str:
    """
    Convert a numpy region ``(r0, r1, c0, c1)`` into a FITS section string.

    Parameters
    ----------
    region : tuple of int
        0-based half-open ``(r0, r1, c0, c1)``, as returned by `parse_fits_section`.

    Returns
    -------
    str
        ``'[c0+1:c1,r0+1:r1]'``.
    """
    r0, r1, c0, c1 = region
    return f"[{c0 + 1}:{c1},{r0 + 1}:{r1}]"


def flip_region(
    region: Region, shape: Tuple[int, int], flip_rows: bool, flip_cols: bool
) -> Region:
    """
    Map a region through ``np.flip`` of the image it refers to.

    Parameters
    ----------
    region : tuple of int
        0-based half-open ``(r0, r1, c0, c1)``.
    shape : tuple of int
        ``(nrows, ncols)`` of the image.
    flip_rows : bool
        The image is flipped along axis 0 (``np.flip(data, axis=0)``).
    flip_cols : bool
        The image is flipped along axis 1 (``np.flip(data, axis=1)``).

    Returns
    -------
    tuple of int
        The same pixels in the flipped image.
    """
    r0, r1, c0, c1 = region
    nrows, ncols = shape
    if flip_rows:
        r0, r1 = nrows - r1, nrows - r0
    if flip_cols:
        c0, c1 = ncols - c1, ncols - c0
    return r0, r1, c0, c1
