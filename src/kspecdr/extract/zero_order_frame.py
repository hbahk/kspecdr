"""
Zero-order fibre fluxes of one raw Isoplane frame, for real-time use.

Chains the steps of a zero-order frame: ``write_isoplane_converted_image`` (fibre table from
the slit map) -> ``make_im`` (overscan only) -> ``make_zo``. Run as a command, it prints the
result as one JSON line on stdout, so that a process without kspecdr (the ICS IsoPlane server)
can run it in the kspecdr environment right after each exposure::

    python -m kspecdr.extract.zero_order_frame RAW.fits --outdir DIR [--ref REF_zo.fits]
        [--sections-from ICS_FRAME.fits] [--set KEY=VALUE ...]

Products in ``DIR``: ``c<frame>.fits`` (converted), ``c<frame>_im.fits`` and
``c<frame>_zo.fits``. The JSON line is :func:`zero_order_summary` plus ``ok``, ``raw`` and
``zofile``; on failure it is ``{"ok": false, "error": ...}`` and the exit status is 1.
"""

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from astropy.io import fits
from astropy.table import Table

from ..inst.fibermap import make_fiber_table
from ..inst.isoplane import write_isoplane_converted_image
from ..preproc.make_im import make_im
from .zero_order import make_zo

logger = logging.getLogger(__name__)

# ZOFLUX columns in the summary (lower-case keys), when present; FLUX_ERR is added
SUMMARY_COLUMNS = (
    "FIBNO", "LABEL", "TYPE", "XPOS", "YPOS", "FLUX", "SNR", "BKG", "PEAK", "DETECTED",
    "ZOFLAG",
)
# ZOFLUX header cards in the summary: card -> key
SUMMARY_META = {
    "ZOMETHOD": "method", "ZOREF": "reference", "ZOXPOS": "xpos", "ZOPITCH": "pitch",
    "ZODX": "dx", "ZODY": "dy", "ZONDET": "ndet",
}


def _convert(raw: Path, out: Path, fibre_table: Table) -> None:
    """Convert a raw frame; frames without ICS cards are classed as object frames."""
    has_ics = "ICSFUNC" in fits.getheader(raw)
    write_isoplane_converted_image(
        str(raw), str(out), None if has_ics else "MFOBJECT", fiber_table=fibre_table
    )


def _copy_sections(conv: Path, source_raw: Path, fibre_table: Table) -> None:
    """Give ``conv`` the DATASEC/BIASSEC of ``source_raw`` after the same conversion."""
    with tempfile.TemporaryDirectory() as tmp:
        src_conv = Path(tmp) / "c_sections_source.fits"
        _convert(source_raw, src_conv, fibre_table)
        src = fits.getheader(src_conv)
    # Into memory and closed before the overwrite: Windows refuses to replace an open file
    with fits.open(conv, memmap=False) as opened:
        hdul = fits.HDUList([hdu.copy() for hdu in opened])
    if hdul[0].data.shape != fits.getdata(source_raw).shape[-2:]:
        raise ValueError(f"{source_raw.name} has a different readout shape")
    for key in ("DATASEC", "BIASSEC"):
        hdul[0].header[key] = (src[key], f"from {source_raw.name}")
    hdul.writeto(conv, overwrite=True)
    logger.info("Sections from %s: DATASEC %s, BIASSEC %s", source_raw.name,
                src["DATASEC"], src["BIASSEC"])


def reduce_zero_order_frame(
    raw_filename: str,
    outdir: str,
    ref_filename: Optional[str] = None,
    sections_from: Optional[str] = None,
    zo_args: Optional[Dict[str, Any]] = None,
    converted_dir: Optional[str] = None,
) -> Table:
    """
    Convert, ``make_im`` and ``make_zo`` one raw zero-order Isoplane frame.

    Parameters
    ----------
    raw_filename : str
        Raw LightField FITS frame. Frames written by the ICS server carry ``DATASEC`` /
        ``BIASSEC``; frames saved straight from LightField do not (see ``sections_from``).
    outdir : str
        Directory for ``c<frame>_im.fits`` and ``c<frame>_zo.fits`` (created if needed).
    ref_filename : str, optional
        ``*_zo.fits`` of a zero-order flat: reference mode of :func:`make_zo`
        (``ZO_REF_FILENAME``). Star frames need one.
    sections_from : str, optional
        ICS frame of the same readout whose ``DATASEC`` / ``BIASSEC`` are used when the frame
        has none.
    zo_args : dict, optional
        More :func:`make_zo` keys (``ISO_FIB_PITCH``, ``ZO_APHW_X``, ...).
    converted_dir : str, optional
        Directory for the converted ``c<frame>.fits`` (default ``outdir``).

    Returns
    -------
    astropy.table.Table
        The ``ZOFLUX`` table of :func:`make_zo`, with ``meta["ZOFILE"]`` set to its file.

    Raises
    ------
    ValueError
        If the frame has no ``BIASSEC`` and no ``sections_from`` is given.
    """
    raw = Path(raw_filename)
    out = Path(outdir)
    conv_dir = Path(converted_dir) if converted_dir else out
    out.mkdir(parents=True, exist_ok=True)
    conv_dir.mkdir(parents=True, exist_ok=True)
    conv = conv_dir / f"c{raw.stem}.fits"
    im = out / f"c{raw.stem}_im.fits"
    zo = out / f"c{raw.stem}_zo.fits"

    fibre_table = make_fiber_table(None)
    _convert(raw, conv, fibre_table)
    if "BIASSEC" not in fits.getheader(conv):
        if not sections_from:
            raise ValueError(
                f"{raw.name} has no BIASSEC (not written by the ICS server): "
                "give sections_from"
            )
        _copy_sections(conv, Path(sections_from), fibre_table)

    make_im(str(conv), im_filename=str(im), use_bias=True, verbose=False)

    args = {**(zo_args or {}), "IMAGE_FILENAME": str(im), "ZOFLUX_FILENAME": str(zo)}
    if ref_filename:
        args["ZO_REF_FILENAME"] = str(ref_filename)
    table = make_zo(args)
    table.meta["ZOFILE"] = str(zo)
    return table


def _plain(value: Any) -> Any:
    """A JSON value: bytes as str, numpy scalars as Python, non-finite floats as None."""
    if isinstance(value, bytes):
        return value.decode().strip()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def zero_order_summary(table: Table) -> Dict[str, Any]:
    """
    JSON-ready summary of a ``ZOFLUX`` table: the method cards and one list per column.

    Returns
    -------
    dict
        ``nfib``, the cards of ``SUMMARY_META`` present in ``table.meta`` (``method``,
        ``pitch``, ``ndet``, ...), and ``fibres``: ``{column: [value per fibre]}`` with the
        lower-case ``SUMMARY_COLUMNS`` and ``flux_err`` (square root of ``FLUX_VAR``), in
        slit order.
    """
    out = {"nfib": len(table)}
    for card, key in SUMMARY_META.items():
        if card in table.meta:
            out[key] = _plain(table.meta[card])
    fibres = {
        name.lower(): [_plain(v) for v in table[name]]
        for name in SUMMARY_COLUMNS
        if name in table.colnames
    }
    fibres["flux_err"] = [_plain(v) for v in np.sqrt(np.asarray(table["FLUX_VAR"], float))]
    out["fibres"] = fibres
    return out


def _parse_set(items) -> Dict[str, Any]:
    """``["KEY=VALUE", ...]`` -> dict; values read as JSON when they parse, else as text."""
    out = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--set {item!r} is not KEY=VALUE")
        try:
            out[key.strip().upper()] = json.loads(value)
        except json.JSONDecodeError:
            out[key.strip().upper()] = value
    return out


def main(argv=None) -> int:
    """Command-line entry point; see the module docstring."""
    ap = argparse.ArgumentParser(
        prog="python -m kspecdr.extract.zero_order_frame",
        description="Zero-order fibre fluxes of one raw Isoplane frame, as one JSON line.",
    )
    ap.add_argument("raw", help="raw LightField FITS frame")
    ap.add_argument("--outdir", required=True, help="directory for the products")
    ap.add_argument("--ref", help="*_zo.fits of a zero-order flat (reference mode)")
    ap.add_argument("--sections-from", help="ICS frame with DATASEC/BIASSEC for this readout")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="make_zo key, e.g. ISO_FIB_PITCH=6.74 (repeatable)")
    ap.add_argument("-v", "--verbose", action="store_true", help="log INFO to stderr")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(name)s: %(message)s", stream=sys.stderr)
    try:
        table = reduce_zero_order_frame(
            a.raw, a.outdir, ref_filename=a.ref, sections_from=a.sections_from,
            zo_args=_parse_set(a.set),
        )
        result = {"ok": True, "raw": str(a.raw), "zofile": table.meta["ZOFILE"],
                  **zero_order_summary(table)}
        status = 0
    except Exception as e:
        logger.exception("zero-order measurement of %s failed", a.raw)
        result = {"ok": False, "raw": str(a.raw), "error": f"{type(e).__name__}: {e}"}
        status = 1
    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()
    return status


if __name__ == "__main__":
    sys.exit(main())
