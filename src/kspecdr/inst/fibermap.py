"""
K-SPEC fibre map: positioner holes, hole labels and the Isoplane slit.

The focal plate has 73 positioner holes, numbered 1-73 and labelled by plate row and column
(``C4``, ``D2``, ...), plus fixed sky-fibre holes. Only some of the fibres reach the Isoplane
slit; the fibre map (``data/fibers/isoplane_fibermap_<date>.txt``) gives, per hole, the hole
number, label, fibre number, whether the positioner is locked, the role on the slit (``obj``
or ``sky``) and the slit position (Isoplane fibre number).

The ICS assign files (``<project>_2627_<tile>.assign.txt``) list one target per positioner
hole, by label. :func:`make_fiber_table` joins an assign file to the fibre map and returns the
``FIBRES`` table in slit order, ready for
:func:`kspecdr.inst.isoplane.write_isoplane_converted_image`. The fibre map decides which
holes reach the slit and which are sky; targets the assign file puts on sky or off-slit holes
are ignored (and logged).
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from astropy.table import Table

from ..constants import FIBER_TYPE_NONE, FIBER_TYPE_PROGRAM, FIBER_TYPE_SKY

logger = logging.getLogger(__name__)

FIBERMAP_DIR = Path(__file__).parent.parent.parent.parent / "data" / "fibers"
DEFAULT_FIBERMAP = FIBERMAP_DIR / "isoplane_fibermap_20261012.txt"

# Columns of the ICS assign file (no header): SCIOBS/sciobscli.py load_target
ASSIGN_COLUMNS = ("LABEL", "X", "Y", "MAG", "CLASS", "FLAG", "RA", "DEC")

PathLike = Union[str, Path]


def read_fibermap(path: Optional[PathLike] = None) -> Table:
    """
    Read a K-SPEC fibre map.

    Parameters
    ----------
    path : str or Path, optional
        Fibre map file. Default: ``DEFAULT_FIBERMAP`` (positioner mapping of 2026-10-12).

    Returns
    -------
    astropy.table.Table
        One row per hole with columns ``HOLE`` (1-73), ``LABEL``, ``FIBNUM``, ``LOCKED``,
        ``ROLE`` (``obj``, ``sky`` or ``''``), ``LC2``, ``LC3``, ``LCNUM`` and ``SLIT``.
        Missing values (``-`` in the file) are 0 for the integer columns and ``''`` for
        ``ROLE``, ``LC2`` and ``LC3``: ``HOLE == 0`` is a fixed sky-fibre hole, ``SLIT == 0``
        a fibre that does not reach the slit, ``LOCKED`` is 1 for a positioner fixed in place.
    """
    path = Path(path) if path is not None else DEFAULT_FIBERMAP
    raw = Table.read(path, format="ascii.basic", comment="#", fill_values=[("-", "0")])
    tab = Table(meta={"FIBERMAP": path.name})
    for name in raw.colnames:
        col = raw[name]
        is_int = name in ("HOLE", "FIBNUM", "LOCKED", "LCNUM", "SLIT")
        if hasattr(col, "filled"):
            col = col.filled(0 if is_int else "")
        tab[name] = np.asarray(col, dtype=int if is_int else str)

    _check_fibermap(tab, path)
    return tab


def _check_fibermap(tab: Table, path: Path) -> None:
    """Raise ValueError on duplicate holes, labels or slit positions."""
    for name, values in (
        ("LABEL", tab["LABEL"]),
        ("HOLE", tab["HOLE"][tab["HOLE"] > 0]),
        ("SLIT", tab["SLIT"][tab["SLIT"] > 0]),
    ):
        uniq, counts = np.unique(values, return_counts=True)
        if np.any(counts > 1):
            raise ValueError(f"{path.name}: duplicate {name}: {list(uniq[counts > 1])}")
    on_slit = tab["SLIT"] > 0
    bad = on_slit != np.isin(tab["ROLE"], ("obj", "sky"))
    if np.any(bad):
        raise ValueError(f"{path.name}: ROLE and SLIT disagree for {list(tab['LABEL'][bad])}")
    nslit = int(on_slit.sum())
    if nslit and sorted(tab["SLIT"][on_slit]) != list(range(1, nslit + 1)):
        raise ValueError(f"{path.name}: slit positions are not 1..{nslit}")


def hole_label(hole: int, fibermap: Optional[Table] = None) -> str:
    """
    Label of a positioner hole number.

    Parameters
    ----------
    hole : int
        Positioner hole number (1-73).
    fibermap : Table, optional
        From :func:`read_fibermap`; read from ``DEFAULT_FIBERMAP`` if not given.

    Returns
    -------
    str
        Hole label, e.g. ``'C4'`` for hole 1.
    """
    fibermap = read_fibermap() if fibermap is None else fibermap
    match = fibermap["LABEL"][fibermap["HOLE"] == int(hole)]
    if len(match) == 0 or int(hole) <= 0:
        raise KeyError(f"No positioner hole {hole} in {fibermap.meta.get('FIBERMAP')}")
    return str(match[0])


def hole_number(label: str, fibermap: Optional[Table] = None) -> int:
    """
    Positioner hole number of a hole label.

    Parameters
    ----------
    label : str
        Hole label, e.g. ``'C4'``.
    fibermap : Table, optional
        From :func:`read_fibermap`; read from ``DEFAULT_FIBERMAP`` if not given.

    Returns
    -------
    int
        Hole number 1-73, or 0 for a fixed sky-fibre hole (no positioner).
    """
    fibermap = read_fibermap() if fibermap is None else fibermap
    match = fibermap["HOLE"][fibermap["LABEL"] == label.strip().upper()]
    if len(match) == 0:
        raise KeyError(f"No hole labelled {label!r} in {fibermap.meta.get('FIBERMAP')}")
    return int(match[0])


def slit_fibers(fibermap: Optional[Table] = None) -> Table:
    """
    Fibres on the Isoplane slit, in slit order (row ``i`` = Isoplane fibre ``i + 1``).

    Parameters
    ----------
    fibermap : Table, optional
        From :func:`read_fibermap`; read from ``DEFAULT_FIBERMAP`` if not given.

    Returns
    -------
    astropy.table.Table
        The fibre map rows with ``SLIT > 0``, sorted by ``SLIT``.
    """
    fibermap = read_fibermap() if fibermap is None else fibermap
    slit = fibermap[fibermap["SLIT"] > 0]
    slit.sort("SLIT")
    return slit


def read_assign(path: PathLike) -> Table:
    """
    Read an ICS fibre assign file (``<project>_2627_<tile>.assign.txt``).

    Parameters
    ----------
    path : str or Path
        Assign file: no header, one positioner hole per line with the columns
        ``ASSIGN_COLUMNS`` (hole label, focal-plane x and y in mm, magnitude, target class,
        flag, RA and Dec in degrees), as read by the ICS ``SCIOBS.load_target``.

    Returns
    -------
    astropy.table.Table
        Columns ``ASSIGN_COLUMNS``.
    """
    tab = Table.read(
        path, format="ascii.no_header", names=ASSIGN_COLUMNS, comment="#", guess=False
    )
    tab["LABEL"] = np.char.upper(np.asarray(tab["LABEL"], dtype=str))
    uniq, counts = np.unique(tab["LABEL"], return_counts=True)
    if np.any(counts > 1):
        raise ValueError(f"{Path(path).name}: duplicate hole labels {list(uniq[counts > 1])}")
    return tab


def make_fiber_table(
    assign: Union[PathLike, Table, None],
    fibermap: Optional[Table] = None,
    class_types: Optional[dict] = None,
) -> Table:
    """
    ``FIBRES`` table of the Isoplane slit from an assign file and the fibre map.

    Parameters
    ----------
    assign : str, Path, Table or None
        Assign file (see :func:`read_assign`) or its table. None gives a table without
        targets (calibration frames): object fibres get ``TYPE='P'`` and no coordinates.
    fibermap : Table, optional
        From :func:`read_fibermap`; read from ``DEFAULT_FIBERMAP`` if not given.
    class_types : dict, optional
        Assign ``CLASS`` -> fibre ``TYPE`` (``kspecdr.constants``), e.g. ``{8: 'C'}`` for
        spectrophotometric standards. Classes not listed are program targets (``'P'``).

    Returns
    -------
    astropy.table.Table
        One row per slit fibre in slit order, with ``TYPE``, ``NAME`` (hole label),
        ``MAGNITUDE``, ``RA``, ``DEC`` (deg), ``X``, ``Y`` (mm, focal plane), ``CLASS`` and
        ``FLAG`` (from the assign file; NaN / -1 where there is no target) and the fibre-map
        columns ``SLIT``, ``HOLE``, ``LABEL``, ``FIBNUM``, ``LOCKED``.

    Notes
    -----
    Rules, in order: ``ROLE == 'sky'`` fibres are ``TYPE='S'`` whatever the assign file
    says (they sit on locked positioners or fixed sky holes); ``obj`` fibres take the
    target of their hole label from the assign file, or ``TYPE='N'`` when it has none.
    """
    fibermap = read_fibermap() if fibermap is None else fibermap
    class_types = {} if class_types is None else class_types
    slit = slit_fibers(fibermap)
    nfib = len(slit)

    if assign is None:
        targets = None
    else:
        targets = assign if isinstance(assign, Table) else read_assign(assign)
        by_label = {str(lab): i for i, lab in enumerate(targets["LABEL"])}

        unknown = sorted(set(by_label) - set(fibermap["LABEL"]))
        if unknown:
            logger.warning("Assign holes not in the fibre map %s: %s",
                           fibermap.meta.get("FIBERMAP"), unknown)

    out = Table()
    out["TYPE"] = np.full(nfib, FIBER_TYPE_PROGRAM, dtype="U1")
    out["NAME"] = np.asarray(slit["LABEL"], dtype="U20")
    for name in ("MAGNITUDE", "RA", "DEC", "X", "Y"):
        out[name] = np.full(nfib, np.nan)
    out["CLASS"] = np.full(nfib, -1, dtype=int)
    out["FLAG"] = np.full(nfib, -1, dtype=np.int64)
    for name in ("SLIT", "HOLE", "LABEL", "FIBNUM", "LOCKED"):
        out[name] = slit[name]

    sky_targets = []
    for i, row in enumerate(slit):
        label = str(row["LABEL"])
        if row["ROLE"] == "sky":
            out["TYPE"][i] = FIBER_TYPE_SKY
            if targets is not None and label in by_label:
                sky_targets.append(label)
            continue
        if targets is None:
            continue
        if label not in by_label:
            out["TYPE"][i] = FIBER_TYPE_NONE
            logger.warning("Slit fibre %d (%s) has no target in the assign file", i + 1, label)
            continue
        t = targets[by_label[label]]
        out["TYPE"][i] = class_types.get(int(t["CLASS"]), FIBER_TYPE_PROGRAM)
        out["MAGNITUDE"][i] = t["MAG"]
        out["RA"][i] = t["RA"]
        out["DEC"][i] = t["DEC"]
        out["X"][i] = t["X"]
        out["Y"][i] = t["Y"]
        out["CLASS"][i] = t["CLASS"]
        out["FLAG"][i] = t["FLAG"]

    if targets is not None:
        if sky_targets:
            logger.info("Assign targets on sky fibres, ignored: %s", sky_targets)
        off_slit = sorted(set(by_label) - set(slit["LABEL"]))
        logger.info(
            "Fibre table: %d slit fibres (%s); %d assign targets not on the slit",
            nfib,
            ", ".join(f"{k}={v}" for k, v in zip(*np.unique(out["TYPE"], return_counts=True))),
            len(off_slit),
        )

    out.meta["FIBERMAP"] = fibermap.meta.get("FIBERMAP", "")
    return out
