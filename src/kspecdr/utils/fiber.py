"""Fiber-related helpers."""

from typing import Any, Dict

import numpy as np


def parse_fib_overrides(args: Dict[str, Any]) -> Dict[int, str]:
    """
    Parse FIB<no> arguments into a fiber type override map.

    Returns
    -------
    dict
        Mapping of 1-based fiber number to single-character type code.
    """
    overrides: Dict[int, str] = {}
    for key, value in args.items():
        if not isinstance(key, str) or not key.startswith("FIB"):
            continue
        suffix = key[3:]
        if not suffix.isdigit():
            continue
        fibno = int(suffix)
        if fibno <= 0:
            continue
        if value is None:
            continue
        ftype = str(value).strip()
        if not ftype:
            continue
        overrides[fibno] = ftype[0].upper()
    return overrides


def get_override_from_args(args: Dict[str, Any]) -> Dict[int, str]:
    """
    Return cached overrides from args or parse and cache them.
    """
    if args is None:
        return {}
    overrides = args.get("_FIB_TYPE_OVERRIDES")
    if overrides is None:
        overrides = parse_fib_overrides(args)
        args["_FIB_TYPE_OVERRIDES"] = overrides
    return overrides


def apply_fiber_overrides(
    fiber_types: np.ndarray, overrides: Dict[int, str]
) -> np.ndarray:
    """
    Apply 1-based fiber overrides to a fiber type array.
    """
    if fiber_types is None or not overrides:
        return fiber_types

    for fibno, ftype in overrides.items():
        idx = fibno - 1
        if 0 <= idx < len(fiber_types):
            fiber_types[idx] = ftype
    return fiber_types
