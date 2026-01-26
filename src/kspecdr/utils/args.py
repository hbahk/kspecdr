"""Argument parsing, validation, and normalization utilities."""

from pathlib import Path
from typing import Any, Dict, Iterable

from .fiber import parse_fib_overrides


def init_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize and normalize args in place.

    - Parse and cache fiber type overrides from FIB<no> keys.
    """
    if args is None:
        return {}

    # Cache fiber overrides for later use.
    if "_FIB_TYPE_OVERRIDES" not in args:
        args["_FIB_TYPE_OVERRIDES"] = parse_fib_overrides(args)

    return args


def validate_files_exist(
    args: Dict[str, Any], keys: Iterable[str], *, required: bool
) -> None:
    """
    Validate that files referenced by args exist.

    Parameters
    ----------
    args : dict
        Argument dictionary.
    keys : iterable
        Argument keys to validate.
    required : bool
        Whether missing values should raise immediately.
    """
    for key in keys:
        value = args.get(key, "")
        if not value:
            if required:
                raise FileNotFoundError(f"{key} not specified in arguments.")
            continue
        if not Path(value).exists():
            raise FileNotFoundError(f"Cannot find {key} file {value}")


def validate_reduce_object_args(args: Dict[str, Any]) -> None:
    """
    Sanity checks matching 2dfdr REDUCE_OBJECT_ARG_CHECKS.
    """
    required = ("TLMAP_FILENAME", "WAVEL_FILENAME", "FFLAT_FILENAME")
    validate_files_exist(args, required, required=True)

    if args.get("USEDARKIM", False) and args.get("DARK_FILENAME"):
        validate_files_exist(args, ("DARK_FILENAME",), required=True)

    optional = ("BIAS_FILENAME", "LFLAT_FILENAME", "THPUT_FILENAME")
    validate_files_exist(args, optional, required=False)
