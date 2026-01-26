"""Argument parsing and normalization utilities."""

from typing import Any, Dict

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
