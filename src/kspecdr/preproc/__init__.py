"""
Preprocessing Module

This module contains preprocessing functions for astronomical data reduction.
It provides tools for converting raw instrument data into calibrated image files.
"""

from .make_im import MakeIM, make_im

__all__ = ['MakeIM', 'make_im'] 