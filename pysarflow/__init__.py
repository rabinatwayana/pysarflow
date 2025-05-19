# -*- coding: utf-8 -*-
"""
pysarflow package initialization.

This package provides utility functions for processing
Synthetic Aperture Radar (SAR) data.

It includes modules for handling Ground Range Detected (GRD)
and Single Look Complex (SLC) data formats.

Modules:
- grd: Contains functions for GRD data processing (e.g., summing GRD images).
- slc: Contains functions for SLC data processing (e.g., summing SLC images).

"""

from .grd import sum_grd
from .slc import sum_slc

__all__ = ["sum_grd", "sum_slc"]
