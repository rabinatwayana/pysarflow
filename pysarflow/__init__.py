# -*- coding: utf-8 -*-
"""
pysarflow package initialization.

This package provides utility functions for processing
Synthetic Aperture Radar (SAR) data.

It includes modules for handling Ground Range Detected (GRD)
and Single Look Complex (SLC) data formats.

Modules:
- grd: Contains functions for GRD data processing.
- slc: Contains functions for SLC data processing.

"""

from .grd import read_grd_product
from .slc import read_slc_product

__all__ = ["read_grd_product"]
