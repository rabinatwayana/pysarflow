# -*- coding: utf-8 -*-
"""
Preprocessing Sentinel1 GRD Data

This Python script enables users to perform essential preprocessing steps
on Sentinel-1 GRD data, including thermal noise removal, radiometric calibration,
and terrain correction.

It accepts Sentinel-1 products in the .SAFE format.

The script depends on the 'esa_snappy' library, which must be installed
in the Python environment where this script is executed.

This file can also be imported as a module and contains the following functions:
    Major:
        * read_grd_product
    Supporting:
        *
"""

from esa_snappy import ProductIO
import os


def read_grd_product(product_path):
    """
    Reads a Sentinel-1 GRD product using SNAP's ProductIO.

    This function checks whether the provided product path exists on disk, then
    attempts to load the product using ProductIO.readProduct. If the product
    cannot be read, it raises a RuntimeError with details about the failure.

    Args:
        product_path (str): Path to the Sentinel-1 SAFE format product directory or file.

    Returns:
        product: The SNAP product object representing the loaded Sentinel-1 data.

    Raises:
        FileNotFoundError: If the specified product path does not exist.
        RuntimeError: If reading the product fails due to unexpected errors.
    """
    try:
        if not os.path.exists(product_path):
            raise FileNotFoundError(f"Product path does not exist: {product_path}")
        product = ProductIO.readProduct(product_path)
        return product
    except Exception as e:
        raise RuntimeError(
            f"An error occurred while reading the grd product: {str(e)}"
        ) from e
