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

import os
import esa_snappy
import numpy as np
import matplotlib.pyplot as plt

from esa_snappy import Product, ProductIO, ProductUtils, WKTReader, HashMap, GPF, jpy


# Loads the SNAP operators globally when the module is imported
GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()

def read_SAFE_product(file_path):
    """
    Reads a Sentinel-1 SAR GRD product from a .SAFE directory or a .zip archive.

    This function utilizes ESA SNAP's `ProductIO.readProduct` to load the SAR data
    into a SNAP Product object, which is the base object for further processing.

    Args:
        file_path (str): The file path to the Sentinel-1 .SAFE directory (unzipped)
                        or the .zip archive containing the GRD product.

    Returns:
        esa_snappy.Product: A SNAP Product object representing the loaded SAR data.

    Raises:
        FileNotFoundError: If the specified zip_path does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Product path not found: {file_path}")
    
    print(f"Reading SAR product from: {file_path}...")
    product = ProductIO.readProduct(file_path)
    print("\tProduct read successfully.")
    return product