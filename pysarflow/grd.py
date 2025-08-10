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


def apply_orbit_file(product):
    """
    Applies precise satellite orbit file to the SAR product.

    This operation orthorectifies the SAR product to improve accuracy by using 
    precise orbit information.

    Args:
        product (esa_snappy.Product): The input SAR Product object.

    Returns:
        esa_snappy.Product: The product with orbit file applied.
    """
    print('\tApplying Orbit File...')
    parameters = HashMap() 
    parameters.put('orbitType', 'Sentinel Precise (Auto Download)') # 'Sentinel Precise (Auto Download) specifically for Sentinel-1
    parameters.put('continueOnFail', 'false') # Do not continue if orbit file application fails
    
    output = GPF.createProduct('Apply-Orbit-File', parameters, product)
    print('\tOrbit File applied.')
    return output

def border_noise_removal(product) :
    """
    Removes border noise from the SAR product.

    Border noise appears as a low-backscatter band along the image edges 
    and can affect SAR measurements near scene boundaries. This operation 
    trims the noisy border region to improve data quality.

    Args:
        product (esa_snappy.Product): The input SAR Product object.

    Returns:
        esa_snappy.Product: The product after border noise removal.
    """
    print('\tPerforming border noise removal...')
    parameters = HashMap()
    parameters.put('borderLimit', '500')
    parameters.put('trimThreshold', '0.5')
    output = GPF.createProduct('Remove-GRD-Border-Noise', parameters, product)
    print('\tBorder noise removed.')
    return output

def radiometric_calibration(product, polarization, pols_selected) :
    """
    Performs radiometric calibration on the SAR product.

    Calibration converts the raw SAR data into radar brightness values.

    Args:
        product (esa_snappy.Product): The input SAR Product object.
        polarization (str): the desired output polarization type.
        pols_selected (str): the polarizations to be calibrated

    Returns:
        esa_snappy.Product: The radiometrically calibrated SAR Product object.

    Raises:
        ValueError: If an unsupported 'polarization' type is provided.
    """
    print(f'\tRadiometric calibration for polarization(s): {pols_selected}...')
    parameters = HashMap()
    parameters.put('outputSigmaBand', True)
    parameters.put('outputImageScaleInDb', False) # Output linear scale, not dB

    # Determine source bands based on the input polarization type
    if polarization == 'DH':  # Dual-horizontal: HH, HV
        parameters.put('sourceBands', 'Intensity_HH,Intensity_HV')
    elif polarization == 'DV': # Dual-vertical: VH, VV
        parameters.put('sourceBands', 'Intensity_VH,Intensity_VV')
    elif polarization == 'SH' or polarization == 'HH': # Single-horizontal: HH
        parameters.put('sourceBands', 'Intensity_HH')
    elif polarization == 'SV' or polarization == 'VV': # Single-vertical: VV
        parameters.put('sourceBands', 'Intensity_VV')
    else:
        raise ValueError(f"Unsupported polarization type: {polarization}. "
                         "Please use 'DH', 'DV', 'SH'/'HH', or 'SV'/'VV'.")
    
    # This parameter directly controls which output bands are generated
    parameters.put('selectedPolarisations', pols_selected) 

    output = GPF.createProduct("Calibration", parameters, product)
    print('\tRadiometric calibration completed.')
    return output

def conversion_to_db(product):
    """
    Converts SAR backscatter values from linear scale to decibels (dB).

    This transformation is commonly applied to Sentinel-1 GRD products 
    after radiometric calibration, as dB units are easier to interpret 
    and compare across scenes.

    Args:
        product (snappy.Product): Input SAR product in linear scale.

    Returns:
        esa_snappy.Product: Product with backscatter values expressed in dB.
    """
    parameters = HashMap()
    output = GPF.createProduct('linearToFromdB', parameters, product)
    print('\tConversion complete.')
    return output

def export(Product, output_path) -> None:
    """
    Exports a SNAP product as a GeoTIFF file.

    Args:
        product (snappy.Product): The processed SNAP product to export.
        output_path (str): Destination .tif file path.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Exporting product to {output_path} (GeoTIFF)...")
    ProductIO.writeProduct(product, output_path, "GeoTIFF")
    print("Export complete.")