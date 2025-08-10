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
from .utils import extract_bbox

# Loads the SNAP operators globally when the module is imported
GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()

def read_grd_product(file_path):
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

def subset_AOI(product, bbox=[], file_path=None) :
    """
    The raw image is too large to process, theredore to reduce resources 
    required to process, the product is subset to a specific AOI.

    Args:
        product (esa_snappy.Product): Input SAR product.
        bbox (list): Bounding box as [minLon, minLat, maxLon, maxLat].

    Returns:
        esa_snappy.Product: Subsetted product.

    Raises:
        ValueError: If bbox is None or invalid.
        Exception: If both bbox or file_path is not passed
    """
    if bbox:
        if len(bbox) != 4:
            raise ValueError("bbox must be a list of [minLon, minLat, maxLon, maxLat]")
        geometry_wkt = (
            f"POLYGON(({bbox[0]} {bbox[1]}, {bbox[2]} {bbox[1]}, "
            f"{bbox[2]} {bbox[3]}, {bbox[0]} {bbox[3]}, {bbox[0]} {bbox[1]}))"
            )
    elif file_path:
        geometry_wkt=extract_bbox(file_path=file_path)
    else:
        raise Exception(f"Either bbox or file_path should be provided")

    print('\tSubsetting using bounding box:', bbox)

    geometry = WKTReader().read(geometry_wkt)
    HashMap = jpy.get_type('java.util.HashMap')
    GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()
    parameters = HashMap()
    parameters.put('copyMetadata', True)
    parameters.put('geoRegion', geometry)
    output = GPF.createProduct('Subset', parameters, product)
    print('\tProduct subsetted.')
    return output

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

def thermal_noise_removal(product) :
    """
    Removes thermal noise from the SAR product.

    Thermal noise is a constant noise floor that affects SAR images, especially
    in low-backscatter areas. This step removes this noise, improving the signal-to-noise ratio.

    Args:
        product (esa_snappy.Product): The input SAR Product object.

    Returns:
        esa_snappy.Product: The product after thermal noise removal.
    """
    print('\tPerforming thermal noise removal...')
    parameters = HashMap()
    parameters.put('removeThermalNoise', True)
    output = GPF.createProduct('ThermalNoiseRemoval', parameters, product)
    print('\tThermal noise removed.')
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

def speckle_filter(product, filterSizeY='5', filterSizeX='5', filter="Lee", dampingFactor='2', estimateENL='true', enl='1.0', numLooksStr='1',targetWindowSizeStr='3x3',sigmaStr='0.9',anSize='50'):
    """
    Apply speckle filtering to a SAR product using the specified filter parameters.

    This function uses the Sentinel-1 toolbox's Speckle-Filter operator via GPF(Graph Processing Framework) to reduce speckle noise in SAR imagery.

    Parameters:
    -----------
    product : Product
        The input SAR product to be filtered.

    filterSizeY : str, optional
        Filter size in the Y direction (default is '5').

    filterSizeX : str, optional
        Filter size in the X direction (default is '5').

    filter : str, optional
        Type of speckle filter to apply. Common options include "Lee", "Refined Lee", etc.
        Default is "Lee".

    dampingFactor : str, optional
        Damping factor used by the filter (default is '2').

    estimateENL : str, optional
        Whether to estimate the Equivalent Number of Looks (ENL) from the data.
        Accepts 'true' or 'false' (default is 'true').

    enl : str, optional
        ENL value to use if estimateENL is false (default is '1.0').

    numLooksStr : str, optional
        Number of looks in the data (default is '1').

    targetWindowSizeStr : str, optional
        Target window size for filtering, typically in the form '3x3' (default is '3x3').

    sigmaStr : str, optional
        Sigma parameter for filter sensitivity (default is '0.9').

    anSize : str, optional
        Analysis window size used in the filtering process (default is '50').

    Returns:
    --------
    Product
        The filtered SAR product after applying the speckle filter.
    """
    parameters = HashMap()
    parameters.put('sourceBands','Sigma0_VV')
    parameters.put('filter',filter)
    parameters.put('filterSizeX', filterSizeX)
    parameters.put('filterSizeY', filterSizeY)
    parameters.put('dampingFactor',dampingFactor)
    parameters.put('estimateENL',estimateENL)
    parameters.put('enl',enl)
    parameters.put('numLooksStr',numLooksStr)
    parameters.put('targetWindowSizeStr',targetWindowSizeStr)
    parameters.put('sigmaStr',sigmaStr)
    parameters.put('anSize',anSize)
    speckle_filter_output = GPF.createProduct('Speckle-Filter',parameters,product)
    print('\tSpeckle filter completed.')
    return speckle_filter_output


def terrain_correction(product,demName='SRTM 3Sec',pixelSpacingInMeter=10.0,sourceBands='Sigma0_VV'):
    parameters = HashMap()
    parameters.put('demName',demName)
    parameters.put('pixelSpacingInMeter', pixelSpacingInMeter)
    parameters.put('sourceBands',sourceBands)
    tc_output = GPF.createProduct("Terrain-Correction", parameters,product)
    print('\tTerrain correction completed.')
    return tc_output

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
