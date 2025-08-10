# -*- coding: utf-8 -*-
"""
Preprocessing Sentinel1 SLC Data

This Python script enables users to perform essential preprocessing steps
on Sentinel-1 SLC data, including thermal noise removal, radiometric calibration,
and terrain correction.

It accepts Sentinel-1 products in the .SAFE format.

The script depends on the 'esa_snappy' library, which must be installed
in the Python environment where this script is executed.

This file can also be imported as a module and contains the following functions:
    Major:
        * read_slc_product
    Supporting:
        *
"""

from esa_snappy import ProductIO
import os


def read_slc_product(product_path):
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
    

def temporal_baseline(product1_path, product2_path):
    """
    Calculate and print the temporal baseline between two Sentinel-1 products.

    The temporal baseline is defined as the absolute difference in days between 
    the acquisition times (start times) of the two SAR products. This metric is 
    commonly used in interferometric SAR (InSAR) analysis to assess the temporal 
    separation between image acquisitions.

    Parameters:
    ----------
    product1_path : str
        File path to the first Sentinel-1 product (e.g., the master image).
    product2_path : str
        File path to the second Sentinel-1 product (e.g., the slave image).

    Returns:
    It prints the temporal baseline
    None
        The function prints the temporal baseline in days and does not return a value.

    Notes:
    -----
    - This function uses `read_product` to open the Sentinel-1 products.
    - It assumes that both products contain valid start time metadata.
    - Resources are released after processing by calling `.dispose()` on each product.
    """
    product1 =  read_product(product1_path)
    product2 =  read_product(product2_path)
    master_time = product1.getStartTime()
    slave_time = product2.getStartTime()
    temporal_baseline = abs(slave_time.getMJD() - master_time.getMJD())

    print(f"Temporal Baseline: {temporal_baseline:.1f} days")

    product1.dispose()
    product2.dispose()
    return


def interferogram(product):
    """
    Generate an interferogram from a Sentinel-1 interferometric product.

    This function uses the SNAP Graph Processing Framework (GPF) to create 
    an interferogram, which represents the phase difference between two 
    co-registered SAR images. The interferogram is an essential step in 
    interferometric SAR (InSAR) processing for deriving surface deformation, 
    elevation models, or coherence analysis.

    Parameters:
    ----------
    product : org.esa.snap.core.datamodel.Product
        The co-registered Sentinel-1 product (usually the output of the 
        "Back-Geocoding" operator with two coregistered images).

    Returns:
    -------
    output : org.esa.snap.core.datamodel.Product
        The product containing the generated interferogram and optional 
        coherence band.

    Notes:
    -----
    - The function uses predefined parameters for flat-earth phase removal, 
      polynomial fitting, and orbit interpolation.
    - Coherence estimation is enabled by default.
    - Pixel size is set to be square.
    - Uncomment and customize the window size parameters if you want to control 
      the coherence estimation resolution.
    - This function is typically followed by Goldstein filtering and phase 
      unwrapping steps in an InSAR workflow.
    """
    parameters = HashMap()
    print('Creating interferogram ...')
    parameters.put("Subtract flat-earth phase", True)
    parameters.put("Degree of \"Flat Earth\" polynomial", 5)
    parameters.put("Number of \"Flat Earth\" estimation points", 501)
    parameters.put("Orbit interpolation degree", 3)
    parameters.put("Include coherence estimation", True)
    parameters.put("Square Pixel", True)
    parameters.put("Independent Window Sizes", False)
    #parameters.put("Coherence Azimuth Window Size", 10)
    #parameters.put("Coherence Range Window Size", 2)
    output = GPF.createProduct("Interferogram", parameters, product) 
    print("Interferogram created!")
    return output

def goldstein_phase_filtering(product):
    """
    Apply Goldstein Phase Filtering to an interferogram.

    Goldstein filtering is used in InSAR processing to enhance the signal-to-noise 
    ratio of the interferometric phase. This filter suppresses noise while preserving 
    the phase fringes, improving the quality of unwrapping and subsequent deformation 
    analysis.

    Parameters:
    ----------
    product : org.esa.snap.core.datamodel.Product
        The input product containing the interferometric phase, typically the 
        output of the "Interferogram" operator.

    Returns:
    -------
    output : org.esa.snap.core.datamodel.Product
        The product with Goldstein phase filtering applied.

    Notes:
    -----
    - `alpha`: Controls the filtering strength. Higher values result in stronger 
      filtering (default is 1.0).
    - `FFTSizeString`: Defines the FFT window size used for filtering (default is '64').
    - `windowSizeString`: Defines the size of the filtering window (default is '3').
    - `useCoherenceMask`: If set to True, filtering is applied only where coherence 
      exceeds the given threshold.
    - `coherenceThreshold`: Minimum coherence value used if coherence masking is enabled 
      (default is 0.2, but ignored if `useCoherenceMask` is False).

    This function is typically used after interferogram generation and before phase 
    unwrapping in an InSAR processing chain.
    """
    parameters = HashMap()
    print('Apply Goldstein Phase Filtering...')
    parameters.put('alpha', 1.0)
    parameters.put('FFTSizeString', '64')
    parameters.put('windowSizeString', '3')
    parameters.put('useCoherenceMask', False)
    parameters.put('coherenceThreshold', 0.2)  
    output = GPF.createProduct("GoldsteinPhaseFiltering", parameters, product)
    print("Goldstein Phase Filtering applied!")
    return output

def phase_to_elevation(product, DEM):
    """
    Convert unwrapped interferometric phase to elevation using a Digital Elevation Model (DEM).

    This function uses the SNAP Graph Processing Framework (GPF) to apply the 
    "Phase to Elevation" operator. It transforms unwrapped phase data into an 
    elevation map by referencing a known DEM. This step is commonly used in 
    Differential InSAR (DInSAR) or when generating DEMs from SAR data.

    Parameters:
    ----------
    product : org.esa.snap.core.datamodel.Product
        The input product containing unwrapped interferometric phase, typically 
        the result of a phase unwrapping operator.

    DEM : str
        The name of the DEM to use for reference (e.g., 'SRTM 3Sec', 'Copernicus 30m', 
        or path to an external DEM file).

    Returns:
    -------
    output : org.esa.snap.core.datamodel.Product
        The product containing the elevation map derived from phase data.

    Notes:
    -----
    - The DEM is used to geocode the elevation output and aid in the transformation.
    - Bilinear interpolation is used to resample the DEM for better accuracy.
    - `externalDEMNoDataValue` is set to 0.0, which defines how to handle no-data pixels 
      in the DEM.
    - This step assumes the input phase has already been unwrapped and filtered.
    """
   
    parameters = HashMap()
    print('Turning Phase to Elevation...')
    parameters.put('demName', DEM)
    parameters.put('demResamplingMethod', 'BILINEAR_INTERPOLATION')
    parameters.put('externalDEMNoDataValue', 0.0)
    output = GPF.createProduct("PhaseToElevation", parameters, product)
    print("Phase to Elevation applied!")
    return output