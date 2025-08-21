# -*- coding: utf-8 -*-
"""Preprocessing Sentinel1 GRD Data.

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
import numpy as np
import matplotlib.pyplot as plt
from esa_snappy import ProductIO, WKTReader, HashMap, GPF, jpy
from .utils import extract_bbox, convert_0_to_nan

# Loads the SNAP operators globally when the module is imported
GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()


def read_grd_product(file_path):
    """
    Read a Sentinel-1 SAR GRD product from a .SAFE directory or a .zip archive.

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


def subset_AOI(product, bbox=[], file_path=None):
    """
    Subset the product to a specific AOI to reduce resource usage for large images.

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
        geometry_wkt = extract_bbox(file_path=file_path)
    else:
        raise Exception("Either bbox or file_path should be provided")

    print("\tSubsetting using bounding box:", bbox)

    geometry = WKTReader().read(geometry_wkt)
    HashMap = jpy.get_type("java.util.HashMap")
    GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()
    parameters = HashMap()
    parameters.put("copyMetadata", True)
    parameters.put("geoRegion", geometry)
    output = GPF.createProduct("Subset", parameters, product)

    width = output.getSceneRasterWidth()
    height = output.getSceneRasterHeight()

    print("Columns (width):", width)
    print("Rows (height):", height)

    print("\tProduct subsetted.")
    return output


def apply_orbit_file(product):
    """
    Apply precise satellite orbit file to the SAR product.

    This operation orthorectifies the SAR product to improve accuracy by using
    precise orbit information.

    Args:
        product (esa_snappy.Product): The input SAR Product object.

    Returns:
        esa_snappy.Product: The product with orbit file applied.
    """
    print("\tApplying Orbit File...")
    parameters = HashMap()
    parameters.put(
        "orbitType", "Sentinel Precise (Auto Download)"
    )  # 'Sentinel Precise (Auto Download) specifically for Sentinel-1
    parameters.put(
        "continueOnFail", "false"
    )  # Do not continue if orbit file application fails

    output = GPF.createProduct("Apply-Orbit-File", parameters, product)
    print("\tOrbit File applied.")
    return output


def thermal_noise_removal(product):
    """
    Remove thermal noise from the SAR product.

    Thermal noise is a constant noise floor that affects SAR images, especially
    in low-backscatter areas. This step removes this noise, improving the signal-to-noise ratio.

    Args:
        product (esa_snappy.Product): The input SAR Product object.

    Returns:
        esa_snappy.Product: The product after thermal noise removal.
    """
    print("\tPerforming thermal noise removal...")
    parameters = HashMap()
    parameters.put("removeThermalNoise", True)
    output = GPF.createProduct("ThermalNoiseRemoval", parameters, product)
    print("\tThermal noise removed.")
    return output


def border_noise_removal(product):
    """Remove border noise from the SAR product.

    Border noise appears as a low-backscatter band along the image edges
    and can affect SAR measurements near scene boundaries. This operation
    trims the noisy border region to improve data quality.

    Args:
        product (esa_snappy.Product): The input SAR Product object.

    Returns:
        esa_snappy.Product: The product after border noise removal.
    """
    print("\tPerforming border noise removal...")
    parameters = HashMap()
    parameters.put("borderLimit", "500")
    parameters.put("trimThreshold", "0.5")
    output = GPF.createProduct("Remove-GRD-Border-Noise", parameters, product)
    print("\tBorder noise removed.")
    return output


def radiometric_calibration(product, polarization, pols_selected):
    """
    Perform radiometric calibration on the SAR product.

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
    print(f"\tRadiometric calibration for polarization(s): {pols_selected}...")
    parameters = HashMap()
    parameters.put("outputSigmaBand", True)
    parameters.put("outputImageScaleInDb", False)  # Output linear scale, not dB

    # Determine source bands based on the input polarization type
    if polarization == "DH":  # Dual-horizontal: HH, HV
        parameters.put("sourceBands", "Intensity_HH,Intensity_HV")
    elif polarization == "DV":  # Dual-vertical: VH, VV
        parameters.put("sourceBands", "Intensity_VH,Intensity_VV")
    elif polarization == "SH" or polarization == "HH":  # Single-horizontal: HH
        parameters.put("sourceBands", "Intensity_HH")
    elif polarization == "SV" or polarization == "VV":  # Single-vertical: VV
        parameters.put("sourceBands", "Intensity_VV")
    else:
        raise ValueError(
            f"Unsupported polarization type: {polarization}. "
            "Please use 'DH', 'DV', 'SH'/'HH', or 'SV'/'VV'."
        )

    # This parameter directly controls which output bands are generated
    parameters.put("selectedPolarisations", pols_selected)

    output = GPF.createProduct("Calibration", parameters, product)
    print("\tRadiometric calibration completed.")
    return output


def speckle_filter(product, filterSizeY="5", filterSizeX="5", filter="Lee"):
    """Apply speckle filtering to a SAR product using the specified filter parameters.

    This function uses the Sentinel-1 toolbox's Speckle-Filter operator via GPF
    (Graph Processing Framework) to reduce speckle noise in SAR imagery.

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

    Returns:
    --------
    Product
        The filtered SAR product after applying the speckle filter.
    """

    band_name = list(product.getBandNames())
    parameters = HashMap()
    parameters.put("sourceBands", band_name[0])
    parameters.put("filter", filter)
    parameters.put("filterSizeX", filterSizeX)
    parameters.put("filterSizeY", filterSizeY)
    parameters.put("dampingFactor", "2")
    parameters.put("estimateENL", "true")
    parameters.put("enl", "1.0")
    parameters.put("numLooksStr", "1")
    parameters.put("targetWindowSizeStr", "3x3")
    parameters.put("sigmaStr", "0.9")
    parameters.put("anSize", "50")
    speckle_filter_output = GPF.createProduct("Speckle-Filter", parameters, product)
    width = speckle_filter_output.getSceneRasterWidth()
    height = speckle_filter_output.getSceneRasterHeight()

    print("Columns (width):", width)
    print("Rows (height):", height)

    print("\tSpeckle filter completed.")
    return speckle_filter_output


def terrain_correction(
    product,
    demName="SRTM 3Sec",
    pixelSpacingInMeter=10.0,
    demResamplingMethod="BILINEAR_INTERPOLATION",
    imgResamplingMethod="BILINEAR_INTERPOLATION",
):
    """
    Performs terrain correction on a SAR product using a specified DEM.
    This step corrects geometric distortions caused by the side-looking
    geometry of SAR sensors and normalizes pixel spacing.

    Args:
        product : Product
            Input SAR product.
        demName : str, optional
            DEM source to use (e.g., 'SRTM 3Sec', 'Copernicus 30m').
        pixelSpacingInMeter : float, optional
            Output pixel spacing in meters. Default is 10.0.
        demResamplingMethod : str, optional
            Resampling method for DEM
            (e.g., 'NEAREST_NEIGHBOUR', 'BILINEAR_INTERPOLATION', 'CUBIC_CONVOLUTION').
        imgResamplingMethod : str, optional
            Resampling method for image bands
            (e.g., 'NEAREST_NEIGHBOUR', 'BILINEAR_INTERPOLATION', 'CUBIC_CONVOLUTION').

    Returns:
        esa_snappy.Product: Terrain-corrected product.

    Raises:
        RuntimeError: If terrain correction fails during processing.
    """
    try:
        band_names = list(product.getBandNames())
        parameters = HashMap()
        parameters.put("demName", demName)
        parameters.put("nodataValueAtSea", False)
        parameters.put("sourceBands", band_names[0])
        parameters.put("saveDem", False)  # Avoid saving DEM band
        parameters.put("saveLatLon", False)
        parameters.put(
            "imgResamplingMethod", imgResamplingMethod
        )  # Ensure smooth interpolation
        parameters.put(
            "pixelSpacingInMeter", pixelSpacingInMeter
        )  # Ensure smooth interpolation
        parameters.put(
            "demResamplingMethod", demResamplingMethod
        )  # Ensure smooth interpolation
        parameters.put("noDataValue", -9999.0)
        tc_output = GPF.createProduct("Terrain-Correction", parameters, product)
        tc_output = convert_0_to_nan(tc_output)
        # width  = tc_output.getSceneRasterWidth()
        # height = tc_output.getSceneRasterHeight()
        # print("Columns (width):", width)
        # print("Rows (height):", height)
        print("\tTerrain correction completed.")
        return tc_output
    except Exception as e:
        raise RuntimeError(f"Terrain correction failed: {e}")


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
    output = GPF.createProduct("linearToFromdB", parameters, product)
    print("\tConversion complete.")
    return output


def maskPermanentWater(product):
    """
    Masks permanent water areas in a SAR product using land cover data.

    The function first adds a land cover band to the input product, then creates a
    binary water mask based on GlobCover classification (210 = water). Finally, it
    applies the mask to the SAR product, creating a permanent water masked while
    preserving NoData values.

    Args:
        product (esa_snappy.Product): Input SAR product.

    Returns:
        esa_snappy.Product: SAR product with permanent water masked, containing
                            a 'Permanent_Water_Masked' band.

    Raises:
        RuntimeError: If masking or band math operations fail.
    """
    try:
        print("Masking permanent waster based on landcover data")
        # Add land cover band
        parameters = HashMap()
        parameters.put("landCoverNames", "GlobCover")
        mask_with_land_cover = GPF.createProduct("AddLandCover", parameters, product)
        del parameters

        # Create binary water band
        BandDescriptor = jpy.get_type(
            "org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor"
        )
        parameters = HashMap()
        targetBand = BandDescriptor()
        targetBand.name = "BinaryWater"
        targetBand.type = "uint8"
        targetBand.expression = "(land_cover_GlobCover == 210) ? 0 : 1"
        targetBands = jpy.array(
            "org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor", 1
        )
        targetBands[0] = targetBand
        parameters.put("targetBands", targetBands)
        water_mask = GPF.createProduct("BandMaths", parameters, mask_with_land_cover)

        del parameters
        parameters = HashMap()
        BandDescriptor = jpy.get_type(
            "org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor"
        )
        water_mask.addBand(product.getBand("Sigma0_VV"))
        targetBand = BandDescriptor()
        targetBand.name = "Permanent_Water_Masked"
        targetBand.type = "float32"
        targetBand.expression = (
            "(Sigma0_VV == -9999.0) ? -9999.0 : ((BinaryWater == 1) ? Sigma0_VV : 0)"
        )
        # targetBand.expression = '(BinaryWater == 1) ? Sigma0_VV : 0'
        # targetBand.expression = '(BinaryWater == 1) ? Difference_Band : 0'

        targetBands = jpy.array(
            "org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor", 1
        )
        targetBands[0] = targetBand
        parameters.put("targetBands", targetBands)
        product_masked = GPF.createProduct("BandMaths", parameters, water_mask)
        print("\tMasking permanent water complete.")
        return product_masked
    except Exception as e:
        raise RuntimeError(f"Terrain correction failed: {e}")


def export(product, output_path) -> None:
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
    print("\tExport complete.")


def stack(master_product, slave_product):
    """
    Collocates two SAR products (master and slave) into a single stacked product.

    This creates a new product where the bands from the master and slave products
    are aligned spatially. Optionally, the master and slave components can be renamed
    to avoid conflicts.

    Args:
        master_product (esa_snappy.Product): The reference SAR product.
        slave_product (esa_snappy.Product): The secondary SAR product to
        be collocated with the master.

    Returns:
        esa_snappy.Product: A new product containing collocated bands from both master and slave products.

    Raises:
        RuntimeError: If the Collocate operation fails.
    """
    parameters = HashMap()
    parameters.put("targetProductType", "Collocated")
    parameters.put("resamplingType", "NEAREST_NEIGHBOUR")  # or 'Bilinear', 'Bicubic'
    parameters.put("renameMasterComponents", True)
    parameters.put("renameSlaveComponents", True)

    stacked = GPF.createProduct(
        "Collocate", parameters, [master_product, slave_product]
    )
    return stacked


def band_difference(product_stacked):
    """
    Computes the difference between two bands of a stacked SAR product.

    A new band named 'Difference_Band' is created, calculated as:
        Difference_Band = band2 - band1

    Args:
        product_stacked (esa_snappy.Product): Input SAR product with at least two bands (e.g., pre- and post-event).

    Returns:
        esa_snappy.Product: A new product containing the 'Difference_Band'.

    Raises:
        RuntimeError: If the BandMaths operation fails.
    """

    try:
        band_names = list(product_stacked.getBandNames())
        BandDescriptor = jpy.get_type(
            "org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor"
        )

        # Create a BandDescriptor object
        band_def = BandDescriptor()
        band_def.name = "Difference_Band"
        band_def.type = "float32"
        band_def.expression = f"{band_names[1]} - {band_names[0]}"  # Ensure these band names exist in product_stacked
        band_def.noDataValue = 0.0
        band_def.description = "Post - Pre difference"

        # Create a Java array of BandDescriptor
        Array = jpy.array("org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor", 1)
        targetBands = Array
        targetBands[0] = band_def  # Assign band_def to the first index of the array

        # Parameters HashMap
        parameters = jpy.get_type("java.util.HashMap")()
        parameters.put("targetBands", targetBands)

        # Run BandMaths
        diff_product = GPF.createProduct("BandMaths", parameters, product_stacked)
        return diff_product
    except Exception as e:
        raise RuntimeError(f"Terrain correction failed: {e}")


def plotBand(
    product1, band_name1, vmin=None, vmax=None, cmap=plt.cm.binary, figsize=(10, 10)
):
    """
    Plots one or two SNAP Product bands side by side.
    Axes ticks are shown, and each subplot has its own title.

    Args:
        product1: First SNAP Product.
        band_name1: Band name for first product.
        product2: Optional second SNAP Product.
        band_name2: Band name for second product.
        vmin, vmax: Colormap limits.
        cmap: Colormap.
        figsize: Figure size.

    Returns:
        list of matplotlib.image.AxesImage: The image plot objects.
    """

    def get_band_data(product, band_name):
        band = product.getBand(band_name)
        if band is None:
            raise ValueError(f"Band '{band_name}' not found in product.")
        w, h = band.getRasterWidth(), band.getRasterHeight()
        data = np.zeros(w * h, np.float32)
        band.readPixels(0, 0, w, h, data)
        data.shape = (h, w)
        return data

    data1 = get_band_data(product1, band_name1)

    plt.figure(figsize=figsize)
    img1 = plt.imshow(data1, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xticks()
    plt.yticks()
    plt.title(f"Band: {band_name1}")
    plt.show()
    return [img1]


def generateFloodMask(product_masked, threshold=0.1):
    """
    Generates a flood mask from a masked difference band in a SAR product.

    The function creates a new band called 'flooded', where pixels are classified as:
        - -9999.0: NoData
        - 0: Permanent water
        - 1: Flooded
        - 2: Others

    Args:
        product_masked (esa_snappy.Product): Input SAR product with masked difference band.
        threshold (float, optional): Threshold for identifying flooded pixels. Default is 0.1.

    Returns:
        esa_snappy.Product: A new product with a 'flooded' band representing the flood mask.

    Raises:
        RuntimeError: If the BandMaths operation fails.
    """
    try:
        parameters = HashMap()
        BandDescriptor = jpy.get_type(
            "org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor"
        )

        targetBand = BandDescriptor()
        targetBand.name = "flooded"
        targetBand.type = "float32"

        # Threshold the masked difference band
        targetBand.expression = f"(Permanent_Water_Masked == -9999.0 ? -9999.0 : (Permanent_Water_Masked == 0 ? 0 : (Permanent_Water_Masked > {threshold} ? 1 : 2)))"

        targetBands = jpy.array(
            "org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor", 1
        )
        targetBands[0] = targetBand
        parameters.put("targetBands", targetBands)

        binary_flood = GPF.createProduct("BandMaths", parameters, product_masked)
        return binary_flood
    except Exception as e:
        raise RuntimeError(f"Terrain correction failed: {e}")


def preprocess_grd_product(file_path, config):
    """
    Preprocess a Sentinel-1 GRD product using a given configuration.

    The function performs a series of preprocessing steps including AOI subsetting,
    orbit file application, thermal and border noise removal, radiometric calibration,
    speckle filtering, and terrain correction.

    Args:
        file_path (str): Path to the Sentinel-1 GRD product file.
        config (dict): Configuration dictionary containing preprocessing parameters, such as:
            - aoi_bbox (list): Bounding box [minLon, minLat, maxLon, maxLat].
            - polarization (str, optional): Polarization to calibrate,
              e.g., "VV" or "VH". Defaults to "VV".
            - pols_selected (list, optional): List of polarizations to include. Defaults to None.
            - speckle_filter (str, optional): Speckle filter type. Defaults to "Lee".
            - filterSizeX (int or str, optional): Filter kernel size in X direction. Defaults to 5.
            - filterSizeY (int or str, optional): Filter kernel size in Y direction. Defaults to 5.
            - demName (str, optional): DEM name for terrain correction. Defaults to "SRTM 3Sec".
            - pixelSpacingInMeter (float or str, optional): Pixel spacing for terrain correction.
              Defaults to 10.0.
            - demResamplingMethod (str, optional): Resampling method for DEM.
              Defaults to "BILINEAR_INTERPOLATION".
            - imgResamplingMethod (str, optional): Resampling method for image.
              Defaults to "BILINEAR_INTERPOLATION".

    Returns:
        esa_snappy.Product: Preprocessed Sentinel-1 product.

    Raises:
        RuntimeError: If any step of the preprocessing fails.
    """
    try:
        aoi_bbox = config["aoi_bbox"]
        polarization = config.get("polarization", "VV")
        pols_selected = config.get("pols_selected", None)
        filter = config.get("speckle_filter", "Lee")
        filterSizeX = config.get("filterSizeX", "5")
        filterSizeY = config.get("filterSizeY", "5")
        demName = config.get("demName", "SRTM 3Sec")
        pixelSpacingInMeter = config.get("pixelSpacingInMeter", "10.0")
        demResamplingMethod = config.get(
            "demResamplingMethod", "BILINEAR_INTERPOLATION"
        )
        imgResamplingMethod = config.get(
            "imgResamplingMethod", "BILINEAR_INTERPOLATION"
        )

        product = read_grd_product(file_path)

        # 1. Subset AOI (optional)
        if aoi_bbox:
            product = subset_AOI(product=product, bbox=config["aoi_bbox"])

        # 2. Apply corrections
        product = apply_orbit_file(product)
        product = thermal_noise_removal(product)
        product = border_noise_removal(product)

        # 3. Radiometric Calibration
        product = radiometric_calibration(
            product, polarization=polarization, pols_selected=pols_selected
        )

        # 4. Speckle Filter
        product = speckle_filter(
            product,
            filter=filter,
            filterSizeX=filterSizeX,
            filterSizeY=filterSizeY,
        )

        # 5. Terrain Correction
        product = terrain_correction(
            product,
            demName=demName,
            pixelSpacingInMeter=pixelSpacingInMeter,
            demResamplingMethod=demResamplingMethod,
            imgResamplingMethod=imgResamplingMethod,
        )

        return product
    except Exception as e:
        raise RuntimeError(f"Terrain correction failed: {e}")
