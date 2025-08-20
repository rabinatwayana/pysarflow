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

import os
import esa_snappy
from esa_snappy import GPF
from esa_snappy import ProductIO, GeoPos, PixelPos, WKTReader
from esa_snappy import HashMap
from esa_snappy import jpy
from time import *
import numpy as np
import math
from pathlib import Path
from shapely.geometry import Point, Polygon, box
from shapely import wkt as _wkt
import geopandas as gpd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from datetime import datetime


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

def burst_for_geometry(product, safe_dir, geom, subswath=None):
    """
    Determine TOPS burst index (or range) for a geometry in a Sentinel-1 IW SLC.

    Args:
      product   : SNAP Product (ProductIO.readProduct(...))
      safe_dir  : path to the *.SAFE folder (str or Path)
      geom      : Shapely Point/Polygon, or WKT string, or bbox tuple (minlon,minlat,maxlon,maxlat)
      subswath  : optional 'IW1'|'IW2'|'IW3' to force a swath

    Returns (dict):
      {
        'swath': 'IW1'|'IW2'|'IW3',
        'band_name': 'Intensity_IW1_VV',   # band used
        'linesPerBurst': 1495,
        'numberOfBursts': 10,
        'burst': 5,              # for Point
        'firstBurst': 4,         # for Polygon
        'lastBurst': 6,          # for Polygon
        'geom_type': 'Point'|'Polygon'
      }
    """

    # --- normalize geometry ---
    if isinstance(geom, (list, tuple)) and len(geom) == 4:
        geom = box(*geom)
    elif isinstance(geom, str):
        geom = _wkt.loads(geom)
    elif not isinstance(geom, (Point, Polygon)):
        raise ValueError("geom must be Point/Polygon, WKT string, or bbox tuple")

    is_point = isinstance(geom, Point)
    aoi = geom if not is_point else geom

    names = list(product.getBandNames())

    # --- pick a band that actually covers the geometry (auto-detect swath if needed) ---
    def band_contains_point(band, lat, lon):
        gc = band.getGeoCoding()
        if gc is None: return False, None
        pp = gc.getPixelPos(GeoPos(float(lat), float(lon)), None)
        ok = (pp is not None and math.isfinite(pp.x) and math.isfinite(pp.y)
              and 0 <= pp.x < band.getRasterWidth()
              and 0 <= pp.y < band.getRasterHeight())
        return ok, pp

    # center we’ll use to probe coverage
    probe = (aoi.y, aoi.x) if is_point else (aoi.centroid.y, aoi.centroid.x)

    sw_order = [subswath] if subswath in ("IW1", "IW2", "IW3") else ["IW1", "IW2", "IW3"]
    chosen_sw, chosen_band_name, chosen_pp = None, None, None

    for sw in sw_order:
        cands = [n for n in names if sw in n]
        for name in cands:
            ok, pp = band_contains_point(product.getBand(name), *probe)
            if ok:
                chosen_sw, chosen_band_name, chosen_pp = sw, name, pp
                break
        if chosen_sw:
            break

    if chosen_sw is None:
        raise ValueError("Geometry is outside IW1/IW2/IW3 coverage for this product.")

    band = product.getBand(chosen_band_name)
    gc = band.getGeoCoding()
    W, H = band.getRasterWidth(), band.getRasterHeight()

    # --- read annotation XML for this swath (namespace-agnostic) ---
    safe_dir = Path(safe_dir)
    ann_dir = safe_dir / "annotation"
    xml_files = sorted(ann_dir.glob(f"*{chosen_sw.lower()}*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No annotation XML found for {chosen_sw} in {ann_dir}")
    root = ET.parse(xml_files[0]).getroot()

    def find_text_any(root, local):
        for el in root.iter():
            if el.tag.rsplit('}', 1)[-1] == local and el.text:
                return el.text.strip()
        return None

    txt_lpb = find_text_any(root, "linesPerBurst")
    bursts_nodes = [el for el in root.iter() if el.tag.rsplit('}', 1)[-1] == "burst"]
    numberOfBursts = int(find_text_any(root, "numberOfBursts")) if find_text_any(root, "numberOfBursts") else len(bursts_nodes)

    if txt_lpb:
        linesPerBurst = int(txt_lpb)
    else:
        # try infer from firstAzimuthLine step, else fallback to H / numBursts
        first_lines = []
        for b in bursts_nodes[:2]:
            for el in b.iter():
                if el.tag.rsplit('}', 1)[-1] == "firstAzimuthLine" and el.text:
                    first_lines.append(int(el.text)); break
        linesPerBurst = max(1, (first_lines[1] - first_lines[0]) if len(first_lines) == 2 else H // max(1, numberOfBursts))

    out = {
        "swath": chosen_sw,
        "band_name": chosen_band_name,
        "linesPerBurst": linesPerBurst,
        "numberOfBursts": numberOfBursts,
        "geom_type": "Point" if is_point else "Polygon",
    }

    # --- map geometry to pixel rows and compute burst index / range ---
    def clamp_b(b): return max(1, min(numberOfBursts, b))

    if is_point:
        y = int(chosen_pp.y)
        out["burst"] = clamp_b(y // linesPerBurst + 1)
    else:
        # use exterior vertices (and centroid) to get min/max valid rows
        ys = []
        for lon, lat in list(aoi.exterior.coords) + [(aoi.centroid.x, aoi.centroid.y)]:
            pp = gc.getPixelPos(GeoPos(float(lat), float(lon)), None)
            if (pp is not None and math.isfinite(pp.y) and 0 <= pp.y < H):
                ys.append(pp.y)
        if not ys:
            raise ValueError("AOI polygon does not intersect the chosen sub-swath.")
        y_min, y_max = min(ys), max(ys)
        out["firstBurst"] = clamp_b(int(y_min) // linesPerBurst + 1)
        out["lastBurst"]  = clamp_b(int(y_max) // linesPerBurst + 1)

    return out

def topsar_split(product, burst_dict, pols=None, output_complex=True):
    """
    Run TOPSAR-Split using burst indices from burst_for_geometry(...).
    """
    Integer = jpy.get_type('java.lang.Integer')

    band = burst_dict['band_name']              # e.g. 'i_IW1_VH'
    swath = next(iw for iw in ['IW1','IW2','IW3','EW1','EW2','EW3','EW4','EW5'] if iw in band)

    if pols is None:
        pols = band.split('_')[-1]              # infer polarisation from band name

    if burst_dict['geom_type'] == 'Point':
        fb = lb = int(burst_dict['burst'])
    else:
        fb, lb = int(burst_dict['firstBurst']), int(burst_dict['lastBurst'])
        if fb > lb: fb, lb = lb, fb

    # Build parameters
    params = HashMap()
    params.put('subswath', swath)
    params.put('selectedPolarisations', pols)
    params.put('firstBurstIndex', Integer(fb))
    params.put('lastBurstIndex', Integer(lb))
    params.put('outputComplex', bool(output_complex))

    # Run TOPSAR-Split
    output = GPF.createProduct("TOPSAR-Split", params, product)
    print(f"TOPSAR-Split applied: {swath} bursts {fb}–{lb} ({pols})")
    return output


def apply_orbit(product,
                orbit_type="Sentinel Precise (Auto Download)"):
    """
    Run SNAP 'Apply-Orbit-File' on a Product.

    orbit_type options commonly used:
      - "Sentinel Precise (Auto Download)"   # preferred
      - "Sentinel Restituted (Auto Download)"# fallback if precise not yet available
      - "DORIS Precise VOR (ENVISAT)" etc.   # for other missions

    Returns a new Product with orbits applied.
    """
    Boolean = jpy.get_type('java.lang.Boolean')
    Integer = jpy.get_type('java.lang.Integer')

    params = HashMap()
    params.put("orbitType", orbit_type)
    params.put("polyDegree", Integer(3))
    params.put("continueOnFail", Boolean(True))

    out = GPF.createProduct("Apply-Orbit-File", params, product)
    print(f"Apply-Orbit-File: {orbit_type}")
    return out

def back_geocoding(products, dem_name="SRTM 1Sec HGT", ext_dem=None):
    """
    Run SNAP Back-Geocoding on master + list of slaves.

    Args:
        master   : SNAP Product (Apply-Orbit-File already done)
        slaves   : list of SNAP Products (Apply-Orbit-File already done)
        dem_name : name of DEM in SNAP auxdata (default SRTM 1Sec HGT)
        ext_dem  : optional external DEM product

    Returns:
        SNAP Product with master + co-registered slaves
    """
    print("Running Back-Geocoding...")

    params = HashMap()
    params.put("demName", dem_name)
    params.put("demResamplingMethod", "BILINEAR_INTERPOLATION")
    params.put("resamplingType", "BILINEAR_INTERPOLATION")
    params.put("maskOutAreaWithoutElevation", True)
    params.put("outputDerampDemodPhase", True)
    params.put("disableReramp", False)


    if ext_dem is not None:
        params.put("externalDEMFile", ext_dem)

    output = GPF.createProduct("Back-Geocoding", params, products) 
    print("Back geocoding applied!")
    return output

def run_esd(product, preset="default", **overrides):
    """
    Enhanced Spectral Diversity (ESD) with sensible defaults + optional overrides.
    - product: Back-Geocoding output (master + slave(s))
    - preset:  'default' | 'robust' | 'fast'
    - overrides: any operator key you want to force
    """
    params = HashMap()
    for k, v in esd_params(preset).items():
        params.put(k, v)
    # user overrides win if provided
    for k, v in overrides.items():
        params.put(k, v)
    esd = GPF.createProduct("Enhanced-Spectral-Diversity", params, product)
    return esd


def esd_params(preset):
    Boolean = jpy.get_type('java.lang.Boolean')
    if preset == "robust":
        # more forgiving in low coherence, a bit slower
        return {
            "cohThreshold": 0.15,            # default often ~0.2
            "xCorrThreshold": 0.05,
            "fineWinWidthStr": "512",
            "fineWinHeightStr": "512",
            "fineWinAccAzimuth": "16",
            "fineWinAccRange": "16",
            "fineWinOversampling": "128",
            "estimateAzimuthShift": Boolean(True),
            "estimateRangeShift":   Boolean(False),
            "doNotWriteTargetBands": Boolean(False),
        }
    if preset == "fast":
        # quicker; good for previews
        return {
            "cohThreshold": 0.2,
            "xCorrThreshold": 0.1,
            "fineWinWidthStr": "256",
            "fineWinHeightStr": "256",
            "estimateAzimuthShift": Boolean(True),
            "estimateRangeShift":   Boolean(False),
            "doNotWriteTargetBands": Boolean(True),  # smaller output
        }
    # default
    return {
        # let SNAP defaults mostly apply; set only stable keys
        "estimateAzimuthShift": Boolean(True),
        "estimateRangeShift":   Boolean(False),
        "doNotWriteTargetBands": Boolean(False),
    }


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

def topsar_deburst(product, polarization):  
    """
    Apply TOPSAR deburst operation to a Sentinel-1 product.

    This function removes burst discontinuities in TOPSAR acquisitions 
    by merging bursts into a seamless image for the specified polarization. 
    It is a necessary preprocessing step for Sentinel-1 TOPSAR IW and EW 
    data before further interferometric or geocoding analysis.

    Parameters
    ----------
    product : snappy.Product
        The input Sentinel-1 product to which the deburst operation will be applied.
    polarization : str
        The polarization channel to process (e.g., 'VV', 'VH', 'HH', 'HV').

    Returns
    -------
    snappy.Product
        The deburst-processed Sentinel-1 product.
    """
    parameters = HashMap()
    print('Apply TOPSAR Deburst...')
    parameters.put("Polarisations", polarization)
    output = GPF.createProduct("TOPSAR-Deburst", parameters, product)
    print("TOPSAR Deburst applied!")
    return output
  
  
def topsar_deburst(product, polarization):  
    """
    Apply TOPSAR deburst operation to a Sentinel-1 product.

    This function removes burst discontinuities in TOPSAR acquisitions 
    by merging bursts into a seamless image for the specified polarization. 
    It is a necessary preprocessing step for Sentinel-1 TOPSAR IW and EW 
    data before further interferometric or geocoding analysis.

    Parameters
    ----------
    product : snappy.Product
        The input Sentinel-1 product to which the deburst operation will be applied.
    polarization : str
        The polarization channel to process (e.g., 'VV', 'VH', 'HH', 'HV').

    Returns
    -------
    snappy.Product
        The deburst-processed Sentinel-1 product.
    """
    parameters = HashMap()
    print('Apply TOPSAR Deburst...')
    parameters.put("Polarisations", polarization)
    output = GPF.createProduct("TOPSAR-Deburst", parameters, product)
    print("TOPSAR Deburst applied!")
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

def terrain_correction(product, DEM):
    """
    Apply terrain correction to a SAR product using a specified DEM.

    Terrain correction removes geometric distortions caused by topography and sensor 
    viewing geometry. This step geocodes the image into a map coordinate system 
    and ensures that pixel locations align with their true geographic position.

    Parameters
    ----------
    product : snappy.Product
        The SAR product to which terrain correction will be applied.
    DEM : str
        The name of the Digital Elevation Model (e.g., 'SRTM 3Sec' or a custom DEM) 
        to be used for terrain correction.

    Returns
    -------
    snappy.Product
        The terrain-corrected product.

    Notes
    -----
    - The DEM is saved as part of the output product.
    - Areas with missing DEM values are assigned an external no-data value (0.0).
    - This step is typically performed near the end of the preprocessing chain to 
      produce a geocoded product suitable for analysis and visualization.
    """
    parameters = HashMap()
    print('Applying Terrain Correction...')
    parameters.put('demName', DEM)
    parameters.put('saveDEM', True)
    parameters.put('externalDEMNoDataValue', 0.0)
    output = GPF.createProduct("Terrain-Correction", parameters, product)
    print("Terrain Correction applied!")
    return output
