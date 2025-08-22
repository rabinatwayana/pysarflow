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
from matplotlib.colors import hsv_to_rgb
import subprocess

Integer = jpy.get_type('java.lang.Integer')


def read_slc_product(product_path):
    """
    Reads a Sentinel-1 SLC product using SNAP's ProductIO.

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
            f"An error occurred while reading the slc product: {str(e)}"
        ) from e

def burst_for_geometry(product, safe_dir, geom, subswath=None):
    """
    Determine TOPS burst index (or range) for a geometry in a Sentinel-1 IW SLC.

    Args:
      product   : Input SAR product.
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

    Raises:

        ValueError: 
        - If geom is not a Shapely Point/Polygon, a WKT string, or a 4-element bbox ``(minx, miny, maxx, maxy)``
        - If the geometry lies outside the products IW1/IW2/IW3 coverage.
        - If the AOI polygon does not intersect the chosen sub-swath.

        FileNotFoundError:
        - If no annotation XML file is found for the selected sub-swath

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
        "sub-swath": chosen_sw,
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

    Args:
      product : Input SAR product. Output of the burst_for_geometry
      burst_dict : output from the burst_for_geometry function
      pols = polarization

    Returns
        Product restricted to the specified sub-swath, burst range, and polarisations.
    """
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
    parameters = HashMap()
    parameters.put('subswath', swath)
    parameters.put('selectedPolarisations', pols)
    parameters.put('firstBurstIndex', Integer(fb))
    parameters.put('lastBurstIndex', Integer(lb))
    parameters.put('outputComplex', bool(output_complex))

    # Run TOPSAR-Split
    output = GPF.createProduct("TOPSAR-Split", parameters, product)
    print(f"TOPSAR-Split applied: {swath} bursts {fb}–{lb} ({pols})")
    return output


def apply_orbit(product,
                orbit_type="Sentinel Precise (Auto Download)"):
    """
    Run SNAP 'Apply-Orbit-File' on a Product.

    Args:
        product:
            Input product to process. Ouput of TOPSAR-split
        orbit_type:
            The orbit source/type to use. Common values include:
              - `"Sentinel Precise (Auto Download)"` – preferred.
              - `"Sentinel Restituted (Auto Download)"` – fallback if precise is not yet available.
              - `"DORIS Precise VOR (ENVISAT)"` – for other missions.

    Returns:
        Product: A new SNAP `Product` with orbits applied.
    """
    Boolean = jpy.get_type('java.lang.Boolean')
    Integer = jpy.get_type('java.lang.Integer')

    parameters = HashMap()
    parameters.put("orbitType", orbit_type)
    parameters.put("polyDegree", Integer(3))
    parameters.put("continueOnFail", Boolean(True))

    out = GPF.createProduct("Apply-Orbit-File", parameters, product)
    print(f"Apply-Orbit-File: {orbit_type}")
    return out

def back_geocoding(products, dem_name="SRTM 1Sec HGT", ext_dem=None):
    """
    Run SNAP Back-Geocoding on master + slave(s).

    Args:
        master   : SNAP Product (Apply-Orbit-File already done). Output of the apply orbit
        slaves   : list of SNAP Products (Apply-Orbit-File already done). Output of the apply orbit 
        dem_name : name of DEM in SNAP auxdata (default SRTM 1Sec HGT)
        ext_dem  : optional external DEM product

    Returns:
        SNAP Product with master + co-registered slave or slaves
    """
    print("Running Back-Geocoding...")

    parameters = HashMap()
    parameters.put("demName", dem_name)
    parameters.put("demResamplingMethod", "BILINEAR_INTERPOLATION")
    parameters.put("resamplingType", "BILINEAR_INTERPOLATION")
    parameters.put("maskOutAreaWithoutElevation", True)
    parameters.put("outputDerampDemodPhase", True)
    parameters.put("disableReramp", False)


    if ext_dem is not None:
        parameters.put("externalDEMFile", ext_dem)

    output = GPF.createProduct("Back-Geocoding", parameters, products) 
    print("Back geocoding applied!")
    return output

def enhanced_spectral_diversity(product, preset="default", **overrides):
    """
    Enhanced Spectral Diversity (ESD) with sensible defaults + optional overrides.

     Args:
        product:
            A SNAP `Product` — usually the output of Back-Geocoding
            containing the master and one or more slave images.
        preset:
            Convenience preset passed to `esd_parameters(preset)` that supplies a
            default parameter set. Expected values:
            - `"default"` - full computation.
            - `"fast"`  – less lighter computation.
            - `"faster"`    – lighter computation.
        **overrides:
            Any ESD operator parameter you want to force/override from the
            preset (e.g., `cohWinAz=5`, `cohWinRg=10`, `maxIterations=25`,
            etc.). Keys must match the operator's parameter names.

    Returns:
        Product: A new SNAP `Product` with ESD refinement applied.
    """
    parameters = HashMap()
    for k, v in esd_parameters(preset).items():
        parameters.put(k, v)
    # user overrides win if provided
    for k, v in overrides.items():
        parameters.put(k, v)
    esd = GPF.createProduct("Enhanced-Spectral-Diversity", parameters, product)
    return esd


def esd_parameters(preset): #enhanced spectral diversity parameters
    Boolean = jpy.get_type('java.lang.Boolean')
    if preset == "fast":
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
    if preset == "faster":
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
        "fineWinWidthStr": "512",
        "fineWinHeightStr": "512",
        "fineWinAccAzimuth": "16",
        "fineWinAccRange": "16",
        "fineWinOversampling": "128",
        "esdEstimator": "Periodogram",
        "weightFunc": "Inv Quadratic",
        "temporalBaselineType": "Number of images",
        "integrationMethod": "L1 and L2",
        "xCorrThreshold": 0.1,
        "cohThreshold": 0.3,
        "overallRangeShift": 0.0,
        "overallAzimuthShift": 0.0,
        "numBlocksPerOverlap": Integer(10),
        "maxTemporalBaseline":Integer(2),
        "doNotWriteTargetBands": False,
        "useSuppliedRangeShift": False,
        "useSuppliedAzimuthShift": False

    }

def temporal_baseline(product1_path, product2_path):
    """
    Calculate and print the temporal baseline between two Sentinel-1 products.

    The temporal baseline is defined as the absolute difference in days between 
    the acquisition times (start times) of the two SAR products. This metric is 
    commonly used in interferometric SAR (InSAR) analysis to assess the temporal 
    separation between image acquisitions.

    Args:
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
    product1 =  read_slc_product(product1_path)
    product2 =  read_slc_product(product2_path)
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

    Args:
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

    Args
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

def multilooking(product, n_rg=3, n_az=1, source_bands=None, output_intensity=False):
    """
    SNAP 'Multilook' operator.

    Args:
      product         : SNAP Product (typically AFTER TOPSAR-Deburst)
      n_rg (int)      : number of looks in range (speckle ↓, res ↓)
      n_az (int)      : number of looks in azimuth
      source_bands    : optional list/CSV of bands to process (e.g. 'i_*,q_*,coh_*')
      output_intensity: for complex inputs, also write intensity bands

    Returns:
      SNAP Product with multilooked bands
    """
    Integer = jpy.get_type('java.lang.Integer')
    Boolean = jpy.get_type('java.lang.Boolean')

    parameters = HashMap()
    parameters.put('nRgLooks', Integer(n_rg))
    parameters.put('nAzLooks', Integer(n_az))
    parameters.put('outputIntensity', Boolean(output_intensity))
    if source_bands:
        if isinstance(source_bands, (list, tuple)):
            source_bands = ",".join(source_bands)
        parameters.put('sourceBands', str(source_bands))

    return GPF.createProduct('Multilook', parameters, product)

def goldstein_phase_filtering(product):
    """
    Apply Goldstein Phase Filtering to an interferogram.

    Goldstein filtering is used in InSAR processing to enhance the signal-to-noise 
    ratio of the interferometric phase. This filter suppresses noise while preserving 
    the phase fringes, improving the quality of unwrapping and subsequent deformation 
    analysis.

    Args:
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

def snaphu_export(xml_path, new_input_file, new_target_folder):
    """
    Modify a SNAP XML graph to update input and output paths, then execute it using GPT.

    This function updates the SNAP XML workflow file by:
    - Replacing the file path in the 'Read' node with the given input file.
    - Replacing the target folder in the 'SnaphuExport' node with the given output folder.
    After updating, it saves the modified XML and runs the graph using the `gpt` command-line tool.

    Args
    ----------
    xml_path : str
        Path to the SNAP XML graph file to be modified and executed.
    new_input_file : str
        Path to the new input file (to replace the one in the 'Read' node).
    new_target_folder : str
        Path to the new target folder (to replace the one in the 'SnaphuExport' node).

    Returns
    -------
    None
        The function prints progress updates and the output of the `gpt` execution.

    Raises
    ------
    FileNotFoundError
        If the provided `xml_path` does not exist.
    xml.etree.ElementTree.ParseError
        If the XML file cannot be parsed.
    subprocess.CalledProcessError
        If the `gpt` command execution fails.

    Notes
    -----
    - Requires SNAP's Graph Processing Tool (`gpt`) to be installed and available in the system PATH.
    - The XML graph must contain nodes with IDs 'Read' and 'SnaphuExport' for the function to work correctly.
    """
    print("Snaphu exporting...")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for node in root.findall(".//node[@id='Read']/parameters/file"):
        node.text = new_input_file

    for node in root.findall(".//node[@id='SnaphuExport']/parameters/targetFolder"):
        node.text = new_target_folder

    tree.write(xml_path, encoding="UTF-8", xml_declaration=True)

    gpt_command = "gpt"  
    result = subprocess.run([gpt_command, xml_path], capture_output=True, text=True, check=True)
    print("Processing complete.\nOutput:\n", result.stdout)

def snaphu_unwrapping(conf_file_path, snaphu_exe_path, output_directory):
    """
    Run the SNAPHU unwrapping process using a configuration file.

    This function reads a SNAPHU configuration file, extracts the command line 
    arguments, and executes the SNAPHU binary with those arguments in the specified 
    output directory. It ensures the working directory exists before execution and 
    reports success or failure after completion.

    Args
    ----------
    conf_file_path : str
        Path to the SNAPHU configuration file (typically generated by SNAP or manually prepared).
    snaphu_exe_path : str
        Path to the SNAPHU executable to be used for unwrapping.
    output_directory : str
        Directory where SNAPHU will be executed and output files will be stored.

    Returns
    -------
    bool
        True if SNAPHU unwrapping completed successfully (return code 0), 
        False otherwise.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    IndexError
        If the configuration file has fewer than 7 lines (expected command at line 7).
    subprocess.SubprocessError
        If there is an unexpected error while executing the SNAPHU command.

    Notes
    -----
    - The function assumes that the unwrapping command is located on line 7 of the configuration file.
    - Any leading '#' or 'snaphu' prefixes in the command line will be stripped before execution.
    - Requires SNAPHU to be compiled and available at the specified `snaphu_exe_path`.
    """
    with open(conf_file_path, 'r') as file:
        lines = file.readlines()
        
    if len(lines) <= 6:
        raise IndexError("Configuration file doesn't have enough lines")
        
    command_line = lines[6].strip()
    
    if command_line.startswith('#'):
        command_line = command_line[1:].strip()
    if command_line.startswith('snaphu'):
        command_line = command_line[6:].strip()
    
    full_command = f'"{snaphu_exe_path}" {command_line}'
    
    print(f"Running SNAPHU command: {full_command}")
    print(f"Working directory: {output_directory}")
    
    os.makedirs(output_directory, exist_ok=True)
    
    result = subprocess.run(
        full_command,
        cwd=output_directory,
        shell=True,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("SNAPHU unwrapping completed successfully!")
        print("Output:", result.stdout)
    else:
        print("SNAPHU unwrapping failed!")
        print("Error:", result.stderr)
        print("Return code:", result.returncode)
    return result.returncode == 0

def snaphu_import(source_product, unwrapped_product): 
    """
    Import SNAPHU unwrapped interferogram results back into SNAP.

    This function uses SNAP's Graph Processing Framework (GPF) to merge the 
    unwrapped interferogram produced by SNAPHU with the original source product. 
    The resulting product contains the unwrapped phase information, making it 
    available for further processing within SNAP.

    Args
    ----------
    source_product : snappy.Product
        The original interferogram product before SNAPHU unwrapping.
    unwrapped_product : snappy.Product
        The product containing the SNAPHU unwrapped interferogram.

    Returns
    -------
    snappy.Product
        A SNAP product containing the imported unwrapped phase data.

    Notes
    -----
    - The parameter `'doNotKeepWrapped'` is set to False to keep the wrapped phase 
      along with the unwrapped phase.
    - Requires SNAP's snappy module and the GPF operator `'SnaphuImport'`.
    """ 
    parameters = HashMap()
    print("SNAPHU importing...")
    parameters.put('doNotKeepWrapped', False)
    products = [source_product, unwrapped_product]
    output = GPF.createProduct('SnaphuImport', parameters, products)
    print("SNAPHU imported...")
    return output

def phase_to_elevation(product, DEM):
    """
    Convert unwrapped interferometric phase to elevation using a Digital Elevation Model (DEM).

    This function uses the SNAP Graph Processing Framework (GPF) to apply the 
    "Phase to Elevation" operator. It transforms unwrapped phase data into an 
    elevation map by referencing a known DEM. This step is commonly used in 
    Differential InSAR (DInSAR) or when generating DEMs from SAR data.

    Args:
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

def snaphu_export(xml_path, new_input_file, new_target_folder):
    """
    Modify a SNAP XML graph to update input and output paths, then execute it using GPT.

    This function updates the SNAP XML workflow file by:
    - Replacing the file path in the 'Read' node with the given input file.
    - Replacing the target folder in the 'SnaphuExport' node with the given output folder.
    After updating, it saves the modified XML and runs the graph using the `gpt` command-line tool.

    Parameters
    ----------
    xml_path : str
        Path to the SNAP XML graph file to be modified and executed.
    new_input_file : str
        Path to the new input file (to replace the one in the 'Read' node).
    new_target_folder : str
        Path to the new target folder (to replace the one in the 'SnaphuExport' node).

    Returns
    -------
    None
        The function prints progress updates and the output of the `gpt` execution.

    Raises
    ------
    FileNotFoundError
        If the provided `xml_path` does not exist.
    xml.etree.ElementTree.ParseError
        If the XML file cannot be parsed.
    subprocess.CalledProcessError
        If the `gpt` command execution fails.

    Notes
    -----
    - Requires SNAP's Graph Processing Tool (`gpt`) to be installed and available in the system PATH.
    - The XML graph must contain nodes with IDs 'Read' and 'SnaphuExport' for the function to work correctly.
    """
    print("Snaphu exporting...")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for node in root.findall(".//node[@id='Read']/parameters/file"):
        node.text = new_input_file

    for node in root.findall(".//node[@id='SnaphuExport']/parameters/targetFolder"):
        node.text = new_target_folder

    tree.write(xml_path, encoding="UTF-8", xml_declaration=True)

    gpt_command = "gpt"  
    result = subprocess.run([gpt_command, xml_path], capture_output=True, text=True, check=True)
    print("Processing complete.\nOutput:\n", result.stdout)

def snaphu_unwrapping(conf_file_path, snaphu_exe_path, output_directory):
    """
    Run the SNAPHU unwrapping process using a configuration file.

    This function reads a SNAPHU configuration file, extracts the command line 
    arguments, and executes the SNAPHU binary with those arguments in the specified 
    output directory. It ensures the working directory exists before execution and 
    reports success or failure after completion.

    Parameters
    ----------
    conf_file_path : str
        Path to the SNAPHU configuration file (typically generated by SNAP or manually prepared).
    snaphu_exe_path : str
        Path to the SNAPHU executable to be used for unwrapping.
    output_directory : str
        Directory where SNAPHU will be executed and output files will be stored.

    Returns
    -------
    bool
        True if SNAPHU unwrapping completed successfully (return code 0), 
        False otherwise.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    IndexError
        If the configuration file has fewer than 7 lines (expected command at line 7).
    subprocess.SubprocessError
        If there is an unexpected error while executing the SNAPHU command.

    Notes
    -----
    - The function assumes that the unwrapping command is located on line 7 of the configuration file.
    - Any leading '#' or 'snaphu' prefixes in the command line will be stripped before execution.
    - Requires SNAPHU to be compiled and available at the specified `snaphu_exe_path`.
    """
    with open(conf_file_path, 'r') as file:
        lines = file.readlines()
        
    if len(lines) <= 6:
        raise IndexError("Configuration file doesn't have enough lines")
        
    command_line = lines[6].strip()
    
    if command_line.startswith('#'):
        command_line = command_line[1:].strip()
    if command_line.startswith('snaphu'):
        command_line = command_line[6:].strip()
    
    full_command = f'"{snaphu_exe_path}" {command_line}'
    
    print(f"Running SNAPHU command: {full_command}")
    print(f"Working directory: {output_directory}")
    
    os.makedirs(output_directory, exist_ok=True)
    
    result = subprocess.run(
        full_command,
        cwd=output_directory,
        shell=True,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("SNAPHU unwrapping completed successfully!")
        print("Output:", result.stdout)
    else:
        print("SNAPHU unwrapping failed!")
        print("Error:", result.stderr)
        print("Return code:", result.returncode)
    return result.returncode == 0

def snaphu_import(source_product, unwrapped_product): 
    """
    Import SNAPHU unwrapped interferogram results back into SNAP.

    This function uses SNAP's Graph Processing Framework (GPF) to merge the 
    unwrapped interferogram produced by SNAPHU with the original source product. 
    The resulting product contains the unwrapped phase information, making it 
    available for further processing within SNAP.

    Parameters
    ----------
    source_product : snappy.Product
        The original interferogram product before SNAPHU unwrapping.
    unwrapped_product : snappy.Product
        The product containing the SNAPHU unwrapped interferogram.

    Returns
    -------
    snappy.Product
        A SNAP product containing the imported unwrapped phase data.

    Notes
    -----
    - The parameter `'doNotKeepWrapped'` is set to False to keep the wrapped phase 
      along with the unwrapped phase.
    - Requires SNAP's snappy module and the GPF operator `'SnaphuImport'`.
    """ 
    parameters = HashMap()
    print("SNAPHU importing...")
    parameters.put('doNotKeepWrapped', False)
    products = [source_product, unwrapped_product]
    output = GPF.createProduct('SnaphuImport', parameters, products)
    print("SNAPHU imported...")
    return output  
def terrain_correction_slc(product, DEM):
    """
    Apply terrain correction to a SAR product using a specified DEM.

    Terrain correction removes geometric distortions caused by topography and sensor 
    viewing geometry. This step geocodes the image into a map coordinate system 
    and ensures that pixel locations align with their true geographic position.

    Args
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

def save_product(product, filename, output_dir="_results", fmt="BEAM-DIMAP"):
    """
    Save a SNAP product to disk.

    Args
    ----------
    product : snappy.Product
        The SNAP product to save.
    filename : str
        Output filename (without extension).
    output_dir : str, optional
        Directory where results will be saved (default: "_results").
    fmt : str, optional
        Output format (default: "BEAM-DIMAP").

    Returns
    -------
    str
        The full output path where the product was saved.
    """
    out_path = f"{output_dir}/{filename}"
    print(f"Saving product to {out_path} ({fmt})...")
    ProductIO.writeProduct(product, out_path, fmt)
    print("Product saved successfully.")
    return out_path


def plot(
    dim_path,
    i_band=None, q_band=None, coh_band=None,
    downsample=1,
    fade=(0.2, 0.8),          # (min,max) brightness from coherence
    title="",
    save_path=None,
    return_rgb=False,
    ax=None,
):
    """
    Visualize an interferogram as SNAP-like rainbow (HSV). This can be used to plot any interferogram outputs

    Args:
      dim_path   : path to the .dim file (next to the .data/ folder)
      i_band     : name of the real (i) interferogram band; auto-detected if None
      q_band     : name of the imag (q) interferogram band; auto-detected if None
      coh_band   : name of the coherence band; auto-detected if None
      downsample : integer stride for quick viewing (e.g., 2, 4)
      fade       : tuple (v_min, v_max) mapping coherence → value (brightness)
      title      : plot title
      save_path  : if set, write the RGB to this path (e.g., 'phase.png')
      return_rgb : if True, return the RGB numpy array
      ax         : optional matplotlib axes to draw on

    Returns:
      rgb (H,W,3) array if return_rgb=True, else None.
    """
    p = ProductIO.readProduct(dim_path)
    try:
        # --- band auto-detect (if names not given) ---
        names = list(p.getBandNames())
        def pick(prefix):
            for n in names:
                nn = n.lower()
                if nn.startswith(prefix):  # strict startswith
                    return n
            for n in names:                 # fallback: contains
                if prefix in n.lower():
                    return n
            return None

        i_band  = i_band  or pick('i_ifg')
        q_band  = q_band  or pick('q_ifg')
        coh_band = coh_band or pick('coh_')

        if not (i_band and q_band and coh_band):
            raise ValueError(
                f"Could not find required bands. "
                f"i_band={i_band}, q_band={q_band}, coh_band={coh_band}. "
                f"Available: {names[:12]}{' ...' if len(names)>12 else ''}"
            )

        w, h = p.getSceneRasterWidth(), p.getSceneRasterHeight()
        buf = np.zeros(w*h, np.float32)

        bi = p.getBand(i_band); bi.readPixels(0,0,w,h,buf); i = buf.reshape(h,w).copy()
        bq = p.getBand(q_band); bq.readPixels(0,0,w,h,buf); q = buf.reshape(h,w).copy()
        bc = p.getBand(coh_band); bc.readPixels(0,0,w,h,buf); coh = buf.reshape(h,w).copy()

        if downsample and downsample > 1:
            s = slice(None, None, int(downsample))
            i, q, coh = i[s, s], q[s, s], coh[s, s]

        # --- phase → HSV ---
        phase = np.arctan2(q, i)                         # [-pi, +pi]
        hue   = (phase + np.pi) / (2*np.pi)              # [0,1]
        sat   = np.ones_like(hue)
        vmin, vmax = fade
        val   = vmin + (vmax - vmin) * np.clip(coh, 0, 1)

        rgb = hsv_to_rgb(np.dstack([hue, sat, val]))

        # --- plot ---
        if ax is None:
            plt.figure(figsize=(10, 6))
            ax = plt.gca()
        ax.imshow(rgb, origin="upper")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title)

        if save_path:
            # matplotlib expects 0..1 floats; rgb already is
            plt.imsave(save_path, rgb)

        return rgb if return_rgb else None

    finally:
        try:
            p.dispose()
        except Exception:
             pass