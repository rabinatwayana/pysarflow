# -*- coding: utf-8 -*-
"""It includes common functionalities that would be used in other modules."""

from pystac_client import Client
from pystac import ItemCollection

import requests
from PIL import Image
from io import BytesIO
import numpy as np
import math

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import shape

import rioxarray 
import odc.stac
import hvplot.pandas

import os
import xarray as xr
import xml.etree.ElementTree as ET


def s3_to_https(s3_url):
    '''
    Convert an S3 URL (s3://bucket/key) to its equivalent HTTPS URL for web access
    '''
    if not s3_url.startswith("s3://"):
        raise ValueError("Not a valid S3 path.")
    s3_path = s3_url.replace("s3://", "")
    bucket, *key_parts = s3_path.split("/")
    key = "/".join(key_parts)
    return f"https://{bucket}.s3.amazonaws.com/{key}"


def items_to_geodataframe(items):
    ''' 
        Convert a list of STAC Items into a GeoDataFrame
    '''
    _items = []
    for i in items:
        _i = {
            "id": i["id"],
            "type": i["type"],
            "stac_version": i["stac_version"],
            "links": i["links"],
            "stac_extensions": i["stac_extensions"],
            "bbox": i["bbox"],
            "geometry": shape(i["geometry"]),
        }
        _items.append(_i)
    return gpd.GeoDataFrame(_items, geometry="geometry")

def plot_bbox(data, *args, **kwargs):
    '''
    plot aoi on a map with background tiles
    source: https://pystac-client.readthedocs.io/en/stable/tutorials/stac-metadata-viz.html
    '''
    return data.hvplot.polygons(
        *args,
        geo=True,
        projection="GOOGLE_MERCATOR",
        xaxis=None,
        yaxis=None,
        frame_width=600,
        frame_height=600,
        fill_alpha=0,
        line_width=4,
        **kwargs,
    )
        
def plot_thumbnails(df_assets, df_properties, n_cols=4):
    '''
        Plot image thumbnails from assets DataFrame in a grid layout with titles.

        Parameters:
        - df_assets (pd.DataFrame): DataFrame containing asset information including S3 thumbnail URLs and IDs.
        - df_properties (pd.DataFrame): DataFrame containing additional properties for assets
        - n_cols (Integer): number of columns in a row to plot image
        The function:
        - Converts S3 URLs to HTTPS URLs to fetch thumbnail images.
        - Retrieves and displays thumbnails in a grid (default 4 columns).
        - Sets titles using asset IDs and orbit pass information.
    '''
    try:
        if df_assets is None:
            raise ValueError("Asset DataFrame is not initialized. Ensure data is loaded before proceeding.")
        thumbnails = []
        titles = []

        # Collect thumbnails and titles
        for _, item in df_assets.iterrows():
            s3_url = item["thumbnail"]
            https_url = s3_to_https(s3_url)
            try:
                response = requests.get(https_url)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content)).convert("RGB")
                thumbnails.append(img)
                
                properties = df_properties[df_properties["id"] == item["id"]]
                orbit_state = properties.iloc[0]["sat:orbit_state"]

                titles.append(f'{item["id"].split("_")[-3]}, Pass: {orbit_state}')

            except Exception as e:
                print(f"Failed to load thumbnail for {item.id}: {e}")

        # Determine grid layout
        n_images = len(thumbnails)
        n_cols = n_cols
        n_rows = math.ceil(n_images / n_cols)

        # Plot thumbnails in grid
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        for idx, ax in enumerate(axes):
            if idx < n_images:
                ax.imshow(thumbnails[idx])
                ax.set_title(titles[idx], fontsize=9)
                ax.axis('off')
            else:
                ax.axis('off')  # Hide unused subplots

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Unexpected error occurred in plot_thumbnails:", str(e))

def parse_thermal_noise_removal_lut(safe_folder_path):
    """
    Parsing Lookup Table (LUT) for thermal noise removal raeding the noise xml file.

    This function reads noise related XML files within the 'annotation/calibration' folder in the
    provided SAFE directory path,
    and returns an xarray.Dataset with LUT for available polarizations

    Arguments:
        safe_folder_path (str): Path to the root directory of the Sentinel-1 SAFE format product.

    Returns:
        xarray.Dataset: A dataset containing the LUT for noise removal for available polarizations
        (e.g., 'VV', 'VH')

    Raises Exception:
        FileNotFoundError: If the 'calibration' folder or required XML files are missing,
    """

    calibration_path = os.path.join(safe_folder_path, "annotation/calibration/")
    if not os.path.exists(calibration_path):
        raise FileNotFoundError(f"'calibration' folder not found inside {safe_folder_path}")

    # Find XML files starting with 's1a-iw-grd' (case insensitive)
    xml_files = [f for f in os.listdir(calibration_path) if f.lower().startswith('noise') and f.lower().endswith('.xml')]
    if not xml_files:
        raise FileNotFoundError("No suitable calibration XML files found in 'calibration' folder")
    lut_dict={}
    for xml_file in xml_files:
        polarizations = ["vv", "vh", "hh", "hv"]
        band_name = None
        for pol in polarizations:
            if pol in xml_file.lower():
                band_name = pol.upper()
                break

        if not band_name:
            raise FileNotFoundError(f"Polarization type not found in file name: {xml_file}")

        print(f"Reading xml for {band_name} band")

        xml_path = os.path.join(calibration_path, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        lines = []
        pixels = None
        noise_values = []

        for calib_vec in root.findall('.//noiseRangeVector'):
            line = int(calib_vec.find('line').text)
            pixel_str = calib_vec.find('pixel').text.strip()
            noise_range_str = calib_vec.find("noiseRangeLut").text.strip()

            pixels = [int(x) for x in pixel_str.split()]
            noise = [float(x) for x in noise_range_str.split()]

            lines.append(line)
            noise_values.append(noise)
        noise_array = np.array(noise_values)
        lut = xr.DataArray(noise_array, coords={"line": lines, "pixel": pixels}, dims=["line", "pixel"])
        lut_dict[band_name]= lut    
    lut_ds = xr.Dataset(lut_dict)
    print("Thermal noise removal LUT created successfully")
    return lut_ds

def parse_radiometric_calibration_lut(safe_folder_path, representation_type="sigmaNought"):
    """
    Parsing LUT for radiometric calibration bt reading the calibration xml files.

    This function reads calibration related XML files within the 'annotation/calibration' folder in the
    provided SAFE directory path,
    and returns an xarray.Dataset with LUT for available polarizations based on the representation type as required.

    Arguments:
        safe_folder_path (str): Path to the root directory of the Sentinel-1 SAFE format product.
        representation_type (str, optional): Type of backscatter representation to be used. 
        Options are:
            - 'sigmaNought' (default)
            - 'betaNought'
            - 'gamma'

    Returns:
        xarray.Dataset: A dataset containing the LUT for radiometric calibration for available polarizations
        (e.g., 'VV', 'VH')

    Raises Exception:
        Exception: representation_type is not valid
        FileNotFoundError: If the 'calibration' folder or required XML files are missing,
    """

    supporting_representation_types=["sigmaNought","betaNought","gamma"]
    if representation_type not in supporting_representation_types:
        raise Exception(f"representation_type {representation_type} is not supported. Supporting types are {supporting_representation_types}")

    calibration_path = os.path.join(safe_folder_path, "annotation/calibration/")
    if not os.path.exists(calibration_path):
        raise FileNotFoundError(f"'calibration' folder not found inside {safe_folder_path}")

    # Find XML files starting with 's1a-iw-grd' (case insensitive)
    xml_files = [f for f in os.listdir(calibration_path) if f.lower().startswith('calibration') and f.lower().endswith('.xml')]
    if not xml_files:
        raise FileNotFoundError("No suitable calibration XML files found in 'calibration' folder")
    lut_dict={}
    for xml_file in xml_files:

        polarizations = ["vv", "vh", "hh", "hv"]
        band_name = None
        for pol in polarizations:
            if pol in xml_file.lower():
                band_name = pol.upper()
                break

        if not band_name:
            raise FileNotFoundError(f"Polarization type not found in file name: {xml_file}")

        print(f"Reading calibration for {band_name} band")

        xml_path = os.path.join(calibration_path, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        lines = []
        pixels = None
        correction_values = []

        for calib_vec in root.findall('.//calibrationVector'):
            line = int(calib_vec.find('line').text)
            pixel_str = calib_vec.find('pixel').text.strip()
            value_str = calib_vec.find(representation_type).text.strip()

            pixels = [int(x) for x in pixel_str.split()]
            values = [float(x) for x in value_str.split()]

            lines.append(line)
            correction_values.append(values)

        beta_array = np.array(correction_values)
        lut = xr.DataArray(beta_array, coords={"line": lines, "pixel": pixels}, dims=["line", "pixel"])
        lut_dict[band_name]= lut    
    lut_ds = xr.Dataset(lut_dict)
    print("Radiometric calibration LUT created successfully")
    return lut_ds

def apply_correction(type, ds, lut_ds):
    """
    This is the common function to apply thermal noise removal and raduometric correction

    Arguments:
        type (str): Type of correction
        Options are:
            - 'thermal_noise_removal'
            - 'radiometric_calibration'
        representation_type (str, optional): Type of backscatter representation to be used. 
        Options are:
            - 'sigmaNought' (default)
            - 'betaNought'
            - 'gamma'
    
    Returns:
        xarray.Dataset: A dataset containing corrected backscatter values for available polarizations
    """
    corrected_dict={}
    for pol in list(ds.keys()):
        da= ds[pol]
        lut=lut_ds[pol]

        # Creating full image grid
        image_lines = np.arange(da.shape[0])   
        image_pixels = np.arange(da.shape[1])  

        # Interpolate the LUT on these coordinates (numeric interpolation):
        lut_interp = lut.interp(
            line=image_lines,
            pixel=image_pixels,
            method='linear',
            kwargs={"fill_value": "extrapolate"}  # to extrapolate near edges if needed
        )

        if da.dims != lut_interp.dims:
            da = da.rename({'y': 'line', 'x': 'pixel'}) 

        # Apply correction using respective formula
        if type=="thermal_noise_removal":
            # source: https://sentinels.copernicus.eu/documents/247904/2142675/Thermal-Denoising-of-Products-Generated-by-Sentinel-1-IPF.pdf#page=18.10
            corrected_dict[pol] = xr.where(da**2 - lut_interp > 0, da**2 - lut_interp, 0)
        elif type=="radiometric_calibration":
            #SOURCE: https://sentinels.copernicus.eu/documents/247904/685163/S1-Radiometric-Calibration-V1.0.pdf
            corrected_dict[pol] = da / (lut_interp**2) # Since da is thermally corrected, no need of da**2
        
    calibrated_ds= xr.Dataset(corrected_dict)
    return calibrated_ds