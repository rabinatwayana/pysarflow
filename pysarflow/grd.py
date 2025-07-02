# -*- coding: utf-8 -*-
"""It includes functions for handling Ground Range Detected (GRD) datasets."""
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

from .utils_grd import items_to_geodataframe, apply_correction

import os
import xarray as xr
import zipfile
import xml.etree.ElementTree as ET

from pathlib import Path
from .utils_grd import download_orbit_file, update_annotations_orbit, add_precise_orbit_coords


def sum_grd(a, b):
    """Dockstring here."""
    return a + b


class Sentinel1GRDProcessor:
    """
    A processor class to search and explore Sentinel-1 GRD data from the Earth Search STAC API.

    Attributes
    ----------
    api_url : str
        The URL of the STAC API endpoint. Defaults to the Earth Search AWS endpoint.
    client : Client
        An instance of the STAC API client connected to the specified api_url.
    search : Search
        The STAC API search object resulting from a data query.
    items : list
        A list of search results as dictionaries (STAC items).
    item_collection : ItemCollection
        A collection of STAC items from the search results.
    gdf_aoi : GeoDataFrame
        GeoDataFrame representing the Area of Interest (AOI) geometry.
    gdf_metadata : GeoDataFrame
        GeoDataFrame containing spatial metadata.
    df_properties : pandas.DataFrame
        DataFrame holding properties of each STAC item.
    df_assets : pandas.DataFrame
        DataFrame holding asset URLs for each STAC item.

    Methods
    -------
    search_data(aoi: dict, datetime: str):
        Searches Sentinel-1 GRD data within a given Area of Interest (AOI) and datetime range
    """

    def __init__(self, api_url="https://earth-search.aws.element84.com/v1"):
        """
        Initializes the Sentinel1GRDProcessor with a connection to the specified STAC API.

        Parameters
        ----------
        api_url : str, optional
            The URL of the STAC API endpoint to connect to Earth Search AWS.
        """
         
        self.api_url = api_url
        self.client = Client.open(api_url)

        self.search = None
        self.items = None
        self.item_collection = None
        self.gdf_aoi = None
        self.gdf_metadata = None
        self.df_properties = None
        self.df_assets = None

    def search_data(self, aoi: dict, datetime: str,):

        """
        Search Sentinel-1 GRD data using STAC API for a specified area and time range.
        
        Parameters
        ----------
        aoi : dict
            GeoJSON defining the Area of Interest for the search.
        datetime : str
            datetime range string for filtering the data, e.g., "2023-01-01/2023-01-31".
        """
        self.aoi=aoi
        self.datetime=datetime

        self.search = self.client.search(
            collections=["sentinel-1-grd"],
            intersects=self.aoi,
            datetime=self.datetime
        )
        
        self.items = list(self.search.items_as_dicts())
        self.item_collection = ItemCollection(self.search.get_items())

        # Convert AOI to GeoDataFrame
        self.gdf_aoi = gpd.GeoDataFrame([{"geometry": shape(self.aoi)}])

        # Convert items to GeoDataFrame and property/asset DataFrames
        self.gdf_metadata = items_to_geodataframe(self.items)

        item_properties = []
        item_assets = []

        for item in self.item_collection:
            item_properties.append({"id": item.id, **item.properties})
            item_assets.append({"id": item.id, **{k: v.href for k, v in item.assets.items()}})

        self.df_properties = pd.DataFrame(item_properties)
        self.df_assets = pd.DataFrame(item_assets)

   
    def read_grd_data(self, safe_path, extract_to):
        """
        Loads Sentinel-1 GRD SAR data from a SAFE folder or ZIP file into an xarray dataset,
        and adds the acquisition start time from annotation XML as a 'time' coordinate and attribute.

        Args:
            safe_path (str or Path): Path to the .SAFE folder or .zip file
            extract_to (str or Path): Directory to extract zip file contents to

        Returns:
            xarray.Dataset: dataset containing SAR bands (e.g. VV, VH) with time coordinate

        Raises:
            FileNotFoundError if required folders or files are missing.
        """

        safe_path = Path(safe_path)

        # If input is a ZIP file, extract it to your desired directory 
        if safe_path.suffix.lower() == ".zip":
            print(f"Extracting zip file: {safe_path}")
            extract_dir = Path(extract_to)
            extract_dir.mkdir(exist_ok=True)

            with zipfile.ZipFile(safe_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            # Find the extracted .SAFE folder
            safe_dirs = [d for d in os.listdir(extract_dir) if d.endswith(".SAFE") and (extract_dir / d).is_dir()]
            if not safe_dirs:
                raise FileNotFoundError(f"No .SAFE folder found inside extracted contents of {safe_path}")
            safe_folder_path = extract_dir / safe_dirs[0]
        else:
            # If not a ZIP, treat safe_path as the .SAFE directory itself
            safe_folder_path = safe_path

        # Check if 'measurement' subdirectory exists; where GeoTIFF files are stored
        measurement_path = safe_folder_path / "measurement"
        if not measurement_path.exists():
            raise FileNotFoundError(f"'measurement' folder not found inside {safe_folder_path}")

        annotation_path = safe_folder_path / "annotation"
        if not annotation_path.exists():
            raise FileNotFoundError(f"'annotation' folder not found inside {safe_folder_path}")

        tiff_files = [f for f in os.listdir(measurement_path) if f.lower().endswith((".tif", ".tiff"))]
        if not tiff_files:
            raise FileNotFoundError(f"No GeoTIFF files found in 'measurement' folder of {safe_folder_path}")

        data_vars = {}

        for tiff in tiff_files:
            pols = ["vv", "vh", "hh", "hv"]
            band_name = None
            for pol in pols:
                if pol in tiff.lower():
                    band_name = pol.upper()
                    break
            if not band_name:
                raise ValueError(f"Polarization not found in filename: {tiff}")

            tiff_path = measurement_path / tiff
            print(f"Loading band {band_name} from {tiff_path}")
            da = rioxarray.open_rasterio(tiff_path)
            if "band" in da.dims and len(da.band) == 1:
                da = da.squeeze("band", drop=True)
            data_vars[band_name] = da

        ds = xr.Dataset(data_vars)

        # Find an annotation XML corresponding to the band (e.g., VV)
        # Use the first annotation file that matches the polarization of the bands loaded.
        pol_files = [f for f in os.listdir(annotation_path) if f.lower().endswith(".xml")]
        if not pol_files:
            raise FileNotFoundError(f"No XML annotation files found in {annotation_path}")

        # Try to match annotation with band VV or VH, else just pick the first
        annotation_file = None
        for pol in ["vv", "vh", "hh", "hv"]:
            for f in pol_files:
                if pol in f.lower():
                    annotation_file = annotation_path / f
                    break
            if annotation_file:
                break
        if annotation_file is None:
            annotation_file = annotation_path / pol_files[0]

        # Parse startTime from annotation XML
        root = ET.parse(annotation_file).getroot()
        start_time_elem = root.find(".//startTime")
        if start_time_elem is None:
            raise ValueError("No startTime found in annotation XML")

        start_time = pd.to_datetime(start_time_elem.text)

        # Assign start time as coordinate and attribute
        ds = ds.assign_coords(time=[start_time])
        ds.attrs["startTime"] = start_time.isoformat()

        return ds

    def apply_orbit_file(self, ds, safe_folder_path, save_dir, overwrite=True):
        """
        Apply precise orbit file to a Sentinel-1 dataset.

        Args:
            ds (xarray.Dataset): Dataset to apply orbit data to.
            safe_folder_path (str or Path): Path to the Sentinel-1 SAFE folder or ZIP filename.
            save_dir (str or Path): Where to save downloaded EOF orbit files.
            overwrite (bool): Whether to overwrite annotation XMLs in SAFE folder.

        Returns:
            xarray.Dataset: Dataset with interpolated orbit coordinates.
        """
        orbit_files = download_orbit_file(safe_folder_path, save_dir)
        update_annotations_orbit(safe_folder_path, orbit_files[0], overwrite=overwrite)

        annotation_folder = Path(safe_folder_path) / "annotation"
        first_annotation = sorted(annotation_folder.glob("*.xml"))[0]

        return add_precise_orbit_coords(ds, first_annotation)

    
    def remove_thermal_noise(self, ds, lut_ds):

        """
        Removes thermal noise from Sentinel-1 SAR GRD data using a provided lookup table (LUT).

        This function applies thermal noise correction to each band in the input dataset
        using the specified LUT dataset. 
        
        It relies on a function "apply_correction" with the correction type set to "thermal_noise_removal".

        Args:
            ds (xarray.Dataset): The input dataset containing one or more polarization bands 
                (e.g., 'VV', 'VH') as data variables which is result of function "load_sentinel1_data".
            lut_ds (xarray.Dataset or dict): The thermal noise lookup table used for correction, 
                which is the result of function "parse_thermal_noise_removal_lut".

        Returns:
            xarray.Dataset: A dataset with thermal noise removed from each polarization band.
        """

        result=apply_correction("thermal_noise_removal",ds, lut_ds)
        print("Thermal noise removed successfully")
        return result


    def radiometric_calibration(self, ds, lut_ds):
        """
        Perform radiometric calibration of Sentinel-1 SAR GRD data using a provided lookup table (LUT).

        This function applies radiometric calibration to each band in the input dataset
        using the specified LUT dataset. 
        
        It relies on a function "apply_correction" with the correction type set to "radiometric_calibration".

        Args:
            ds (xarray.Dataset): The input dataset containing one or more polarization bands 
                (e.g., 'VV', 'VH') as data variables which is result of function "load_sentinel1_data".
            lut_ds (xarray.Dataset or dict): The thermal noise lookup table used for correction, 
                which is the result of function "parse_radiometric_calibration_lut".

        Returns:
            xarray.Dataset: A dataset with thermal noise removed from each polarization band.
        """

        result= apply_correction("radiometric_calibration", ds, lut_ds)
        print("Radiometric calibration completed successfully")
        return result

        
