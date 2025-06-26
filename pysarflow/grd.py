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

from .utils_grd import items_to_geodataframe

import os
import rioxarray
import xarray as xr
import zipfile
from pathlib import Path


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
        Loads Sentinel-1 GRD SAR data into an xarray Dataset.

        Parameters:
        -----------
        safe_path (str or Path): Path to the .SAFE folder or .zip file
        extract_to (str or Path): Directory to extract zip file contents to; 
                                 for windows: the Sentinel safe file already 
                                 has really long name so a short file path recommended i.e. 'C:\Temp'

        Returns:
        --------
        xarray.Dataset
        """
        safe_path = Path(safe_path)

        # If input is a ZIP file, extract it to your desired directory 
        # 
        if safe_path.suffix.lower() == ".zip":
            print(f"Extracting zip file: {safe_path}")
            extract_dir = Path(extract_to)
            extract_dir.mkdir(exist_ok=True)
            
            # extract the zip file
            with zipfile.ZipFile(safe_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

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
        print("Sentinel-1 data loaded successfully")
        return ds

