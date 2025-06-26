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
