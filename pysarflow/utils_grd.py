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