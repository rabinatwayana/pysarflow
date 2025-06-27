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
import xarray as xr
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
import re

from eof.download import download_eofs  #pip install sentineleof
from datetime import datetime
from scipy.interpolate import interp1d


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

# ----- download and apply orbit file -----

    def download_orbit_file(self, safe_folder, save_dir):
        """
        Extract date and mission from Sentinel-1 SAFE or zip filename,
        download corresponding precise orbit files, and return paths.
        
        Args:
            safe_folder (str): Sentinel-1 SAFE folder or zip filename.
            save_dir (str): Directory to save downloaded orbit files.
            
        Returns:
            list: List of paths to downloaded orbit files.
        """
        # Extract mission (e.g., S1A, S1B, S1C)
        mission = safe_folder[:3]
        
        # Extract date string in format YYYYMMDD from filename
        match = re.search(r'_(\d{8})T', safe_folder)
        if not match:
            raise ValueError("Date not found in filename. Expected pattern '_YYYYMMDDT'.")
        date_str = match.group(1)
        
        # Format date as YYYY-MM-DD
        date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        
        # Download orbit files
        orbit_files = download_eofs([date_formatted], [mission], save_dir=save_dir)

        return orbit_files
    

    def update_annotations_orbit(self, safe_folder, eof_orbit_file, overwrite=True):
        """
            Parses the EOF orbit file and replaces the orbitList in all annotation XML files.

            Args:
                safe_folder (str or Path): Path to Sentinel-1 SAFE folder.
                eof_orbit_file (str or Path): Path to precise orbit EOF file.
                overwrite (bool): Overwrite original annotation XMLs if True.

            Returns:
                None
            """
        # parsing the EOF file
        tree = ET.parse(eof_orbit_file)
        root = tree.getroot()
        osv_list = root.find(".//List_of_OSVs")
        if osv_list is None:
            raise ValueError("No OSV list found in EOF file")

        precise_orbits = []
        for osv in osv_list.findall("OSV"):
            utc_text = osv.find("UTC").text
            utc_clean = re.sub(r"^UTC=", "", utc_text)
            time = datetime.fromisoformat(utc_clean)
            pos = np.array([float(osv.find(tag).text) for tag in ["X", "Y", "Z"]])
            vel = np.array([float(osv.find(tag).text) for tag in ["VX", "VY", "VZ"]])
            precise_orbits.append({"time": time, "position": pos, "velocity": vel})
        
        # --- Apply orbits to each annotation file ---
        safe_folder = Path(safe_folder)
        annotation_folder = safe_folder / "annotation"
        xml_files = list(annotation_folder.glob("*.xml"))
        if not xml_files:
            raise FileNotFoundError(f"No XML files found in {annotation_folder}")

        for xml_file in xml_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            orbit_list = root.find(".//orbitList")
            if orbit_list is None:
                print(f"No orbitList found in {xml_file}, skipping.")
                continue

            orbit_list.clear()
            for sv in precise_orbits:
                orbit = ET.SubElement(orbit_list, "orbit")
                ET.SubElement(orbit, "time").text = sv["time"].isoformat()
                ET.SubElement(orbit, "frame").text = "Earth Fixed"

                pos_el = ET.SubElement(orbit, "position")
                for coord, label in zip(sv["position"], ["x", "y", "z"]):
                    ET.SubElement(pos_el, label).text = f"{coord:.6e}"

                vel_el = ET.SubElement(orbit, "velocity")
                for coord, label in zip(sv["velocity"], ["x", "y", "z"]):
                    ET.SubElement(vel_el, label).text = f"{coord:.6e}"

            out_path = xml_file if overwrite else xml_file.with_name(xml_file.stem + "_updated.xml")
            tree.write(out_path, encoding="utf-8", xml_declaration=True)
            print(f"Updated orbitList in {out_path}")


    def apply_orbit(self, ds, annotation_xml_path):
        """
        Applies precise satellite orbit data to an xarray Dataset by interpolating positions
        and velocities from an annotation XML file.

        Args:
            ds (xarray.Dataset): The dataset to which orbit data will be assigned.
            annotation_xml_path (str or Path): Path to the Sentinel-1 annotation XML file
                                            containing precise orbit state vectors.

        Returns:
            xarray.Dataset: Dataset with new coordinates:
                            - sat_pos_{x,y,z}: satellite positions (m)
                            - sat_vel_{x,y,z}: satellite velocities (m/s)
        """
        # Parse orbit state vectors from XML
        root = ET.parse(annotation_xml_path).getroot()
        orbits_xml = root.find(".//orbitList")
        orbits = []
        for orbit in orbits_xml.findall("orbit"):
            time = datetime.fromisoformat(orbit.find("time").text)
            pos = np.array([float(orbit.find(f"position/{axis}").text) for axis in ("x", "y", "z")])
            vel = np.array([float(orbit.find(f"velocity/{axis}").text) for axis in ("x", "y", "z")])
            orbits.append({"time": time, "position": pos, "velocity": vel})

        # Get timestamps from dataset or metadata
        if "time" in ds.coords:
            times = [pd.to_datetime(t).to_pydatetime() for t in ds.time.values]
        else:
            start_time_str = ds.attrs.get("startTime")
            if not start_time_str:
                raise ValueError("No time info in dataset or metadata")
            times = [datetime.fromisoformat(start_time_str)]

        # Prepare for interpolation
        base_time = orbits[0]["time"]
        orbit_secs = np.array([(o["time"] - base_time).total_seconds() for o in orbits])
        pos_array = np.vstack([o["position"] for o in orbits])
        vel_array = np.vstack([o["velocity"] for o in orbits])
        target_secs = np.array([(t - base_time).total_seconds() for t in times])

        # Interpolate positions and velocities
        interp_pos = [interp1d(orbit_secs, pos_array[:, i], fill_value="extrapolate")(target_secs) for i in range(3)]
        interp_vel = [interp1d(orbit_secs, vel_array[:, i], fill_value="extrapolate")(target_secs) for i in range(3)]

        # Assign interpolated orbit data as coordinates in the dataset
        ds = ds.assign_coords(
            sat_pos_x=("time", interp_pos[0]),
            sat_pos_y=("time", interp_pos[1]),
            sat_pos_z=("time", interp_pos[2]),
            sat_vel_x=("time", interp_vel[0]),
            sat_vel_y=("time", interp_vel[1]),
            sat_vel_z=("time", interp_vel[2]),
        )

        return ds
