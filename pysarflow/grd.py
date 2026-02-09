# -*- coding: utf-8 -*-
"""It includes functions for handling Ground Range Detected (GRD) datasets."""
from pystac_client import Client
from pystac import ItemCollection

import requests
from PIL import Image
from io import BytesIO
import numpy as np
import math

import rasterio
from rasterio.transform import rowcol
from rasterio.warp import transform_bounds

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import shape

import rioxarray
import xarray as xr 
import odc.stac
import hvplot.pandas

from .utils_grd import items_to_geodataframe, apply_correction, parse_radiometric_calibration_lut,parse_thermal_noise_removal_lut, get_ipf_version, get_acquisition_mode, get_calibration_constant, compute_scaling_factor, parse_noise_vectors, build_noise_map, lee_filter,download_orbit_file, update_annotations_orbit, add_precise_orbit_coords

import os
import xarray as xr
import zipfile
import xml.etree.ElementTree as ET
from scipy.ndimage import uniform_filter

from pathlib import Path

# ----------- SENTINEL-1 GRD PROCESSOR --------------
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
    
    # ----------- DATA SEARCH --------------
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

   
   # ----------- DATA LOADING --------------
    def read_grd_data(self, safe_path, extract_to=None):
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

        # NEW
        vv_tiff_path = None  # Track the VV band file path for later use in subsetting

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

            # NEW
            if band_name == "VV":
                vv_tiff_path = tiff_path

        ds = xr.Dataset(data_vars)

        # NEW!
        # Store the VV TIFF path in attributes (used by subset function)
        if vv_tiff_path:
            ds.attrs["vv_tiff_path"] = str(vv_tiff_path)

        # NEW!
        if vv_tiff_path:
            with rasterio.open(vv_tiff_path) as src:
                if src.gcps and len(src.gcps[0]) > 0:
                    ds.attrs["gcps"] = src.gcps[0]           # list of GCPs
                    ds.attrs["gcps_crs"] = src.gcps[1] or "EPSG:4326"
                    
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
        print("Data loaded successfully")
        return ds


    # ----------- SUBSET TO AOI --------------
    def subset_aoi(self, ds, bbox: list[float]):
        """
        Subset Sentinel-1 GRD xarray Dataset to  AOI using lon/lat bbox.
        
        Features:
        - Uses GCPs from original TIFF to map bbox corners → pixel range
        - Subsets ALL bands (VV, VH, etc.) using the same pixel window
        - Adds rough geographic coordinates (linear interpolation)
        - Works around common Sentinel-1 GRD affine/GCP issues
        
        Args:
            ds: xarray.Dataset from read_grd_data (needs 'vv_tiff_path' in attrs)
            bbox: [min_lon, min_lat, max_lon, max_lat] in EPSG:4326
        
        Returns:
            xr.Dataset: subsetted version with approximate coords
        """
        # --- check bbox validity ---
        if len(bbox) != 4:
            raise ValueError("bbox must be [min_lon, min_lat, max_lon, max_lat]")
        
        min_lon, min_lat, max_lon, max_lat = bbox
        
        # enforce expected EPSG since GCPs are usually in EPSG:4326 
        if not (-180 <= min_lon <= 180 and -90 <= min_lat <= 90 and
                -180 <= max_lon <= 180 and -90 <= max_lat <= 90):
            raise ValueError(
                "bbox coordinates are expected in EPSG:4326 (lon/lat). "
                f"Got {bbox}"
            )

        print(f"Subsetting to AOI (EPSG:4326): [{min_lon:.4f}, {min_lat:.4f}, {max_lon:.4f}, {max_lat:.4f}]")
        
        # Get TIFF path from attributes
        tiff_path = ds.attrs.get("vv_tiff_path")
        if not tiff_path:
            raise ValueError(
                "No 'vv_tiff_path' found in ds.attrs.\n"
                f"Available keys: {list(ds.attrs.keys())}\n"
                "Make sure read_grd_data stores it as ds.attrs['vv_tiff_path']"
            )
        print(f"Using TIFF for geolocation: {tiff_path}")

        # Open TIFF and get GCPs
        with rasterio.open(tiff_path) as src:
            if not src.gcps or len(src.gcps[0]) == 0:
                raise ValueError("No GCPs found in TIFF — cannot estimate subset location")

            gcps, gcps_crs = src.gcps
            gcps_crs = gcps_crs or "EPSG:4326"

            # Show approximate scene extent from GCPs (for debugging)
            lons = [gcp.x for gcp in gcps]
            lats = [gcp.y for gcp in gcps]
            print(f"Scene approx lon range (from GCPs): {min(lons):.4f} – {max(lons):.4f}")
            print(f"Scene approx lat range (from GCPs): {min(lats):.4f} – {max(lats):.4f}")

            left, bottom, right, top = min_lon, min_lat, max_lon, max_lat

            # Create approximate affine from GCPs (used only for rowcol)
            from rasterio.transform import from_gcps
            approx_transform = from_gcps(gcps)

            # Map bbox corners to pixel coordinates
            corner_lons = [left, right, right, left]
            corner_lats = [top, top, bottom, bottom]  # top = max lat, bottom = min lat

            rows = []
            cols = []

            for lon, lat in zip(corner_lons, corner_lats):
                try:
                    row, col = rowcol(
                        approx_transform,
                        lon, lat,
                        precision=0.1,  # tolerance for non-exact fit
                        op=float
                    )
                    rows.append(row)
                    cols.append(col)
                except Exception as e:
                    print(f"Corner ({lon:.4f}, {lat:.4f}) could not be mapped: {e}")
        if not rows:
            raise ValueError("None of the bbox corners could be mapped to pixels — likely no overlap")

        # Take min/max row/col
        row_start = max(0, int(min(rows)))
        row_stop  = min(src.height, int(max(rows)) + 1)
        col_start = max(0, int(min(cols)))
        col_stop  = min(src.width, int(max(cols)) + 1)

        height = row_stop - row_start
        width  = col_stop - col_start

        # Slice all bands using the computed window
        subset_vars = {}
        for band_name in ds.data_vars:
            subset_da = ds[band_name].isel(
                y=slice(row_start, row_stop),
                x=slice(col_start, col_stop)
            )

            # Rough geographic coordinates (linear from bbox)
            approx_x = np.linspace(left, right, subset_da.sizes["x"])
            approx_y = np.linspace(top, bottom, subset_da.sizes["y"])[::-1]  # flip for north-up
            subset_da = subset_da.assign_coords(x=approx_x, y=approx_y)
            subset_da = subset_da.rio.write_crs("EPSG:4326")

            subset_vars[band_name] = subset_da

        subset_ds = xr.Dataset(subset_vars)
        subset_ds.attrs = ds.attrs.copy()

        print(f"subset shape (y, x): {subset_da.sizes['y']}, {subset_da.sizes['x']}")

        return subset_ds


    # ----------- APPLY ORBIT FILE --------------
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

    
    # ----------- DATA CORRECTIONS : THERMAL NOISE REMOVAL --------------
    def remove_thermal_noise(self, safe_folder,ds):

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
        # thermal_lut_ds=parse_thermal_noise_removal_lut(safe_folder)
        # print(thermal_lut_ds, thermal_lut_ds)
        # result=apply_correction("thermal_noise_removal",ds, thermal_lut_ds)
        # print("Thermal noise removed successfully")
        # return result
    
        # def remove_thermal_noise(safe_folder, ds):
        # """
        # Remove thermal noise from Sentinel-1 bands (VV/VH/etc.).
        # Ensures line/pixel coordinates exist.
        # """
        # # Make sure dataset has coordinates
        
        for pol in ds.data_vars:
            da = ds[pol]
            if 'line' not in da.coords or 'pixel' not in da.coords:
                lines = np.arange(da.shape[0])
                pixels = np.arange(da.shape[1])
                ds[pol] = xr.DataArray(da.values, dims=("line", "pixel"),
                                    coords={"line": lines, "pixel": pixels})

        thermal_lut_ds = parse_thermal_noise_removal_lut(safe_folder)
        corrected_ds = apply_correction("thermal_noise_removal", ds, thermal_lut_ds)
        print("Thermal noise removed successfully")
        return corrected_ds

    # ----------- DATA CORRECTIONS : BORDER NOISE REMOVAL --------------
    def remove_border_noise(self, safe_folder, ds, blocksize=2000, threshold=0.5):
        """
        Remove Sentinel-1 GRD border noise from an xarray.Dataset.

        Parameters
            safe_folder (str or Path): Path to Sentinel-1 SAFE folder.
            ds(xarray.Dataset): GRD dataset with dimensions (line, pixel) or (y, x).
            blocksize (int, optional): Border size in pixels (default: 2000).
            threshold (float, optional): Threshold to keep pixels after noise subtraction (default: 0.5).

        Returns
            xarray.Dataset: Noise-corrected dataset.
        """
        scaling_factor, ipf_version = compute_scaling_factor(safe_folder)

        if ipf_version >= 2.9:
            print(f"[INFO] IPF version {ipf_version} ≥ 2.9 → border noise removal is usually not necessary.")
            return ds
        else:
            print(f"[INFO] IPF version {ipf_version} < 2.9 → removing border noise...")

        noise_vectors = parse_noise_vectors(safe_folder)

        # Determine shape
        lines = ds.dims.get('line', ds.dims.get('y'))
        samples = ds.dims.get('pixel', ds.dims.get('x'))
        shape = (lines, samples)

        # Build noise map
        noise_map = build_noise_map(noise_vectors, shape, blocksize)

        corrected = {}
        for band in ds.data_vars:
            arr = ds[band].values.astype(float)
            mask = np.ones_like(arr, dtype=bool)

            # Top and bottom borders
            for slc in [slice(0, blocksize), slice(lines - blocksize, lines)]:
                arr_part = arr[slc, :]
                noise_part = noise_map[slc, :]
                denoised = arr_part**2 - noise_part * scaling_factor
                mask[slc, :] &= denoised > threshold

            # Left and right borders
            for slc in [slice(0, blocksize), slice(samples - blocksize, samples)]:
                arr_part = arr[:, slc]
                noise_part = noise_map[:, slc]
                denoised = arr_part**2 - noise_part * scaling_factor
                mask[:, slc] &= denoised > threshold

            arr_masked = np.where(mask, arr, 0)
            corrected[band] = (ds[band].dims, arr_masked)

        return xr.Dataset(corrected, coords=ds.coords)

    # ----------- RADIOMETRIC CALIBRATION --------------
    def radiometric_calibration(self, safe_folder, ds, representation_type="sigmaNought"):
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
        sigma_nought_lut=parse_radiometric_calibration_lut(safe_folder, representation_type)
        result= apply_correction("radiometric_calibration", ds, sigma_nought_lut)
        print("Radiometric calibration completed successfully")
        return result
    
    # ----------- SPECKLE FILTERING --------------
    def speckle_filter(self, ds, method = 'lee', size=7):
        """
        Apply speckle filtering to all bands in xarray.Dataset.
        Currently supports only Lee filter as filtering method

        Parameters
            ds (xarray.Dataset): Input dataset with dimensions ('line','pixel') or ('y','x').
            method (str, optional): Filtering method ('lee' supported, default).
            size (int, optional): Window size (default: 7).

        Returns
            xarray.Dataset : Dataset with new variables '{band}_filtered'.
        """
        if method != 'lee':
            raise NotImplementedError(f"Speckle filter method '{method}' not implemented.")

        filtered_vars = {}
        for band in ds.data_vars:
            arr = ds[band]

            # Create valid data mask (avoid propagating NaNs)
            valid_mask = np.isfinite(arr)
            arr_filled = arr.fillna(0)

            # Apply Lee filter using xarray.apply_ufunc
            filtered = xr.apply_ufunc(
                lee_filter,
                arr_filled,
                kwargs={'size': size},
                input_core_dims=[['line', 'pixel']],   # adjust if dimensions are ('y','x')
                output_core_dims=[['line', 'pixel']],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[arr.dtype]
            )

            # Restore original NaNs
            filtered = filtered.where(valid_mask)

            filtered_vars[f"{band}_filtered"] = filtered

        return ds.assign(**filtered_vars)
    
    # ----------- CONVERSION TO DECIBEL (dB) --------------
    def convert_to_db(self, ds, bands=['VV', 'VH'], floor=1e-10):
        """
        Convert specified bands in an xarray.Dataset to decibel (dB) scale.
        Adds new variables with '_db' suffix to the dataset.

        Parameters
            ds (xarray.Dataset): Input dataset containing power or intensity bands.
            bands (list of str, optional): Names of bands to convert. Defaults to ['VV', 'VH'].
            floor (float, optional): Minimum value to avoid log(0). Default is 1e-10.

        Returns
            xarray.Dataset :Dataset with additional bands in dB (e.g., 'VV_db', 'VH_db').
        """
        new_vars = {}
        for band in bands:
            if band in ds:
                db = 10 * np.log10(np.maximum(ds[band], floor))
                new_vars[f"{band}_db"] = db
            else:
                print(f"Warning: band '{band}' not found in dataset; skipping.")
        return ds.assign(**new_vars)


