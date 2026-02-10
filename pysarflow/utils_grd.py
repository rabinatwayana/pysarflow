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
import re

from pathlib import Path
from datetime import datetime
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree
from eof.download import download_eofs # from sentineleof package


# ----------- s3 URL CONVERSION --------------
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


#  ----------- STAC ITEMS TO GEODATAFRAME --------------
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

# ----------- BOUNDING BOX PLOTTING --------------
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

# ----------- THUMBNAIL PLOTTING --------------        
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


# ----------------- SUBSETTING TO AOI --------------
def parse_geolocation_grid(annotation_path: str):
    """
    Parse sparse geolocation grid from Sentinel-1 annotation XML.

    Args:
        annotation_path: Path to the annotation XML file

    Returns:
        dict containing:
            - latitude: 2D numpy array of latitudes
            - longitude: 2D numpy array of longitudes
            - lines: 1D array of sparse line indices
            - pixels: 1D array of sparse pixel indices
    """
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    
    points = []
    for point in root.findall(".//geolocationGridPoint"):
        points.append({
            'line': float(point.find("line").text),
            'pixel': float(point.find("pixel").text),
            'latitude': float(point.find("latitude").text),
            'longitude': float(point.find("longitude").text),
            'height': float(point.find("height").text)
        })
    
    lines = sorted(set(p['line'] for p in points))
    pixels = sorted(set(p['pixel'] for p in points))
    
    lat_grid = np.zeros((len(lines), len(pixels)))
    lon_grid = np.zeros((len(lines), len(pixels)))
    
    for p in points:
        i = lines.index(p['line'])
        j = pixels.index(p['pixel'])
        lat_grid[i, j] = p['latitude']
        lon_grid[i, j] = p['longitude']
    
    return {
        'latitude': lat_grid,
        'longitude': lon_grid,
        'lines': np.array(lines),
        'pixels': np.array(pixels)
    }


def build_geo_kdtree(lat_grid: np.ndarray, lon_grid: np.ndarray):
    """
    Build KDTree to invert lat/lon → sparse line/pixel indices.

    Args:
        lat_grid: 2D array of latitude values
        lon_grid: 2D array of longitude values

    Returns:
        tree: cKDTree
        n_lines: number of sparse lines
        n_pixels: number of sparse pixels
    """
    n_lines, n_pixels = lat_grid.shape
    points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    tree = cKDTree(points)
    return tree, n_lines, n_pixels

# ----------- LUT PARSING FOR THERMAL NOISE REMOVAL AND RADIOMETRIC CALIBRATION --------------
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

# ----------- RADIOMETRIC CALIBRATION --------------
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
        azimuth_times=[]

        for calib_vec in root.findall('.//calibrationVector'):
            line = int(calib_vec.find('line').text)
            pixel_str = calib_vec.find('pixel').text.strip()
            value_str = calib_vec.find(representation_type).text.strip()
            # azimuth_time = calib_vec.find('azimuthTime').text

            pixels = [int(x) for x in pixel_str.split()]
            values = [float(x) for x in value_str.split()]

            lines.append(line)
            correction_values.append(values)
            # azimuth_times.append(azimuth_time)

        beta_array = np.array(correction_values)
        lut = xr.DataArray(beta_array, coords={"line": lines, "pixel": pixels}, dims=["line", "pixel"])
        
        lut_dict[band_name]= lut    
    lut_ds = xr.Dataset(lut_dict)
    print("Radiometric calibration LUT created successfully")
    return lut_ds


def parse_beta_lut(safe_folder_path, representation_type="betaNought"):
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
        azimuth_times=[]

        for calib_vec in root.findall('.//calibrationVector'):
            # line = int(calib_vec.find('line').text)
            pixel_str = calib_vec.find('pixel').text.strip()
            value_str = calib_vec.find(representation_type).text.strip()
            azimuth_time = calib_vec.find('azimuthTime').text
            print(azimuth_time,"khjkhjk")

            pixels = [int(x) for x in pixel_str.split()]
            values = [float(x) for x in value_str.split()]

            # lines.append(line)
            correction_values.append(values)
            azimuth_times.append(azimuth_time)

        beta_array = np.array(correction_values)
        lut = xr.DataArray(beta_array, coords={"line": azimuth_times, "pixel": pixels}, dims=["line", "pixel"])

        # lut = xr.DataArray(beta_array, coords={"azimuth_time": azimuth_times, "ground_range": pixels}, dims=["azimuth_time", "ground_range"])
        
        lut_dict[band_name]= lut    
    lut_ds = xr.Dataset(lut_dict)
    print("Radiometric calibration LUT created successfully")
    return lut_ds


# -----------APPLY THERMAL NOISE REMOVAL AND RADIOMETRIC CALIBRATION --------------
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
    corrected_dict = {}
    for pol in ds.data_vars:
        da = ds[pol]
        if pol not in lut_ds:
            corrected_dict[pol] = da
            continue
        lut = lut_ds[pol]

        # Align LUT to dataset
        lut_aligned = lut.reindex(
            line=da.line,
            pixel=da.pixel,
            method="nearest",
            fill_value=0
        )

        if type == "thermal_noise_removal":
            corrected_dict[pol] = xr.where(da**2 - lut_aligned > 0, da**2 - lut_aligned, 0)
        elif type == "radiometric_calibration":
            corrected_dict[pol] = da / (lut_aligned**2)

    return xr.Dataset(corrected_dict)


# ---------- ORBIT FILE HANDLING --------------
def download_orbit_file(safe_folder, save_dir):
    """
    Extract date and mission from Sentinel-1 SAFE or zip filename,
    download corresponding precise orbit files, and return paths.
    """
    filename = os.path.basename(safe_folder)

    # Extract mission (e.g., S1A, S1B, S1C) from filename
    match_mission = re.match(r'(S1[ABC])_', filename)
    if not match_mission:
        raise ValueError("Mission not found in filename. Expected pattern 'S1A_', 'S1B_', or 'S1C_'.")
    mission = match_mission.group(1)

    # Extract date string in format YYYYMMDD from filename
    match_date = re.search(r'_(\d{8})T', filename)
    if not match_date:
        raise ValueError("Date not found in filename. Expected pattern '_YYYYMMDDT'.")
    date_str = match_date.group(1)
    date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

    # Download orbit file
    orbit_files = download_eofs([date_formatted], [mission], save_dir=save_dir)
    return orbit_files


def update_annotations_orbit(safe_folder, eof_orbit_file, overwrite=True):
    """
    Parses the EOF orbit file and replaces the orbitList in all annotation XML files.

    Args:
        safe_folder (str or Path): Path to Sentinel-1 SAFE folder.
        eof_orbit_file (str or Path): Path to precise orbit EOF file.
        overwrite (bool): Overwrite original annotation XMLs if True.

    Returns:
        None
    """
    #parsing the EOF file
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

    annotation_folder = Path(safe_folder) / "annotation"
    xml_files = [f for f in annotation_folder.glob("*.xml") if "_updated" not in f.stem]
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

        stem = xml_file.stem
        if stem.endswith("_updated"):
            stem = stem[:-8]

        out_path = xml_file.with_name(stem + "_updated.xml")
        if not overwrite and out_path.exists():
            out_path = xml_file.with_name(stem + "_updated_2.xml")
        tree.write(out_path, encoding="utf-8", xml_declaration=True)
        print(f"Updated orbitList in {out_path}")

def add_precise_orbit_coords(ds, annotation_xml_path):
    """
    Applies precise satellite orbit data to an xarray Dataset by interpolating positions
    and velocities from an annotation XML file.

    Args:
        ds (xarray.Dataset): The dataset to which orbit data will be assigned.
        annotation_xml_path (str or Path): Path to updated annotation XML.

    Returns:
        xarray.Dataset: Dataset with added orbit position and velocity coordinates.
                        - sat_pos_{x,y,z}: satellite positions (m)
                        - sat_vel_{x,y,z}: satellite velocities (m/s)
    """
    root = ET.parse(annotation_xml_path).getroot()
    orbits_xml = root.find(".//orbitList")
    orbits = []
    for orbit in orbits_xml.findall("orbit"):
        time = datetime.fromisoformat(orbit.find("time").text)
        pos = np.array([float(orbit.find(f"position/{axis}").text) for axis in ("x", "y", "z")])
        vel = np.array([float(orbit.find(f"velocity/{axis}").text) for axis in ("x", "y", "z")])
        orbits.append({"time": time, "position": pos, "velocity": vel})

    if "time" in ds.coords:
        times = [pd.to_datetime(t).to_pydatetime() for t in ds.time.values]
    else:
        start_time_str = ds.attrs.get("startTime")
        if not start_time_str:
            raise ValueError("No time info in dataset or metadata")
        times = [datetime.fromisoformat(start_time_str)]

    base_time = orbits[0]["time"]
    orbit_secs = np.array([(o["time"] - base_time).total_seconds() for o in orbits])
    pos_array = np.vstack([o["position"] for o in orbits])
    vel_array = np.vstack([o["velocity"] for o in orbits])
    target_secs = np.array([(t - base_time).total_seconds() for t in times])

    interp_pos = [interp1d(orbit_secs, pos_array[:, i], fill_value="extrapolate")(target_secs) for i in range(3)]
    interp_vel = [interp1d(orbit_secs, vel_array[:, i], fill_value="extrapolate")(target_secs) for i in range(3)]

    ds = ds.assign_coords(
        sat_pos_x=("time", interp_pos[0]),
        sat_pos_y=("time", interp_pos[1]),
        sat_pos_z=("time", interp_pos[2]),
        sat_vel_x=("time", interp_vel[0]),
        sat_vel_y=("time", interp_vel[1]),
        sat_vel_z=("time", interp_vel[2]),
    )
    return ds



# ----------- BORDER NOISE REMOVAL --------------
def get_ipf_version(safe_folder):
    """
    Extract the Sentinel-1 IPF processor version from manifest.safe.

    Parameters:
        safe_folder(str or Path): Path to Sentinel-1 SAFE folder.

    Returns:
        IPF processor version, e.g., 2.91.
    """
    manifest = Path(safe_folder) / "manifest.safe"
    with open(manifest, 'r', encoding='utf-8') as f:
        text = f.read()

    match = re.search(r'name="Sentinel-1 IPF"\s+version="([\d.]+)"', text)
    if match:
        version = float(match.group(1))
        return version
    raise RuntimeError("IPF version not found in manifest.safe")

def get_acquisition_mode(safe_folder):
    """
    Extract the acquisition mode (e.g., IW or EW) from manifest.safe.

    Parameters
    safe_folder(str or Path) : Path to Sentinel-1 SAFE folder.

    Returns
        Acquisition mode (str).
    """
    manifest_path = Path(safe_folder) / "manifest.safe"
    tree = ET.parse(manifest_path)
    root = tree.getroot()

    namespaces = {prefix: uri for event, (prefix, uri) in ET.iterparse(manifest_path, events=['start-ns'])}
    ns_uri = namespaces.get('s1sarl1')
    if ns_uri is None:
        raise RuntimeError("Namespace prefix 's1sarl1' not found in manifest.safe")

    instr_mode_tag = f'{{{ns_uri}}}instrumentMode'
    mode_tag = f'{{{ns_uri}}}mode'

    instr_mode_elem = root.find(f'.//{instr_mode_tag}')
    if instr_mode_elem is not None:
        mode_elem = instr_mode_elem.find(mode_tag)
        if mode_elem is not None:
            return mode_elem.text.strip()
    raise RuntimeError("Acquisition mode not found in manifest.safe")

def get_calibration_constant(safe_folder):
    """
    Extract the calibration constant (ADN) from calibration XML.

    Parameters
    safe_folder(str or Path) : Path to Sentinel-1 SAFE folder.

    Returns
        Calibration constant (float).
    """
    calib_files = list(Path(safe_folder).glob('annotation/calibration/*calibration-s1*-grd-*.xml'))
    if not calib_files:
        raise RuntimeError("Calibration XML files not found in SAFE folder")
    tree = ET.parse(calib_files[0])
    root = tree.getroot()
    dn_elem = root.find('.//calibrationVector/dn')
    if dn_elem is None or not dn_elem.text:
        raise RuntimeError("Calibration constant (dn) not found in calibration XML")
    adn = float(dn_elem.text.strip().split()[0])
    return adn

def compute_scaling_factor(safe_folder):
    """
    Compute noise scaling factor based on IPF version and acquisition mode.

    Parameters
    safe_folder (str or Path) : Path to Sentinel-1 SAFE folder.

    Returns
        scaling_factor (tuple): float, ipf_version: float 
    """
    ipf_version = get_ipf_version(safe_folder)
    acq_mode = get_acquisition_mode(safe_folder)
    adn = get_calibration_constant(safe_folder)

    knoise = {'IW': 75088.7, 'EW': 56065.87}
    if acq_mode not in knoise:
        raise RuntimeError(f"Unsupported acquisition mode: {acq_mode}")

    if ipf_version < 2.34:
        scaling_factor = knoise[acq_mode] * adn
    else:
        scaling_factor = knoise[acq_mode] * adn * adn

    return scaling_factor, ipf_version

def parse_noise_vectors(safe_folder):
    """
    Parse noise vectors from noise annotation XML files.

    Parameters
    safe_folder(str or Path):Path to Sentinel-1 SAFE folder.

    Returns
    list of tuples: (line_number, pixel_positions, noise_values)
    """
    noise_files = list(Path(safe_folder).glob('annotation/calibration/*noise*.xml'))
    if not noise_files:
        raise RuntimeError("No noise annotation XML files found in SAFE folder")

    noise_vectors = []
    for nf in noise_files:
        tree = ET.parse(nf)
        root = tree.getroot()
        for nv in root.findall('.//noiseVector'):
            line = int(nv.find('line').text)
            pixels = [int(x) for x in nv.find('pixel').text.strip().split()]
            noise_vals = [float(x) for x in nv.find('noiseLut').text.strip().split()]
            noise_vectors.append((line, pixels, noise_vals))
    return noise_vectors

def build_noise_map(noise_vectors, shape, blocksize):
    """
    Build a noise map array for image borders by interpolating noise vectors.

    Parameters
        noise_vectors (list):Output from parse_noise_vectors.
        shape (tuple): (lines, samples) shape of the dataset.
        blocksize (int): Size of border in pixels.

    Returns
        Noise map of same shape as image.(np.ndarray)
    """
    lines, samples = shape
    noise_map = np.zeros((lines, samples), dtype=float)

    vector_lines = np.array([nv[0] for nv in noise_vectors])

    for line, pixels, noises in noise_vectors:
        if line < blocksize or line >= lines - blocksize:
            interp = np.interp(np.arange(samples), pixels, noises)
            noise_map[line, :] = interp
    return noise_map



# ----------- FILTERING --------------
def lee_filter(img, size):
    """
    Apply Lee filter with specified window size.
    Adapted from https://stackoverflow.com/questions/39785970/speckle-lee-filter-in-python

    Parameters
        img(np.ndarray) :mInput image (2D array).
        size (int): Window size (e.g., 5, 7).

    Returns
        np.ndarray: Speckle-filtered image.
    """
    img_mean = uniform_filter(img, size)
    img_sqr_mean = uniform_filter(img ** 2, size)
    img_variance = img_sqr_mean - img_mean ** 2

    # Estimate overall variance from image
    overall_variance = np.mean(img_variance)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)

    return img_output