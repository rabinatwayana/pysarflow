# -*- coding: utf-8 -*-
"""
It includes common or secondary functionalities that would be used in other modules.

"""
import shapefile
import pygeoif
from esa_snappy import jpy, GPF

def get_subswath(aoi, product):
    """
    Identify the Sentinel-1 subswath (e.g., IW1, IW2, or IW3) covering the given Area of Interest (AOI).

    This function determines which Sentinel-1 interferometric wide (IW) subswath contains 
    the centroid of a given AOI. This is useful when selecting or cropping data specific 
    to one of the three IW subswaths.

    Parameters:
    ----------
    aoi : str | shapely.geometry.Polygon | list[tuple[float, float]]
        The Area of Interest to test for subswath coverage. It can be:
        - A WKT polygon string (e.g., 'POLYGON((lon lat, lon lat, ...))')
        - A Shapely `Polygon` object
        - A list or tuple of (lon, lat) coordinate pairs

    product : org.esa.snap.core.datamodel.Product
        The Sentinel-1 product containing IW subswath bands.

    Returns:
    -------
    result : str
        The name of the subswath ('IW1', 'IW2', or 'IW3') that contains the AOI centroid.
        Returns 'No subswath for the AOI' if none of the available subswaths cover it.

    Notes:
    -----
    - The function uses the product's band names to infer available subswaths.
    - It checks whether the AOI centroid lies within the bounds of each subswath's geocoded raster.
    - Requires that the product bands include geocoding information.
    """
    if isinstance(aoi, str) and aoi.startswith('POLYGON'):
        coords_str = aoi.replace('POLYGON((', '').replace('))', '')
        coord_pairs = coords_str.split(',')
        coords = []
        for pair in coord_pairs:
            lon, lat = map(float, pair.strip().split())
            coords.append((lon, lat))
        aoi_polygon = Polygon(coords)
    elif isinstance(aoi, Polygon):
        aoi_polygon = aoi
    elif isinstance(aoi, (list, tuple)):
        aoi_polygon = Polygon(aoi)
    else:
        raise ValueError("AOI must be a WKT string, Shapely Polygon, or list of (lon, lat) tuples")
    
    centroid = aoi_polygon.centroid
    
    band_names = list(product.getBandNames())
    subswaths = set()
    
    for band_name in band_names:
        if '_IW' in band_name:
            sw_part = band_name.split('_IW')[1][:1]
            if sw_part.isdigit():
                subswaths.add(f"IW{sw_part}")
                
    result = "No subswath for the AOI"
    for subswath in sorted(subswaths):
        subswath_bands = [band for band in band_names if f'_IW{subswath[-1]}_' in band]
        if not subswath_bands:
            continue
            
        band = product.getBand(subswath_bands[0])
        geo_coding = band.getGeoCoding()
        
        if geo_coding:
            pixel_pos = geo_coding.getPixelPos(esa_snappy.GeoPos(centroid.y, centroid.x), None)
            width = band.getRasterWidth()
            height = band.getRasterHeight()
            
            if 0 <= pixel_pos.x < width and 0 <= pixel_pos.y < height:
                result = subswath
    
    return result


def extract_bbox(file_path):
    """
    Extracts a bounding box from a shapefile and returns it as a WKT polygon.

    Args:
        file_path (str): Path to the shapefile (.shp) containing the boundary geometry.

    Returns:
        str: WKT string representing the bounding polygon for use in SNAP SubsetOp.
    """
    r = shapefile.Reader("data/island_boundary2.shp")
    g=[]
    for s in r.shapes():
        g.append(pygeoif.geometry.as_shape(s))
    m = pygeoif.MultiPoint(g)
    wkt = str(m.wkt).replace("MULTIPOINT","POLYGON(") + ")"
    SubsetOp = jpy.get_type('org.esa.snap.core.gpf.common.SubsetOp')
    bounding_wkt = wkt
    return bounding_wkt


def convert_0_to_nan(product):
    """
    Convert all zero values in the bands of a Sentinel-1 product to NaN (represented as -9999.0).

    This function iterates over all bands in the input product and replaces
    pixels with a value of 0 with -9999.0, which is commonly used as the NoData value
    in SNAP products. The data type of each band is set to float32 to accommodate NaN values.

    Args:
        product (esa_snappy.Product): Sentinel-1 product whose zero values need to be converted.

    Returns:
        esa_snappy.Product: A new product with zero values replaced by -9999.0 in all bands.
    """
    band_names = list(product.getBandNames())
    BandDescriptor = jpy.get_type('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor')

    # Create Java array of BandDescriptor
    Array = jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', len(band_names))

    for i, name in enumerate(band_names):
        band_def = BandDescriptor()
        band_def.name = name   # avoid overwrite
        band_def.type = 'float32'
        band_def.expression = f"{name} == 0 ? -9999.0 : {name}"
        band_def.noDataValue = -9999.0     # match replacement value
        Array[i] = band_def

    # Parameters HashMap
    parameters = jpy.get_type('java.util.HashMap')()
    parameters.put('targetBands', Array)

    # Run BandMaths
    updated_product = GPF.createProduct('BandMaths', parameters, product)
    return updated_product

def extract_info(product_path):
    """
    Extract and display basic information about a Sentinel-1 product.

    Parameters
    ----------
    product_path : str
        Path to the Sentinel-1 product file.

    Prints
    ------
    - Product name
    - Product type
    - Product description
    - Scene width and height
    - Acquisition start and end time
    - Number of bands and their names

    Notes
    -----
    This function uses the SNAP API to read the product and display its metadata. 
    It disposes of the product after extraction to free resources. 
    Useful for quickly inspecting a product before further processing.
    """
    product = read_product(product_path)
    print("Product name:", product.getName())
    print("Product type:", product.getProductType())
    print("Description:", product.getDescription())
    #print("Beam Mode:", check_beam_mode(product_path))
    print("Scene width:", product.getSceneRasterWidth())
    print("Scene height:", product.getSceneRasterHeight())

    metadata = product.getMetadataRoot()
    print("Start time:", product.getStartTime())
    print("End time:", product.getEndTime())

    print("\n Bands")
    count = 0
    band_names = product.getBandNames()
    number_bands = len(list(band_names))
    print("Number of bands:", number_bands)
    while count < number_bands:
        band = product.getBand(band_names[count])
        print("Band name:", band.getName())
        count += 1

    product.dispose()
    print("\n")
    return

