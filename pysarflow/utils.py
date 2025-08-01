# -*- coding: utf-8 -*-
"""
It includes common or secondary functionalities that would be used in other modules.

"""
def get_subswath(aoi, product):
    ):
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