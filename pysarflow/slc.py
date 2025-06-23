# -*- coding: utf-8 -*-

"""It includes functions for handling Single Look Complex (SLC) datasets."""


def sum_slc(a, b):
    """Dockstring here."""
    return a + b

def extract_info(product_path):
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
    number_bands=len(list(band_names))
    print("Number of bands:", number_bands)
    while count<number_bands:
        band = product.getBand(band_names[count])
        print("Band name:", band.getName())
        count = count + 1   
    product.dispose()
    print("\n")
    return

def temporal_baseline(product1_path, product2_path):
    product1 =  read_product(product1_path)
    product2 =  read_product(product2_path)
    master_time = product1.getStartTime()
    slave_time = product2.getStartTime()
    temporal_baseline = abs(slave_time.getMJD() - master_time.getMJD())

    print(f"Temporal Baseline: {temporal_baseline:.1f} days")

    product1.dispose()
    product2.dispose()
    return

def read_product(product_path):
    product = ProductIO.readProduct(product_path)
    return product

def write(product, save_path, extension):
    print("Writing the product...")
    #GeoTIFF = .tif
    #BEAM-DIMAP = .dim
    ProductIO.writeProduct(product, save_path, extension)
    print("Product written!")

def get_subswath(aoi, product):
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

def check_same_subswath(product1, product2, aoi):
    subswath1 = get_subswath(aoi, product1)
    subswath2 = get_subswath(aoi, product2)
    if subswath1 == subswath2:
        print("Both product use the same subswath:", get_subswath(aoi, product1))
    else:
        print("For product", product1.getName(), "you'll need also to use to use the", get_subswath(aoi, product2), "subswath")
        print("For product", product2.getName(), "you'll need also to use to use the", get_subswath(aoi, product1), "subswath")

def topsar_split(product, subswath, polarization):
    parameters = HashMap()
    print('Apply TOPSAR Split...')
    parameters.put('subswath', subswath)
    #parameters.put('firstBurstIndex', 1) 
    #parameters.put('lastBurstIndex', 8) 
    parameters.put('selectedPolarisations', polarization)
    output = GPF.createProduct("TOPSAR-Split", parameters, product)
    print("TOPSAR Split applied!")
    return output

def apply_orbit_file(product):
    parameters = HashMap()
    print('Apply Orbit File...')
    parameters.put("Orbit State Vectors", "Sentinel Precise (Auto Download)")
    parameters.put("Polynomial Degree", 3)
    parameters.put("Do not fail if new orbit file is not found", True)
    output = GPF.createProduct("Apply-Orbit-File", parameters, product)
    print("Orbit File applied!") 
    return output

def back_geocoding(products, DEM="SRTM 1Sec HGT (Auto Download)"):
    parameters = HashMap()
    print('Back geocoding ...')
    parameters.put("Digital Elevation Model", DEM)
    parameters.put("demResamplingMethod", "BILINEAR_INTERPOLATION")
    parameters.put("resamplingType", "BILINEAR_INTERPOLATION")
    parameters.put("maskOutAreaWithoutElevation", True)
    parameters.put('disableSpectralDiversity', True)
    parameters.put("outputDerampDemodPhase", True)
    parameters.put("disableReramp", False)
    #parameters.put("The list of source bands", "")
    output = GPF.createProduct("Back-Geocoding", parameters, products) 
    print("Back geocoding applied!")
    return output

def enhanced_spectral_diversity(product):
    parameters = HashMap()
    print('Enhancing Spectral Diversity ...')
    parameters.put("fineWinWidthStr", "512")
    parameters.put("fineWinHeightStr", "512")
    parameters.put("fineWinAccAzimuth", "16")
    parameters.put("fineWinAccRange", "16")
    parameters.put("fineWinOversampling", "128")
    parameters.put("esdEstimator", "Periodogram")
    parameters.put("weightFunc", "Inv Quadratic")
    parameters.put("temporalBaselineType", "Number of images")
    parameters.put("integrationMethod", "L1 and L2")
    parameters.put("xCorrThreshold", 0.1)
    parameters.put("cohThreshold", 0.3)
    parameters.put("overallRangeShift", 0.0)
    parameters.put("overallAzimuthShift", 0.0)
    Integer = jpy.get_type('java.lang.Integer')
    parameters.put("numBlocksPerOverlap", Integer(10))
    parameters.put("maxTemporalBaseline", Integer(2))
    parameters.put("doNotWriteTargetBands", False)
    parameters.put("useSuppliedRangeShift", False)
    parameters.put("useSuppliedAzimuthShift", False)
    output = GPF.createProduct("Enhanced-Spectral-Diversity", parameters, product)
    print("Enhanced Spectral Diversity applied!")
    return output

def interferogram(product):
    parameters = HashMap()
    print('Creating interferogram ...')
    parameters.put("Subtract flat-earth phase", True)
    parameters.put("Degree of \"Flat Earth\" polynomial", 5)
    parameters.put("Number of \"Flat Earth\" estimation points", 501)
    parameters.put("Orbit interpolation degree", 3)
    parameters.put("Include coherence estimation", True)
    parameters.put("Square Pixel", True)
    parameters.put("Independent Window Sizes", False)
    #parameters.put("Coherence Azimuth Window Size", 10)
    #parameters.put("Coherence Range Window Size", 2)
    output = GPF.createProduct("Interferogram", parameters, product) 
    print("Interferogram created!")
    return output

def topsar_deburst(product, polarization):  
    parameters = HashMap()
    print('Apply TOPSAR Deburst...')
    parameters.put("Polarisations", polarization)
    output=GPF.createProduct("TOPSAR-Deburst", parameters, product)
    print("TOPSAR Deburst applied!")
    return output

def goldstein_phase_filtering(product):
    parameters = HashMap()
    print('Apply Goldstein Phase Filtering...')
    parameters.put('alpha', 1.0)
    parameters.put('FFTSizeString', '64')
    parameters.put('windowSizeString', '3')
    parameters.put('useCoherenceMask', False)
    parameters.put('coherenceThreshold', 0.2)  
    output = GPF.createProduct("GoldsteinPhaseFiltering", parameters, product)
    print("Goldstein Phase Filtering applied!")
    return output