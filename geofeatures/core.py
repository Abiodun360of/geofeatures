import numpy as np
import geopandas as gpd
from rasterstats import zonal_stats
import rasterio


def extract_zonal_features(raster_path, vector_gdf, stats=("mean", "std", "min", "max", "count")):
    """
    Extract zonal statistics from a raster for each polygon in a GeoDataFrame.

    Parameters
    ----------
    raster_path : str
        Path to the raster file (GeoTIFF).
    vector_gdf : geopandas.GeoDataFrame
        Polygons to compute statistics within.
    stats : tuple of str
        Statistics to compute (mean, std, min, max, count, etc.)

    Returns
    -------
    geopandas.GeoDataFrame
        Original vector_gdf with new statistic columns appended.
    """
    results = zonal_stats(vector_gdf, raster_path, stats=stats)
    stats_df = gpd.pd.DataFrame(results)
    return vector_gdf.reset_index(drop=True).join(stats_df)


def compute_ndvi(nir_band, red_band):
    """
    Compute NDVI from NIR and Red bands.

    Parameters
    ----------
    nir_band, red_band : array-like (numpy or xarray)
        Near-infrared and red reflectance bands, same shape.

    Returns
    -------
    array-like
        NDVI values, range -1 to 1.
    """
    nir = nir_band.astype("float32")
    red = red_band.astype("float32")
    return (nir - red) / (nir + red + 1e-6)


def ndvi_zonal_stats(nir_band, red_band, vector_gdf, raster_template_path, stats=("mean", "std", "min", "max", "count")):
    """
    Convenience function: compute NDVI from NIR/Red bands, then extract
    zonal statistics for each polygon in one call.

    Parameters
    ----------
    nir_band, red_band : array-like
        Near-infrared and red reflectance bands (same shape, e.g. from rioxarray).
    vector_gdf : geopandas.GeoDataFrame
        Polygons to compute statistics within.
    raster_template_path : str
        Path to save the intermediate NDVI raster (needed by rasterstats).
    stats : tuple of str
        Statistics to compute.

    Returns
    -------
    geopandas.GeoDataFrame
        vector_gdf with NDVI statistic columns appended.
    """
    ndvi = compute_ndvi(nir_band, red_band)
    ndvi.rio.to_raster(raster_template_path)
    return extract_zonal_features(raster_template_path, vector_gdf, stats=stats)


def compute_evi(nir_band, red_band, blue_band, G=2.5, C1=6.0, C2=7.5, L=1.0):
    """
    Compute EVI (Enhanced Vegetation Index).
    Improves on NDVI in dense vegetation, corrects for atmospheric and soil effects.
    Formula: G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L)
    """
    nir = nir_band.astype("float32")
    red = red_band.astype("float32")
    blue = blue_band.astype("float32")
    return G * (nir - red) / (nir + C1 * red - C2 * blue + L + 1e-6)


def compute_gndvi(nir_band, green_band):
    """
    Compute GNDVI (Green Normalized Difference Vegetation Index).
    Sensitive to chlorophyll concentration.
    Formula: (NIR - Green) / (NIR + Green)
    """
    nir = nir_band.astype("float32")
    green = green_band.astype("float32")
    return (nir - green) / (nir + green + 1e-6)


def compute_mndwi(green_band, swir_band):
    """
    Compute MNDWI (Modified Normalized Difference Water Index).
    Better than NDWI at detecting water in urban/built-up areas.
    Formula: (Green - SWIR) / (Green + SWIR)
    """
    green = green_band.astype("float32")
    swir = swir_band.astype("float32")
    return (green - swir) / (green + swir + 1e-6)


def compute_ndmi(nir_band, swir_band):
    """
    Compute NDMI (Normalized Difference Moisture Index).
    Measures vegetation water/moisture content.
    Formula: (NIR - SWIR) / (NIR + SWIR)
    """
    nir = nir_band.astype("float32")
    swir = swir_band.astype("float32")
    return (nir - swir) / (nir + swir + 1e-6)


def compute_bsi(swir_band, red_band, nir_band, blue_band):
    """
    Compute BSI (Bare Soil Index).
    Highlights bare/exposed soil.
    Formula: ((SWIR + Red) - (NIR + Blue)) / ((SWIR + Red) + (NIR + Blue))
    """
    swir = swir_band.astype("float32")
    red = red_band.astype("float32")
    nir = nir_band.astype("float32")
    blue = blue_band.astype("float32")
    return ((swir + red) - (nir + blue)) / ((swir + red) + (nir + blue) + 1e-6)


def compute_nbr(nir_band, swir2_band):
    """
    Compute NBR (Normalized Burn Ratio).
    Used to assess burn severity / fire damage.
    Formula: (NIR - SWIR2) / (NIR + SWIR2)
    Note: SWIR2 refers to the longer-wavelength SWIR band (e.g. Sentinel-2 B12).
    """
    nir = nir_band.astype("float32")
    swir2 = swir2_band.astype("float32")
    return (nir - swir2) / (nir + swir2 + 1e-6)


def compute_ndwi(green_band, nir_band):
    """
    Compute NDWI (Normalized Difference Water Index).
    Highlights water bodies. Formula: (Green - NIR) / (Green + NIR)
    """
    green = green_band.astype("float32")
    nir = nir_band.astype("float32")
    return (green - nir) / (green + nir + 1e-6)


def compute_savi(nir_band, red_band, L=0.5):
    """
    Compute SAVI (Soil-Adjusted Vegetation Index).
    Formula: ((NIR - Red) / (NIR + Red + L)) * (1 + L)
    """
    nir = nir_band.astype("float32")
    red = red_band.astype("float32")
    return ((nir - red) / (nir + red + L + 1e-6)) * (1 + L)


def compute_ndbi(swir_band, nir_band):
    """
    Compute NDBI (Normalized Difference Built-up Index).
    Formula: (SWIR - NIR) / (SWIR + NIR)
    """
    swir = swir_band.astype("float32")
    nir = nir_band.astype("float32")
    return (swir - nir) / (swir + nir + 1e-6)


def merge_shapefiles(gdf_list, target_crs=None):
    """
    Merge multiple GeoDataFrames into one, reprojecting to a common CRS first.

    Parameters
    ----------
    gdf_list : list of geopandas.GeoDataFrame
        GeoDataFrames to merge. Can have different CRSs.
    target_crs : str or None
        CRS to reproject all inputs to before merging. If None, uses the
        CRS of the first GeoDataFrame in the list.

    Returns
    -------
    geopandas.GeoDataFrame
        Combined GeoDataFrame with all input features.
    """
    if not gdf_list:
        raise ValueError("gdf_list cannot be empty")

    if target_crs is None:
        target_crs = gdf_list[0].crs

    reprojected = [gdf.to_crs(target_crs) for gdf in gdf_list]
    merged = gpd.pd.concat(reprojected, ignore_index=True)
    return gpd.GeoDataFrame(merged, crs=target_crs)


def dissolve_by_attribute(vector_gdf, attribute, agg_func="first"):
    """
    Dissolve (merge) polygons that share the same value in a given attribute column.

    Parameters
    ----------
    vector_gdf : geopandas.GeoDataFrame
        Input polygons.
    attribute : str
        Column name to dissolve by (e.g. "state_name").
    agg_func : str or dict, default "first"
        Aggregation function for other columns during dissolve.

    Returns
    -------
    geopandas.GeoDataFrame
        Dissolved GeoDataFrame, one row per unique attribute value.
    """
    if attribute not in vector_gdf.columns:
        raise ValueError(f"Column '{attribute}' not found in vector_gdf")

    return vector_gdf.dissolve(by=attribute, aggfunc=agg_func).reset_index()


def clip_vector(vector_gdf, clip_boundary_gdf):
    """
    Clip a GeoDataFrame to the boundary of another GeoDataFrame,
    automatically handling CRS mismatches.

    Parameters
    ----------
    vector_gdf : geopandas.GeoDataFrame
        The data to be clipped.
    clip_boundary_gdf : geopandas.GeoDataFrame
        The boundary to clip to.

    Returns
    -------
    geopandas.GeoDataFrame
        Clipped GeoDataFrame, reprojected to match vector_gdf's original CRS.
    """
    if vector_gdf.crs != clip_boundary_gdf.crs:
        clip_boundary_gdf = clip_boundary_gdf.to_crs(vector_gdf.crs)

    return gpd.clip(vector_gdf, clip_boundary_gdf)


def clip_raster_by_vector(raster_path, vector_gdf, output_path):
    """
    Clip a raster to the boundary of a vector GeoDataFrame, handling
    CRS mismatches automatically.
    """
    import rasterio.mask

    with rasterio.open(raster_path) as src:
        if vector_gdf.crs != src.crs:
            vector_gdf = vector_gdf.to_crs(src.crs)

        geometries = vector_gdf.geometry.values

        out_image, out_transform = rasterio.mask.mask(src, geometries, crop=True)
        out_meta = src.meta.copy()

    out_meta.update({
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

    with rasterio.open(output_path, "w", **out_meta) as dst:
        dst.write(out_image)

    return output_path


def convert_vector_format(input_path, output_path):
    """
    Convert a vector file between formats (Shapefile, GeoJSON, GPKG, KML).
    Format is inferred from the output_path file extension.
    """
    driver_map = {
        ".shp": "ESRI Shapefile",
        ".geojson": "GeoJSON",
        ".json": "GeoJSON",
        ".gpkg": "GPKG",
        ".kml": "KML",
    }

    ext = "." + output_path.rsplit(".", 1)[-1].lower()
    if ext not in driver_map:
        raise ValueError(f"Unsupported output extension '{ext}'. Supported: {list(driver_map.keys())}")

    driver = driver_map[ext]
    gdf = gpd.read_file(input_path)

    if driver == "KML" and gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    try:
        gdf.to_file(output_path, driver=driver)
    except Exception as e:
        raise ValueError(
            f"Failed to write '{driver}' format. Your GDAL/OGR build may not support this driver. "
            f"Original error: {e}"
        )

    return gdf


def load_kmz(path):
    """
    Load a KMZ file (zipped KML) into a GeoDataFrame.
    """
    import zipfile
    import tempfile
    import os

    with zipfile.ZipFile(path, "r") as z:
        kml_names = [n for n in z.namelist() if n.lower().endswith(".kml")]
        if not kml_names:
            raise ValueError(f"No .kml file found inside {path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            z.extract(kml_names[0], tmpdir)
            kml_path = os.path.join(tmpdir, kml_names[0])
            gdf = gpd.read_file(kml_path, driver="KML")

    return gdf


def load_vector(path, target_crs=None, fix_invalid=True):
    """
    Load a vector file into a GeoDataFrame, with optional CRS reprojection
    and automatic geometry repair.
    """
    gdf = gpd.read_file(path)

    if gdf.crs is None:
        raise ValueError(f"No CRS found in {path} — cannot proceed safely without a defined CRS")

    if fix_invalid:
        invalid_mask = ~gdf.geometry.is_valid
        if invalid_mask.any():
            gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].buffer(0)

    if target_crs is not None:
        gdf = gdf.to_crs(target_crs)

    return gdf


def load_raster_as_array(path):
    """
    Load a raster file and return both the pixel array and its metadata.
    """
    with rasterio.open(path) as src:
        data = src.read(1)
        meta = {"crs": src.crs, "transform": src.transform}
    return data, meta


def compute_slope(dem_array, pixel_size, units="degrees"):
    """
    Compute slope from a DEM array.

    Parameters
    ----------
    dem_array : numpy.ndarray
        2D array of elevation values.
    pixel_size : float
        Ground resolution of one pixel (e.g. 30 for 30m DEM).
    units : str, default "degrees"
        "degrees" or "percent" or "radians".

    Returns
    -------
    numpy.ndarray
        Slope values, same shape as dem_array (edges will have reduced accuracy).
    """
    dem = dem_array.astype("float64")
    dy, dx = np.gradient(dem, pixel_size)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))

    if units == "radians":
        return slope_rad
    elif units == "degrees":
        return np.degrees(slope_rad)
    elif units == "percent":
        return np.tan(slope_rad) * 100
    else:
        raise ValueError("units must be 'degrees', 'radians', or 'percent'")


def compute_aspect(dem_array, pixel_size):
    """
    Compute aspect (compass direction of slope) from a DEM array.

    Parameters
    ----------
    dem_array : numpy.ndarray
        2D array of elevation values.
    pixel_size : float
        Ground resolution of one pixel.

    Returns
    -------
    numpy.ndarray
        Aspect in degrees (0-360, where 0/360 = North, 90 = East, etc.)
        Flat areas (zero gradient) return -1.
    """
    dem = dem_array.astype("float64")
    dy, dx = np.gradient(dem, pixel_size)

    aspect_rad = np.arctan2(dy, -dx)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = 90.0 - aspect_deg
    aspect_deg = np.where(aspect_deg < 0, aspect_deg + 360, aspect_deg)

    flat_mask = (dx == 0) & (dy == 0)
    aspect_deg = np.where(flat_mask, -1, aspect_deg)

    return aspect_deg


def compute_hillshade(dem_array, pixel_size, azimuth=315, altitude=45):
    """
    Compute hillshade (simulated illumination) from a DEM array.

    Parameters
    ----------
    dem_array : numpy.ndarray
        2D array of elevation values.
    pixel_size : float
        Ground resolution of one pixel.
    azimuth : float, default 315
        Sun direction in degrees (0-360, 315 = NW, standard default).
    altitude : float, default 45
        Sun angle above horizon in degrees.

    Returns
    -------
    numpy.ndarray
        Hillshade values 0-255 (8-bit shading), same shape as dem_array.
    """
    dem = dem_array.astype("float64")
    dy, dx = np.gradient(dem, pixel_size)

    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect_rad = np.arctan2(dy, -dx)

    azimuth_rad = np.radians(360.0 - azimuth + 90)
    altitude_rad = np.radians(altitude)

    shaded = (
        np.sin(altitude_rad) * np.cos(slope_rad)
        + np.cos(altitude_rad) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad)
    )

    hillshade = 255 * (shaded + 1) / 2
    return np.clip(hillshade, 0, 255)


def distance_to_nearest(source_gdf, target_gdf, distance_col_name="dist_to_nearest"):
    """
    Compute the distance from each geometry in source_gdf to its nearest
    geometry in target_gdf.

    Parameters
    ----------
    source_gdf : geopandas.GeoDataFrame
        Geometries to measure distance FROM (e.g. farm parcels).
    target_gdf : geopandas.GeoDataFrame
        Geometries to measure distance TO (e.g. roads, rivers).
    distance_col_name : str, default "dist_to_nearest"
        Name of the new column to add with distance values.

    Returns
    -------
    geopandas.GeoDataFrame
        source_gdf with an added distance column. Units match the CRS
        (e.g. meters if projected CRS, degrees if geographic CRS —
        caller should ensure both inputs use a projected CRS for
        meaningful distances).
    """
    if source_gdf.crs != target_gdf.crs:
        target_gdf = target_gdf.to_crs(source_gdf.crs)

    result = source_gdf.copy()
    distances = []

    for geom in source_gdf.geometry:
        dists_to_all = target_gdf.geometry.distance(geom)
        distances.append(dists_to_all.min())

    result[distance_col_name] = distances
    return result


def reproject_raster(input_path, output_path, target_crs, resampling_method="nearest"):
    """
    Reproject a raster to a new CRS.

    Parameters
    ----------
    input_path : str
        Path to the input raster.
    output_path : str
        Path to write the reprojected raster.
    target_crs : str
        Target CRS (e.g. "EPSG:4326").
    resampling_method : str, default "nearest"
        One of: "nearest", "bilinear", "cubic".

    Returns
    -------
    str
        The output_path, for convenience chaining.
    """
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    resampling_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
    }
    if resampling_method not in resampling_map:
        raise ValueError(f"resampling_method must be one of {list(resampling_map.keys())}")

    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": target_crs,
            "transform": transform,
            "width": width,
            "height": height
        })

        with rasterio.open(output_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=resampling_map[resampling_method]
                )

    return output_path


def resample_raster(input_path, output_path, target_resolution):
    """
    Resample a raster to a new pixel resolution (same CRS).
    """
    from rasterio.enums import Resampling as ResamplingEnum

    with rasterio.open(input_path) as src:
        scale_factor = src.res[0] / target_resolution
        new_width = int(src.width * scale_factor)
        new_height = int(src.height * scale_factor)

        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=ResamplingEnum.bilinear
        )

        new_transform = src.transform * src.transform.scale(
            (src.width / new_width), (src.height / new_height)
        )

        kwargs = src.meta.copy()
        kwargs.update({
            "height": new_height,
            "width": new_width,
            "transform": new_transform
        })

        with rasterio.open(output_path, "w", **kwargs) as dst:
            dst.write(data)

    return output_path
