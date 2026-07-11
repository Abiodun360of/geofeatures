import numpy as np
import geopandas as gpd
from rasterstats import zonal_stats


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
