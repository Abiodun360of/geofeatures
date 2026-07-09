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
