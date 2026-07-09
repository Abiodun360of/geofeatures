import numpy as np
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.transform import from_bounds
import pytest

from geofeatures.core import compute_ndvi, extract_zonal_features


def test_compute_ndvi_known_values():
    # NIR=0.8, Red=0.2 -> NDVI = (0.8-0.2)/(0.8+0.2) = 0.6
    nir = np.array([[0.8]], dtype="float32")
    red = np.array([[0.2]], dtype="float32")
    ndvi = compute_ndvi(nir, red)
    assert np.isclose(ndvi[0, 0], 0.6, atol=1e-3)


def test_compute_ndvi_range():
    nir = np.random.uniform(0, 1, (10, 10)).astype("float32")
    red = np.random.uniform(0, 1, (10, 10)).astype("float32")
    ndvi = compute_ndvi(nir, red)
    assert ndvi.min() >= -1.001
    assert ndvi.max() <= 1.001


def test_extract_zonal_features_known_constant_raster(tmp_path):
    # Raster of constant value 5.0 everywhere
    raster_path = tmp_path / "constant.tif"
    data = np.full((10, 10), 5.0, dtype="float32")
    transform = from_bounds(0, 0, 10, 10, 10, 10)

    with rasterio.open(
        raster_path, "w", driver="GTiff",
        height=10, width=10, count=1, dtype="float32",
        crs="EPSG:4326", transform=transform
    ) as dst:
        dst.write(data, 1)

    polygon = gpd.GeoDataFrame({"name": ["test"]}, geometry=[box(1, 1, 9, 9)], crs="EPSG:4326")
    result = extract_zonal_features(str(raster_path), polygon, stats=("mean", "min", "max"))

    assert np.isclose(result["mean"].iloc[0], 5.0)
    assert np.isclose(result["min"].iloc[0], 5.0)
    assert np.isclose(result["max"].iloc[0], 5.0)
