import numpy as np
import geopandas as gpd
from shapely.geometry import box
import rasterio
import rasterio.transform
from rasterio.transform import from_bounds
import pytest

from geofeatures.core import compute_ndvi, extract_zonal_features, compute_evi, compute_gndvi, compute_mndwi, compute_ndmi, compute_bsi, compute_nbr, compute_ndwi, compute_savi, compute_ndbi, merge_shapefiles, dissolve_by_attribute, clip_vector, clip_raster_by_vector


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


def test_compute_evi_known_values():
    # NIR=0.6, Red=0.2, Blue=0.1 -> 2.5*(0.6-0.2)/(0.6+6*0.2-7.5*0.1+1) = 2.5*0.4/2.05 = 0.4878
    nir = np.array([[0.6]], dtype="float32")
    red = np.array([[0.2]], dtype="float32")
    blue = np.array([[0.1]], dtype="float32")
    evi = compute_evi(nir, red, blue)
    assert np.isclose(evi[0, 0], 0.4878, atol=1e-3)


def test_compute_gndvi_known_values():
    # NIR=0.7, Green=0.3 -> (0.7-0.3)/(0.7+0.3) = 0.4
    nir = np.array([[0.7]], dtype="float32")
    green = np.array([[0.3]], dtype="float32")
    gndvi = compute_gndvi(nir, green)
    assert np.isclose(gndvi[0, 0], 0.4, atol=1e-3)


def test_compute_mndwi_known_values():
    # Green=0.5, SWIR=0.2 -> (0.5-0.2)/(0.5+0.2) = 0.4286
    green = np.array([[0.5]], dtype="float32")
    swir = np.array([[0.2]], dtype="float32")
    mndwi = compute_mndwi(green, swir)
    assert np.isclose(mndwi[0, 0], 0.4286, atol=1e-3)


def test_compute_ndmi_known_values():
    # NIR=0.6, SWIR=0.3 -> (0.6-0.3)/(0.6+0.3) = 0.3333
    nir = np.array([[0.6]], dtype="float32")
    swir = np.array([[0.3]], dtype="float32")
    ndmi = compute_ndmi(nir, swir)
    assert np.isclose(ndmi[0, 0], 0.3333, atol=1e-3)


def test_compute_bsi_known_values():
    # SWIR=0.4, Red=0.3, NIR=0.5, Blue=0.1 -> ((0.4+0.3)-(0.5+0.1))/((0.4+0.3)+(0.5+0.1)) = (0.7-0.6)/(0.7+0.6) = 0.0769
    swir = np.array([[0.4]], dtype="float32")
    red = np.array([[0.3]], dtype="float32")
    nir = np.array([[0.5]], dtype="float32")
    blue = np.array([[0.1]], dtype="float32")
    bsi = compute_bsi(swir, red, nir, blue)
    assert np.isclose(bsi[0, 0], 0.0769, atol=1e-3)


def test_compute_nbr_known_values():
    # NIR=0.7, SWIR2=0.2 -> (0.7-0.2)/(0.7+0.2) = 0.5556
    nir = np.array([[0.7]], dtype="float32")
    swir2 = np.array([[0.2]], dtype="float32")
    nbr = compute_nbr(nir, swir2)
    assert np.isclose(nbr[0, 0], 0.5556, atol=1e-3)


def test_compute_ndwi_known_values():
    # Green=0.5, NIR=0.1 -> (0.5-0.1)/(0.5+0.1) = 0.6667
    green = np.array([[0.5]], dtype="float32")
    nir = np.array([[0.1]], dtype="float32")
    ndwi = compute_ndwi(green, nir)
    assert np.isclose(ndwi[0, 0], 0.6667, atol=1e-3)


def test_compute_savi_known_values():
    # NIR=0.8, Red=0.2, L=0.5 -> ((0.8-0.2)/(0.8+0.2+0.5)) * 1.5 = 0.6
    nir = np.array([[0.8]], dtype="float32")
    red = np.array([[0.2]], dtype="float32")
    savi = compute_savi(nir, red, L=0.5)
    assert np.isclose(savi[0, 0], 0.6, atol=1e-3)


def test_compute_ndbi_known_values():
    # SWIR=0.5, NIR=0.3 -> (0.5-0.3)/(0.5+0.3) = 0.25
    swir = np.array([[0.5]], dtype="float32")
    nir = np.array([[0.3]], dtype="float32")
    ndbi = compute_ndbi(swir, nir)
    assert np.isclose(ndbi[0, 0], 0.25, atol=1e-3)


def test_merge_shapefiles():
    from shapely.geometry import Point

    gdf1 = gpd.GeoDataFrame({"name": ["A"]}, geometry=[Point(0, 0)], crs="EPSG:4326")
    gdf2 = gpd.GeoDataFrame({"name": ["B"]}, geometry=[Point(1, 1)], crs="EPSG:4326")

    merged = merge_shapefiles([gdf1, gdf2])

    assert len(merged) == 2
    assert set(merged["name"]) == {"A", "B"}
    assert merged.crs == gdf1.crs


def test_merge_shapefiles_different_crs():
    from shapely.geometry import Point

    gdf1 = gpd.GeoDataFrame({"name": ["A"]}, geometry=[Point(0, 0)], crs="EPSG:4326")
    gdf2 = gpd.GeoDataFrame({"name": ["B"]}, geometry=[Point(500000, 500000)], crs="EPSG:32631")

    merged = merge_shapefiles([gdf1, gdf2], target_crs="EPSG:4326")

    assert len(merged) == 2
    assert merged.crs.to_epsg() == 4326


def test_dissolve_by_attribute():
    from shapely.geometry import box

    gdf = gpd.GeoDataFrame(
        {"region": ["north", "north", "south"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(0, -1, 1, 0)],
        crs="EPSG:4326"
    )

    dissolved = dissolve_by_attribute(gdf, "region")

    assert len(dissolved) == 2
    assert set(dissolved["region"]) == {"north", "south"}


def test_dissolve_by_attribute_missing_column():
    from shapely.geometry import box

    gdf = gpd.GeoDataFrame({"region": ["north"]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")

    with pytest.raises(ValueError):
        dissolve_by_attribute(gdf, "nonexistent_column")


def test_clip_vector():
    from shapely.geometry import box

    data = gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[box(0, 0, 2, 2), box(5, 5, 7, 7)],
        crs="EPSG:4326"
    )
    boundary = gpd.GeoDataFrame({"id": [0]}, geometry=[box(-1, -1, 3, 3)], crs="EPSG:4326")

    clipped = clip_vector(data, boundary)

    # Only the first box (0,0,2,2) overlaps the boundary (-1,-1,3,3)
    assert len(clipped) == 1
    assert clipped.iloc[0]["id"] == 1


def test_clip_raster_by_vector(tmp_path):
    from shapely.geometry import box

    raster_path = tmp_path / "test_raster.tif"
    data = np.ones((10, 10), dtype="float32")
    transform = rasterio.transform.from_bounds(0, 0, 10, 10, 10, 10)

    with rasterio.open(
        raster_path, "w", driver="GTiff",
        height=10, width=10, count=1, dtype="float32",
        crs="EPSG:4326", transform=transform
    ) as dst:
        dst.write(data, 1)

    clip_boundary = gpd.GeoDataFrame(
        {"id": [1]}, geometry=[box(0, 0, 5, 10)], crs="EPSG:4326"
    )

    output_path = tmp_path / "clipped.tif"
    clip_raster_by_vector(str(raster_path), clip_boundary, str(output_path))

    with rasterio.open(output_path) as clipped_src:
        clipped_data = clipped_src.read(1)
        assert clipped_data.shape[1] <= 6
        assert clipped_data.shape[1] >= 4
