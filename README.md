# GeoFeatures

Lightweight Python library for extracting zonal statistics and vegetation indices (NDVI) from satellite imagery for remote sensing ML pipelines.

## Why

Computing per-polygon raster statistics (e.g. mean NDVI per farm parcel or administrative region) is a repetitive step in almost every remote sensing / geospatial ML workflow. GeoFeatures wraps this into a clean, tested API.

## Install

```bash
pip install geofeatures
```

## Quick example

```python
from geofeatures.core import ndvi_zonal_stats
import geopandas as gpd

polygons = gpd.read_file("parcels.geojson")

result = ndvi_zonal_stats(
    nir_band=nir,
    red_band=red,
    vector_gdf=polygons,
    raster_template_path="ndvi.tif"
)

print(result[["name", "mean", "std"]])
```

## Validated on real data

Tested end-to-end on Sentinel-2 L2A imagery (via Microsoft Planetary Computer) over Ogun State, Nigeria — mean NDVI of 0.29 across ~4M pixels, consistent with mixed agricultural/urban land cover.

## API

- `compute_ndvi(nir_band, red_band)` — compute NDVI from raw bands
- `extract_zonal_features(raster_path, vector_gdf, stats)` — zonal statistics per polygon
- `ndvi_zonal_stats(nir_band, red_band, vector_gdf, raster_template_path)` — convenience wrapper combining both

## License

MIT
