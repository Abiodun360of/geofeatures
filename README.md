# GeoFeatures

Lightweight Python library for extracting zonal statistics and spectral indices from satellite imagery for remote sensing ML pipelines.

## Why

Computing per-polygon raster statistics and spectral indices (e.g. mean NDVI per farm parcel) is a repetitive step in almost every remote sensing / geospatial ML workflow. GeoFeatures wraps this into a clean, tested API.

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

## Spectral indices

| Function | Detects | Inputs |
|---|---|---|
| `compute_ndvi()` | Vegetation health | NIR, Red |
| `compute_evi()` | Dense vegetation | NIR, Red, Blue |
| `compute_savi()` | Sparse vegetation | NIR, Red |
| `compute_gndvi()` | Chlorophyll | NIR, Green |
| `compute_ndwi()` | Surface water | Green, NIR |
| `compute_mndwi()` | Urban water | Green, SWIR |
| `compute_ndmi()` | Vegetation moisture | NIR, SWIR |
| `compute_ndbi()` | Built-up areas | SWIR, NIR |
| `compute_bsi()` | Bare soil | SWIR, Red, NIR, Blue |
| `compute_nbr()` | Burn severity | NIR, SWIR2 |

## Zonal statistics

- `extract_zonal_features(raster_path, vector_gdf, stats)` — statistics per polygon
- `ndvi_zonal_stats(...)` — convenience wrapper combining NDVI + zonal stats

## Validated

Tested end-to-end on real Sentinel-2 L2A imagery (via Microsoft Planetary Computer) over Ogun State, Nigeria. 12 unit tests covering all indices with hand-verified formula outputs.

## License

MIT
