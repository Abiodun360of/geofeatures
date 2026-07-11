# GeoFeatures

Python library for spectral index computation, zonal statistics, and vector/raster operations for remote sensing ML pipelines.

## Install

```bash
pip install geofeatures
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

- `extract_zonal_features(raster_path, vector_gdf, stats)`
- `ndvi_zonal_stats(...)` — convenience wrapper (NDVI + zonal stats in one call)

## Vector operations

- `merge_shapefiles(gdf_list, target_crs=None)` — merge GeoDataFrames, handling CRS mismatches
- `dissolve_by_attribute(vector_gdf, attribute, agg_func="first")` — dissolve polygons by shared attribute
- `clip_vector(vector_gdf, clip_boundary_gdf)` — clip vector to a boundary

## Raster-vector operations

- `clip_raster_by_vector(raster_path, vector_gdf, output_path)` — clip a raster to a vector boundary

## Validated

Tested end-to-end on real Sentinel-2 L2A imagery over Ogun State, Nigeria. 18 unit tests with hand-verified formula outputs and edge-case coverage (CRS mismatches, missing columns, spatial overlap logic).

## License

MIT

## Visualization

- `plot_raster(raster_path, title, cmap)` — quick raster preview
- `plot_vector(vector_gdf, column, title)` — quick vector/choropleth preview
- `plot_zonal_result(vector_gdf, column, title)` — convenience wrapper for zonal stats results

## Format conversion

- `convert_vector_format(input_path, output_path)` — convert between Shapefile, GeoJSON, GPKG, KML
- `load_kmz(path)` — load a KMZ (zipped KML) file
- `load_vector(path, target_crs, fix_invalid)` — load with automatic geometry repair
- `load_raster_as_array(path)` — load raster as numpy array + metadata

## Terrain analysis

- `compute_slope(dem_array, pixel_size, units)` — slope in degrees/percent/radians
- `compute_aspect(dem_array, pixel_size)` — compass direction of slope (0-360°)
- `compute_hillshade(dem_array, pixel_size, azimuth, altitude)` — simulated illumination

## Spatial analysis

- `distance_to_nearest(source_gdf, target_gdf, distance_col_name)` — distance from each feature to nearest target feature

## Raster transformation

- `reproject_raster(input_path, output_path, target_crs, resampling_method)` — reproject to a new CRS
- `resample_raster(input_path, output_path, target_resolution)` — change pixel resolution
