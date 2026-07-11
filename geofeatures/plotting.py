"""
Quick visualization utilities for GeoFeatures outputs.
"""
import matplotlib.pyplot as plt
import rasterio


def plot_raster(raster_path, title=None, cmap="viridis", figsize=(8, 6)):
    """
    Quickly visualize a raster file (e.g. an NDVI or other index output).
    """
    with rasterio.open(raster_path) as src:
        data = src.read(1)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap=cmap)
    plt.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title(title or raster_path)
    ax.axis("off")
    return ax


def plot_vector(vector_gdf, column=None, title=None, cmap="viridis", figsize=(8, 6), legend=True):
    """
    Quickly visualize a GeoDataFrame, optionally colored by an attribute column.
    """
    fig, ax = plt.subplots(figsize=figsize)
    vector_gdf.plot(column=column, cmap=cmap, legend=legend, ax=ax, edgecolor="black", linewidth=0.5)
    ax.set_title(title or "")
    ax.axis("off")
    return ax


def plot_zonal_result(vector_gdf, column, title=None, cmap="RdYlGn", figsize=(8, 6)):
    """
    Convenience wrapper: visualize zonal statistics results as a choropleth map.
    """
    return plot_vector(vector_gdf, column=column, title=title or f"{column} by region", cmap=cmap, figsize=figsize)
