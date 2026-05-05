# ================================================================
# Landsat NDSI Snow Cover Mapping Dashboard
# Fully open-source, non-Google-Earth-Engine workflow
#
# Data source:
#   Microsoft Planetary Computer STAC API
#   Landsat Collection 2 Level-2 Cloud Optimized GeoTIFFs
#
# Main features:
#   1. Draw AOI or upload shapefile / GeoJSON / GPKG
#   2. Search Landsat 4, 5, 7, 8, 9 Collection 2 Level-2 scenes
#   3. Read cloud-hosted COGs locally using rasterio
#   4. Apply QA_PIXEL cloud/shadow/fill/cirrus mask
#   5. Compute NDSI = (Green - SWIR1) / (Green + SWIR1)
#   6. Generate NDSI histogram
#   7. Derive snow threshold using Otsu or manual threshold
#   8. Calculate snow area
#   9. Export NDSI GeoTIFF, snow mask GeoTIFF, snow polygons SHP
#
# Author: Open-source geospatial dashboard template
# ================================================================


# ================================================================
# 1. Imports
# ================================================================

import os
import re
import io
import json
import math
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
import plotly.express as px

from shapely.geometry import shape, mapping
from shapely.ops import unary_union

from streamlit_folium import st_folium
from folium.plugins import Draw

import rasterio
from rasterio.crs import CRS
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.features import geometry_mask, shapes, sieve
from rasterio.warp import transform_bounds

from pystac_client import Client
import planetary_computer as pc


# ================================================================
# 2. Streamlit page configuration
# ================================================================

st.set_page_config(
    page_title="Landsat NDSI Snow Cover Dashboard",
    page_icon="❄️",
    layout="wide"
)

st.title("❄️ Landsat NDSI Snow Cover Mapping Dashboard")
st.caption("Open-source local processing workflow. No Google Earth Engine is used.")

st.markdown(
    """
    Please **draw your area of interest on the map** or **upload a shapefile/GeoJSON/GPKG**.
    The dashboard searches Landsat Collection 2 Level-2 imagery, computes NDSI, estimates snow cover,
    and exports GeoTIFF and Shapefile outputs.
    """
)


# ================================================================
# 3. Constants
# ================================================================

PLANETARY_COMPUTER_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
LANDSAT_COLLECTION = "landsat-c2-l2"

SURFACE_REFLECTANCE_SCALE = 0.0000275
SURFACE_REFLECTANCE_OFFSET = -0.2

LANDSAT_PLATFORMS = {
    "Landsat 4": "landsat-4",
    "Landsat 5": "landsat-5",
    "Landsat 7": "landsat-7",
    "Landsat 8": "landsat-8",
    "Landsat 9": "landsat-9"
}


# ================================================================
# 4. General utility functions
# ================================================================

def safe_name(text: str) -> str:
    """
    Convert user text to a safe filename.
    """
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    return text[:90] if text else "landsat_ndsi_snow"


def zip_directory(directory_path: str, zip_path: str):
    """
    Zip a directory recursively.
    """
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(directory_path):
            for file in files:
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, directory_path)
                zipf.write(full_path, relative_path)


def get_download_bytes(file_path: str):
    """
    Read file bytes for Streamlit download button.
    """
    with open(file_path, "rb") as f:
        return f.read()


# ================================================================
# 5. AOI functions
# ================================================================

def read_uploaded_aoi(uploaded_file, simplify_tolerance: float = 0.0):
    """
    Read uploaded AOI file.

    Supported:
        - zipped shapefile: .zip containing .shp, .shx, .dbf, .prj
        - GeoJSON
        - JSON
        - GPKG

    Returns:
        GeoJSON geometry in EPSG:4326.
    """
    if uploaded_file is None:
        return None

    suffix = Path(uploaded_file.name).suffix.lower()

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, uploaded_file.name)

        with open(input_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        if suffix == ".zip":
            with zipfile.ZipFile(input_path, "r") as zip_ref:
                zip_ref.extractall(tmpdir)

            shp_files = []
            for root, _, files in os.walk(tmpdir):
                for file in files:
                    if file.lower().endswith(".shp"):
                        shp_files.append(os.path.join(root, file))

            if len(shp_files) == 0:
                raise ValueError(
                    "No .shp file found inside ZIP. Please upload a zipped shapefile."
                )

            gdf = gpd.read_file(shp_files[0])

        elif suffix in [".geojson", ".json", ".gpkg"]:
            gdf = gpd.read_file(input_path)

        else:
            raise ValueError(
                "Unsupported AOI format. Please upload ZIP shapefile, GeoJSON, JSON, or GPKG."
            )

    if gdf.empty:
        raise ValueError("Uploaded AOI file is empty.")

    if gdf.crs is None:
        raise ValueError(
            "Uploaded AOI has no CRS. Please define projection before uploading."
        )

    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.notnull()]
    gdf = gdf[~gdf.geometry.is_empty]

    polygon_gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]

    if polygon_gdf.empty:
        raise ValueError("AOI must contain Polygon or MultiPolygon geometry.")

    polygon_gdf["geometry"] = polygon_gdf.geometry.buffer(0)

    geom = unary_union(polygon_gdf.geometry.values)

    if simplify_tolerance > 0:
        geom = geom.simplify(simplify_tolerance, preserve_topology=True)

    return mapping(geom)


def get_drawn_geometry(map_data):
    """
    Extract last drawn polygon from streamlit-folium map output.
    """
    if not map_data:
        return None

    drawings = map_data.get("all_drawings", [])

    if not drawings:
        return None

    last = drawings[-1]
    geom = last.get("geometry", None)

    if geom and geom.get("type") in ["Polygon", "MultiPolygon"]:
        return geom

    return None


def make_base_map(aoi_geojson=None):
    """
    Create Folium base map for AOI drawing and result display.
    """
    center = [32.0, 78.0]
    zoom = 5

    if aoi_geojson:
        try:
            geom = shape(aoi_geojson)
            center = [geom.centroid.y, geom.centroid.x]
            zoom = 8
        except Exception:
            pass

    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="OpenStreetMap",
        control_scale=True
    )

    if aoi_geojson:
        folium.GeoJson(
            aoi_geojson,
            name="AOI Boundary",
            style_function=lambda x: {
                "fillColor": "none",
                "color": "yellow",
                "weight": 3,
                "dashArray": "5,5"
            }
        ).add_to(m)

        try:
            bounds = shape(aoi_geojson).bounds
            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
        except Exception:
            pass

    return m


def estimate_utm_crs_from_aoi(aoi_geojson):
    """
    Estimate a suitable UTM CRS from AOI centroid.
    """
    geom = shape(aoi_geojson)
    lon = geom.centroid.x
    lat = geom.centroid.y

    zone = int((lon + 180) // 6) + 1

    if lat >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone

    return CRS.from_epsg(epsg)


def build_processing_grid(aoi_geojson, resolution_m):
    """
    Build a target projected processing grid fully covering the AOI.

    Returns:
        dst_crs
        dst_transform
        width
        height
        aoi_projected_geometry
        aoi_mask
    """
    dst_crs = estimate_utm_crs_from_aoi(aoi_geojson)

    aoi_geom_4326 = shape(aoi_geojson)

    aoi_gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[aoi_geom_4326],
        crs="EPSG:4326"
    ).to_crs(dst_crs)

    aoi_projected = aoi_gdf.geometry.iloc[0]

    minx, miny, maxx, maxy = aoi_projected.bounds

    width = int(math.ceil((maxx - minx) / resolution_m))
    height = int(math.ceil((maxy - miny) / resolution_m))

    if width <= 0 or height <= 0:
        raise ValueError("Invalid AOI dimensions. Please check the AOI geometry.")

    dst_transform = from_origin(minx, maxy, resolution_m, resolution_m)

    aoi_mask = geometry_mask(
        geometries=[mapping(aoi_projected)],
        out_shape=(height, width),
        transform=dst_transform,
        invert=True
    )

    return dst_crs, dst_transform, width, height, aoi_projected, aoi_mask


# ================================================================
# 6. STAC search functions
# ================================================================

def search_landsat_items(
    aoi_geojson,
    start_date,
    end_date,
    selected_platforms,
    max_cloud_cover,
    max_search_items
):
    """
    Search Landsat C2 L2 STAC items from Microsoft Planetary Computer.
    """
    catalog = Client.open(PLANETARY_COMPUTER_STAC)

    search = catalog.search(
        collections=[LANDSAT_COLLECTION],
        intersects=aoi_geojson,
        datetime=f"{start_date}/{end_date}",
        query={
            "eo:cloud_cover": {
                "lte": float(max_cloud_cover)
            }
        },
        max_items=int(max_search_items)
    )

    items = list(search.items())

    selected_platform_values = [
        LANDSAT_PLATFORMS[p] for p in selected_platforms
    ]

    filtered_items = []
    for item in items:
        platform = item.properties.get("platform", "")
        if platform in selected_platform_values:
            filtered_items.append(item)

    filtered_items = sorted(
        filtered_items,
        key=lambda item: item.properties.get("eo:cloud_cover", 9999)
    )

    return filtered_items


def create_items_table(items):
    """
    Create a dataframe summarizing searched Landsat scenes.
    """
    rows = []

    for i, item in enumerate(items):
        rows.append(
            {
                "index": i,
                "id": item.id,
                "datetime": item.properties.get("datetime", ""),
                "platform": item.properties.get("platform", ""),
                "cloud_cover": item.properties.get("eo:cloud_cover", None),
                "path": item.properties.get("landsat:wrs_path", ""),
                "row": item.properties.get("landsat:wrs_row", "")
            }
        )

    return pd.DataFrame(rows)


# ================================================================
# 7. Raster reading and Landsat masking functions
# ================================================================

def read_cog_to_grid(
    href,
    dst_crs,
    dst_transform,
    width,
    height,
    resampling_method=Resampling.nearest
):
    """
    Read a remote Cloud Optimized GeoTIFF into the target grid.

    This uses WarpedVRT so the source image is reprojected and aligned
    to the dashboard processing grid.
    """
    rasterio_env = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".TIF,.tif,.TIFF,.tiff",
        "GDAL_HTTP_MAX_RETRY": "3",
        "GDAL_HTTP_RETRY_DELAY": "2"
    }

    with rasterio.Env(**rasterio_env):
        with rasterio.open(href) as src:
            with WarpedVRT(
                src,
                crs=dst_crs,
                transform=dst_transform,
                width=width,
                height=height,
                resampling=resampling_method,
                nodata=src.nodata
            ) as vrt:
                arr = vrt.read(1, masked=False)

    return arr


def make_landsat_valid_mask(qa_pixel_array, mask_water=False):
    """
    Create a valid-pixel mask using Landsat QA_PIXEL.

    QA_PIXEL bits used:
        Bit 0 = Fill
        Bit 1 = Dilated cloud
        Bit 2 = Cirrus
        Bit 3 = Cloud
        Bit 4 = Cloud shadow
        Bit 7 = Water, optional

    Important:
        Bit 5 = Snow is not masked because snow is the target class.
    """
    qa = qa_pixel_array.astype(np.uint16)

    fill = (qa & (1 << 0)) != 0
    dilated_cloud = (qa & (1 << 1)) != 0
    cirrus = (qa & (1 << 2)) != 0
    cloud = (qa & (1 << 3)) != 0
    cloud_shadow = (qa & (1 << 4)) != 0

    invalid = fill | dilated_cloud | cirrus | cloud | cloud_shadow

    if mask_water:
        water = (qa & (1 << 7)) != 0
        invalid = invalid | water

    valid = ~invalid

    return valid


def scale_surface_reflectance(raw_array):
    """
    Apply Landsat C2 L2 surface reflectance scale and offset.
    """
    return raw_array.astype("float32") * SURFACE_REFLECTANCE_SCALE + SURFACE_REFLECTANCE_OFFSET


def compute_ndsi(green_reflectance, swir1_reflectance, valid_mask):
    """
    Compute NDSI.

    NDSI = (Green - SWIR1) / (Green + SWIR1)
    """
    denominator = green_reflectance + swir1_reflectance

    ndsi = np.full(green_reflectance.shape, np.nan, dtype="float32")

    safe = valid_mask & np.isfinite(denominator) & (np.abs(denominator) > 1e-6)

    with np.errstate(divide="ignore", invalid="ignore"):
        ndsi[safe] = (
            (green_reflectance[safe] - swir1_reflectance[safe])
            / denominator[safe]
        )

    ndsi[(ndsi < -1.5) | (ndsi > 1.5)] = np.nan

    return ndsi


def process_single_landsat_item(
    item,
    dst_crs,
    dst_transform,
    width,
    height,
    aoi_mask,
    mask_water=False,
    read_rgb=True
):
    """
    Process one Landsat item:
        - sign STAC item
        - read green, swir16, qa_pixel
        - compute valid mask
        - compute NDSI
        - optionally read RGB bands for preview
    """
    signed_item = pc.sign(item)

    required_assets = ["green", "swir16", "qa_pixel"]

    for asset in required_assets:
        if asset not in signed_item.assets:
            raise ValueError(f"Required asset '{asset}' not available in item {item.id}")

    green_raw = read_cog_to_grid(
        signed_item.assets["green"].href,
        dst_crs,
        dst_transform,
        width,
        height,
        resampling_method=Resampling.bilinear
    )

    swir_raw = read_cog_to_grid(
        signed_item.assets["swir16"].href,
        dst_crs,
        dst_transform,
        width,
        height,
        resampling_method=Resampling.bilinear
    )

    qa_raw = read_cog_to_grid(
        signed_item.assets["qa_pixel"].href,
        dst_crs,
        dst_transform,
        width,
        height,
        resampling_method=Resampling.nearest
    )

    valid_mask = make_landsat_valid_mask(
        qa_pixel_array=qa_raw,
        mask_water=mask_water
    )

    valid_mask = valid_mask & aoi_mask & (green_raw != 0) & (swir_raw != 0)

    green_ref = scale_surface_reflectance(green_raw)
    swir_ref = scale_surface_reflectance(swir_raw)

    ndsi = compute_ndsi(
        green_reflectance=green_ref,
        swir1_reflectance=swir_ref,
        valid_mask=valid_mask
    )

    rgb = None

    if read_rgb:
        rgb_assets = ["red", "green", "blue"]

        if all(asset in signed_item.assets for asset in rgb_assets):
            red_raw = read_cog_to_grid(
                signed_item.assets["red"].href,
                dst_crs,
                dst_transform,
                width,
                height,
                resampling_method=Resampling.bilinear
            )

            blue_raw = read_cog_to_grid(
                signed_item.assets["blue"].href,
                dst_crs,
                dst_transform,
                width,
                height,
                resampling_method=Resampling.bilinear
            )

            red_ref = scale_surface_reflectance(red_raw)
            blue_ref = scale_surface_reflectance(blue_raw)

            rgb = np.dstack([red_ref, green_ref, blue_ref]).astype("float32")
            rgb[~valid_mask, :] = np.nan

    valid_pixels = int(np.sum(np.isfinite(ndsi) & aoi_mask))

    return {
        "item_id": item.id,
        "platform": item.properties.get("platform", ""),
        "datetime": item.properties.get("datetime", ""),
        "cloud_cover": item.properties.get("eo:cloud_cover", None),
        "ndsi": ndsi,
        "valid_mask": valid_mask,
        "rgb": rgb,
        "valid_pixels": valid_pixels
    }


def composite_ndsi(scene_results, method="Median composite"):
    """
    Composite NDSI arrays from multiple scenes.
    """
    ndsi_stack = np.stack([scene["ndsi"] for scene in scene_results], axis=0)

    if method == "Median composite":
        composite = np.nanmedian(ndsi_stack, axis=0).astype("float32")

    elif method == "Mean composite":
        composite = np.nanmean(ndsi_stack, axis=0).astype("float32")

    elif method == "Maximum NDSI composite":
        composite = np.nanmax(ndsi_stack, axis=0).astype("float32")

    elif method == "Best single scene":
        composite = scene_results[0]["ndsi"].astype("float32")

    else:
        raise ValueError("Unsupported composite method.")

    return composite


def composite_rgb(scene_results, method="Median composite"):
    """
    Composite RGB arrays for preview only.
    """
    rgb_list = [scene["rgb"] for scene in scene_results if scene["rgb"] is not None]

    if len(rgb_list) == 0:
        return None

    rgb_stack = np.stack(rgb_list, axis=0)

    if method in ["Median composite", "Maximum NDSI composite"]:
        rgb = np.nanmedian(rgb_stack, axis=0).astype("float32")

    elif method == "Mean composite":
        rgb = np.nanmean(rgb_stack, axis=0).astype("float32")

    elif method == "Best single scene":
        rgb = rgb_list[0].astype("float32")

    else:
        rgb = rgb_list[0].astype("float32")

    return rgb


# ================================================================
# 8. Histogram and threshold functions
# ================================================================

def calculate_histogram(ndsi_array, aoi_mask, bins=100):
    """
    Calculate NDSI histogram within AOI.
    """
    values = ndsi_array[np.isfinite(ndsi_array) & aoi_mask]

    if values.size == 0:
        raise ValueError("No valid NDSI pixels found inside AOI.")

    counts, edges = np.histogram(
        values,
        bins=int(bins),
        range=(-1.0, 1.0)
    )

    centers = (edges[:-1] + edges[1:]) / 2.0

    hist_df = pd.DataFrame(
        {
            "NDSI": centers,
            "Pixel_Count": counts
        }
    )

    return hist_df, values


def otsu_threshold_from_histogram(hist_df):
    """
    Compute Otsu threshold from histogram.
    """
    values = hist_df["NDSI"].values.astype("float64")
    counts = hist_df["Pixel_Count"].values.astype("float64")

    total = counts.sum()

    if total <= 0:
        return 0.40

    sum_total = np.sum(values * counts)

    weight_background = 0.0
    sum_background = 0.0
    max_variance = -1.0
    threshold = 0.40

    for i in range(len(values)):
        weight_background += counts[i]

        if weight_background == 0:
            continue

        weight_foreground = total - weight_background

        if weight_foreground == 0:
            break

        sum_background += values[i] * counts[i]

        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        between_class_variance = (
            weight_background
            * weight_foreground
            * (mean_background - mean_foreground) ** 2
        )

        if between_class_variance > max_variance:
            max_variance = between_class_variance
            threshold = values[i]

    return float(threshold)


# ================================================================
# 9. Snow mask and area functions
# ================================================================

def create_snow_mask(
    ndsi_array,
    aoi_mask,
    threshold,
    resolution_m,
    minimum_patch_area_ha=0.0
):
    """
    Create binary snow mask from NDSI threshold.

    Snow = 1
    Non-snow = 0
    Outside AOI / invalid = 255
    """
    valid = np.isfinite(ndsi_array) & aoi_mask
    snow_bool = (ndsi_array >= threshold) & valid

    if minimum_patch_area_ha > 0:
        min_pixels = int(
            max(
                1,
                round((minimum_patch_area_ha * 10000.0) / (resolution_m ** 2))
            )
        )

        sieved = sieve(
            snow_bool.astype("uint8"),
            size=min_pixels,
            connectivity=8
        )

        snow_bool = sieved.astype(bool)

    snow_out = np.full(ndsi_array.shape, 255, dtype="uint8")
    snow_out[valid] = 0
    snow_out[snow_bool] = 1

    return snow_out, valid, snow_bool


def calculate_area_statistics(valid_mask, snow_bool, aoi_mask, resolution_m):
    """
    Calculate AOI, valid-data, and snow-covered area.
    """
    pixel_area_km2 = (resolution_m * resolution_m) / 1_000_000.0

    aoi_area_km2 = float(np.sum(aoi_mask) * pixel_area_km2)
    valid_area_km2 = float(np.sum(valid_mask) * pixel_area_km2)
    snow_area_km2 = float(np.sum(snow_bool) * pixel_area_km2)

    snow_percent_valid = (
        0.0 if valid_area_km2 == 0 else snow_area_km2 / valid_area_km2 * 100.0
    )

    snow_percent_aoi = (
        0.0 if aoi_area_km2 == 0 else snow_area_km2 / aoi_area_km2 * 100.0
    )

    return {
        "AOI area km²": aoi_area_km2,
        "Valid Landsat area km²": valid_area_km2,
        "Snow area km²": snow_area_km2,
        "Snow % of valid Landsat area": snow_percent_valid,
        "Snow % of AOI": snow_percent_aoi
    }


# ================================================================
# 10. Export functions
# ================================================================

def write_ndsi_geotiff(
    ndsi_array,
    output_path,
    dst_crs,
    dst_transform
):
    """
    Write NDSI GeoTIFF.
    """
    output = np.where(
        np.isfinite(ndsi_array),
        ndsi_array,
        -9999.0
    ).astype("float32")

    profile = {
        "driver": "GTiff",
        "height": output.shape[0],
        "width": output.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": dst_crs,
        "transform": dst_transform,
        "nodata": -9999.0,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(output, 1)
        dst.set_band_description(1, "NDSI")


def write_snow_mask_geotiff(
    snow_mask,
    output_path,
    dst_crs,
    dst_transform
):
    """
    Write binary snow mask GeoTIFF.

    Values:
        1 = snow
        0 = non-snow
        255 = nodata / outside AOI / invalid
    """
    profile = {
        "driver": "GTiff",
        "height": snow_mask.shape[0],
        "width": snow_mask.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": dst_crs,
        "transform": dst_transform,
        "nodata": 255,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(snow_mask, 1)
        dst.set_band_description(1, "snow_mask")


def vectorize_snow_mask(
    snow_mask,
    dst_transform,
    dst_crs,
    output_shapefile_path
):
    """
    Convert snow raster mask to shapefile polygons.
    """
    snow_binary = (snow_mask == 1).astype("uint8")
    mask = snow_binary == 1

    records = []

    for geom, value in shapes(
        snow_binary,
        mask=mask,
        transform=dst_transform
    ):
        if int(value) == 1:
            records.append(
                {
                    "geometry": shape(geom),
                    "snow": 1
                }
            )

    if len(records) == 0:
        empty_gdf = gpd.GeoDataFrame(
            {"snow": []},
            geometry=[],
            crs=dst_crs
        )
        empty_gdf.to_file(output_shapefile_path)
        return 0

    gdf = gpd.GeoDataFrame(records, crs=dst_crs)

    gdf["area_ha"] = gdf.geometry.area / 10000.0
    gdf["area_km2"] = gdf.geometry.area / 1_000_000.0

    gdf = gdf.to_crs("EPSG:4326")
    gdf.to_file(output_shapefile_path)

    return len(gdf)


def export_all_outputs(
    ndsi_array,
    snow_mask,
    dst_crs,
    dst_transform,
    export_prefix
):
    """
    Export NDSI GeoTIFF, snow mask GeoTIFF and snow polygon shapefile.

    Returns:
        output_dir
        output_zip_path
        output file paths
    """
    temp_dir = tempfile.mkdtemp(prefix="landsat_ndsi_snow_")
    output_dir = os.path.join(temp_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    prefix = safe_name(export_prefix)

    ndsi_tif = os.path.join(output_dir, f"{prefix}_NDSI.tif")
    snow_tif = os.path.join(output_dir, f"{prefix}_snow_mask.tif")
    snow_shp = os.path.join(output_dir, f"{prefix}_snow_polygons.shp")

    write_ndsi_geotiff(
        ndsi_array=ndsi_array,
        output_path=ndsi_tif,
        dst_crs=dst_crs,
        dst_transform=dst_transform
    )

    write_snow_mask_geotiff(
        snow_mask=snow_mask,
        output_path=snow_tif,
        dst_crs=dst_crs,
        dst_transform=dst_transform
    )

    polygon_count = vectorize_snow_mask(
        snow_mask=snow_mask,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        output_shapefile_path=snow_shp
    )

    zip_path = os.path.join(temp_dir, f"{prefix}_outputs.zip")
    zip_directory(output_dir, zip_path)

    return {
        "output_dir": output_dir,
        "zip_path": zip_path,
        "ndsi_tif": ndsi_tif,
        "snow_tif": snow_tif,
        "snow_shp": snow_shp,
        "polygon_count": polygon_count
    }


# ================================================================
# 11. Preview map functions
# ================================================================

def normalize_rgb(rgb_array, lower=2, upper=98):
    """
    Normalize RGB array for map preview.
    """
    if rgb_array is None:
        return None

    rgb = rgb_array.copy()

    output = np.zeros_like(rgb, dtype="float32")

    for i in range(3):
        band = rgb[:, :, i]
        valid = np.isfinite(band)

        if np.sum(valid) == 0:
            continue

        p_low, p_high = np.nanpercentile(band[valid], [lower, upper])

        if p_high <= p_low:
            continue

        output[:, :, i] = (band - p_low) / (p_high - p_low)

    output = np.clip(output, 0, 1)

    alpha = np.isfinite(rgb[:, :, 0]) & np.isfinite(rgb[:, :, 1]) & np.isfinite(rgb[:, :, 2])
    rgba = np.dstack(
        [
            output[:, :, 0],
            output[:, :, 1],
            output[:, :, 2],
            alpha.astype("float32")
        ]
    )

    return rgba


def ndsi_to_rgba(ndsi_array):
    """
    Create simple RGBA image for NDSI preview.
    """
    valid = np.isfinite(ndsi_array)

    scaled = (ndsi_array + 1.0) / 2.0
    scaled = np.clip(scaled, 0, 1)

    rgba = np.zeros((ndsi_array.shape[0], ndsi_array.shape[1], 4), dtype="float32")

    # Simple brown-yellow-blue gradient approximation
    rgba[:, :, 0] = 1.0 - scaled * 0.7
    rgba[:, :, 1] = 0.6 + scaled * 0.4
    rgba[:, :, 2] = scaled
    rgba[:, :, 3] = valid.astype("float32") * 0.75

    return rgba


def snow_to_rgba(snow_mask):
    """
    Create RGBA overlay for snow mask preview.
    """
    snow = snow_mask == 1

    rgba = np.zeros((snow_mask.shape[0], snow_mask.shape[1], 4), dtype="float32")
    rgba[:, :, 0] = 0.0
    rgba[:, :, 1] = 1.0
    rgba[:, :, 2] = 1.0
    rgba[:, :, 3] = snow.astype("float32") * 0.75

    return rgba


def get_latlon_bounds_from_grid(dst_crs, dst_transform, width, height):
    """
    Convert raster grid bounds to EPSG:4326 bounds for Folium ImageOverlay.
    """
    left = dst_transform.c
    top = dst_transform.f
    right = left + width * dst_transform.a
    bottom = top + height * dst_transform.e

    minlon, minlat, maxlon, maxlat = transform_bounds(
        dst_crs,
        "EPSG:4326",
        left,
        bottom,
        right,
        top,
        densify_pts=21
    )

    return [[minlat, minlon], [maxlat, maxlon]]


def add_array_overlay_to_map(
    fmap,
    array_rgba,
    bounds,
    name,
    opacity=1.0,
    show=True
):
    """
    Add RGBA numpy image overlay to Folium map.
    """
    folium.raster_layers.ImageOverlay(
        image=array_rgba,
        bounds=bounds,
        opacity=opacity,
        name=name,
        interactive=True,
        cross_origin=False,
        zindex=1,
        show=show
    ).add_to(fmap)


# ================================================================
# 12. Sidebar inputs
# ================================================================

with st.sidebar:
    st.header("1. AOI input")

    uploaded_file = st.file_uploader(
        "Upload AOI shapefile ZIP / GeoJSON / GPKG",
        type=["zip", "geojson", "json", "gpkg"]
    )

    simplify_tolerance = st.number_input(
        "AOI simplification tolerance, degrees",
        min_value=0.0,
        max_value=0.05,
        value=0.0,
        step=0.001,
        format="%.4f",
        help="Use only for very complex AOIs. 0 means no simplification."
    )

    st.header("2. Landsat search")

    selected_platforms = st.multiselect(
        "Select Landsat sensors",
        options=list(LANDSAT_PLATFORMS.keys()),
        default=["Landsat 8", "Landsat 9"]
    )

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        start_date = st.date_input(
            "Start date",
            value=pd.to_datetime("2024-01-01")
        )

    with col_d2:
        end_date = st.date_input(
            "End date",
            value=pd.to_datetime("2024-12-31")
        )

    max_cloud_cover = st.slider(
        "Maximum scene cloud cover (%)",
        min_value=0,
        max_value=100,
        value=40,
        step=5
    )

    max_search_items = st.slider(
        "Maximum STAC items to search",
        min_value=10,
        max_value=200,
        value=80,
        step=10
    )

    max_scenes_to_process = st.slider(
        "Maximum scenes to process",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help="Higher values improve composites but increase processing time."
    )

    st.header("3. Processing")

    resolution_m = st.selectbox(
        "Processing resolution",
        options=[30, 60, 90, 120],
        index=0,
        help="30 m is native Landsat resolution. Larger values are faster."
    )

    composite_method = st.selectbox(
        "Composite method",
        options=[
            "Best single scene",
            "Median composite",
            "Mean composite",
            "Maximum NDSI composite"
        ],
        index=1
    )

    mask_water = st.checkbox(
        "Mask water using QA_PIXEL bit 7",
        value=False,
        help="Keep unchecked for glacier/snow mapping unless water confusion is a problem."
    )

    histogram_bins = st.slider(
        "Histogram bins",
        min_value=50,
        max_value=250,
        value=100,
        step=25
    )

    st.header("4. Snow threshold")

    threshold_mode = st.radio(
        "Threshold method",
        options=["Auto Otsu", "Manual"],
        index=0
    )

    manual_threshold = st.number_input(
        "Manual NDSI threshold",
        min_value=-1.0,
        max_value=1.0,
        value=0.40,
        step=0.01,
        format="%.2f"
    )

    minimum_patch_area_ha = st.number_input(
        "Minimum snow patch area to retain, ha",
        min_value=0.0,
        max_value=10000.0,
        value=0.0,
        step=0.5,
        help="Set 0 to retain all snow patches."
    )

    st.header("5. Export")

    export_prefix = st.text_input(
        "Output file prefix",
        value="landsat_ndsi_snow"
    )


# ================================================================
# 13. AOI drawing/upload section
# ================================================================

uploaded_aoi_geojson = None

if uploaded_file is not None:
    try:
        uploaded_aoi_geojson = read_uploaded_aoi(
            uploaded_file=uploaded_file,
            simplify_tolerance=simplify_tolerance
        )
        st.success("Uploaded AOI loaded successfully.")
    except Exception as e:
        st.error(f"AOI upload error: {e}")

st.subheader("Step 1: Draw AOI or Upload Shapefile")

draw_map = make_base_map(uploaded_aoi_geojson)

draw_control = Draw(
    export=False,
    position="topleft",
    draw_options={
        "polyline": False,
        "rectangle": True,
        "polygon": True,
        "circle": False,
        "marker": False,
        "circlemarker": False
    },
    edit_options={
        "edit": True,
        "remove": True
    }
)

draw_control.add_to(draw_map)
folium.LayerControl().add_to(draw_map)

map_data = st_folium(
    draw_map,
    height=520,
    use_container_width=True
)

drawn_aoi_geojson = get_drawn_geometry(map_data)

if uploaded_aoi_geojson:
    selected_aoi_geojson = uploaded_aoi_geojson
    st.info("AOI source: uploaded vector file.")
elif drawn_aoi_geojson:
    selected_aoi_geojson = drawn_aoi_geojson
    st.info("AOI source: drawn polygon.")
else:
    selected_aoi_geojson = None
    st.warning("Please draw your area of interest on the map or upload a shapefile.")


# ================================================================
# 14. Main processing button
# ================================================================

process_button = st.button(
    "Search Landsat and Generate Snow Cover Map",
    type="primary"
)

if process_button:

    if selected_aoi_geojson is None:
        st.error("No AOI selected. Please draw an AOI or upload a vector file.")
        st.stop()

    if len(selected_platforms) == 0:
        st.error("Please select at least one Landsat sensor.")
        st.stop()

    if pd.to_datetime(start_date) >= pd.to_datetime(end_date):
        st.error("Start date must be earlier than end date.")
        st.stop()

    try:
        with st.spinner("Searching Landsat STAC catalog..."):
            items = search_landsat_items(
                aoi_geojson=selected_aoi_geojson,
                start_date=str(start_date),
                end_date=str(end_date),
                selected_platforms=selected_platforms,
                max_cloud_cover=max_cloud_cover,
                max_search_items=max_search_items
            )

        if len(items) == 0:
            st.error(
                "No Landsat scenes found. Try increasing the date range, "
                "cloud cover threshold, or selecting additional Landsat sensors."
            )
            st.stop()

        items_table = create_items_table(items)

        if composite_method == "Best single scene":
            selected_items = items[:1]
        else:
            selected_items = items[:max_scenes_to_process]

        with st.spinner("Building local processing grid..."):
            (
                dst_crs,
                dst_transform,
                width,
                height,
                aoi_projected,
                aoi_mask
            ) = build_processing_grid(
                aoi_geojson=selected_aoi_geojson,
                resolution_m=resolution_m
            )

        estimated_pixels = width * height

        if estimated_pixels > 60_000_000:
            st.warning(
                f"The selected AOI/grid is large: {estimated_pixels:,} pixels. "
                "Processing may be slow. Consider using 60 m, 90 m, or a smaller AOI."
            )

        scene_results = []

        progress = st.progress(0)
        status = st.empty()

        for i, item in enumerate(selected_items):
            status.write(
                f"Processing scene {i + 1} of {len(selected_items)}: {item.id}"
            )

            try:
                scene_result = process_single_landsat_item(
                    item=item,
                    dst_crs=dst_crs,
                    dst_transform=dst_transform,
                    width=width,
                    height=height,
                    aoi_mask=aoi_mask,
                    mask_water=mask_water,
                    read_rgb=True
                )

                if scene_result["valid_pixels"] > 0:
                    scene_results.append(scene_result)

            except Exception as scene_error:
                st.warning(f"Skipped scene {item.id}: {scene_error}")

            progress.progress((i + 1) / len(selected_items))

        status.empty()

        if len(scene_results) == 0:
            st.error("No valid scenes could be processed inside AOI.")
            st.stop()

        with st.spinner("Creating NDSI composite and snow mask..."):
            ndsi_composite = composite_ndsi(
                scene_results=scene_results,
                method=composite_method
            )

            rgb_composite = composite_rgb(
                scene_results=scene_results,
                method=composite_method
            )

            hist_df, ndsi_values = calculate_histogram(
                ndsi_array=ndsi_composite,
                aoi_mask=aoi_mask,
                bins=histogram_bins
            )

            otsu_threshold = otsu_threshold_from_histogram(hist_df)

            if threshold_mode == "Auto Otsu":
                final_threshold = otsu_threshold
            else:
                final_threshold = float(manual_threshold)

            snow_mask, valid_mask, snow_bool = create_snow_mask(
                ndsi_array=ndsi_composite,
                aoi_mask=aoi_mask,
                threshold=final_threshold,
                resolution_m=resolution_m,
                minimum_patch_area_ha=minimum_patch_area_ha
            )

            area_stats = calculate_area_statistics(
                valid_mask=valid_mask,
                snow_bool=snow_bool,
                aoi_mask=aoi_mask,
                resolution_m=resolution_m
            )

        st.session_state["snow_result"] = {
            "items_table": items_table,
            "processed_scenes": pd.DataFrame(
                [
                    {
                        "item_id": s["item_id"],
                        "platform": s["platform"],
                        "datetime": s["datetime"],
                        "cloud_cover": s["cloud_cover"],
                        "valid_pixels": s["valid_pixels"]
                    }
                    for s in scene_results
                ]
            ),
            "aoi_geojson": selected_aoi_geojson,
            "dst_crs": dst_crs,
            "dst_transform": dst_transform,
            "width": width,
            "height": height,
            "aoi_mask": aoi_mask,
            "ndsi": ndsi_composite,
            "rgb": rgb_composite,
            "hist_df": hist_df,
            "otsu_threshold": otsu_threshold,
            "final_threshold": final_threshold,
            "snow_mask": snow_mask,
            "valid_mask": valid_mask,
            "snow_bool": snow_bool,
            "area_stats": area_stats,
            "resolution_m": resolution_m,
            "composite_method": composite_method
        }

        st.success("Snow cover mapping completed.")

    except Exception as e:
        st.error(f"Processing failed: {e}")


# ================================================================
# 15. Results display
# ================================================================

if "snow_result" in st.session_state:

    result = st.session_state["snow_result"]

    st.subheader("Step 2: Landsat Scenes")

    with st.expander("All searched scenes", expanded=False):
        st.dataframe(
            result["items_table"],
            use_container_width=True
        )

    with st.expander("Scenes processed", expanded=True):
        st.dataframe(
            result["processed_scenes"],
            use_container_width=True
        )

    st.subheader("Step 3: Snow Cover Statistics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Processed scenes",
        len(result["processed_scenes"])
    )

    col2.metric(
        "Final threshold",
        f"{result['final_threshold']:.3f}"
    )

    col3.metric(
        "Snow area",
        f"{result['area_stats']['Snow area km²']:.2f} km²"
    )

    col4.metric(
        "Snow % valid area",
        f"{result['area_stats']['Snow % of valid Landsat area']:.2f}%"
    )

    stats_df = pd.DataFrame(
        list(result["area_stats"].items()),
        columns=["Parameter", "Value"]
    )

    st.dataframe(
        stats_df,
        use_container_width=True
    )

    st.caption(
        f"Otsu threshold estimated from NDSI histogram: {result['otsu_threshold']:.3f}"
    )

    st.subheader("Step 4: NDSI Histogram")

    fig = px.bar(
        result["hist_df"],
        x="NDSI",
        y="Pixel_Count",
        title="NDSI Histogram within AOI"
    )

    fig.add_vline(
        x=result["final_threshold"],
        line_dash="dash",
        annotation_text=f"Threshold = {result['final_threshold']:.3f}",
        annotation_position="top right"
    )

    fig.update_layout(
        xaxis_title="NDSI",
        yaxis_title="Pixel count",
        bargap=0.05
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Step 5: Map Preview")

    preview_map = make_base_map(result["aoi_geojson"])

    bounds_4326 = get_latlon_bounds_from_grid(
        dst_crs=result["dst_crs"],
        dst_transform=result["dst_transform"],
        width=result["width"],
        height=result["height"]
    )

    rgb_rgba = normalize_rgb(result["rgb"])
    ndsi_rgba = ndsi_to_rgba(result["ndsi"])
    snow_rgba = snow_to_rgba(result["snow_mask"])

    if rgb_rgba is not None:
        add_array_overlay_to_map(
            fmap=preview_map,
            array_rgba=rgb_rgba,
            bounds=bounds_4326,
            name="Landsat RGB Preview",
            opacity=1.0,
            show=True
        )

    add_array_overlay_to_map(
        fmap=preview_map,
        array_rgba=ndsi_rgba,
        bounds=bounds_4326,
        name="NDSI Preview",
        opacity=0.75,
        show=False
    )

    add_array_overlay_to_map(
        fmap=preview_map,
        array_rgba=snow_rgba,
        bounds=bounds_4326,
        name="Snow Mask",
        opacity=0.85,
        show=True
    )

    folium.GeoJson(
        result["aoi_geojson"],
        name="AOI Boundary",
        style_function=lambda x: {
            "fillColor": "none",
            "color": "yellow",
            "weight": 3,
            "dashArray": "5,5"
        }
    ).add_to(preview_map)

    folium.LayerControl().add_to(preview_map)

    st_folium(
        preview_map,
        height=620,
        use_container_width=True
    )

    st.subheader("Step 6: Export GeoTIFF and Shapefile")

    st.markdown(
        """
        Export package includes:

        1. NDSI GeoTIFF  
        2. Snow mask GeoTIFF  
        3. Snow polygon shapefile  
        4. All shapefile sidecar files zipped together  
        """
    )

    if st.button("Generate Export Files", type="primary"):

        try:
            with st.spinner("Writing GeoTIFF and Shapefile outputs..."):
                export_result = export_all_outputs(
                    ndsi_array=result["ndsi"],
                    snow_mask=result["snow_mask"],
                    dst_crs=result["dst_crs"],
                    dst_transform=result["dst_transform"],
                    export_prefix=export_prefix
                )

            st.success(
                f"Export completed. Snow polygons created: {export_result['polygon_count']}"
            )

            st.session_state["export_result"] = export_result

        except Exception as e:
            st.error(f"Export failed: {e}")

    if "export_result" in st.session_state:
        export_result = st.session_state["export_result"]

        st.download_button(
            label="Download All Outputs as ZIP",
            data=get_download_bytes(export_result["zip_path"]),
            file_name=os.path.basename(export_result["zip_path"]),
            mime="application/zip"
        )

        col_a, col_b = st.columns(2)

        with col_a:
            st.download_button(
                label="Download NDSI GeoTIFF",
                data=get_download_bytes(export_result["ndsi_tif"]),
                file_name=os.path.basename(export_result["ndsi_tif"]),
                mime="image/tiff"
            )

        with col_b:
            st.download_button(
                label="Download Snow Mask GeoTIFF",
                data=get_download_bytes(export_result["snow_tif"]),
                file_name=os.path.basename(export_result["snow_tif"]),
                mime="image/tiff"
            )


# ================================================================
# End of app.py
# ================================================================
