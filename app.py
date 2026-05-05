# ================================================================
# Landsat NDSI Snow Cover Mapping Dashboard
# Streamlit + Google Earth Engine
#
# Main functions:
# 1. Draw AOI on map or upload shapefile/GeoJSON
# 2. Retrieve Landsat Collection 2 Level-2 imagery
# 3. Cloud/shadow masking using QA_PIXEL
# 4. Compute NDSI = (Green - SWIR1) / (Green + SWIR1)
# 5. Generate NDSI histogram
# 6. Estimate snow threshold using Otsu or manual value
# 7. Calculate snow-covered area
# 8. Export NDSI/snow mask as GeoTIFF and snow polygons as SHP
# ================================================================

import os
import re
import json
import zipfile
import tempfile
from pathlib import Path

import ee
import folium
import geopandas as gpd
import pandas as pd
import plotly.express as px
import streamlit as st

from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from folium.plugins import Draw
from streamlit_folium import st_folium


# ================================================================
# 1. Streamlit page configuration
# ================================================================

st.set_page_config(
    page_title="Landsat NDSI Snow Cover Dashboard",
    page_icon="❄️",
    layout="wide"
)

st.title("❄️ Landsat NDSI Snow Cover Mapping Dashboard")

st.markdown(
    """
    Please **draw your area of interest on the map** or **upload a shapefile/GeoJSON**
    to define the processing extent. The dashboard computes snow cover using
    the Normalized Difference Snow Index, NDSI.
    """
)


# ================================================================
# 2. Earth Engine initialization
# ================================================================

@st.cache_resource(show_spinner=False)
def initialize_earth_engine(project_id=None):
    """
    Initialize Google Earth Engine.

    For local use:
        earthengine authenticate
        streamlit run app.py

    For institutional/cloud deployment:
        use a configured Earth Engine project or service account workflow.
    """
    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
        return True, None

    except Exception as first_error:
        try:
            ee.Authenticate()
            if project_id:
                ee.Initialize(project=project_id)
            else:
                ee.Initialize()
            return True, None

        except Exception as auth_error:
            return False, f"{first_error}\n\nAuthentication error:\n{auth_error}"


# ================================================================
# 3. Landsat metadata
# ================================================================

LANDSAT_COLLECTIONS = {
    "Landsat 5 TM - C2 L2 T1": {
        "id": "LANDSAT/LT05/C02/T1_L2",
        "sensor": "Landsat 5 TM",
        "blue": "SR_B1",
        "green": "SR_B2",
        "red": "SR_B3",
        "nir": "SR_B4",
        "swir1": "SR_B5",
        "swir2": "SR_B7"
    },
    "Landsat 7 ETM+ - C2 L2 T1": {
        "id": "LANDSAT/LE07/C02/T1_L2",
        "sensor": "Landsat 7 ETM+",
        "blue": "SR_B1",
        "green": "SR_B2",
        "red": "SR_B3",
        "nir": "SR_B4",
        "swir1": "SR_B5",
        "swir2": "SR_B7"
    },
    "Landsat 8 OLI/TIRS - C2 L2 T1": {
        "id": "LANDSAT/LC08/C02/T1_L2",
        "sensor": "Landsat 8 OLI/TIRS",
        "blue": "SR_B2",
        "green": "SR_B3",
        "red": "SR_B4",
        "nir": "SR_B5",
        "swir1": "SR_B6",
        "swir2": "SR_B7"
    },
    "Landsat 9 OLI-2/TIRS-2 - C2 L2 T1": {
        "id": "LANDSAT/LC09/C02/T1_L2",
        "sensor": "Landsat 9 OLI-2/TIRS-2",
        "blue": "SR_B2",
        "green": "SR_B3",
        "red": "SR_B4",
        "nir": "SR_B5",
        "swir1": "SR_B6",
        "swir2": "SR_B7"
    }
}


# ================================================================
# 4. Utility functions
# ================================================================

def safe_name(text):
    """Create a safe file/export name."""
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    return text[:90] if text else "landsat_ndsi_snow"


def read_uploaded_aoi(uploaded_file, simplify_tolerance=0.0):
    """
    Read uploaded AOI.

    Accepted:
    - zipped shapefile containing .shp, .shx, .dbf, .prj
    - GeoJSON
    - JSON
    - GPKG

    Returns:
    - GeoJSON geometry dictionary in EPSG:4326
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

            if not shp_files:
                raise ValueError(
                    "No .shp file found inside the uploaded ZIP. "
                    "Please upload a zipped shapefile containing .shp, .shx, .dbf and .prj."
                )

            gdf = gpd.read_file(shp_files[0])

        elif suffix in [".geojson", ".json", ".gpkg"]:
            gdf = gpd.read_file(input_path)

        else:
            raise ValueError(
                "Unsupported file type. Please upload a zipped shapefile, GeoJSON, JSON or GPKG."
            )

    if gdf.empty:
        raise ValueError("Uploaded vector file is empty.")

    if gdf.crs is None:
        raise ValueError(
            "Uploaded file has no CRS information. Please define its projection before uploading."
        )

    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.notnull()]
    gdf = gdf[~gdf.geometry.is_empty]

    polygon_gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]

    if polygon_gdf.empty:
        raise ValueError("AOI must contain polygon or multipolygon geometry.")

    try:
        polygon_gdf["geometry"] = polygon_gdf.geometry.buffer(0)
    except Exception:
        pass

    geom = unary_union(polygon_gdf.geometry.values)

    if simplify_tolerance and simplify_tolerance > 0:
        geom = geom.simplify(simplify_tolerance, preserve_topology=True)

    return mapping(geom)


def get_drawn_geometry(map_data):
    """
    Extract the most recent polygon drawn using streamlit-folium.
    """
    if not map_data:
        return None

    drawings = map_data.get("all_drawings", None)

    if drawings and len(drawings) > 0:
        last_drawing = drawings[-1]
        geometry = last_drawing.get("geometry", None)

        if geometry and geometry.get("type") in ["Polygon", "MultiPolygon"]:
            return geometry

    return None


def get_geometry_bounds(geojson_geometry):
    """
    Get bounds from GeoJSON geometry for map zooming.
    """
    geom = shape(geojson_geometry)
    minx, miny, maxx, maxy = geom.bounds
    return [[miny, minx], [maxy, maxx]]


def make_base_map(aoi_geojson=None, zoom=5):
    """
    Create base Folium map.
    """
    center = [32.0, 78.0]

    if aoi_geojson:
        try:
            geom = shape(aoi_geojson)
            center = [geom.centroid.y, geom.centroid.x]
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
            name="AOI",
            style_function=lambda x: {
                "fillColor": "none",
                "color": "yellow",
                "weight": 3,
                "dashArray": "5,5"
            }
        ).add_to(m)

        try:
            m.fit_bounds(get_geometry_bounds(aoi_geojson))
        except Exception:
            pass

    return m


def add_ee_layer_to_folium(folium_map, ee_image, vis_params, name, shown=True, opacity=1.0):
    """
    Add Earth Engine image layer to Folium map.
    """
    map_id = ee.Image(ee_image).getMapId(vis_params)

    folium.raster_layers.TileLayer(
        tiles=map_id["tile_fetcher"].url_format,
        attr="Google Earth Engine",
        name=name,
        overlay=True,
        control=True,
        show=shown,
        opacity=opacity
    ).add_to(folium_map)


# ================================================================
# 5. Landsat preprocessing functions
# ================================================================

def mask_landsat_c2_l2(image):
    """
    Mask unwanted Landsat pixels using QA_PIXEL.

    Masked:
    - Fill
    - Dilated cloud
    - Cirrus
    - Cloud
    - Cloud shadow

    Important:
    Snow bit is NOT masked because snow is the target class.
    """
    qa = image.select("QA_PIXEL")

    fill_bit = 1 << 0
    dilated_cloud_bit = 1 << 1
    cirrus_bit = 1 << 2
    cloud_bit = 1 << 3
    cloud_shadow_bit = 1 << 4

    mask = (
        qa.bitwiseAnd(fill_bit).eq(0)
        .And(qa.bitwiseAnd(dilated_cloud_bit).eq(0))
        .And(qa.bitwiseAnd(cirrus_bit).eq(0))
        .And(qa.bitwiseAnd(cloud_bit).eq(0))
        .And(qa.bitwiseAnd(cloud_shadow_bit).eq(0))
    )

    return image.updateMask(mask)


def scale_landsat_sr_band(image, band_name):
    """
    Apply Landsat Collection 2 Level-2 optical surface reflectance scaling.

    Reflectance = DN * 0.0000275 - 0.2
    """
    return image.select(band_name).multiply(0.0000275).add(-0.2)


def prepare_landsat_collection(meta, start_date, end_date, aoi, max_cloud):
    """
    Prepare one Landsat image collection:
    - filter by AOI
    - filter by date
    - filter by cloud cover metadata
    - apply QA mask
    - scale reflectance
    - rename common bands
    - compute NDSI
    """
    collection = (
        ee.ImageCollection(meta["id"])
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUD_COVER", max_cloud))
    )

    def _prepare(image):
        image = mask_landsat_c2_l2(image)

        blue = scale_landsat_sr_band(image, meta["blue"]).rename("blue")
        green = scale_landsat_sr_band(image, meta["green"]).rename("green")
        red = scale_landsat_sr_band(image, meta["red"]).rename("red")
        nir = scale_landsat_sr_band(image, meta["nir"]).rename("nir")
        swir1 = scale_landsat_sr_band(image, meta["swir1"]).rename("swir1")
        swir2 = scale_landsat_sr_band(image, meta["swir2"]).rename("swir2")

        reflectance = ee.Image.cat([blue, green, red, nir, swir1, swir2])

        ndsi = reflectance.normalizedDifference(["green", "swir1"]).rename("NDSI")

        output = reflectance.addBands(ndsi)

        return (
            output
            .copyProperties(image, image.propertyNames())
            .set("sensor", meta["sensor"])
            .set("collection_id", meta["id"])
        )

    return collection.map(_prepare)


def build_merged_landsat_collection(selected_collections, start_date, end_date, aoi, max_cloud):
    """
    Merge selected Landsat collections into one harmonized ImageCollection.
    """
    prepared_collections = []

    for collection_name in selected_collections:
        meta = LANDSAT_COLLECTIONS[collection_name]
        prepared = prepare_landsat_collection(
            meta=meta,
            start_date=start_date,
            end_date=end_date,
            aoi=aoi,
            max_cloud=max_cloud
        )
        prepared_collections.append(prepared)

    if not prepared_collections:
        raise ValueError("No Landsat collection selected.")

    merged = prepared_collections[0]

    for extra_collection in prepared_collections[1:]:
        merged = merged.merge(extra_collection)

    return merged


def create_composite(collection, method):
    """
    Create composite from ImageCollection.
    """
    if method == "Median composite":
        return collection.median()

    if method == "Mean composite":
        return collection.mean()

    if method == "Minimum cloud single scene":
        return ee.Image(collection.sort("CLOUD_COVER").first())

    if method == "Mosaic":
        return collection.mosaic()

    raise ValueError("Unsupported composite method.")


# ================================================================
# 6. Histogram and threshold functions
# ================================================================

def get_ndsi_histogram(ndsi_image, aoi, scale=30, max_buckets=128):
    """
    Generate NDSI histogram inside AOI.
    """
    histogram_dict = (
        ndsi_image.reduceRegion(
            reducer=ee.Reducer.histogram(maxBuckets=max_buckets),
            geometry=aoi,
            scale=scale,
            bestEffort=True,
            maxPixels=1e13,
            tileScale=4
        )
        .get("NDSI")
        .getInfo()
    )

    if not histogram_dict:
        raise ValueError("Could not generate NDSI histogram. Check AOI, date range and cloud filtering.")

    counts = histogram_dict.get("histogram", [])
    means = histogram_dict.get("bucketMeans", None)

    if means is None:
        bucket_min = histogram_dict.get("bucketMin")
        bucket_width = histogram_dict.get("bucketWidth")
        means = [
            bucket_min + bucket_width * (i + 0.5)
            for i in range(len(counts))
        ]

    df = pd.DataFrame(
        {
            "NDSI": means,
            "Pixel_Count": counts
        }
    )

    df = df.dropna()

    return df


def otsu_threshold_from_histogram(hist_df):
    """
    Compute Otsu threshold from NDSI histogram.
    """
    if hist_df.empty:
        return 0.4

    values = hist_df["NDSI"].astype(float).values
    counts = hist_df["Pixel_Count"].astype(float).values

    total = counts.sum()

    if total <= 0:
        return 0.4

    sum_total = (values * counts).sum()

    weight_background = 0.0
    sum_background = 0.0
    max_between_class_variance = -1.0
    threshold = values[0]

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

        if between_class_variance > max_between_class_variance:
            max_between_class_variance = between_class_variance
            threshold = values[i]

    return float(threshold)


# ================================================================
# 7. Snow mask, area, vectorization and export functions
# ================================================================

def create_snow_mask(ndsi_image, threshold, scale=30, min_patch_area_ha=0.0):
    """
    Create binary snow mask from NDSI threshold.

    Optional:
    Removes isolated snow patches smaller than minimum mapping area.
    """
    snow = ndsi_image.gte(threshold).rename("snow_mask")
    snow = snow.selfMask()

    if min_patch_area_ha and min_patch_area_ha > 0:
        pixel_area_m2 = scale * scale
        min_pixels = max(1, int((min_patch_area_ha * 10000) / pixel_area_m2))

        max_connected_size = max(256, min(4096, min_pixels * 2))

        connected_pixels = snow.connectedPixelCount(
            maxSize=max_connected_size,
            eightConnected=True
        )

        snow = snow.updateMask(connected_pixels.gte(min_pixels))

    return snow.toByte()


def compute_area_statistics(ndsi_image, snow_mask, aoi, scale=30):
    """
    Compute AOI area, valid NDSI area and snow-covered area.
    """
    aoi_area_km2 = ee.Number(aoi.area(1)).divide(1e6).getInfo()

    valid_area_image = ee.Image.pixelArea().rename("area").updateMask(ndsi_image.mask())
    snow_area_image = ee.Image.pixelArea().rename("area").updateMask(snow_mask)

    valid_area_m2 = (
        valid_area_image.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=scale,
            bestEffort=True,
            maxPixels=1e13,
            tileScale=4
        )
        .get("area")
        .getInfo()
    )

    snow_area_m2 = (
        snow_area_image.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=scale,
            bestEffort=True,
            maxPixels=1e13,
            tileScale=4
        )
        .get("area")
        .getInfo()
    )

    valid_area_km2 = 0.0 if valid_area_m2 is None else valid_area_m2 / 1e6
    snow_area_km2 = 0.0 if snow_area_m2 is None else snow_area_m2 / 1e6

    snow_percent_valid = (
        0.0 if valid_area_km2 == 0 else (snow_area_km2 / valid_area_km2) * 100
    )

    snow_percent_aoi = (
        0.0 if aoi_area_km2 == 0 else (snow_area_km2 / aoi_area_km2) * 100
    )

    return {
        "AOI area km²": aoi_area_km2,
        "Valid Landsat area km²": valid_area_km2,
        "Snow area km²": snow_area_km2,
        "Snow % of valid Landsat area": snow_percent_valid,
        "Snow % of AOI": snow_percent_aoi
    }


def snow_mask_to_vectors(snow_mask, aoi, scale, threshold):
    """
    Convert snow raster mask to vector polygons.
    Use carefully for large AOIs because vectorization can be computationally heavy.
    """
    vectors = snow_mask.toInt().reduceToVectors(
        reducer=ee.Reducer.countEvery(),
        geometry=aoi,
        scale=scale,
        geometryType="polygon",
        eightConnected=True,
        labelProperty="snow",
        maxPixels=1e13,
        tileScale=4
    )

    vectors = vectors.map(
        lambda feature: feature.set(
            {
                "threshold": threshold,
                "area_ha": feature.geometry().area(1).divide(10000)
            }
        )
    )

    return vectors


def start_geotiff_export(export_image, aoi, folder, prefix, scale, crs=None):
    """
    Start GeoTIFF export to Google Drive.
    """
    export_params = {
        "image": export_image,
        "description": f"{prefix}_GeoTIFF",
        "folder": folder,
        "fileNamePrefix": prefix,
        "region": aoi,
        "scale": scale,
        "maxPixels": 1e13,
        "fileFormat": "GeoTIFF"
    }

    if crs and crs != "Native":
        export_params["crs"] = crs

    task = ee.batch.Export.image.toDrive(**export_params)
    task.start()
    return task.id


def start_shapefile_export(vector_fc, folder, prefix):
    """
    Start snow polygon shapefile export to Google Drive.
    """
    task = ee.batch.Export.table.toDrive(
        collection=vector_fc,
        description=f"{prefix}_Snow_Polygons_SHP",
        folder=folder,
        fileNamePrefix=f"{prefix}_snow_polygons",
        fileFormat="SHP"
    )

    task.start()
    return task.id


# ================================================================
# 8. Sidebar controls
# ================================================================

with st.sidebar:
    st.header("1. Earth Engine")

    project_id = st.text_input(
        "Earth Engine project ID, optional",
        value="",
        help="Leave blank for default local Earth Engine authentication."
    )

    ee_ok, ee_error = initialize_earth_engine(project_id.strip() or None)

    if not ee_ok:
        st.error("Earth Engine could not be initialized.")
        st.code(ee_error)
        st.stop()

    st.success("Earth Engine initialized.")

    st.header("2. AOI input")

    uploaded_file = st.file_uploader(
        "Upload AOI shapefile ZIP / GeoJSON / GPKG",
        type=["zip", "geojson", "json", "gpkg"]
    )

    simplify_tolerance = st.number_input(
        "AOI simplification tolerance in degrees",
        min_value=0.0,
        max_value=0.05,
        value=0.0,
        step=0.001,
        format="%.4f",
        help="Use small values only for very complex AOIs. 0 means no simplification."
    )

    st.header("3. Landsat settings")

    selected_collections = st.multiselect(
        "Select Landsat datasets",
        options=list(LANDSAT_COLLECTIONS.keys()),
        default=[
            "Landsat 8 OLI/TIRS - C2 L2 T1",
            "Landsat 9 OLI-2/TIRS-2 - C2 L2 T1"
        ]
    )

    col_date_1, col_date_2 = st.columns(2)

    with col_date_1:
        start_date = st.date_input("Start date", value=pd.to_datetime("2024-01-01"))

    with col_date_2:
        end_date = st.date_input("End date", value=pd.to_datetime("2024-12-31"))

    max_cloud = st.slider(
        "Maximum scene cloud cover (%)",
        min_value=0,
        max_value=100,
        value=40,
        step=5
    )

    composite_method = st.selectbox(
        "Composite method",
        [
            "Median composite",
            "Mean composite",
            "Minimum cloud single scene",
            "Mosaic"
        ],
        index=0
    )

    processing_scale = st.selectbox(
        "Processing scale",
        options=[30, 60, 90, 120],
        index=0,
        help="Use 30 m for final analysis. Larger values speed up exploratory processing."
    )

    histogram_buckets = st.slider(
        "Histogram buckets",
        min_value=32,
        max_value=256,
        value=128,
        step=32
    )

    st.header("4. Snow threshold")

    threshold_mode = st.radio(
        "Threshold mode",
        ["Auto Otsu", "Manual"],
        index=0
    )

    manual_threshold = st.number_input(
        "What threshold value would you like to use for snow detection?",
        min_value=-1.0,
        max_value=1.0,
        value=0.40,
        step=0.01,
        format="%.2f"
    )

    min_patch_area_ha = st.number_input(
        "Minimum snow patch area to retain, ha",
        min_value=0.0,
        max_value=1000.0,
        value=0.0,
        step=0.5,
        help="Set 0 to retain all patches."
    )

    st.header("5. Export settings")

    export_folder = st.text_input(
        "Google Drive export folder",
        value="GEE_Snow_Exports"
    )

    export_prefix = safe_name(
        st.text_input(
            "Export file prefix",
            value="landsat_ndsi_snow"
        )
    )

    export_crs = st.selectbox(
        "Export CRS",
        ["Native", "EPSG:4326", "EPSG:32643", "EPSG:32644", "EPSG:32645"],
        index=0,
        help="Use Native unless a specific projection is required."
    )


# ================================================================
# 9. AOI map: upload or draw
# ================================================================

uploaded_aoi_geojson = None

if uploaded_file:
    try:
        uploaded_aoi_geojson = read_uploaded_aoi(
            uploaded_file=uploaded_file,
            simplify_tolerance=simplify_tolerance
        )
        st.success("Uploaded AOI loaded successfully.")
    except Exception as e:
        st.error(f"AOI upload error: {e}")
        uploaded_aoi_geojson = None

st.subheader("Step 1: Draw AOI or upload shapefile")

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
    edit_options={"edit": True, "remove": True}
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
    st.info("AOI source: drawn polygon on map.")
else:
    selected_aoi_geojson = None
    st.warning("Please draw your area of interest on the map or upload a shapefile.")


# ================================================================
# 10. Processing
# ================================================================

process_button = st.button("Generate Snow Cover Map", type="primary")

if process_button:
    if selected_aoi_geojson is None:
        st.error("No AOI selected. Please draw an AOI or upload a shapefile.")
        st.stop()

    if not selected_collections:
        st.error("Please select at least one Landsat dataset.")
        st.stop()

    if pd.to_datetime(start_date) >= pd.to_datetime(end_date):
        st.error("Start date must be earlier than end date.")
        st.stop()

    try:
        with st.spinner("Processing Landsat imagery and computing NDSI..."):
            aoi = ee.Geometry(selected_aoi_geojson)

            landsat_collection = build_merged_landsat_collection(
                selected_collections=selected_collections,
                start_date=str(start_date),
                end_date=str(end_date),
                aoi=aoi,
                max_cloud=max_cloud
            )

            image_count = landsat_collection.size().getInfo()

            if image_count == 0:
                st.error(
                    "No Landsat images found for the selected AOI/date/cloud filter. "
                    "Please increase date range or cloud cover threshold."
                )
                st.stop()

            composite = create_composite(
                collection=landsat_collection,
                method=composite_method
            ).clip(aoi)

            ndsi = composite.select("NDSI").clip(aoi)

            hist_df = get_ndsi_histogram(
                ndsi_image=ndsi,
                aoi=aoi,
                scale=processing_scale,
                max_buckets=histogram_buckets
            )

            otsu_threshold = otsu_threshold_from_histogram(hist_df)

            if threshold_mode == "Auto Otsu":
                final_threshold = otsu_threshold
            else:
                final_threshold = float(manual_threshold)

            snow_mask = create_snow_mask(
                ndsi_image=ndsi,
                threshold=final_threshold,
                scale=processing_scale,
                min_patch_area_ha=min_patch_area_ha
            ).clip(aoi)

            area_stats = compute_area_statistics(
                ndsi_image=ndsi,
                snow_mask=snow_mask,
                aoi=aoi,
                scale=processing_scale
            )

            export_image = (
                ndsi.float()
                .rename("NDSI")
                .addBands(snow_mask.float().rename("snow_mask"))
                .clip(aoi)
            )

            snow_vectors = snow_mask_to_vectors(
                snow_mask=snow_mask,
                aoi=aoi,
                scale=processing_scale,
                threshold=final_threshold
            )

            st.session_state["result"] = {
                "aoi": aoi,
                "aoi_geojson": selected_aoi_geojson,
                "collection": landsat_collection,
                "image_count": image_count,
                "composite": composite,
                "ndsi": ndsi,
                "hist_df": hist_df,
                "otsu_threshold": otsu_threshold,
                "final_threshold": final_threshold,
                "snow_mask": snow_mask,
                "area_stats": area_stats,
                "export_image": export_image,
                "snow_vectors": snow_vectors,
                "processing_scale": processing_scale
            }

        st.success("Snow cover processing completed.")

    except ee.EEException as e:
        st.error(f"Earth Engine processing error: {e}")

    except Exception as e:
        st.error(f"Processing error: {e}")


# ================================================================
# 11. Display results
# ================================================================

if "result" in st.session_state:
    result = st.session_state["result"]

    st.subheader("Step 2: Processing Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Images used", result["image_count"])
    col2.metric("Threshold", f"{result['final_threshold']:.3f}")
    col3.metric("Snow area", f"{result['area_stats']['Snow area km²']:.2f} km²")
    col4.metric("Snow % of valid area", f"{result['area_stats']['Snow % of valid Landsat area']:.2f}%")

    with st.expander("Detailed area statistics", expanded=True):
        stats_df = pd.DataFrame(
            list(result["area_stats"].items()),
            columns=["Parameter", "Value"]
        )
        st.dataframe(stats_df, use_container_width=True)

    st.subheader("Step 3: NDSI Histogram and Threshold")

    hist_df = result["hist_df"]

    fig = px.bar(
        hist_df,
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

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Auto Otsu threshold estimated from histogram: {result['otsu_threshold']:.3f}"
    )

    st.subheader("Step 4: Snow Cover Map")

    result_map = make_base_map(result["aoi_geojson"])

    add_ee_layer_to_folium(
        folium_map=result_map,
        ee_image=result["composite"].select(["red", "green", "blue"]),
        vis_params={
            "bands": ["red", "green", "blue"],
            "min": 0.02,
            "max": 0.35,
            "gamma": 1.2
        },
        name="Landsat RGB Composite",
        shown=True,
        opacity=1.0
    )

    add_ee_layer_to_folium(
        folium_map=result_map,
        ee_image=result["ndsi"],
        vis_params={
            "min": -0.5,
            "max": 1.0,
            "palette": ["7f3b08", "fdb863", "ffffbf", "abd9e9", "2c7bb6"]
        },
        name="NDSI",
        shown=False,
        opacity=0.85
    )

    add_ee_layer_to_folium(
        folium_map=result_map,
        ee_image=result["snow_mask"],
        vis_params={
            "min": 1,
            "max": 1,
            "palette": ["00FFFF"]
        },
        name="Snow Mask",
        shown=True,
        opacity=0.75
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
    ).add_to(result_map)

    folium.LayerControl().add_to(result_map)

    st_folium(
        result_map,
        height=620,
        use_container_width=True
    )

    st.subheader("Step 5: Export Outputs")

    st.markdown(
        """
        Exported GeoTIFF contains two bands:

        1. `NDSI`
        2. `snow_mask`

        Shapefile export contains snow polygons derived from the thresholded snow mask.
        """
    )

    export_col1, export_col2 = st.columns(2)

    with export_col1:
        if st.button("Start GeoTIFF Export to Google Drive"):
            try:
                geotiff_task_id = start_geotiff_export(
                    export_image=result["export_image"],
                    aoi=result["aoi"],
                    folder=export_folder,
                    prefix=export_prefix,
                    scale=result["processing_scale"],
                    crs=export_crs
                )

                st.success(
                    f"GeoTIFF export started. Task ID: {geotiff_task_id}"
                )

            except Exception as e:
                st.error(f"GeoTIFF export error: {e}")

    with export_col2:
        if st.button("Start Snow Polygon SHP Export to Google Drive"):
            try:
                shp_task_id = start_shapefile_export(
                    vector_fc=result["snow_vectors"],
                    folder=export_folder,
                    prefix=export_prefix
                )

                st.success(
                    f"Shapefile export started. Task ID: {shp_task_id}"
                )

            except Exception as e:
                st.error(f"Shapefile export error: {e}")

    with st.expander("Optional small-AOI direct download link"):
        st.warning(
            "Direct download is suitable only for small AOIs. "
            "For larger regions, use Google Drive export tasks."
        )

        if st.button("Create temporary GeoTIFF download link"):
            try:
                download_url = result["export_image"].getDownloadURL(
                    {
                        "scale": result["processing_scale"],
                        "region": result["aoi"],
                        "format": "GEO_TIFF"
                    }
                )

                st.link_button("Download GeoTIFF ZIP", download_url)

            except Exception as e:
                st.error(f"Direct download link error: {e}")


# ================================================================
# End of application
# ================================================================
