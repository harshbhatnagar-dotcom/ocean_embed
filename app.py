import os
import json
import tempfile
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import xarray as xr
import streamlit as st
import plotly.graph_objects as go

import copernicusmarine

from scipy.interpolate import griddata
from tensorflow.keras.models import load_model

from dotenv import load_dotenv

load_dotenv()

# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="OceanEmbed",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = APP_DIR / "models" / "best_oceanembed_model.keras"

DATA_DIR = APP_DIR / "data"

MASK_PATH = DATA_DIR / "common_ocean_mask.npy"
LATITUDE_PATH = DATA_DIR / "latitude.npy"
LONGITUDE_PATH = DATA_DIR / "longitude.npy"
DEPTHS_PATH = DATA_DIR / "depths.npy"
NORMALIZATION_PATH = DATA_DIR / "normalization_stats.json"


# OceanEmbed domain
LAT_MIN = 5.0
LAT_MAX = 30.0

LON_MIN = 45.0
LON_MAX = 105.0

GRID_RESOLUTION = 0.25

PATCH_SIZE = 32

MIN_PATCH_COVERAGE = 0.75


# ============================================================
# COPERNICUS MARINE DATASETS
# ============================================================

SST_DATASET_ID = "METOFFICE-GLO-SST-L4-REP-OBS-SST"
SST_VARIABLE = "analysed_sst"

SSS_DATASET_ID = "cmems_obs-mob_glo_phy-sss_nrt_multi_P1D"
SSS_VARIABLE = "sos"

SLA_DATASET_ID = "cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D"
SLA_VARIABLE = "sla"

CURRENT_DATASET_ID = "cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D-m"
U_VARIABLE = "uo"
V_VARIABLE = "vo"


# ============================================================
# DEPTHS
# ============================================================

TARGET_DEPTHS = np.array(
    [
        0,
        5,
        10,
        20,
        30,
        50,
        75,
        100,
        125,
        150,
        200,
        300,
        500,
        700,
        1000,
    ],
    dtype=np.float32,
)


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🌊 OceanEmbed")

st.markdown(
    """
### Satellite-Based Subsurface Ocean Temperature Reconstruction

OceanEmbed uses satellite surface observations:

**SST + SSS + SLA + Surface Currents → Deep Learning → Subsurface Temperature**

The model reconstructs temperature from the surface down to **1000 m**.
"""
)

st.info(
    "GLORYS is used during model training as the target dataset. "
    "It is NOT used as an input during live inference."
)


# ============================================================
# LOAD STATIC DATA
# ============================================================

@st.cache_resource
def load_static_resources():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    if not MASK_PATH.exists():
        raise FileNotFoundError(
            f"Ocean mask not found:\n{MASK_PATH}"
        )

    if not LATITUDE_PATH.exists():
        raise FileNotFoundError(
            f"Latitude file not found:\n{LATITUDE_PATH}"
        )

    if not LONGITUDE_PATH.exists():
        raise FileNotFoundError(
            f"Longitude file not found:\n{LONGITUDE_PATH}"
        )

    if not DEPTHS_PATH.exists():
        raise FileNotFoundError(
            f"Depth file not found:\n{DEPTHS_PATH}"
        )

    if not NORMALIZATION_PATH.exists():
        raise FileNotFoundError(
            f"Normalization file not found:\n{NORMALIZATION_PATH}"
        )

    model = load_model(MODEL_PATH)

    common_ocean_mask = np.load(MASK_PATH).astype(bool)

    latitude = np.load(LATITUDE_PATH)

    longitude = np.load(LONGITUDE_PATH)

    depths = np.load(DEPTHS_PATH)

    with open(NORMALIZATION_PATH, "r") as f:
        normalization_stats = json.load(f)

    return (
        model,
        common_ocean_mask,
        latitude,
        longitude,
        depths,
        normalization_stats,
    )


# ============================================================
# LOAD RESOURCES
# ============================================================

try:

    (
        model,
        common_ocean_mask,
        latitude,
        longitude,
        depths,
        normalization_stats,
    ) = load_static_resources()

except Exception as e:

    st.error("Failed to load OceanEmbed resources.")

    st.exception(e)

    st.stop()


# ============================================================
# GRID VALIDATION
# ============================================================

if common_ocean_mask.shape != (len(latitude), len(longitude)):

    st.error(
        "Ocean mask dimensions do not match latitude/longitude grid."
    )

    st.stop()


# ============================================================
# COPERNICUS LOGIN
# ============================================================

@st.cache_resource
def copernicus_login():

    username = os.getenv("COPERNICUSMARINE_USERNAME")

    password = os.getenv("COPERNICUSMARINE_PASSWORD")

    if not username or not password:

        raise RuntimeError(
            "Copernicus Marine credentials are not configured.\n\n"
            "Set:\n"
            "COPERNICUSMARINE_USERNAME\n"
            "COPERNICUSMARINE_PASSWORD"
        )

    copernicusmarine.login(
        username=username,
        password=password,
    )

    return True


# ============================================================
# TEMPORARY DATA DOWNLOAD
# ============================================================

def download_copernicus_dataset(
    dataset_id,
    variables,
    start_datetime,
    end_datetime,
    output_directory,
):

    copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=variables,
        minimum_longitude=LON_MIN,
        maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN,
        maximum_latitude=LAT_MAX,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        output_directory=output_directory,
        force_download=True,
    )


# ============================================================
# FIND NETCDF FILE
# ============================================================

def find_netcdf_file(directory):

    directory = Path(directory)

    files = list(directory.rglob("*.nc"))

    if not files:

        raise FileNotFoundError(
            f"No NetCDF file found in {directory}"
        )

    # Usually only one file is created per dataset download
    return files[0]


# ============================================================
# REMOVE NON-SPATIAL DIMENSIONS
# ============================================================

def prepare_data_array(da):

    # Remove singleton dimensions
    da = da.squeeze(drop=True)

    # Handle latitude naming
    rename_map = {}

    if "lat" in da.dims:
        rename_map["lat"] = "latitude"

    if "lon" in da.dims:
        rename_map["lon"] = "longitude"

    if rename_map:
        da = da.rename(rename_map)

    # Sort coordinates
    if "latitude" in da.coords:
        da = da.sortby("latitude")

    if "longitude" in da.coords:
        da = da.sortby("longitude")

    return da


# ============================================================
# REGRID TO TRAINING GRID
# ============================================================

def regrid_to_training_grid(da):

    da = prepare_data_array(da)

    target_lat = xr.DataArray(
        latitude,
        dims="latitude",
        coords={"latitude": latitude},
    )

    target_lon = xr.DataArray(
        longitude,
        dims="longitude",
        coords={"longitude": longitude},
    )

    da_regridded = da.interp(
        latitude=target_lat,
        longitude=target_lon,
        method="linear",
    )

    return da_regridded.values.astype(np.float32)


# ============================================================
# CURRENT SURFACE EXTRACTION
# ============================================================

def extract_surface_current(da):

    da = prepare_data_array(da)

    if "depth" in da.dims:

        try:
            da = da.sel(depth=0, method="nearest")
        except Exception:
            da = da.isel(depth=0)

    return regrid_to_training_grid(da)


# ============================================================
# LOAD LIVE OCEAN DATA
# ============================================================

def load_live_data(selected_date):

    date_string = selected_date.strftime("%Y-%m-%d")

    start_datetime = f"{date_string}T00:00:00"
    end_datetime = f"{date_string}T23:59:59"

    # TemporaryDirectory automatically deletes everything
    # when this function finishes.
    with tempfile.TemporaryDirectory(
        prefix="oceanembed_"
    ) as temp_dir:

        temp_dir = Path(temp_dir)

        # ----------------------------------------------------
        # SST
        # ----------------------------------------------------

        sst_dir = temp_dir / "sst"
        sst_dir.mkdir()

        with st.status(
            "Downloading SST...",
            expanded=False,
        ):

            download_copernicus_dataset(
                dataset_id=SST_DATASET_ID,
                variables=[SST_VARIABLE],
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                output_directory=str(sst_dir),
            )

        sst_file = find_netcdf_file(sst_dir)

        # ----------------------------------------------------
        # SSS
        # ----------------------------------------------------

        sss_dir = temp_dir / "sss"
        sss_dir.mkdir()

        with st.status(
            "Downloading SSS...",
            expanded=False,
        ):

            download_copernicus_dataset(
                dataset_id=SSS_DATASET_ID,
                variables=[SSS_VARIABLE],
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                output_directory=str(sss_dir),
            )

        sss_file = find_netcdf_file(sss_dir)

        # ----------------------------------------------------
        # SLA
        # ----------------------------------------------------

        sla_dir = temp_dir / "sla"
        sla_dir.mkdir()

        with st.status(
            "Downloading SLA...",
            expanded=False,
        ):

            download_copernicus_dataset(
                dataset_id=SLA_DATASET_ID,
                variables=[SLA_VARIABLE],
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                output_directory=str(sla_dir),
            )

        sla_file = find_netcdf_file(sla_dir)

        # ----------------------------------------------------
        # SURFACE CURRENTS
        # ----------------------------------------------------

        current_dir = temp_dir / "currents"
        current_dir.mkdir()

        with st.status(
            "Downloading surface currents...",
            expanded=False,
        ):

            download_copernicus_dataset(
                dataset_id=CURRENT_DATASET_ID,
                variables=[U_VARIABLE, V_VARIABLE],
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                output_directory=str(current_dir),
            )

        current_file = find_netcdf_file(current_dir)

        # ----------------------------------------------------
        # READ DATA
        # ----------------------------------------------------

        with xr.open_dataset(sst_file) as ds:

            sst_da = ds[SST_VARIABLE]

            # SST from Copernicus is Kelvin.
            sst_da = sst_da - 273.15

            sst = regrid_to_training_grid(sst_da)

        with xr.open_dataset(sss_file) as ds:

            sss_da = ds[SSS_VARIABLE]

            # The downloaded SSS product is already represented
            # in the values used by the training pipeline.
            sss = regrid_to_training_grid(sss_da)

        with xr.open_dataset(sla_file) as ds:

            sla_da = ds[SLA_VARIABLE]

            # SLA is meters.
            sla = regrid_to_training_grid(sla_da)

        with xr.open_dataset(current_file) as ds:

            u_da = ds[U_VARIABLE]

            v_da = ds[V_VARIABLE]

            u = extract_surface_current(u_da)

            v = extract_surface_current(v_da)

    # TemporaryDirectory is now automatically deleted.

    return sst, sss, sla, u, v


# ============================================================
# BUILD LIVE VALIDITY MASK
# ============================================================

def build_live_validity_mask(
    sst,
    sss,
    sla,
    u,
    v,
):

    live_valid_mask = (
        np.isfinite(sst)
        & np.isfinite(sss)
        & np.isfinite(sla)
        & np.isfinite(u)
        & np.isfinite(v)
    )

    final_valid_mask = (
        common_ocean_mask
        & live_valid_mask
    )

    return final_valid_mask


# ============================================================
# NORMALIZE FEATURES
# ============================================================

def normalize_features(
    sst,
    sss,
    sla,
    u,
    v,
    valid_mask,
):

    feature_arrays = [
        sst,
        sss,
        sla,
        u,
        v,
    ]

    feature_names = [
        "SST",
        "SSS",
        "SLA",
        "U",
        "V",
    ]

    normalized = []

    for arr, name in zip(
        feature_arrays,
        feature_names,
    ):

        stats = normalization_stats[name]

        mean = float(stats["mean"])

        std = float(stats["std"])

        if std == 0:

            raise ValueError(
                f"Standard deviation for {name} is zero."
            )

        arr_norm = (
            arr - mean
        ) / std

        arr_norm = np.nan_to_num(
            arr_norm,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        normalized.append(
            arr_norm.astype(np.float32)
        )

    X = np.stack(
        normalized,
        axis=-1,
    )

    # Invalid locations are explicitly zeroed.
    X[~valid_mask] = 0.0

    return X


# ============================================================
# EXTRACT PATCHES
# ============================================================

def extract_live_patches(
    X,
    valid_mask,
):

    patches = []

    centers = []

    coverages = []

    half = PATCH_SIZE // 2

    height, width = valid_mask.shape

    for i in range(
        half,
        height - half + 1,
    ):

        for j in range(
            half,
            width - half + 1,
        ):

            mask_patch = valid_mask[
                i - half:i + half,
                j - half:j + half,
            ]

            coverage = float(
                mask_patch.mean()
            )

            if coverage >= MIN_PATCH_COVERAGE:

                patch = X[
                    i - half:i + half,
                    j - half:j + half,
                    :,
                ]

                patches.append(
                    patch
                )

                # IMPORTANT:
                # Store [latitude, longitude]
                centers.append(
                    [
                        float(latitude[i]),
                        float(longitude[j]),
                    ]
                )

                coverages.append(
                    coverage
                )

    if not patches:

        return (
            np.empty(
                (
                    0,
                    PATCH_SIZE,
                    PATCH_SIZE,
                    5,
                ),
                dtype=np.float32,
            ),
            np.empty(
                (0, 2),
                dtype=np.float32,
            ),
            np.empty(
                (0,),
                dtype=np.float32,
            ),
        )

    return (
        np.asarray(
            patches,
            dtype=np.float32,
        ),
        np.asarray(
            centers,
            dtype=np.float32,
        ),
        np.asarray(
            coverages,
            dtype=np.float32,
        ),
    )


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_patches(
    model,
    patches,
):

    if len(patches) == 0:

        raise ValueError(
            "No valid patches available for prediction."
        )

    predictions = model.predict(
        patches,
        batch_size=64,
        verbose=0,
    )

    predictions = np.asarray(
        predictions,
        dtype=np.float32,
    )

    if predictions.ndim != 2:

        raise ValueError(
            f"Unexpected prediction shape: "
            f"{predictions.shape}"
        )

    if predictions.shape[1] != len(depths):

        raise ValueError(
            f"Model returned {predictions.shape[1]} "
            f"outputs, but {len(depths)} depths are expected."
        )

    return predictions


# ============================================================
# RECONSTRUCT FULL MAPS
# ============================================================

def reconstruct_full_maps(
    predictions,
    patch_centers,
    final_valid_mask,
):

    n_depths = predictions.shape[1]

    n_lat = len(latitude)

    n_lon = len(longitude)

    output_maps = np.full(
        (
            n_depths,
            n_lat,
            n_lon,
        ),
        np.nan,
        dtype=np.float32,
    )

    # Grid of complete training coordinates
    lon_grid, lat_grid = np.meshgrid(
        longitude,
        latitude,
    )

    center_lats = patch_centers[:, 0]

    center_lons = patch_centers[:, 1]

    points = np.column_stack(
        [
            center_lats,
            center_lons,
        ]
    )

    target_points = np.column_stack(
        [
            lat_grid.ravel(),
            lon_grid.ravel(),
        ]
    )

    for depth_idx in range(n_depths):

        values = predictions[
            :,
            depth_idx,
        ]

        # Linear interpolation
        interpolated = griddata(
            points,
            values,
            target_points,
            method="linear",
        )

        interpolated = interpolated.reshape(
            n_lat,
            n_lon,
        )

        # Fill remaining holes with nearest-neighbour
        # interpolation where possible.
        missing = ~np.isfinite(
            interpolated
        )

        if np.any(missing):

            nearest_values = griddata(
                points,
                values,
                target_points[missing.ravel()],
                method="nearest",
            )

            interpolated[
                missing
            ] = nearest_values

        # Apply final live validity mask
        interpolated[
            ~final_valid_mask
        ] = np.nan

        output_maps[
            depth_idx
        ] = interpolated.astype(
            np.float32
        )

    return output_maps


# ============================================================
# MAP FIGURE
# ============================================================

def create_temperature_map(
    temperature_map,
    selected_depth,
    selected_date,
):

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=temperature_map,
            x=longitude,
            y=latitude,
            colorbar=dict(
                title="Temperature (°C)"
            ),
            hovertemplate=(
                "Latitude: %{y:.2f}°N"
                "<br>Longitude: %{x:.2f}°E"
                "<br>Temperature: %{z:.2f} °C"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=(
            f"Predicted Subsurface Temperature "
            f"at {selected_depth:.0f} m"
            f"<br><sup>{selected_date}</sup>"
        ),
        xaxis_title="Longitude (°E)",
        yaxis_title="Latitude (°N)",
        height=650,
        margin=dict(
            l=40,
            r=40,
            t=80,
            b=40,
        ),
    )

    return fig


# ============================================================
# PROFILE FIGURE
# ============================================================

def create_profile_figure(
    profile_temperature,
    profile_depths,
    latitude_value,
    longitude_value,
):

    fig = go.Figure()

    # --------------------------------------------------------
    # Temperature profile
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=profile_temperature,
            y=profile_depths,
            mode="lines+markers",
            name="Predicted Temperature",
            line=dict(
                width=3
            ),
            marker=dict(
                size=7
            ),
            hovertemplate=(
                "<b>Temperature:</b> %{x:.2f} °C"
                "<br><b>Depth:</b> %{y:.0f} m"
                "<extra></extra>"
            ),
        )
    )

    # --------------------------------------------------------
    # Create cleaner depth ticks
    # --------------------------------------------------------

    # Show fewer labels to avoid crowding,
    # while keeping all 15 prediction points.
    display_depths = [
        0,
        50,
        100,
        150,
        200,
        300,
        500,
        700,
        1000,
    ]

    fig.update_layout(
        title=(
            f"Predicted Temperature Profile"
            f"<br><sup>"
            f"{latitude_value:.2f}°N, "
            f"{longitude_value:.2f}°E"
            f"</sup>"
        ),

        xaxis=dict(
            title="Temperature (°C)",
            showgrid=True,
        ),

        yaxis=dict(
            title="Depth (m)",
            autorange="reversed",

            # Only show selected depth labels
            tickmode="array",

            tickvals=display_depths,

            ticktext=[
                f"{d} m"
                for d in display_depths
            ],

            showgrid=True,
        ),

        height=600,

        margin=dict(
            l=80,
            r=40,
            t=100,
            b=60,
        ),

        hovermode="closest",

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig
# ============================================================
# CSV CREATION
# ============================================================

def create_prediction_csv(
    temperature_maps,
):

    rows = []

    for depth_idx, depth_value in enumerate(depths):

        temp_map = temperature_maps[
            depth_idx
        ]

        valid_indices = np.where(
            np.isfinite(temp_map)
        )

        lat_indices = valid_indices[0]

        lon_indices = valid_indices[1]

        rows_for_depth = pd.DataFrame(
            {
                "latitude": latitude[
                    lat_indices
                ],
                "longitude": longitude[
                    lon_indices
                ],
                "depth_m": float(depth_value),
                "temperature_C": temp_map[
                    lat_indices,
                    lon_indices,
                ],
            }
        )

        rows.append(
            rows_for_depth
        )

    if not rows:

        return pd.DataFrame()

    return pd.concat(
        rows,
        ignore_index=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ OceanEmbed Settings")

st.sidebar.markdown(
    """
**Study Region**

Latitude: 5°N – 30°N

Longitude: 45°E – 105°E

Resolution: 0.25°
"""
)


# ============================================================
# DATE SELECTION
# ============================================================

st.sidebar.subheader("📅 Select Date")

MIN_DATE = date(2024, 7, 1)

MAX_DATE = date.today()

selected_date = st.sidebar.date_input(
    "Observation date",
    value=date(2025, 1, 15),
    min_value=MIN_DATE,
    max_value=MAX_DATE,
)


# ============================================================
# DEPTH SELECTION
# ============================================================

st.sidebar.subheader("🌊 Select Depth")

selected_depth = st.sidebar.selectbox(
    "Depth",
    options=[
        int(d)
        for d in depths
    ],
    index=0,
)


# ============================================================
# RUN BUTTON
# ============================================================

run_prediction = st.sidebar.button(
    "🚀 Run OceanEmbed",
    type="primary",
    use_container_width=True,
)


# ============================================================
# RESOURCE INFORMATION
# ============================================================

with st.sidebar.expander(
    "ℹ️ Model Information"
):

    st.write(
        f"Input shape: "
        f"`32 × 32 × 5`"
    )

    st.write(
        f"Output depths: "
        f"`{len(depths)}`"
    )

    st.write(
        "Maximum depth: "
        f"`{int(np.max(depths))} m`"
    )

    st.write(
        "Minimum patch coverage: "
        f"`{MIN_PATCH_COVERAGE * 100:.0f}%`"
    )

    st.write(
        "TensorFlow model: "
        "`best_oceanembed_model.keras`"
    )


# ============================================================
# RUN INFERENCE
# ============================================================

if run_prediction:

    # --------------------------------------------------------
    # Login
    # --------------------------------------------------------

    try:

        with st.spinner(
            "Connecting to Copernicus Marine..."
        ):

            copernicus_login()

    except Exception as e:

        st.error(
            "Copernicus Marine login failed."
        )

        st.exception(e)

        st.stop()

    # --------------------------------------------------------
    # Download live data
    # --------------------------------------------------------

    try:

        progress = st.progress(
            0,
            text="Starting live data extraction..."
        )

        progress.progress(
            10,
            text="Downloading live ocean observations..."
        )

        (
            sst,
            sss,
            sla,
            u,
            v,
        ) = load_live_data(
            selected_date
        )

        progress.progress(
            40,
            text="Live data downloaded and regridded."
        )

    except Exception as e:

        st.error(
            "Failed to download or process "
            "Copernicus Marine data."
        )

        st.exception(e)

        st.stop()

    # --------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------

    expected_shape = (
        len(latitude),
        len(longitude),
    )

    arrays = {
        "SST": sst,
        "SSS": sss,
        "SLA": sla,
        "U": u,
        "V": v,
    }

    for name, arr in arrays.items():

        if arr.shape != expected_shape:

            st.error(
                f"{name} has shape {arr.shape}, "
                f"expected {expected_shape}."
            )

            st.stop()

    # --------------------------------------------------------
    # Build validity mask
    # --------------------------------------------------------

    progress.progress(
        50,
        text="Building ocean validity mask..."
    )

    final_valid_mask = build_live_validity_mask(
        sst,
        sss,
        sla,
        u,
        v,
    )

    valid_count = int(
        final_valid_mask.sum()
    )

    total_ocean_cells = int(
        common_ocean_mask.sum()
    )

    if valid_count == 0:

        st.error(
            "No valid ocean cells are available "
            "for the selected date."
        )

        st.stop()

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    progress.progress(
        60,
        text="Normalizing satellite features..."
    )

    X_normalized = normalize_features(
        sst,
        sss,
        sla,
        u,
        v,
        final_valid_mask,
    )

    # --------------------------------------------------------
    # Extract patches
    # --------------------------------------------------------

    progress.progress(
        65,
        text="Extracting 32×32 satellite patches..."
    )

    (
        patches,
        patch_centers,
        patch_coverages,
    ) = extract_live_patches(
        X_normalized,
        final_valid_mask,
    )

    if len(patches) == 0:

        st.error(
            "No valid 32×32 patches were found "
            "for this date."
        )

        st.stop()

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    progress.progress(
        75,
        text="Running OceanEmbed deep learning model..."
    )

    predictions = predict_patches(
        model,
        patches,
    )

    # --------------------------------------------------------
    # Reconstruct maps
    # --------------------------------------------------------

    progress.progress(
        90,
        text="Reconstructing full-area temperature maps..."
    )

    temperature_maps = reconstruct_full_maps(
        predictions,
        patch_centers,
        final_valid_mask,
    )

    progress.progress(
        100,
        text="Prediction completed."
    )

    progress.empty()

    # --------------------------------------------------------
    # Save results in session state
    # --------------------------------------------------------

    st.session_state["temperature_maps"] = (
        temperature_maps
    )

    st.session_state["predictions"] = (
        predictions
    )

    st.session_state["patch_centers"] = (
        patch_centers
    )

    st.session_state["patch_coverages"] = (
        patch_coverages
    )

    st.session_state["sst"] = sst
    st.session_state["sss"] = sss
    st.session_state["sla"] = sla
    st.session_state["u"] = u
    st.session_state["v"] = v

    st.session_state["final_valid_mask"] = (
        final_valid_mask
    )

    st.session_state["selected_date"] = (
        selected_date
    )

    st.session_state["prediction_complete"] = True

    st.success(
        f"OceanEmbed prediction completed for "
        f"{selected_date.strftime('%Y-%m-%d')}."
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

if st.session_state.get(
    "prediction_complete",
    False
):

    temperature_maps = st.session_state[
        "temperature_maps"
    ]

    predictions = st.session_state[
        "predictions"
    ]

    patch_centers = st.session_state[
        "patch_centers"
    ]

    patch_coverages = st.session_state[
        "patch_coverages"
    ]

    final_valid_mask = st.session_state[
        "final_valid_mask"
    ]

    result_date = st.session_state[
        "selected_date"
    ]


    # ========================================================
    # SUMMARY METRICS
    # ========================================================

    st.subheader("📊 Prediction Summary")

    depth_index = int(
        np.where(
            depths == selected_depth
        )[0][0]
    )

    selected_map = temperature_maps[
        depth_index
    ]

    valid_temperature = selected_map[
        np.isfinite(selected_map)
    ]

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Selected Depth",
            f"{selected_depth} m",
        )

    with col2:

        if len(valid_temperature) > 0:

            st.metric(
                "Mean Temperature",
                f"{np.mean(valid_temperature):.2f} °C",
            )

    with col3:

        if len(valid_temperature) > 0:

            st.metric(
                "Minimum",
                f"{np.min(valid_temperature):.2f} °C",
            )

    with col4:

        if len(valid_temperature) > 0:

            st.metric(
                "Maximum",
                f"{np.max(valid_temperature):.2f} °C",
            )


    # ========================================================
    # TEMPERATURE MAP
    # ========================================================

    st.subheader(
        f"🌡️ Predicted Temperature Map — {selected_depth} m"
    )

    map_figure = create_temperature_map(
        selected_map,
        selected_depth,
        result_date,
    )

    st.plotly_chart(
        map_figure,
        use_container_width=True,
    )


    # ========================================================
    # PROFILE SECTION
    # ========================================================

    st.subheader(
        "📍 Predicted Temperature Profile"
    )

    st.write(
        "Select a location to view the predicted "
        "temperature profile from the surface to 1000 m."
    )

    col1, col2 = st.columns(2)

    with col1:

        selected_lat = st.number_input(
            "Latitude (°N)",
            min_value=float(
                np.min(latitude)
            ),
            max_value=float(
                np.max(latitude)
            ),
            value=22.0,
            step=0.25,
            format="%.2f",
        )

    with col2:

        selected_lon = st.number_input(
            "Longitude (°E)",
            min_value=float(
                np.min(longitude)
            ),
            max_value=float(
                np.max(longitude)
            ),
            value=75.0,
            step=0.25,
            format="%.2f",
        )


    # --------------------------------------------------------
    # FIND NEAREST PATCH CENTER
    # --------------------------------------------------------

    if len(patch_centers) > 0:

        patch_latitudes = (
            patch_centers[:, 0]
        )

        patch_longitudes = (
            patch_centers[:, 1]
        )

        distance = (
            (patch_latitudes - selected_lat) ** 2
            +
            (patch_longitudes - selected_lon) ** 2
        )

        nearest_idx = int(
            np.argmin(distance)
        )

        nearest_lat = float(
            patch_latitudes[
                nearest_idx
            ]
        )

        nearest_lon = float(
            patch_longitudes[
                nearest_idx
            ]
        )

        profile_temperature = (
            predictions[
                nearest_idx,
                :
            ].astype(float)
        )

        profile_depths = (
            depths.astype(float)
        )

        profile_figure = create_profile_figure(
            profile_temperature,
            profile_depths,
            nearest_lat,
            nearest_lon,
        )

        st.plotly_chart(
            profile_figure,
            use_container_width=True,
        )

        st.caption(
            f"Nearest valid model patch: "
            f"{nearest_lat:.2f}°N, "
            f"{nearest_lon:.2f}°E"
        )

        # ----------------------------------------------------
        # PROFILE TABLE
        # ----------------------------------------------------

        with st.expander(
            "📋 View predicted temperature values"
        ):

            profile_df = pd.DataFrame(
                {
                    "Depth (m)": profile_depths.astype(
                        int
                    ),
                    "Temperature (°C)": np.round(
                        profile_temperature,
                        2,
                    ),
                }
            )

            st.dataframe(
                profile_df,
                use_container_width=True,
                hide_index=True,
            )


    # ========================================================
    # LIVE INPUT DATA SUMMARY
    # ========================================================

    st.subheader(
        "🛰️ Live Input Data Summary"
    )

    input_arrays = {
        "SST": st.session_state["sst"],
        "SSS": st.session_state["sss"],
        "SLA": st.session_state["sla"],
        "U": st.session_state["u"],
        "V": st.session_state["v"],
    }

    summary_rows = []

    for name, arr in input_arrays.items():

        finite_values = arr[
            np.isfinite(arr)
        ]

        if len(finite_values) == 0:

            summary_rows.append(
                {
                    "Feature": name,
                    "Unit": "",
                    "Minimum": np.nan,
                    "Maximum": np.nan,
                    "Mean": np.nan,
                    "Valid Cells": 0,
                }
            )

            continue

        units = {
            "SST": "°C",
            "SSS": "PSU",
            "SLA": "m",
            "U": "m/s",
            "V": "m/s",
        }

        summary_rows.append(
            {
                "Feature": name,
                "Unit": units[name],
                "Minimum": float(
                    np.min(finite_values)
                ),
                "Maximum": float(
                    np.max(finite_values)
                ),
                "Mean": float(
                    np.mean(finite_values)
                ),
                "Valid Cells": int(
                    len(finite_values)
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
    )


    # ========================================================
    # PATCH INFORMATION
    # ========================================================

    st.subheader(
        "🧩 Model Patch Information"
    )

    patch_col1, patch_col2, patch_col3 = (
        st.columns(3)
    )

    with patch_col1:

        st.metric(
            "Valid Patches",
            f"{len(patch_centers):,}",
        )

    with patch_col2:

        st.metric(
            "Minimum Coverage",
            f"{np.min(patch_coverages) * 100:.1f}%",
        )

    with patch_col3:

        st.metric(
            "Mean Coverage",
            f"{np.mean(patch_coverages) * 100:.1f}%",
        )


    # ========================================================
    # ALL DEPTHS SUMMARY
    # ========================================================

    st.subheader(
        "🌊 Predicted Temperature by Depth"
    )

    all_depth_rows = []

    for depth_idx, depth_value in enumerate(
        depths
    ):

        depth_map = temperature_maps[
            depth_idx
        ]

        values = depth_map[
            np.isfinite(depth_map)
        ]

        if len(values) == 0:
            continue

        all_depth_rows.append(
            {
                "Depth (m)": int(
                    depth_value
                ),
                "Minimum (°C)": round(
                    float(
                        np.min(values)
                    ),
                    2,
                ),
                "Maximum (°C)": round(
                    float(
                        np.max(values)
                    ),
                    2,
                ),
                "Mean (°C)": round(
                    float(
                        np.mean(values)
                    ),
                    2,
                ),
            }
        )

    all_depth_df = pd.DataFrame(
        all_depth_rows
    )

    st.dataframe(
        all_depth_df,
        use_container_width=True,
        hide_index=True,
    )


    # ========================================================
    # DOWNLOAD RESULTS
    # ========================================================

    st.subheader(
        "📥 Export Prediction"
    )

    prediction_csv = create_prediction_csv(
        temperature_maps
    )

    csv_bytes = prediction_csv.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Prediction CSV",
        data=csv_bytes,
        file_name=(
            f"OceanEmbed_prediction_"
            f"{result_date.strftime('%Y-%m-%d')}.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )


else:

    # ========================================================
    # INITIAL SCREEN
    # ========================================================

    st.subheader(
        "🚀 Ready for Live Prediction"
    )

    st.markdown(
        """
Select a date and depth from the sidebar, then click
**Run OceanEmbed**.

The application will:

1. Connect to Copernicus Marine.
2. Download SST.
3. Download SSS.
4. Download SLA.
5. Download surface U/V currents.
6. Convert the observations to the training grid.
7. Apply the training ocean mask.
8. Normalize the five input features.
9. Extract valid 32×32 patches.
10. Run the trained CNN.
11. Predict temperature at all 15 depths.
12. Reconstruct the full North Indian Ocean maps.
13. Display the selected depth.
14. Generate a temperature profile for a selected location.

The downloaded NetCDF files are stored only in a temporary
directory and are automatically deleted after processing.
"""
    )

    st.warning(
        "Select a date and click '🚀 Run OceanEmbed' "
        "to start live inference."
    )