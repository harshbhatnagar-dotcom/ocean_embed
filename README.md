# OceanEmbed

## Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature

**Smart India Hackathon 2026 --- Problem Statement SIH26066**

OceanEmbed is a deep-learning framework for reconstructing **subsurface
ocean temperature profiles from surface satellite observations**. It
combines multiple surface-ocean variables with a custom **2D
Convolutional Neural Network (CNN)** to estimate temperature at 15
depths from the surface to 1000 m.

The current prototype focuses on the **North Indian Ocean**, including
the Arabian Sea and Bay of Bengal.

------------------------------------------------------------------------

## Problem Statement

Direct subsurface ocean-temperature measurements are spatially sparse
because they rely on in-situ observing systems such as ARGO profiling
floats, moored buoys, gliders, and ship observations.

Satellite observations provide broad and frequent coverage of surface
conditions. Surface variables contain signatures of subsurface processes
through circulation, stratification, thermocline displacement, mesoscale
activity, and related physical processes.

OceanEmbed learns relationships between surface observations and
subsurface temperature structure to provide spatially continuous
temperature reconstructions.

## Objectives

-   Reconstruct subsurface ocean temperature from surface satellite
    observations.
-   Predict temperature at multiple depths using one deep-learning
    model.
-   Provide spatially continuous estimates that complement sparse
    in-situ measurements.
-   Validate predictions against independent real ARGO observations.
-   Provide interactive temperature maps and vertical-profile
    visualization.

## Study Region

  Parameter             Value
  --------------------- ---------------------------------
  Region                North Indian Ocean
  Latitude              5°N -- 30°N
  Longitude             45°E -- 105°E
  Spatial Resolution    0.25° × 0.25°
  Grid Size             100 × 240
  Temporal Resolution   Daily
  Training Period       1 January 2025 -- 31 March 2026
  Number of Days        455

## Input Variables

OceanEmbed uses five surface channels:

  Channel   Variable   Description
  --------- ---------- -----------------------------
  1         SST        Sea Surface Temperature
  2         SSS        Sea Surface Salinity
  3         SLA        Sea Level Anomaly
  4         U          East-West Surface Current
  5         V          North-South Surface Current

## Data Sources

Surface-ocean observations are obtained from **Copernicus Marine**
products.

**Training target:** GLORYS Global Ocean Reanalysis subsurface
temperature (`thetao`).

**Independent validation:** Real ARGO profiling-float temperature
observations. ARGO is not used as the training target.

## Target Depths

The model predicts temperature at 15 depths:

`0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m`

Each valid input patch therefore produces a 15-value vertical
temperature profile.

## Data Preprocessing

``` text
Surface Satellite Observations
            ↓
Spatial Subsetting
            ↓
Regridding to 0.25° × 0.25°
            ↓
Common Ocean Mask
            ↓
Validity / Coverage Checking
            ↓
Channel-wise Normalization
            ↓
32 × 32 × 5 Spatial Patches
            ↓
CNN
```

### Ocean Mask

A common ocean mask identifies valid ocean grid cells and prevents land
regions from being treated as ocean observations.

### Normalization

Each surface variable is standardized independently using training
statistics:

`normalized value = (value - mean) / standard deviation`

### Patch Generation

Each model input has shape **32 × 32 × 5**. At 0.25° resolution, this
corresponds to approximately an **8° × 8°** spatial region.

## Model Architecture

OceanEmbed uses a **custom lightweight 2D CNN**, not a pretrained
ResNet/VGG-style model.

``` text
Input: 32 × 32 × 5
        ↓
Conv2D — 32 filters
        ↓
Batch Normalization
        ↓
MaxPooling2D
        ↓
Conv2D — 64 filters
        ↓
Batch Normalization
        ↓
MaxPooling2D
        ↓
Conv2D — 128 filters
        ↓
Batch Normalization
        ↓
Global Average Pooling
        ↓
Dense — 128 neurons
        ↓
Dropout — 0.2
        ↓
Dense — 15 outputs
        ↓
Temperature Profile: 0–1000 m
```

The network has **three convolutional layers** and **12 processing
layers** when convolution, normalization, pooling, global pooling,
dense, and dropout operations are counted individually.

### Training Configuration

  Parameter           Value
  ------------------- ---------------------
  Optimizer           Adam
  Learning Rate       0.001
  Loss Function       Mean Squared Error
  Monitoring Metric   Mean Absolute Error
  Output Units        15
  Output Activation   Linear

## Why CNN?

Satellite observations are spatial grid data. A CNN can learn local
spatial patterns and relationships among neighboring ocean regions.
OceanEmbed therefore uses surrounding spatial context rather than only
the surface values at one point.

## Application Workflow

``` text
User Selects Date
        ↓
Acquire SST + SSS + SLA + U + V
        ↓
Regrid and Preprocess
        ↓
Ocean Mask
        ↓
Normalize
        ↓
Create 32 × 32 Patches
        ↓
OceanEmbed CNN
        ↓
Predict 15 Depth Temperatures
        ↓
Spatial Reconstruction
        ↓
Temperature Map + Vertical Profile
```

## GLORYS Test Results

  Metric     GLORYS Held-Out Test
  -------- ----------------------
  MAE                   0.5252 °C
  RMSE                  0.7217 °C
  Bias                 +0.0829 °C
  R²                       0.9897

## Independent ARGO Validation

Real ARGO observations were processed using quality-controlled pressure
and temperature measurements. Pressure was converted to depth and
observations were interpolated to the model target depths. Profiles were
then spatially matched with valid model centers using a maximum matching
distance of **0.75°**.

  Item                                     Value
  -------------------------------------- -------
  Profiles identified in region/period     5,538
  Processed valid profiles                   411
  Spatially matched profiles                 300
  Unique matched satellite dates             221
  Valid prediction-observation pairs       3,755

### ARGO Results

  Metric     Independent ARGO Observations
  -------- -------------------------------
  MAE                            0.5447 °C
  RMSE                           0.8135 °C
  Bias                          +0.2115 °C
  R²                                0.9860

ARGO was not used as the training target. Because these observations
fall within the model's training calendar period, this is **independent
observational validation**, not a temporal holdout test.

## GLORYS vs ARGO

  Validation Source                 MAE (°C)   RMSE (°C)   Bias (°C)       R²
  ------------------------------- ---------- ----------- ----------- --------
  GLORYS Held-Out Test                0.5252      0.7217     +0.0829   0.9897
  Independent ARGO Observations       0.5447      0.8135     +0.2115   0.9860

## Depth-Wise ARGO Validation

    Depth (m)     N      MAE     RMSE      Bias       R²
  ----------- ----- -------- -------- --------- --------
            0     4   0.2577   0.2693   -0.1871   0.9147
            5   288   0.2887   0.3890   -0.0699   0.9189
           10   289   0.2652   0.3644   -0.0186   0.9273
           20   290   0.3002   0.4432   +0.0373   0.8941
           30   294   0.3893   0.5833   +0.1351   0.8265
           50   294   0.5919   0.8309   +0.2944   0.7269
           75   294   0.9064   1.2148   +0.4321   0.5825
          100   294   1.0858   1.3708   +0.5248   0.5251
          125   294   1.0559   1.3306   +0.6083   0.4645
          150   292   0.7986   1.0061   +0.4210   0.6321
          200   284   0.5529   0.6928   +0.2301   0.8209
          300   280   0.3081   0.4128   +0.0463   0.9084
          500   279   0.2471   0.3206   +0.0262   0.9298
          700   279   0.2417   0.3134   +0.0520   0.9328
         1000     0      ---      ---       ---      ---

No ARGO accuracy metric is reported at 1000 m because the selected
validation profiles did not provide valid observations at that target
depth.

## Application Features

The Streamlit interface is designed to:

-   Select an observation date.
-   Acquire required surface-ocean data.
-   Generate subsurface temperature reconstruction dynamically.
-   Select a depth and visualize its temperature field.
-   Inspect vertical temperature profiles at selected locations.

## Date Selection and Generalization

The CNN is not inherently restricted to the dates used for training. If
all required surface observations are available for another date, the
same preprocessing pipeline can provide them to the trained model.

Dates outside the training period are **out-of-training-period
inference** and should ideally be evaluated against independent
observations before accuracy claims are made.

## Technology Stack

-   Python
-   TensorFlow / Keras
-   2D Convolutional Neural Network
-   NumPy
-   Pandas
-   Xarray
-   NetCDF4
-   SciPy
-   Matplotlib
-   Plotly
-   Streamlit
-   Copernicus Marine
-   GLORYS
-   ARGO

## Key Innovation

OceanEmbed combines multiple surface-ocean observations with spatial
deep learning to reconstruct a multi-depth subsurface temperature
profile. It is intended to **complement, not replace, direct observing
systems such as ARGO**.

## Advantages

-   Uses broadly available surface-ocean observations.
-   Predicts 15 depth temperatures simultaneously.
-   Captures spatial context through 2D CNN patches.
-   Provides spatially continuous reconstruction.
-   Supports rapid learned inference.
-   Evaluated against real ARGO observations.
-   Provides interactive scientific visualization.

## Current Limitations

-   Current model is region-specific to the North Indian Ocean.
-   Predictions depend on availability and quality of required surface
    observations.
-   GLORYS is the training reference, so the model learns relationships
    represented by that reanalysis.
-   Errors are larger in parts of the thermocline region.
-   Current ARGO validation uses individual profiles interpolated to
    target depths, not an official gridded ARGO product.
-   ARGO validation dates overlap the training calendar period, so this
    is not a temporal holdout.
-   No ARGO metric is available at 1000 m for the selected profiles.
-   The current model performs reconstruction rather than long-range
    forecasting.

## Future Scope

-   Extend the training period.
-   Further train and tune the model.
-   Add surface wind information.
-   Explore CNN-LSTM, ConvLSTM, attention, and Transformer
    architectures.
-   Improve thermocline-region reconstruction.
-   Expand independent ARGO validation.
-   Evaluate fully unseen temporal periods.
-   Extend to other ocean basins.
-   Predict additional subsurface variables such as salinity.
-   Improve near-real-time processing.
-   Add uncertainty quantification.

## Scientific Distinction

OceanEmbed performs **subsurface temperature reconstruction**, not
long-range forecasting.

``` text
Observed Surface Ocean State
            ↓
       OceanEmbed
            ↓
Estimated Subsurface Temperature
```

## Summary

OceanEmbed:

-   Uses **5 surface variables**.
-   Processes **32 × 32 × 5** spatial patches.
-   Predicts **15 depths from 0--1000 m**.
-   Operates at **0.25° daily resolution** over the North Indian Ocean.
-   Uses **GLORYS** as its training reference.
-   Uses **real ARGO profiles** for independent observational
    validation.
-   Achieves approximately **0.525 °C MAE** on the held-out GLORYS test
    set.
-   Achieves approximately **0.545 °C MAE** against matched ARGO
    observations.

The goal is a practical spatial reconstruction framework that
complements existing in-situ ocean-observation systems.

------------------------------------------------------------------------

**OceanEmbed**\
*Satellite Embedding-Based Deep Learning Framework for Reconstruction of
Subsurface Ocean Temperature from Surface Satellite Observations*\
**Smart India Hackathon 2026 --- SIH26066**
