# Imputation Report

**Input:** `data/04_master/MASTER_DATASET.csv`  
**Output:** `data/04_master/MASTER_DATASET_IMPUTED.csv`  
**Implementation:** `src/imputation.py`  
**Inspection:** `debug.py`  
**Report date:** 2026-09-24

## Purpose

The master dataset combines weather observations from sensors that do not all
report at the same time. Missing values can therefore represent different
things:

- a physical condition, such as no detected rain or cloud layer;
- a short communication or sensor gap that can reasonably be estimated from
  nearby observations; or
- a longer sensor outage where filling a value would create false data.

The pipeline uses domain-aware rules rather than one global replacement value.
Short gaps are filled only up to 15 rows, which is treated as approximately
15 minutes because the master data is minute-aligned. Longer gaps remain
missing and are reported as sensor outages.

## Input and output structure

The inspected output contains **521,735 rows** and **40 columns**. The
`debug.py` output reports `Datetime` as `object` because it reads the saved
CSV directly. During imputation, `src/imputation.py` parses `Datetime` with
`pd.to_datetime`, sorts the data chronologically, and then writes it back to
CSV.

The output columns include:

- the original weather fields;
- `is_observed_<field>` flags created before imputation;
- circular wind-direction features: `Wind_Dir_sin` and `Wind_Dir_cos`;
- reconstructed `Wind_Dir_Imputed`.

## What is used, when, and where

| Data group | Columns | Method | Condition / limit | Reason |
|---|---|---|---|---|
| Structural cloud state | `Cloud_Base_1`, `Cloud_Base_2`, `Cloud_Base_3` | Fill with `25000` | Every missing value | `25000` feet is used as the project convention for clear/no-clouds detected. |
| Structural precipitation state | `Rain_Intensity`, `Rain_Sum` | Fill with `0.0` | Every missing value | Missing rain fields are treated as no measured rain in this pipeline. |
| Categorical observations | `Present_Weather_Code`, `OCTA1`, `OCTA2`, `OCTA3` | Forward fill | Maximum 15 consecutive rows | The latest categorical observation is carried across a short reporting gap. |
| Wind direction | `Wind_Dir` | Convert degrees to sine/cosine, interpolate vectors | Maximum 15 consecutive rows | Circular directions must not be linearly interpolated as raw degrees across 0°/360°. |
| Continuous sensor values | `Pressure_QNH`, `Temperature`, `Humidity`, `DewPoint`, `Visibility_MOR`, `RVR`, `Wind_Speed`, `Wind_Gust` | Linear interpolation | Maximum 15 consecutive rows | Nearby measurements provide a reasonable estimate for short gaps. |
| Long outages | All remaining fields | No imputation | Gaps longer than 15 rows, or no surrounding values | Avoids inventing long-term sensor readings. |

## Detailed methods

### 1. Chronological preparation

Before any imputation:

1. `Datetime` is converted to pandas datetime values.
2. Rows are sorted by `Datetime`.
3. The index is reset.

This ensures forward filling and interpolation operate in time order.

### 2. Cloud-base imputation

The three cloud-base fields are filled with `25000`:

```text
Cloud_Base_1 -> 25000
Cloud_Base_2 -> 25000
Cloud_Base_3 -> 25000
```

This is a structural/domain rule, not a statistical estimate. It represents
the project's selected upper ceiling for a clear or undetected cloud layer.
It should be reviewed if missing cloud values can also mean a disconnected
ceilometer or an unavailable sensor.

### 3. Rain imputation

`Rain_Intensity` and `Rain_Sum` are filled with `0.0`. This treats missing
precipitation measurements as no rain. The rule is useful for downstream
rain-event analysis, but it should not be used if the source system uses
missing values to indicate a failed rain sensor.

### 4. Categorical forward fill

The following categorical fields use the previous known value for at most 15
rows:

- `Present_Weather_Code`
- `OCTA1`
- `OCTA2`
- `OCTA3`

No value is fabricated for a longer outage. Leading missing values, where no
previous value exists, remain missing.

### 5. Wind direction encoding

Wind direction is measured in degrees and is circular. Direct interpolation
would produce incorrect values at the north boundary; for example, the
shortest path from `359°` to `1°` is not `180°`.

The implementation:

1. Converts `Wind_Dir` to numeric, coercing invalid text to missing.
2. Converts degrees to radians.
3. Creates:
   - `Wind_Dir_sin = sin(direction)`
   - `Wind_Dir_cos = cos(direction)`
4. Linearly interpolates both components for up to 15 rows.
5. Reconstructs a readable angle:
   - `Wind_Dir_Imputed = atan2(sin, cos)` converted to degrees and normalized
     to `[0, 360)`.

The sine and cosine fields are the preferred features for machine-learning
models because they preserve circular continuity. `Wind_Dir_Imputed` is a
human-readable reconstructed value.

### 6. Continuous sensor interpolation

Linear interpolation is applied to:

- `Pressure_QNH`
- `Temperature`
- `Humidity`
- `DewPoint`
- `Visibility_MOR`
- `RVR`
- `Wind_Speed`
- `Wind_Gust`

Only gaps within the 15-row limit are filled. Missing values at the beginning
or end of a series, and gaps beyond the limit, can remain null.

## Observability and provenance

Before imputation, `src/merging.py` creates an
`is_observed_<column>` flag for every measured weather field:

- `1` means the value was present in the merged source data.
- `0` means it was missing before imputation.

These flags are retained in the output so models and analysts can distinguish
measured values from imputed values. They should not be recomputed after
imputation.

## Observed output statistics

The supplied `debug.py` output shows the following non-null counts:

| Column group | Output non-null count | Interpretation |
|---|---:|---|
| `Datetime` | 521,735 | Complete time index |
| Cloud bases and `OCTA1`–`OCTA3` | 521,735 each | Structural cloud fill produced complete fields |
| `Present_Weather_Code` | 521,735 | Categorical fill produced a complete field |
| `Rain_Intensity`, `Rain_Sum` | 521,735 each | Rain defaults produced complete fields |
| `Pressure_QNH` | 484,704 | 37,031 values remain missing after the 15-row limit |
| `Temperature`, `Humidity`, `DewPoint` | 485,092 each | 36,643 values remain missing in each field |
| `RVR`, `Visibility_MOR` | 480,541 each | 41,194 values remain missing in each field |
| `Wind_Speed`, `Wind_Gust` | 160,930 each | Long wind-sensor gaps remain missing |
| `Wind_Dir` | 160,854 | Original/reconstructed raw direction field is not overwritten |
| `Wind_Dir_sin`, `Wind_Dir_cos`, `Wind_Dir_Imputed` | 160,927 each | Direction vectors and reconstructed angle available where direction data supports them |

The remaining nulls are expected under the 15-row protection rule and should
be treated as genuine long-duration gaps unless separately investigated.

## Important assumptions and limitations

1. **Rows are assumed to be minute-aligned.** The code uses `limit=15`, which
   means 15 rows, not a time-duration-aware 15-minute window. If timestamps
   are irregular, the limit does not guarantee 15 actual minutes.
2. **No station/group boundary is used during imputation.** The data is sorted
   globally by `Datetime`; forward fill and interpolation are not grouped by
   runway, sensor, station, or source file. This is safe only if the master
   dataset represents one aligned observation stream per timestamp.
3. **Structural defaults are semantic choices.** Filling missing cloud bases
   with `25000` and missing rain values with `0.0` can hide a sensor outage if
   the source uses missing values for both “none observed” and “not available.”
4. **Observed flags preserve provenance but do not identify the method used.**
   A future enhancement could add method-specific flags such as
   `is_imputed_<column>` or `imputation_method_<column>`.
5. **Wind direction is not replaced in place.** The original `Wind_Dir` remains
   available, while the derived vector and reconstructed fields support
   modeling and interpretation.

## Recommended validation

Before using the output for modeling:

- verify that the master data has one row per intended minute;
- verify whether the 15-row limit corresponds to the actual sampling rate;
- review the structural cloud and rain conventions with domain experts;
- compare imputed values against held-out observed values;
- monitor the remaining-null counts as a sensor-health indicator; and
- preserve the original master file and the observed flags for auditability.
