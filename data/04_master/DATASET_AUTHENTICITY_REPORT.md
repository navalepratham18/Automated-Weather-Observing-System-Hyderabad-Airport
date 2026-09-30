# Master Dataset Authenticity and Usability Report

**Scope:** `data/04_master/*`  
**Assessment date:** 2026-09-29  
**Compared files:**

- `MASTER_DATASET.csv` — merged master, closest to the values available in the
  processed/raw `.his` inputs.
- `MASTER_DATASET_IMPUTED.csv` — output of `src/imputation.py`.
- `MASTER_DATASET_CLEANED.csv` — output of `src/clean_data.py`, built from the
  imputed file.

## Executive conclusion

### Which file contains the most authentic data?

**`MASTER_DATASET.csv` is the most authentic file.** It preserves the observed
values and the original missingness pattern. It should be the source of truth
for auditing, scientific interpretation, validation, and any claim about what
the weather station actually recorded.

However, it is **not the best standalone training matrix** because the entire
available December 2024 segment is empty for the measured weather variables.
That gap is not repaired by the merge; it is present in the raw master output
itself.

### Which file is most practical for model training?

**`MASTER_DATASET_CLEANED.csv` is more convenient for algorithms**, because most
numeric columns have no blank values. It is not more authentic: it contains
synthetic values created by interpolation, forward filling, rolling medians,
global medians, zero defaults, and the fixed cloud-base value `25000`.

The recommended practice is therefore:

1. Keep `MASTER_DATASET.csv` as the immutable provenance/ground-truth layer.
2. Use a derived modeling dataset only after adding explicit imputation masks
   and documenting every transformation.
3. Do not evaluate predictions against imputed target values as though they
   were observations.

## Dataset inventory and structural checks

| File | Rows | Columns | Datetime range | Duplicate timestamps |
|---|---:|---:|---|---:|
| `MASTER_DATASET.csv` | 521,735 | 37 | 2024-12-02 17:23 to 2026-01-31 06:16 | 0 |
| `MASTER_DATASET_IMPUTED.csv` | 521,735 | 39 | Same | 0 |
| `MASTER_DATASET_CLEANED.csv` | 521,735 | 39 | Same | 0 |

The row count and time range are preserved across all three files. The
additional columns in the derived files are the wind-direction sine/cosine
features and `Wind_Dir_Imputed`; the cleaned file also removes the original
`Wind_Dir` column.

The master is not a continuous full-calendar record. The observed timeline
contains December 2024, then starts again in February 2025, and ends in
January 2026. Missing calendar days/months should not be silently interpreted
as normal station readings.

The timestamp sequence is also not perfectly continuous at one-minute
resolution. Audit results identify 14 two-minute gaps, 7 three-minute gaps,
one four-minute gap, gaps of approximately 163 and 298 minutes, and one gap
of approximately **89,110 minutes (61.9 days)**. The derived files preserve
these gaps; filling cells does not recreate missing timestamps.

## Authenticity and missingness findings

### Raw master (`MASTER_DATASET.csv`)

Overall blank rates in the raw master are:

| Variable group | Blank rate |
|---|---:|
| Cloud base 1 / OCTA1 | 43.31% / 43.50% |
| Cloud base 2 / OCTA2 | 73.13% / 73.69% |
| Cloud base 3 / OCTA3 | 90.07% / 90.71% |
| Pressure, temperature, humidity, dew point | 7.04–7.11% |
| Present weather and rain fields | 7.87% |
| RVR and visibility | 8.04% |
| Wind speed, direction, gust | about 69.17% |

The missingness is variable-specific and therefore informative. The higher
missingness in cloud layers 2/3 and wind fields may reflect sensor availability
or source-file coverage rather than random noise.

Counting empty and whitespace-only values as missing gives approximately
**3.6 million missing measurement cells** in the raw master. This missingness
should be retained as a property of the data-generating process, not hidden by
the number of populated cells in a derived file.

### December 2024 outage

For the 36,637 December 2024 rows in the raw master, all 18 measured weather
value columns are blank (100% missing). This is consistent with the user's
observation and should be treated as a sensor/data-availability outage unless
the source files demonstrate otherwise.

The pipeline's existing quality report also records incomplete conversion
coverage: 12,127 interim CSV outputs from 13,315 raw `.his` files (91.08%).
Before concluding that December is irrecoverable, compare every raw `.his` file
with its expected interim CSV and re-run the missing conversions. A missing
interim output is a processing defect, whereas a successfully converted file
whose measurements are blank is a source-data outage.

## What the derived files do correctly

The derived pipeline has several sound ideas:

- It sorts records chronologically before time-based operations.
- It preserves `is_observed_*` columns rather than deleting provenance fields.
- It treats wind direction as a circular quantity by creating sine/cosine
  components before interpolation.
- It limits the first interpolation pass to 15 rows/minutes.
- It uses domain defaults for rain (`0`) and an aviation clear-sky ceiling
  (`25000`) instead of arbitrary constants.
- It keeps the raw master separate from the modeling-oriented output.

These choices can make a model train successfully without changing the raw
record.

## What may be wrong or misleading

### 1. Imputed values can be mistaken for observations

`MASTER_DATASET_CLEANED.csv` has zero blanks in most numeric columns, but this
means missing records have been replaced, not recovered. In particular:

- Cloud bases are filled with `25000`, which encodes “clear/no cloud detected”
  and is not equivalent to a measured 25,000-foot cloud base.
- Rain fields are filled with zero. A missing rain sensor report is not always
  proof that rainfall was zero.
- Short continuous gaps are linearly interpolated.
- Longer gaps are forward-filled for up to three hours, then replaced by a
  global median.
- Wind values are zero-filled, which is indistinguishable from calm wind
  unless the observation mask is used.
- Categorical fields are forward-filled and then defaulted to zero.

The cleaned file is therefore a **modeling representation**, not an authentic
station record.

### 2. Observation flags are not fully trustworthy

`src/merging.py` creates flags with `merged_df[col].notna()`. The source rows
contain whitespace strings such as `" "`; pandas can retain these as non-null
strings. Consequently, a field can be visually blank while its
`is_observed_*` flag is `1`. The first rows of the December master demonstrate
this pattern for OCTA fields.

Before creating flags, normalize whitespace and known missing tokens, for
example by stripping strings and converting empty strings to `NaN`. Rebuild the
raw master after this fix; do not patch the flags only in the cleaned file.

### 3. The cleaned file drops raw wind direction

Dropping `Wind_Dir` after deriving sine/cosine is reasonable for a neural
network, but it removes an easy audit field from the cleaned artifact. Retain
the raw direction and add a separate `Wind_Dir_Is_Imputed` or equivalent mask
if human review is important.

### 4. Cleaning can introduce distribution distortion

IQR capping changes extreme weather events. That can suppress exactly the
visibility, wind, rain, or temperature events that are operationally
important. The fixed clear-sky value can also create a large artificial spike
in cloud-base distributions.

For aviation/weather use, flag outliers first and retain the original value.
Only cap or remove values in a separate feature view, with before/after counts.

### 5. Some aggregation rules need domain confirmation

In `src/merging.py`:

- `Wind_Dir: last` avoids invalid arithmetic averaging, but a circular mean or
  source-specific report selection may be more appropriate.
- `Rain_Sum: max` is only correct if the source field is a resettable
  within-window accumulation and the maximum represents the desired daily
  value. If it is a running cumulative counter, `last` or a validated
  difference is usually more meaningful.
- `Wind_Gust: max` and `Rain_Intensity: max` are plausible for an interval but
  should be verified against the source sampling/reporting semantics.

### 6. Row-based limits are unsafe across timestamp gaps

The imputation code uses `limit=15` and the cleaning code uses a 30-row
rolling window or `limit=180`. These are only equivalent to 15 minutes,
30 minutes, and three hours when every row is exactly one minute apart.
Because the master has irregular intervals and a 61.9-day gap, these should
be changed to time-aware windows or applied only within contiguous segments.

### 7. Potential leakage in model preparation

`src/train_dl.py` fills values and fits scalers before the chronological
train/validation/test split. It also derives global quantiles for target
clipping before the split. This allows future-period information to influence
earlier training data.

Fit imputers, outlier thresholds, and scalers on the training period only.
Keep validation and test targets restricted to genuinely observed target rows.

## Recommended data products

| Purpose | Recommended input | Rules |
|---|---|---|
| Provenance, audit, reporting | `MASTER_DATASET.csv` | Never overwrite; preserve raw values and corrected observation masks |
| Exploratory plots | Raw master plus explicit observed masks | Plot imputed and observed values differently |
| Training with engineered features | A new versioned derived file | Add per-column imputation indicators and retain raw direction |
| Honest forecasting evaluation | Raw master | Score only on observed target values; exclude outage periods |
| Operational deployment | Cleaned/engineered file | Declare defaults, monitor missingness, and preserve an imputation log |

## Improvements with highest value

1. **Reconcile raw-to-interim coverage.** Reprocess the 1,188 raw files with
   missing interim outputs and generate a machine-readable coverage report.
2. **Normalize missing tokens before flagging.** Strip whitespace, map `""`,
   `"NA"`, `"null"`, and similar tokens to `NaN`, then recompute all
   `is_observed_*` fields.
3. **Separate observation from fill.** For every value column, add
   `is_imputed_<column>` and ideally `imputation_method_<column>`.
4. **Use training-only fitting.** Learn medians, IQR bounds, interpolation
   policies, and scalers from the training segment only.
5. **Preserve target authenticity.** Do not train/evaluate a target on values
   filled with `25000`, a median, or a forward fill without clearly labeling
   those rows.
6. **Validate physical constraints.** Check humidity bounds, wind direction
   range, non-negative rain, dew-point/temperature consistency, and cloud-layer
   ordering. Report violations rather than silently clipping them.
7. **Version transformations.** Store the source filename, pipeline version,
   transformation method, and timestamp for each generated dataset.
8. **Use gap-aware splits.** Split by contiguous dates or complete operational
   periods, not only by row count, and exclude the December outage from
   ordinary performance claims.

## Final recommendation

Use **`MASTER_DATASET.csv` as the authentic reference dataset** and
**`MASTER_DATASET_CLEANED.csv` only as a convenience/modeling derivative**.
Do not choose the cleaned file because it has fewer blanks; that metric mainly
measures how aggressively it replaces missing data. The strongest workflow is
to rebuild the raw observation masks, recover any missing `.his` conversions,
create a new feature dataset with explicit imputation indicators, and evaluate
models only against observed targets.
