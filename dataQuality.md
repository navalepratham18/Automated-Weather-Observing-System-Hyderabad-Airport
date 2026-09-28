# Data Quality Report: `data/02_interim`

**Report date:** 2026-09-24  
**Source:** `data/01_raw`  
**Interim data:** `data/02_interim`

## Executive summary

The interim dataset contains **12,127 CSV files** generated from **13,315 raw `.his` files**. This gives a file-level coverage of **91.08%**. A total of **1,188 raw files do not currently have a corresponding CSV output**.

The missing files are not empty or unreadable based on the configured parser: all 1,188 tested missing files were successfully parsed with the ingestion settings. They should therefore be reprocessed rather than discarded.

## File inventory

| Metric | Result |
|---|---:|
| Raw `.his` files | 13,315 |
| Interim `.csv` files | 12,127 |
| Missing interim outputs | 1,188 |
| File coverage | 91.08% |
| Empty raw files | 0 |
| Empty interim CSV files | 0 |
| Raw data size | 6,359,599,380 bytes |
| Interim data size | 1,427,490,580 bytes |

## Content profile

| Metric | Result |
|---|---:|
| Total data rows | 44,976,651 |
| Minimum rows per CSV | 1 |
| Maximum rows per CSV | 7,080 |
| Median rows per CSV | 5,760 |
| Mean rows per CSV | 3,708.80 |
| Distinct schemas | 7 |
| Distinct column names | 19 |
| Files without a header | 0 |
| Files with duplicate header names | 0 |
| Invalid datetime values | 0 |
| Invalid numeric values | 0 |

### Schema distribution

| Columns | Files | Column names |
|---:|---:|---|
| 1 | 4,165 | `Datetime` |
| 4 | 2,217 | `Datetime`, `Wind_Speed`, `Wind_Dir`, `Wind_Gust` |
| 3 | 1,514 | `Datetime`, `RVR`, `Visibility_MOR` |
| 5 | 1,541 | `Datetime`, `Pressure_QNH`, `Temperature`, `Humidity`, `DewPoint` |
| 4 | 1,541 | `Datetime`, `Present_Weather_Code`, `Rain_Intensity`, `Rain_Sum` |
| 7 | 784 | `Datetime`, `Cloud_Base_1`, `Cloud_Base_2`, `Cloud_Base_3`, `OCTA1`, `OCTA2`, `OCTA3` |
| 7 | 365 | `Datetime`, `Cloud_Base_1`, `OCTA1`, `Cloud_Base_2`, `OCTA2`, `Cloud_Base_3`, `OCTA3` |

The schema table uses the number of actual fields, so the wind, visibility,
PTU, and present-weather groups include their `Datetime` field in the total.

### Field-level statistics

Statistics below are calculated across all non-empty values in the interim
CSV files. `Present_Weather_Code` is categorical and is therefore reported
only by completeness.

| Column | Files | Non-empty values | Missing values | Minimum | Maximum | Mean |
|---|---:|---:|---:|---:|---:|---:|
| `Datetime` | 12,127 | 44,976,651 | 0 | — | — | — |
| `Wind_Speed` | 2,217 | 10,103,143 | 2,572,493 | 0.00 | 32.63 | 7.4476 |
| `Wind_Dir` | 2,217 | 10,103,143 | 2,572,493 | 0.00 | 360.00 | 188.6343 |
| `Wind_Gust` | 2,217 | 10,103,094 | 2,572,542 | 0.00 | 43.35 | 10.9096 |
| `Pressure_QNH` | 1,541 | 1,938,526 | 260,376 | 1001.46 | 1023.22 | 1012.4645 |
| `Temperature` | 1,541 | 1,929,402 | 269,500 | 8.40 | 38.90 | 25.0479 |
| `Humidity` | 1,541 | 1,929,402 | 269,500 | 0.00 | 100.00 | 67.7488 |
| `DewPoint` | 1,541 | 1,929,401 | 269,501 | -36.68 | 25.38 | 17.7604 |
| `Present_Weather_Code` | 1,541 | 7,609,972 | 1,186,115 | categorical | categorical | — |
| `Rain_Intensity` | 1,541 | 7,609,971 | 1,186,116 | 0.00 | 154.58 | 0.0670 |
| `Rain_Sum` | 1,541 | 7,609,970 | 1,186,117 | 0.00 | 99.99 | 21.4257 |
| `RVR` | 1,514 | 7,589,000 | 1,059,279 | 100.00 | 2100.00 | 2096.7632 |
| `Visibility_MOR` | 1,514 | 7,589,052 | 1,059,227 | 0.00 | 10000.00 | 8533.6390 |
| `Cloud_Base_1` | 1,149 | 3,027,582 | 3,532,531 | 0.00 | 25000.00 | 5810.1113 |
| `Cloud_Base_2` | 1,149 | 1,417,501 | 5,142,612 | 200.00 | 25000.00 | 7877.8857 |
| `Cloud_Base_3` | 1,149 | 464,770 | 6,095,343 | 600.00 | 25000.00 | 9650.1721 |
| `OCTA1` | 1,149 | 3,027,582 | 3,532,531 | 1.00 | 9.00 | 3.6144 |
| `OCTA2` | 1,149 | 1,417,501 | 5,142,612 | 3.00 | 8.00 | 6.1140 |
| `OCTA3` | 1,149 | 464,770 | 6,095,343 | 5.00 | 8.00 | 7.3279 |

Missing values are concentrated in optional sensor fields, especially the
second and third cloud layers. They should not automatically be treated as
zero; downstream processing should preserve them as missing observations.

## Distribution by year and month

| Year | Month | Interim CSV files |
|---|---|---:|
| 2024 | Dec | 1,515 |
| 2025 | Feb | 31 |
| 2025 | Mar | 961 |
| 2025 | Apr | 930 |
| 2025 | May | 961 |
| 2025 | Jun | 930 |
| 2025 | Jul | 964 |
| 2025 | Aug | 973 |
| 2025 | Sep | 952 |
| 2025 | Oct | 992 |
| 2025 | Nov | 959 |
| 2025 | Dec | 974 |
| 2026 | Jan | 985 |
| **Total** |  | **12,127** |

## Missing-output analysis

The 1,188 missing outputs are distributed as follows:

| Source folder | Missing files |
|---|---:|
| `2024/Dec` | 33 |
| `2025/Feb` | 1 |
| `2025/Mar` | 102 |
| `2025/Apr` | 145 |
| `2025/May` | 130 |
| `2025/Jun` | 78 |
| `2025/Jul` | 61 |
| `2025/Aug` | 81 |
| `2025/Sep` | 142 |
| `2025/Oct` | 125 |
| `2025/Nov` | 117 |
| `2025/Dec` | 70 |
| `2026/Jan` | 103 |
| **Total** | **1,188** |

Each missing source file was checked using the configured ingestion settings:

- Delimiter: tab (`\t`)
- Header rows skipped: 1
- Bad lines: error

All 1,188 missing files parsed successfully during this check. This indicates incomplete processing or missing output generation, not a parser-level data-format failure.

## Quality checks

### Passed

- The interim data is organized by source year and month.
- No zero-byte raw `.his` files were found.
- No zero-byte interim `.csv` files were found.
- The missing raw files are parseable with the current ingestion configuration.
- The output path design preserves duplicate filenames from different months and years.

### Outstanding

- Reprocess the 1,188 missing raw files.
- Add an ingestion summary that reports discovered, converted, skipped, and failed files.
- Add a post-run coverage check that compares every raw relative path with its expected interim CSV path.
- Validate row counts and column schemas between raw inputs and generated CSVs.
- Add content checksums if duplicate-content detection is required. Identical filenames alone should not be treated as duplicates because monthly source files can have different contents.

## Recommended acceptance criteria

The interim dataset should be considered complete when:

1. The number of CSV outputs equals the number of raw `.his` inputs.
2. Every raw relative path has a matching CSV relative path under `data/02_interim`.
3. No conversion errors are reported.
4. No output CSV is empty.
5. Row-count and schema checks pass for all converted files.
