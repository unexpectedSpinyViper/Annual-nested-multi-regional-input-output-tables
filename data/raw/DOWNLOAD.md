# Third-party input data — how to obtain it

These datasets are **not redistributed** in this repository (size and third-party
licensing). Download the exact versions below and place them at the indicated paths,
then run `run_all.sh`. After downloading, record a checksum of each file
(`sha256sum <file>`) so the exact inputs can be cited in the data descriptor.

The small, hand-curated correspondence inputs **are** shipped, under
[`correspondence/`](correspondence/) — do not re-download those.

| Dataset | Version used | Place at | Source |
|---|---|---|---|
| WiNDC national + state accounts | v4.1, file dated 2025-03-17 | `data/raw/GTAPWiNDC/data/core/WiNDCdatabase.gdx` (+ `windc_base.gdx`) | https://old.windc.wisc.edu/downloads/version_4_1/core/ |
| OECD inter-country IO tables (SML) | 2025 edition (archives built 2026-01-12) | `data/interim/IOT/OCDE ICIO/<range>_SML/<year>_SML.csv` | https://oe.cd/icio |
| BEA regional accounts, table SAGDP2 | last updated 2026-04-09 | `data/raw/BEA/SAGDP/SAGDP2__ALL_AREAS_1997_2025.csv` | https://apps.bea.gov/regional |
| BEA regional accounts, table **CAGDP2** (county GDP) | latest available | `data/raw/BEA/CAGDP2.zip` | https://apps.bea.gov/regional/zip/CAGDP2.zip |
| Census 2020 **county population centroids** | 2020 Census | `data/raw/census/CenPop2020_Mean_CO.txt` | https://www2.census.gov/geo/docs/reference/cenpop2020/county/CenPop2020_Mean_CO.txt |
| Census **Gazetteer 2023**, counties | 2023 vintage | `data/raw/census/2023_Gaz_counties_national.zip` | https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_counties_national.zip |
| BEA regional accounts, table **SAPCE1** (PCE by state) | 1997–2024, file dated 2025-09-15 | `data/raw/BEA/SAPCE/SAPCE1__ALL_AREAS_1997_2024.csv` | https://apps.bea.gov/regional/zip/SAPCE.zip |
| **Census of Governments** 2017, finance summary table 1 | 2017 Census of Governments | `data/raw/Census_gov_finances/17slsstab1a.xlsx` and `…1b.xlsx` | https://www2.census.gov/programs-surveys/gov-finances/tables/2017/summary-tables/ |
| **Commodity Flow Survey** 2017, public-use microdata | 2017 PUF | `data/raw/CFS/CFS 2017 PUF CSV.csv` | https://www.census.gov/data/datasets/2017/econ/cfs/historical-datasets.html |
| **Freight Analysis Framework**, state O-D | FAF5.7.1, 2017 base year | `data/raw/FAF5/FAF5.7.1_State.csv` | https://www.bts.gov/faf |

Notes:

- **The last four are benchmarks, not inputs.** SAPCE1 and the Census of Governments tables
  are used only by `validation/final_demand_allocator.ipynb`, to score the candidate
  final-demand allocators against an external measurement; the Commodity Flow Survey and
  the Freight Analysis Framework only by `validation/cfs_faf_validation.ipynb` and
  `plots/gamma_vs_observations.py`, to confront the reconstructed inter-state goods
  flows with an independent observation. None of them enters the delivered tables, and the
  pipeline runs without them.
- The BEA zip endpoint is per-table-family, not per-table: `SAPCE.zip` contains SAPCE1 to
  SAPCE5, and `SAPCE1.zip` returns an HTML error page rather than a 404.
- The Census summary table is split alphabetically over two files, `1a` (Alabama to
  Mississippi) and `1b` (the rest); both are needed for the 51 regions.

- **WiNDC → GAMS.** Reading the `.gdx` requires a licensed GAMS install (the Python
  `gamsapi`/`gamspy_base` packages are only bindings). *Recommended*: add a one-off
  export of the needed WiNDC parameters to CSV/parquet and deposit that export with the
  dataset, so readers without a GAMS licence can still reproduce the chain.
- **OECD SML CSVs** are dropped under `data/interim/IOT/OCDE ICIO/`; step
  `01_convert_oecd.ipynb` converts them to parquet in place.
- **Economic centroids.** The 51 reference points of the gravity distance matrix are the
  **GDP-weighted economic centroids** of each region, built by `02_economic_centroids.ipynb`
  from the last three inputs (CAGDP2 county GDP + the 2020 county population centroids;
  the 2023 Gazetteer supplies anchors for the Connecticut planning regions). This replaces
  the earlier state-capital reference points, which were hard-coded and needed no download.
  In the data descriptor, CAGDP2 and the county centroids move from *diagnostic-only* to
  *inputs of the delivered series*.
