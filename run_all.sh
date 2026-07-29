#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reproduce the delivered series from raw inputs to the 26 nested-MRIOT files.
#
# Prerequisites:
#   1. Environment installed:   pip install -r requirements.txt && pip install -e .
#   2. Third-party inputs downloaded into data/  (see data/raw/DOWNLOAD.md)
#   3. A licensed GAMS install (needed to read the WiNDC .gdx in steps 10-11)
#
# Executed notebooks are written to _executed/ (git-ignored); the versioned
# source notebooks under pipeline/ keep their outputs stripped.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/interim data/final _executed

run () {
  echo "======================================================================"
  echo "== $1"
  echo "======================================================================"
  jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 \
    --output-dir=_executed "$1"
}

# 0 — preprocessing: WiNDC<->OECD concordance, OECD ICIO csv -> parquet,
#     GDP-weighted economic centroids (reference points of the gravity distance matrix)
run pipeline/00_concordance.ipynb
run pipeline/01_convert_oecd.ipynb
run pipeline/02_economic_centroids.ipynb

# 1 — intra-US bilateral reconstruction (WiNDC two-layer gravity + RAS)
run pipeline/10_windc_national.ipynb
run pipeline/11_build_intraus_v3.ipynb
run pipeline/12_pipeline_allyears.ipynb

# 2 — nesting into the OECD ICIO world table
run pipeline/20_aggregation.ipynb
run pipeline/21_harmonization.ipynb
run pipeline/22_nesting.ipynb

# 3 — collect the 26 delivered files for deposit
cp data/interim/IOT/nested_mriot_v3.1_RAS/nested_mriot_*.parquet data/final/
echo
echo "Done. Delivered series in data/final/ ($(ls data/final/nested_mriot_*.parquet 2>/dev/null | wc -l) files)."
