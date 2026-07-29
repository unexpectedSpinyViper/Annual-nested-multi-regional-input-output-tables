#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reproduce the delivered series from raw inputs to the 26 nested-MRIOT files.
#
# Prerequisites:
#   1. Environment installed:   pip install -r requirements.txt && pip install -e .
#   2. Third-party inputs downloaded into data/  (see data/raw/DOWNLOAD.md)
#   3. A licensed GAMS install (needed to read the WiNDC .gdx)
#
# The intra-US construction needs about 64 GB; on a cluster run
# `sbatch pipeline/run_series.sbatch` instead, which is stages 1-3 of this script
# on a compute node.
#
# Executed notebooks are written to _executed/ (git-ignored); the versioned
# source notebooks under pipeline/ keep their outputs stripped.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/interim data/final _executed

# The notebooks do `from paths import ROOT`, which prerequisite 1 provides. Export it
# as well so the chain also runs from a plain checkout, without the editable install.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

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

# 1 — sub-national accounts: densification and identity verification
run pipeline/10_windc_national.ipynb

# 2 — intra-US bilateral reconstruction, 26 years
#     Two-layer (trade + margin) doubly constrained gravity on the national commodity
#     pools, RAS, column calibration onto industry output, OECD/SNA value-added
#     convention. This is the step that produces grav_fric_v3.1_RAS, the input of 20.
#     `11_build_intraus_v3` and `12_pipeline_allyears` document the construction and the
#     variants that were screened, but they write the earlier v1.2 schema and are not
#     part of the delivered chain -- run them for the exposition, not for the series.
python -u pipeline/build_series.py

# 3 — nesting into the OECD ICIO world table, and closure of the state cost accounts
run pipeline/20_aggregation.ipynb
run pipeline/21_harmonization.ipynb
run pipeline/22_nesting.ipynb

# 4 — collect the 26 delivered files for deposit
cp data/interim/IOT/nested_mriot_v3.1_RAS/nested_mriot_*.parquet data/final/
echo
echo "Done. Delivered series in data/final/ ($(ls data/final/nested_mriot_*.parquet 2>/dev/null | wc -l) files)."
