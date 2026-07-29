#!/bin/bash
# Re-deliver the whole series on a compute node: intra-US build -> aggregation ->
# harmonisation -> nesting + closure -> data/final/.
#
#   sbatch pipeline/run_series.sbatch
#
# This is run_all.sh minus the preprocessing steps (concordance, OECD csv->parquet,
# economic centroids), which are cheap and already done. It is the heavy part: the
# intra-US construction is rebuilt for all 26 years.
#SBATCH --job-name=mriot-series
#SBATCH --account=torch_pr_1108_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.log

set -euo pipefail

export TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/scratch/$USER/.matplotlib}"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate cascading-impacts

cd "${SLURM_SUBMIT_DIR:-$PWD}"
# Under sbatch, $0 is a copy of this script in the SLURM spool, so a path relative
# to it lands outside the repository. Submit from the repository root.
if [ ! -f pyproject.toml ] || [ ! -d pipeline ]; then
  echo "error: submit this script from the repository root (got $PWD)" >&2
  exit 1
fi
mkdir -p _executed data/final

# `paths` lives in src/ and the notebooks import it directly. That normally comes from
# `pip install -e .` (see README); export it here so the job also runs in an environment
# where the editable install was never done.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

run_nb () {
  echo "=============================================================="
  echo "== $1"
  echo "=============================================================="
  jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 \
    --output-dir=_executed "$1"
}

echo "Host: $(hostname) | Mem: ${SLURM_MEM_PER_NODE:-?} MB | CPUs: ${SLURM_CPUS_PER_TASK:-?}"
date

echo "=============================================================="
echo "== stage 2-3: intra-US construction, 26 years"
echo "=============================================================="
python -u pipeline/build_series.py

run_nb pipeline/20_aggregation.ipynb
run_nb pipeline/21_harmonization.ipynb
run_nb pipeline/22_nesting.ipynb

cp data/interim/IOT/nested_mriot_v3.1_RAS/nested_mriot_*.parquet data/final/
echo
echo "Delivered series in data/final/ ($(ls data/final/nested_mriot_*.parquet | wc -l) files)."
date
echo "DONE"
