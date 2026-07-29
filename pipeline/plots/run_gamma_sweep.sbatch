#!/bin/bash
# Recompute the friction-exponent sweep behind gamma_sensitivity_extended.png and
# gamma_per_sector.png. Each build in the sweep is a full intra-US construction, so this
# is the most expensive job of the repository.
#
#   sbatch pipeline/plots/run_gamma_sweep.sbatch report     # the dense uniform sweep
#   sbatch pipeline/plots/run_gamma_sweep.sbatch obs        # bilateral structure vs CFS/FAF
#   sbatch pipeline/plots/run_gamma_sweep.sbatch grid       # build-only bilateral grid
#
# The three are independent and can run in parallel. Outputs land in
# pipeline/plots/outputs_gamma/, which is versioned so the figure can be redrawn
# without repeating the sweep.
#SBATCH --job-name=gamma-sweep
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
if [ ! -f pyproject.toml ] || [ ! -d pipeline ]; then
  echo "error: submit this script from the repository root (got $PWD)" >&2
  exit 1
fi
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

case "${1:-report}" in
  report) SCRIPT=gamma_sensitivity_report.py ;;
  obs)    SCRIPT=gamma_vs_observations.py ;;
  grid)   SCRIPT=gamma_bilateral_grid.py ;;
  *) echo "unknown target '$1' (report|obs|grid)" >&2; exit 1 ;;
esac

echo "Host: $(hostname) | Mem: ${SLURM_MEM_PER_NODE:-?} MB | CPUs: ${SLURM_CPUS_PER_TASK:-?}"
date
python -u "pipeline/plots/$SCRIPT"
date
echo "DONE"
