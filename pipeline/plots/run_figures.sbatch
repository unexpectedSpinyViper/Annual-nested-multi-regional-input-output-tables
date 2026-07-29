#!/bin/bash
# Regenerate every figure of the data descriptor on a compute node.
#   sbatch pipeline/plots/run_figures.sbatch          all of them
#   sbatch pipeline/plots/run_figures.sbatch fig05    just one script
#
# The light scripts (fig01, fd allocator) also run on a login node; the ones that
# rebuild the intra-US table or read a delivered file do not.
#SBATCH --job-name=descriptor-figs
#SBATCH --account=torch_pr_1108_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.log

set -euo pipefail

export TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/scratch/$USER/.matplotlib}"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate cascading-impacts

cd "$(dirname "$0")/../.."          # repository root

SCRIPTS=(
  fig01_workflow.py                 # workflow schematic
  fig02_national_pool.py            # national commodity pool marginals
  fig03_sagdp2_shares.py            # the production share matrix S
  fig04_nested_layout.py            # layout of a delivered file
  fig05_balance.py                  # closure of the state cost accounts
  fig06_checks_series.py            # per-year accounting checks
  fig07_interstate_structure.py     # the reconstructed sub-national table
  fig08_distance_variants.py        # Supplementary: the four distance references
  plot_source_diff_figure.py        # structural distance between the two sources
  plot_harmonization_figures.py     # the harmonisation multipliers
  plot_gamma_figures.py             # friction-exponent sensitivity
)

if [ $# -gt 0 ]; then
  SCRIPTS=()
  for pat in "$@"; do
    for f in pipeline/plots/${pat}*.py; do SCRIPTS+=("$(basename "$f")"); done
  done
fi

echo "Host: $(hostname) | Mem: ${SLURM_MEM_PER_NODE:-?} MB | CPUs: ${SLURM_CPUS_PER_TASK:-?}"
for s in "${SCRIPTS[@]}"; do
  echo "=============================================================="
  echo "== $s"
  echo "=============================================================="
  python -u "pipeline/plots/$s"
done

echo "DONE"
