"""Bundle every figure into one archive, ready to pull off the cluster.

Sorts what ``figures/`` holds into three folders, writes a manifest saying which
script produced each file and where the manuscript uses it, and packs the lot into a
single dated zip at the repository root:

    figures_bundle_<date>.zip
      manuscript/      the 18 figures of the data descriptor
      supplementary/   the Supplementary figures
      diagnostics/     the exploratory plots the pipeline notebooks leave behind
      data/            the CSV and JSON tables behind the figures
      MANIFEST.csv     file, folder, size, last modified, producing script

Downloading it is then one action instead of forty. Run it after
``run_figures.sbatch``, which is what regenerates the figures themselves.

    python pipeline/plots/collect_figures.py
    python pipeline/plots/collect_figures.py --tex ../path/to/descriptor.tex
    python pipeline/plots/collect_figures.py --outdir ~/somewhere --no-zip
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import shutil
import zipfile
from pathlib import Path

from _figpaths import FIG_DIR, ROOT

# Figures the data descriptor includes, in the order it includes them. Passing --tex
# re-derives this from the manuscript instead, which is the check to run if the
# manuscript has moved on.
MANUSCRIPT = [
    "figure1_workflow.png",
    "national-pool marginal per commodity.png",
    "fig3_sagdp2_shares.png",
    "fig1_nested_layout.png",
    "fig6_checks_series.png",
    "oecd_windc_blocks_comparison.png",
    "fig_source_structural_distance.png",
    "fig5_balance.png",
    "frobenius_relative_divergence_oecd_windc.png",
    "vector_multipliers_harmonization.png",
    "Z_multipliers_harmonization.png",
    "interstate_structure.png",
    "fig_cfs_faf_validation.png",
    "gamma_sensitivity_extended.png",
    "fd_allocator_choice_2017.png",
    "fd_allocator_bias.png",
    "fd_gov_referents.png",
    "fd_allocator_timeseries.png",
]

SUPPLEMENTARY = [
    "figS_source_diff_variants.png",
    "gamma_per_sector.png",
    "fig_distance_variants.png",
    "fig_distance_variants.pdf",
    "fd_allocator_delivered_delta_2017.png",
    "fd_breadth_mechanism_2017.png",
]

# What wrote each file, so a reader of the bundle can regenerate any single figure.
PRODUCER = {
    "figure1_workflow.png": "pipeline/plots/fig01_workflow.py",
    "national-pool marginal per commodity.png": "pipeline/plots/fig02_national_pool.py",
    "fig3_sagdp2_shares.png": "pipeline/plots/fig03_sagdp2_shares.py",
    "fig1_nested_layout.png": "pipeline/plots/fig04_nested_layout.py",
    "fig5_balance.png": "pipeline/plots/fig05_balance.py",
    "fig6_checks_series.png": "pipeline/plots/fig06_checks_series.py",
    "checks_series.csv": "pipeline/plots/fig06_checks_series.py",
    "interstate_structure.png": "pipeline/plots/fig07_interstate_structure.py",
    "fig_distance_variants.png": "pipeline/plots/fig08_distance_variants.py",
    "fig_distance_variants.pdf": "pipeline/plots/fig08_distance_variants.py",
    "centroids_four_variants.csv": "pipeline/plots/fig08_distance_variants.py",
    "distance_matrix_comparison.csv": "pipeline/plots/fig08_distance_variants.py",
    "fig_source_structural_distance.png": "pipeline/plots/plot_source_diff_figure.py",
    "figS_source_diff_variants.png": "pipeline/plots/plot_source_diff_figure.py",
    "oecd_windc_blocks_comparison.png": "pipeline/plots/plot_harmonization_figures.py",
    "frobenius_relative_divergence_oecd_windc.png": "pipeline/plots/plot_harmonization_figures.py",
    "vector_multipliers_harmonization.png": "pipeline/plots/plot_harmonization_figures.py",
    "Z_multipliers_harmonization.png": "pipeline/plots/plot_harmonization_figures.py",
    "harmonisation_multipliers.csv": "pipeline/plots/plot_harmonization_figures.py",
    "gamma_sensitivity_extended.png": "pipeline/plots/plot_gamma_figures.py",
    "gamma_per_sector.png": "pipeline/plots/plot_gamma_figures.py",
    "network_stats_2017.csv": "pipeline/plots/table_network_stats.py",
    "fig_cfs_faf_validation.png": "pipeline/validation/cfs_faf_validation.ipynb",
    "cfs_faf_metrics_2017.csv": "pipeline/validation/cfs_faf_validation.ipynb",
}
for _f in ("fd_allocator_choice_2017.png", "fd_allocator_bias.png", "fd_gov_referents.png",
           "fd_allocator_timeseries.png", "fd_allocator_delivered_delta_2017.png",
           "fd_breadth_mechanism_2017.png", "fd_allocator_results_2017.json",
           "fd_allocator_tradeoffs.csv"):
    PRODUCER[_f] = "pipeline/validation/final_demand_allocator.ipynb"

DATA_SUFFIXES = {".csv", ".json"}


def manuscript_from_tex(tex: Path):
    """The figures a .tex actually includes, in order, without duplicates."""
    src = tex.read_text()
    found = re.findall(r"includegraphics(?:\[[^\]]*\])?\{figures/([^}]*)\}", src)
    return list(dict.fromkeys(found))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tex", type=Path,
                    help="derive the manuscript list from this .tex instead of the built-in one")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="where to build the bundle (default: the repository root)")
    ap.add_argument("--no-zip", action="store_true", help="leave the folder, do not zip it")
    args = ap.parse_args()

    manuscript = MANUSCRIPT
    if args.tex:
        manuscript = manuscript_from_tex(args.tex)
        print(f"manuscript list taken from {args.tex}: {len(manuscript)} figures")

    stamp = dt.date.today().isoformat()
    root = (args.outdir or ROOT).expanduser()
    bundle = root / f"figures_bundle_{stamp}"
    if bundle.exists():
        shutil.rmtree(bundle)

    present = {p.name: p for p in FIG_DIR.iterdir() if p.is_file()}
    missing = [f for f in manuscript if f not in present]

    def folder_of(name):
        if name in manuscript:
            return "manuscript"
        if Path(name).suffix.lower() in DATA_SUFFIXES:
            return "data"
        if name in SUPPLEMENTARY:
            return "supplementary"
        return "diagnostics"

    rows = []
    for name, path in sorted(present.items()):
        sub = folder_of(name)
        dest = bundle / sub
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest / name)
        rows.append({
            "file": name,
            "folder": sub,
            "size_kb": round(path.stat().st_size / 1024),
            "modified": dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(" ", "seconds"),
            "produced_by": PRODUCER.get(name, ""),
            "manuscript_order": manuscript.index(name) + 1 if name in manuscript else "",
        })

    with open(bundle / "MANIFEST.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    counts = {}
    for r in rows:
        counts[r["folder"]] = counts.get(r["folder"], 0) + 1
    print(f"\n{len(rows)} files collected from {FIG_DIR.relative_to(ROOT)}/")
    for k in ("manuscript", "supplementary", "diagnostics", "data"):
        if k in counts:
            print(f"  {k:<15} {counts[k]:3d}")
    if missing:
        print("\n  MISSING from figures/, not in the bundle:")
        for m in missing:
            print(f"    {m}")
        print("  run `sbatch pipeline/plots/run_figures.sbatch` to regenerate them")

    target = bundle
    if not args.no_zip:
        target = bundle.with_suffix(".zip")
        if target.exists():
            target.unlink()
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(bundle.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(bundle.parent))
        shutil.rmtree(bundle)

    size = target.stat().st_size / 1e6 if target.is_file() else sum(
        p.stat().st_size for p in target.rglob("*") if p.is_file()) / 1e6
    print(f"\n-> {target}  ({size:.1f} MB)")
    print("\nPull it onto your machine with, from a terminal on your machine:")
    print(f"  scp {_host()}:{target} ~/Downloads/")
    print("or, to keep a folder in sync without re-copying what has not changed:")
    print(f"  rsync -avz --delete {_host()}:{FIG_DIR}/ ~/Downloads/mriot-figures/")


def _host():
    """Best guess at the host to scp from; edit if your ssh alias differs."""
    import getpass
    import socket
    return f"{getpass.getuser()}@{socket.getfqdn()}"


if __name__ == "__main__":
    main()
