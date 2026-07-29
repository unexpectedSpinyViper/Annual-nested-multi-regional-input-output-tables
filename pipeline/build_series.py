"""Stage 2-3 — build the intra-United-States table for every delivered year.

Runs the v3.1_RAS construction once per year: two-layer (trade + margin) doubly
constrained gravity on the national commodity pools, RAS to the observed marginals,
column calibration onto industry output, then the OECD/SNA value-added convention.
The logic lives in ``gamma_sweep.build_v31_ras``; this module only drives it over the
series and writes the delivered schema.

Output: ``data/interim/IOT/IOT_USA/grav_fric_v3.1_RAS/IOT_<year>.npz``, the input of
step 20 (aggregation).

Reference points of the gravity distance matrix are the GDP-weighted economic
centroids of ``02_economic_centroids.ipynb``; run that step first.

Needs the WiNDC GDX and a licensed GAMS install. About 64 GB and a few minutes per
year, so run it on a compute node:

    sbatch pipeline/run_build_series.sbatch
    python pipeline/build_series.py --years 2017        # or one year, directly
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

_PIPELINE = Path(__file__).resolve().parent
sys.path.insert(0, str(_PIPELINE))
try:                                          # after `pip install -e .`
    from paths import ROOT
except ModuleNotFoundError:                   # plain checkout, no install
    sys.path.insert(0, str(_PIPELINE.parent / "src"))
    from paths import ROOT
import gamma_sweep as gs  # noqa: E402

WINDC_VER = "v3.1_RAS"
GAMMA = 1.0                       # the delivered friction exponent
YEARS = range(1997, 2023)         # 1997 is the first sub-national year, 2022 the last
                                  # year of the global table
OUT_DIR = ROOT / f"data/interim/IOT/IOT_USA/grav_fric_{WINDC_VER}"


def build_year(year, gamma=GAMMA):
    """One year of the delivered intra-US table, as the saved schema."""
    table = gs.build_v31_ras(year, gamma)
    # regions/sectors are module state, not part of the returned dict
    table["regions"] = np.array(gs.regions)
    table["sectors"] = np.array(gs.sectors)
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", nargs="*", type=int, default=list(YEARS),
                    help="years to build (default: the whole delivered series)")
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--force", action="store_true",
                    help="rebuild years whose output already exists")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gs.setup(verbose=True)
    print(f"writing to {OUT_DIR.relative_to(ROOT)}  (gamma = {args.gamma})", flush=True)

    built = skipped = 0
    for year in args.years:
        out = OUT_DIR / f"IOT_{year}.npz"
        if out.exists() and not args.force:
            print(f"[{year}] exists, skipped", flush=True)
            skipped += 1
            continue
        t0 = time.time()
        table = build_year(year, args.gamma)

        # the accounting check of the stage, reported per year rather than asserted:
        # margin sectors carry a known residual against direct output, which the
        # column calibration of the next step absorbs
        Z, F = table["Z"], table["F"]
        row = Z.sum(1) + F.sum(1) + table["EX"]
        col = Z.sum(0) + table["M_interm"] + table["VA"] + table["taxes"]
        m = row > 0.1
        gap = float(np.abs(row[m] - col[m]).sum() / row[m].sum() * 100)

        np.savez_compressed(out, **table)
        print(f"[{year}] cost-identity gap {gap:5.2f}%  "
              f"Z {Z.sum()/1e3:8.2f} tn$  VA {table['VA'].sum()/1e3:6.2f} tn$  "
              f"-> {out.name}  ({time.time()-t0:.0f}s)", flush=True)
        built += 1
        del table, Z, F
        gc.collect()

    print(f"\n{built} year(s) built, {skipped} skipped -> {OUT_DIR}")


if __name__ == "__main__":
    main()
