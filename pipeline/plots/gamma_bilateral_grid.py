"""Build-only gamma grid: how the *bilateral* state-pair structure moves with gamma.

Only the intra-US build step is run (no harmonisation, no nesting), which is what
determines the state-to-state structure, so a fine grid is cheap. Produces, per
gamma: the Nevada -> California export share (all commodities and goods only, the
quantity the Commodity Flow Survey comparison reports), the total variation of the
bilateral share matrix against the delivered gamma = 1 build, and the mean
per-origin partner-share total variation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _figpaths  # noqa: F401  (puts pipeline/ on sys.path)
import gamma_sweep as gs  # noqa: E402

YEAR = 2017
GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
OUT = Path(__file__).resolve().parent / "outputs_gamma"
OUT.mkdir(parents=True, exist_ok=True)

GOODS_KEYS = ("Manufacture", "Mining", "Oil and gas", "Agriculture", "primary metals")


def main():
    gs.setup()
    w2p = gs.build_windc_to_proposed()
    sectors = list(gs.sectors)
    goods = np.array([any(k in w2p.get(s, "") for k in GOODS_KEYS) for s in sectors])
    print(f"{goods.sum()} goods sectors of {len(sectors)}", flush=True)

    states = list(gs.regions)
    n, S = gs.n, gs.S
    iNV, iCA = states.index("NV"), states.index("CA")

    mats = {}
    rows = []
    for g in GRID:
        print(f"-- gamma {g}", flush=True)
        tab = gs.build_v31_ras(YEAR, g)
        Zst = tab["Z"].reshape(n, S, n, S)
        Z_all = Zst.sum(axis=(1, 3))                    # (origin state, dest state)
        Z_gds = Zst[:, goods, :, :].sum(axis=(1, 3))    # goods rows only
        for M in (Z_all, Z_gds):
            np.fill_diagonal(M, 0.0)
        mats[g] = Z_all
        rows.append({
            "gamma": g,
            "NV_to_CA_all_pct": 100 * Z_all[iNV, iCA] / Z_all[iNV].sum(),
            "NV_to_CA_goods_pct": 100 * Z_gds[iNV, iCA] / Z_gds[iNV].sum(),
            "CA_from_NV_all_pct": 100 * Z_all[iNV, iCA] / Z_all[:, iCA].sum(),
            "mean_export_share_to_CA_goods_pct":
                100 * np.mean([Z_gds[i, iCA] / Z_gds[i].sum()
                               for i in range(n) if i != iCA and Z_gds[i].sum() > 0]),
        })
        del tab, Zst

    ref = mats[1.0] / mats[1.0].sum()
    for r in rows:
        M = mats[r["gamma"]]
        P = M / M.sum()
        r["tv_pairs_vs_gamma1"] = 0.5 * np.abs(P - ref).sum()
        tvs = []
        for i in range(n):
            a, b = M[i], mats[1.0][i]
            if a.sum() > 0 and b.sum() > 0:
                tvs.append(0.5 * np.abs(a / a.sum() - b / b.sum()).sum())
        r["mean_partner_tv_vs_gamma1"] = float(np.mean(tvs))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "gamma_bilateral_grid.csv", index=False)
    np.savez_compressed(OUT / "gamma_bilateral_matrices.npz",
                        states=np.array(states),
                        **{f"g{g}": mats[g] for g in GRID})
    print(df.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
