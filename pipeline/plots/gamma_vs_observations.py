"""Bilateral structure of the reconstructed block as a function of gamma, against CFS/FAF.

Companion to gamma_sensitivity_report.py. For every friction exponent of the sweep (and
every commodity-specific scenario) it rebuilds the intra-US table up to the harmonised
stage -- the nesting step is not needed here -- and measures the *bilateral* object that
the external validation judges:

  * the 51x51 state-pair matrix of goods flows (Z + F over the 19 goods sectors), exactly
    the object compared with CFS and FAF in cfs_faf_validation.ipynb;
  * its structural L1 distance to the delivered gamma = 1 build, to CFS and to FAF, on the
    same 0..2 scale, plus rank/log correlations and top-partner recovery;
  * the California over-concentration diagnostics (mean export share to CA, NV -> CA), and
    the leading destination of every state, on goods flows and on all-sector intermediate
    shipments.

Outputs to pipeline/plots/outputs_gamma/:
  gamma_bilateral_vs_obs.csv      one row per build
  gamma_leading_partner.csv       build x origin state -> leading destination
  gamma_ca_shares.csv             build x origin state -> export share to CA
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import _figpaths  # noqa: F401  (puts pipeline/ on sys.path)
import gamma_sweep as gs  # noqa: E402

YEAR = 2017
GAMMAS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
ROOT = gs.ROOT
CFS_CSV = ROOT / "data/raw/CFS/CFS 2017 PUF CSV.csv"
FAF_CSV = ROOT / "data/raw/FAF5/FAF5.7.1_State.csv"
OUT = Path(__file__).resolve().parent / "outputs_gamma"
OUT.mkdir(parents=True, exist_ok=True)

FIPS = {'01':'AL','02':'AK','04':'AZ','05':'AR','06':'CA','08':'CO','09':'CT','10':'DE',
        '11':'DC','12':'FL','13':'GA','15':'HI','16':'ID','17':'IL','18':'IN','19':'IA',
        '20':'KS','21':'KY','22':'LA','23':'ME','24':'MD','25':'MA','26':'MI','27':'MN',
        '28':'MS','29':'MO','30':'MT','31':'NE','32':'NV','33':'NH','34':'NJ','35':'NM',
        '36':'NY','37':'NC','38':'ND','39':'OH','40':'OK','41':'OR','42':'PA','44':'RI',
        '45':'SC','46':'SD','47':'TN','48':'TX','49':'UT','50':'VT','51':'VA','53':'WA',
        '54':'WV','55':'WI','56':'WY'}
GOODS_EXTRA = {"Agriculture, hunting, forestry, fishing and related",
               "Mining, except oil & gas", "Oil and gas extraction", "primary metals"}


# ── metric helpers, identical to cfs_faf_validation.ipynb ─────────────────────
def shares(M, off):
    s = M.copy(); np.fill_diagonal(s, 0.0); return s / s.sum()


def struct_L1(A, B, off):
    return float(np.abs(shares(A, off)[off] - shares(B, off)[off]).sum())


def logcorr(A, B, off):
    a, b = A[off], B[off]; m = (a > 0) & (b > 0)
    return float(stats.pearsonr(np.log(a[m]), np.log(b[m]))[0])


def spear(A, B, off):
    a, b = A[off], B[off]; m = (a > 0) & (b > 0)
    return float(stats.spearmanr(a[m], b[m])[0])


def rownorm(M):
    X = M.copy(); np.fill_diagonal(X, 0.0)
    r = X.sum(1, keepdims=True); r[r == 0] = 1; return X / r


def top_recovery(A, B, axis="row"):
    XA, XB = A.copy(), B.copy()
    np.fill_diagonal(XA, -1); np.fill_diagonal(XB, -1)
    if axis == "row":
        return float((XA.argmax(1) == XB.argmax(1)).mean())
    return float((XA.argmax(0) == XB.argmax(0)).mean())


def partner_L1(A, B):
    RA, RB = rownorm(A), rownorm(B)
    d = np.abs(RA - RB).sum(1)
    return float(d[d > 0].mean()) if (d > 0).any() else 0.0


def load_observed(nr, IDX):
    def mat_from_pairs(dct):
        M = np.zeros((nr, nr))
        for (o, dd), v in dct.items():
            if o in IDX and dd in IDX:
                M[IDX[o], IDX[dd]] = v
        return M

    agg = {}
    for ch in pd.read_csv(CFS_CSV,
                          usecols=["ORIG_STATE", "DEST_STATE", "EXPORT_YN",
                                   "SHIPMT_VALUE", "WGT_FACTOR"],
                          dtype={"ORIG_STATE": str, "DEST_STATE": str, "EXPORT_YN": str},
                          chunksize=1_500_000):
        ch = ch[ch["EXPORT_YN"] == "N"].copy()
        ch["o"] = ch["ORIG_STATE"].map(FIPS); ch["d"] = ch["DEST_STATE"].map(FIPS)
        ch = ch.dropna(subset=["o", "d"])
        ch["v"] = ch["SHIPMT_VALUE"] * ch["WGT_FACTOR"]
        for k, v in ch.groupby(["o", "d"])["v"].sum().items():
            agg[k] = agg.get(k, 0) + v
    M_cfs = mat_from_pairs(agg)

    aggf = {}
    for ch in pd.read_csv(FAF_CSV,
                          usecols=["dms_origst", "dms_destst", "trade_type", "value_2017"],
                          dtype={"dms_origst": str, "dms_destst": str, "trade_type": str},
                          chunksize=1_500_000):
        ch = ch[ch["trade_type"] == "1"].copy()
        ch["o"] = ch["dms_origst"].map(FIPS); ch["d"] = ch["dms_destst"].map(FIPS)
        ch = ch.dropna(subset=["o", "d"])
        for k, v in ch.groupby(["o", "d"])["value_2017"].sum().items():
            aggf[k] = aggf.get(k, 0) + v
    return M_cfs, mat_from_pairs(aggf)


def build_matrices(gamma, label=None):
    """build -> RAS -> VA conv -> aggregate -> harmonise; return the state-pair matrices."""
    table = gs.build_v31_ras(YEAR, gamma)
    agg = gs.aggregate_windc(table, gs.build_windc_to_proposed())
    harm, _ = gs.harmonize_payload(YEAR, agg)
    secs = [str(s) for s in harm["proposed_sectors"]]
    states = [str(r) for r in harm["regions"]]
    ns, nr = len(secs), len(states)
    goods = [i for i, s in enumerate(secs) if s.startswith("Manufacture") or s in GOODS_EXTRA]
    Z4 = harm["Z"].reshape(nr, ns, nr, ns)
    F3 = harm["F"].reshape(nr, ns, nr, -1)
    M_goods = Z4[:, goods, :, :].sum((1, 3)) + F3[:, goods, :, :].sum((1, 3))
    M_all = Z4.sum((1, 3))
    del table, agg, harm, Z4, F3
    gc.collect()
    return M_goods, M_all, states, len(goods)


def main():
    gs.setup()
    M1_goods, M1_all, states, n_goods = build_matrices(1.0)
    nr = len(states)
    IDX = {s: i for i, s in enumerate(states)}
    off = ~np.eye(nr, dtype=bool)
    print(f"{n_goods} goods sectors, {nr} regions", flush=True)

    M_cfs, M_faf = load_observed(nr, IDX)
    print("observed matrices loaded", flush=True)

    builds = [(f"g{g}", g, None) for g in GAMMAS]

    # commodity-specific scenarios, same draws as gamma_sensitivity_report.py
    w2p = gs.build_windc_to_proposed()
    HEAVY = ("Manufacture", "Mining", "Oil and gas", "Agriculture",
             "Construction", "Electricity", "primary metals")

    def sector_class(code):
        prop = w2p.get(code, "")
        if any(k in prop for k in HEAVY):
            return "heavy"
        if "transport" in prop.lower() or "Warehousing" in prop:
            return "transport"
        return "service"

    SECTORS = list(gs.sectors)
    classes = {s: sector_class(s) for s in SECTORS}
    ECON = {"service": 0.3, "transport": 1.2, "heavy": 2.5}
    BANDS = {"service": (0.1, 0.8), "transport": (0.8, 1.6), "heavy": (1.8, 3.0)}
    rng_s = np.random.default_rng(1)
    to_vec = lambda fn: np.array([fn(s) for s in SECTORS], float)  # noqa: E731
    scen = {"economic": to_vec(lambda s: ECON[classes[s]]),
            "structured_rand": to_vec(lambda s: rng_s.uniform(*BANDS[classes[s]])),
            "random_0": np.random.default_rng(10).uniform(0.0, 3.0, len(SECTORS)),
            "random_1": np.random.default_rng(11).uniform(0.0, 3.0, len(SECTORS)),
            "random_2": np.random.default_rng(12).uniform(0.0, 3.0, len(SECTORS))}
    builds += [(f"scen_{k}", v, k) for k, v in scen.items()]

    rows, lead_rows, ca_rows = [], [], []
    for key, gamma, label in builds:
        print(f"\n===== {key} =====", flush=True)
        if key == "g1.0":
            Mg, Ma = M1_goods, M1_all
        else:
            Mg, Ma, _, _ = build_matrices(gamma, label=label)

        ca = rownorm(Mg)[:, IDX["CA"]]
        r = {"build": key,
             "gamma": gamma if np.isscalar(gamma) else float(np.mean(gamma)),
             "L1_vs_gamma1": struct_L1(Mg, M1_goods, off),
             "L1_vs_CFS": struct_L1(Mg, M_cfs, off),
             "L1_vs_FAF": struct_L1(Mg, M_faf, off),
             "pearson_log_CFS": logcorr(Mg, M_cfs, off),
             "spearman_CFS": spear(Mg, M_cfs, off),
             "exp_partner_L1_CFS": partner_L1(Mg, M_cfs),
             "top_exp_recovery_CFS": top_recovery(Mg, M_cfs, "row"),
             "top_imp_recovery_CFS": top_recovery(Mg, M_cfs, "col"),
             "top_exp_recovery_FAF": top_recovery(Mg, M_faf, "row"),
             "mean_CA_export_share": float(np.nanmean(ca)),
             "NV_to_CA_goods": float(ca[IDX["NV"]]),
             "intra_share_goods": float(np.trace(Mg) / Mg.sum())}

        for tag, M in (("goods", Mg), ("all_interm", Ma)):
            X = M.copy(); np.fill_diagonal(X, 0.0)
            lead = np.array(states)[X.argmax(1)]
            vc = pd.Series(lead).value_counts()
            r[f"n_distinct_leaders_{tag}"] = int(vc.size)
            for st in ("CA", "NY", "TX"):
                r[f"leader_{st}_{tag}"] = int(vc.get(st, 0))
            for o, d in zip(states, lead):
                lead_rows.append({"build": key, "scope": tag, "origin": o, "leader": d})
        rows.append(r)
        for o, v in zip(states, ca):
            ca_rows.append({"build": key, "origin": o, "share_to_CA": v})
        del Mg, Ma
        gc.collect()

    # the observed matrices, as reference rows
    for nm, M in (("CFS", M_cfs), ("FAF", M_faf)):
        ca = rownorm(M)[:, IDX["CA"]]
        X = M.copy(); np.fill_diagonal(X, 0.0)
        lead = np.array(states)[X.argmax(1)]
        vc = pd.Series(lead).value_counts()
        rows.append({"build": nm, "gamma": np.nan,
                     "L1_vs_gamma1": struct_L1(M, M1_goods, off),
                     "L1_vs_CFS": struct_L1(M, M_cfs, off),
                     "L1_vs_FAF": struct_L1(M, M_faf, off),
                     "mean_CA_export_share": float(np.nanmean(ca)),
                     "NV_to_CA_goods": float(ca[IDX["NV"]]),
                     "n_distinct_leaders_goods": int(vc.size),
                     "leader_CA_goods": int(vc.get("CA", 0)),
                     "leader_NY_goods": int(vc.get("NY", 0)),
                     "leader_TX_goods": int(vc.get("TX", 0))})
        for o, d in zip(states, lead):
            lead_rows.append({"build": nm, "scope": "goods", "origin": o, "leader": d})
        for o, v in zip(states, ca):
            ca_rows.append({"build": nm, "origin": o, "share_to_CA": v})

    pd.DataFrame(rows).to_csv(OUT / "gamma_bilateral_vs_obs.csv", index=False)
    pd.DataFrame(lead_rows).to_csv(OUT / "gamma_leading_partner.csv", index=False)
    pd.DataFrame(ca_rows).to_csv(OUT / "gamma_ca_shares.csv", index=False)
    print(pd.DataFrame(rows).to_string())
    print("\nwritten to", OUT, flush=True)


if __name__ == "__main__":
    main()
