"""Deep gamma-sensitivity analysis for the data descriptor.

Runs the full intra-US -> nested pipeline over a dense grid of uniform friction
exponents gamma, plus five per-sector (heterogeneous) friction scenarios, and
measures for each build:

  * structural metrics of the intra-US block (intra share, flow-weighted mean
    trade distance, Gini of interstate flows);
  * the response of the delivered table to a -30% final-demand / capacity /
    primary-input shock in one state under Leontief, IIM and Ghosh, decomposed
    into the shocked state, the other 50 US regions, and the rest of the world;
  * the *distribution* of that response across the 50 other US states and across
    foreign countries: per-region shares, rank correlation and total variation
    against the gamma = 1 build, top-5 concentration, Gini.

Outputs (written to pipeline/plots/outputs_gamma/):
  gamma_summary.csv          one row per (build, model): amplitude metrics
  gamma_structure.csv        one row per build: structural metrics
  gamma_state_losses.csv     build x model x region loss (M$), US states + ROW
  gamma_distribution.csv     one row per (build, model): distribution metrics
  gamma_sector_losses.csv    build x model x sector loss over the other-US block

Usage:  python gamma_sensitivity_report.py          (run under SLURM, ~64 GB)
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import _figpaths  # noqa: F401  (puts pipeline/ on sys.path)
import gamma_sweep as gs  # noqa: E402

YEAR = 2017
SHOCK_REGION = "NY"
THETA = 0.30
GAMMAS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
MODELS = ("Leontief", "IIM", "Ghosh")

OUT = Path(__file__).resolve().parent / "outputs_gamma"
OUT.mkdir(parents=True, exist_ok=True)


# ── distribution helpers ──────────────────────────────────────────────────────
def _gini(a):
    a = np.sort(np.abs(np.asarray(a, float).ravel()))
    if a.sum() <= 0:
        return np.nan
    k = a.size
    return (2 * np.arange(1, k + 1) - k - 1).dot(a) / (k * a.sum())


def region_losses(comp, prefix, mask_us, mask_world):
    """Loss aggregated per region (US 2-letter code, foreign 3-letter code)."""
    out = {}
    for model in MODELS:
        s = pd.Series(comp[model].to_numpy(), index=prefix)
        out[model] = s.groupby(level=0).sum()
    df = pd.DataFrame(out)
    df.index.name = "region"
    return df


# ── run one build ─────────────────────────────────────────────────────────────
def analyse(build_key, gamma, shares_pivot, label=None):
    res = gs.run_pipeline_for_gamma(YEAR, gamma, shares_pivot, label=label, verbose=True)
    struct = gs.structural_metrics(res["table"])
    blocks = gs.load_nested_blocks(res["nested"])
    imp = gs.compute_impacts(blocks, SHOCK_REGION, THETA)

    prefix = blocks["prefix"]
    reg = region_losses(imp["comp"], prefix, blocks["mask_us"], blocks["mask_world"])
    reg.insert(0, "build", build_key)

    # sector profile of the other-US spillover (aggregated over the 50 states)
    us_other = (prefix != SHOCK_REGION) & blocks["mask_us"]
    secname = np.array([s.split("_", 1)[1] for s in blocks["sectors"]])
    sec = pd.DataFrame({m: pd.Series(imp["comp"][m].to_numpy()[us_other],
                                     index=secname[us_other]).groupby(level=0).sum()
                        for m in MODELS})
    sec.index.name = "sector"
    sec.insert(0, "build", build_key)

    summ = imp["summary"]
    rows = []
    for m in MODELS:
        rows.append({
            "build": build_key,
            "gamma": gamma if np.isscalar(gamma) else np.nan,
            "model": m,
            "direct": summ.loc[f"{SHOCK_REGION}_direct", m],
            "other_US": summ.loc["other_US_spillover", m],
            "world": summ.loc["world_spillover", m],
            "TOTAL": summ.loc["TOTAL", m],
        })
    srow = {"build": build_key,
            "gamma": gamma if np.isscalar(gamma) else np.nan,
            **{k: v for k, v in struct.items() if k != "Z_rs"},
            "leontief_repro_err": imp["repro"]["leontief_err"],
            "ghosh_repro_err": imp["repro"]["ghosh_err"]}

    # state-pair flow matrix, kept for the two extreme uniform builds only
    Z_rs = struct["Z_rs"] if build_key in ("g0.1", "g1.0", "g3.0") else None

    del res, blocks, imp
    gc.collect()
    return pd.DataFrame(rows), srow, reg.reset_index(), sec.reset_index(), Z_rs


def main():
    gs.setup()
    shares_pivot = gs.nest_v31.build_shares_pivot()

    # per-sector scenarios, identical construction to gamma_pipeline_comparison.ipynb
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

    def to_vec(fn):
        return np.array([fn(s) for s in SECTORS], dtype=float)

    rng_s = np.random.default_rng(1)
    scen = {
        "economic": to_vec(lambda s: ECON[classes[s]]),
        "structured_rand": to_vec(lambda s: rng_s.uniform(*BANDS[classes[s]])),
        "random_0": np.random.default_rng(10).uniform(0.0, 3.0, len(SECTORS)),
        "random_1": np.random.default_rng(11).uniform(0.0, 3.0, len(SECTORS)),
        "random_2": np.random.default_rng(12).uniform(0.0, 3.0, len(SECTORS)),
    }
    pd.DataFrame({"sector": SECTORS,
                  "class": [classes[s] for s in SECTORS],
                  **{k: v for k, v in scen.items()}}).to_csv(
        OUT / "gamma_scenarios_vectors.csv", index=False)

    summaries, structures, regions, sectors_ = [], [], [], []
    zrs = {}

    for g in GAMMAS:
        key = f"g{g}"
        print(f"\n===== uniform gamma = {g} =====", flush=True)
        s, st, r, sc, Z = analyse(key, g, shares_pivot)
        summaries.append(s); structures.append(st); regions.append(r); sectors_.append(sc)
        if Z is not None:
            zrs[key] = Z

    for name, vec in scen.items():
        key = f"scen_{name}"
        print(f"\n===== scenario {name} =====", flush=True)
        s, st, r, sc, _ = analyse(key, vec, shares_pivot, label=name)
        summaries.append(s); structures.append(st); regions.append(r); sectors_.append(sc)

    summary = pd.concat(summaries, ignore_index=True)
    structure = pd.DataFrame(structures)
    reg_all = pd.concat(regions, ignore_index=True)
    sec_all = pd.concat(sectors_, ignore_index=True)

    summary.to_csv(OUT / "gamma_summary.csv", index=False)
    structure.to_csv(OUT / "gamma_structure.csv", index=False)
    reg_all.to_csv(OUT / "gamma_state_losses.csv", index=False)
    sec_all.to_csv(OUT / "gamma_sector_losses.csv", index=False)
    np.savez_compressed(OUT / "gamma_state_pair_flows.npz",
                        states=np.array(list(gs.regions)), **zrs)

    # ── distribution metrics against the gamma = 1 build ──────────────────────
    ref_key = "g1.0"
    dist_rows = []
    for key in reg_all["build"].unique():
        sub = reg_all[reg_all.build == key].set_index("region")
        ref = reg_all[reg_all.build == ref_key].set_index("region")
        for m in MODELS:
            for scope, sel in (("us_other", [r for r in sub.index
                                             if len(r) == 2 and r != SHOCK_REGION]),
                               ("world", [r for r in sub.index if len(r) == 3])):
                a = sub.loc[sel, m].abs()
                b = ref.loc[sel, m].abs()
                pa, pb = a / a.sum(), b / b.sum()
                order = pb.sort_values(ascending=False).index
                dist_rows.append({
                    "build": key, "model": m, "scope": scope,
                    "total_M$": sub.loc[sel, m].sum(),
                    "tv_vs_gamma1": 0.5 * (pa - pb).abs().sum(),
                    "spearman_vs_gamma1": spearmanr(a.values, b.values).statistic,
                    "pearson_log_vs_gamma1": np.corrcoef(
                        np.log10(a.clip(lower=1e-6)), np.log10(b.clip(lower=1e-6)))[0, 1],
                    "gini": _gini(a.values),
                    "top5_share": pa.sort_values(ascending=False).head(5).sum(),
                    "top5_share_of_gamma1_top5": pa.loc[order[:5]].sum(),
                    "max_abs_share_change_pp": 100 * (pa - pb).abs().max(),
                    "argmax_share_change": (pa - pb).abs().idxmax(),
                    "top1": pa.idxmax(),
                })
    pd.DataFrame(dist_rows).to_csv(OUT / "gamma_distribution.csv", index=False)
    print("\nwritten to", OUT, flush=True)


if __name__ == "__main__":
    main()
