"""Figures for the friction-exponent section of the data descriptor.

  figures/gamma_sensitivity_extended.png  — main text: what gamma does to the table
                                   (top row) against what it does to the
                                   propagation models (bottom row).
  figures/gamma_per_sector.png   — supplement: commodity-specific frictions
                                   inside the envelope of the uniform sweep.

Inputs are the CSVs written by gamma_sensitivity_report.py and
gamma_bilateral_grid.py.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _figpaths import FIG_DIR

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs_gamma"
FIGDIR = FIG_DIR

# Categorical hues, fixed order, one per model; checked for CVD separation
# (min OKLab dE = 9.0 over deuter-/prot-/tritanopia, >= 8 target).
C = {"Leontief": "#0072B2", "IIM": "#D55E00", "Ghosh": "#009E73"}
INK, MUTED, GRID = "#1a1a1a", "#5b5b5b", "#d8d8d8"
MODELS = ["Leontief", "IIM", "Ghosh"]

# Structural distance of the delivered table to the two observations, in the L1
# convention of the external-validation subsection (0..2; /2 = mass misallocated).
# Read from the external-validation output rather than pinned, so the two reference
# lines cannot drift away from the delivered table the way a literal does.
def _observed_L1():
    path = FIG_DIR / "cfs_faf_metrics_2017.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run validation/cfs_faf_validation.ipynb first; it "
            f"measures the distance of the delivered table to CFS and FAF.")
    t = pd.read_csv(path).set_index("comparison")["struct_L1"]
    return float(t["recon vs CFS"]), float(t["recon vs FAF"])


CFS_L1, FAF_L1 = _observed_L1()

mpl.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def style(ax, title=None, xlabel=r"friction exponent $\gamma$", ylabel=None):
    if title:
        ax.set_title(title, loc="left", pad=7)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=0.5, linewidth=0.6)
    ax.set_axisbelow(True)


def label_end(ax, x, y, text, color, dx=0.06, va="center"):
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset fontsize",
                color=color, fontsize=8, va=va, ha="left", clip_on=False)


def main():
    summ = pd.read_csv(OUT / "gamma_summary.csv")
    struct = pd.read_csv(OUT / "gamma_structure.csv")
    dist = pd.read_csv(OUT / "gamma_distribution.csv")
    grid = pd.read_csv(OUT / "gamma_bilateral_grid.csv")

    uni = summ[summ.build.str.startswith("g")].dropna(subset=["gamma"])
    ust = struct.dropna(subset=["gamma"]).sort_values("gamma")
    g = ust["gamma"].to_numpy()

    # ── Figure 1 — main text ─────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.4))

    ax = axes[0, 0]
    ax.plot(g, ust["mean_trade_distance_km"], "o-", color=INK, lw=1.6, ms=4.5)
    style(ax, "a  Mean inter-state shipping distance", ylabel="km, flow-weighted")

    ax = axes[0, 1]
    ax.plot(g, ust["interstate_gini"], "o-", color=INK, lw=1.6, ms=4.5)
    style(ax, "b  Concentration of inter-state flows", ylabel="Gini of state-pair flows")

    ax = axes[0, 2]
    gg = grid.sort_values("gamma")
    ax.plot(gg["gamma"], 2 * gg["tv_pairs_vs_gamma1"], "o-", color=INK, lw=1.6, ms=4.5,
            label=r"distance to the $\gamma=1$ build")
    ax.axhline(CFS_L1, color=C["IIM"], lw=1.4, ls="--")
    ax.axhline(FAF_L1, color=C["Ghosh"], lw=1.4, ls=":")
    label_end(ax, 3.0, CFS_L1, "  distance to CFS", C["IIM"])
    label_end(ax, 3.0, FAF_L1 + 0.045, "  distance to FAF", C["Ghosh"])
    style(ax, "c  Displacement of the bilateral structure",
          ylabel=r"$L_1$ distance of state-pair shares")
    ax.legend(loc="upper center", fontsize=8)

    ax = axes[1, 0]
    NUDGE = {"Leontief": 0.0, "IIM": -0.16, "Ghosh": 0.16}
    for m in MODELS:
        d = uni[uni.model == m].sort_values("gamma")
        ref = d.loc[d.gamma == 1.0, "TOTAL"].iloc[0]
        y = 100 * (d["TOTAL"] / ref - 1)
        ax.plot(d["gamma"], y, "o-", color=C[m], lw=1.6, ms=4.5, label=m)
        ax.annotate(f" {m}", (d["gamma"].iloc[-1], y.iloc[-1] + NUDGE[m]),
                    xytext=(4, 0), textcoords="offset points",
                    color=C[m], fontsize=8, va="center", ha="left", clip_on=False)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylim(-1.5, 1.5)
    style(ax, "d  Total propagated loss",
          ylabel=r"% deviation from the $\gamma=1$ build")

    ax = axes[1, 1]
    for m in MODELS:
        d = uni[uni.model == m].sort_values("gamma")
        r1 = d.loc[d.gamma == 1.0].iloc[0]
        ax.plot(d["gamma"], 100 * (d["direct"] / r1["direct"] - 1), "-",
                color=C[m], lw=1.6, label=f"{m}, shocked state")
        ax.plot(d["gamma"], 100 * (d["other_US"] / r1["other_US"] - 1), "--",
                color=C[m], lw=1.6, label=f"{m}, other 50 regions")
        ax.plot(d["gamma"], 100 * (d["world"] / r1["world"] - 1), ":",
                color=C[m], lw=1.4)
    ax.axhline(0, color=MUTED, lw=0.8)
    style(ax, "e  Where the loss falls", ylabel=r"% deviation from the $\gamma=1$ build")
    h = [plt.Line2D([], [], color=MUTED, ls=s, lw=1.5) for s in ("-", "--", ":")]
    leg = ax.legend(h, ["shocked state", "other 50 US regions", "rest of world"],
                    loc="lower left", fontsize=8)
    ax.add_artist(leg)
    ax.legend([plt.Line2D([], [], color=C[m], lw=1.6) for m in MODELS], MODELS,
              loc="upper left", fontsize=8)

    ax = axes[1, 2]
    for m in MODELS:
        d = dist[(dist.model == m) & (dist.scope == "us_other")]
        d = d[d.build.str.match(r"g[\d.]+$")].copy()
        d["gamma"] = d.build.str[1:].astype(float)
        d = d.sort_values("gamma")
        ax.plot(d["gamma"], 100 * d["tv_vs_gamma1"], "o-", color=C[m], lw=1.6, ms=4.5,
                label=m)
        w = dist[(dist.model == m) & (dist.scope == "world")]
        w = w[w.build.str.match(r"g[\d.]+$")].copy()
        w["gamma"] = w.build.str[1:].astype(float)
        w = w.sort_values("gamma")
        ax.plot(w["gamma"], 100 * w["tv_vs_gamma1"], ":", color=C[m], lw=1.4)
    style(ax, "f  Reallocation of the spillover between regions",
          ylabel=r"% of spillover mass changing region")
    h = [plt.Line2D([], [], color=MUTED, ls=s, lw=1.5) for s in ("-", ":")]
    ax.legend(h + [plt.Line2D([], [], color=C[m], lw=1.5) for m in MODELS],
              ["across the 50 other US regions", "across foreign countries"] + MODELS,
              loc="upper center", fontsize=8, ncol=1)

    fig.tight_layout(w_pad=2.4, h_pad=2.6)
    fig.savefig(FIGDIR / "gamma_sensitivity_extended.png")
    plt.close(fig)
    print("wrote", FIGDIR / "gamma_sensitivity_extended.png")

    # ── Figure 2 — supplement: commodity-specific frictions ──────────────────
    scen = summ[summ.build.str.startswith("scen_")].copy()
    scen["name"] = scen.build.str.replace("scen_", "", regex=False)
    sst = struct[struct.build.str.startswith("scen_")].copy()
    sst["name"] = sst.build.str.replace("scen_", "", regex=False)
    NICE = {"economic": "transportability", "structured_rand": "structured random",
            "random_0": "random 0", "random_1": "random 1", "random_2": "random 2"}
    order = ["economic", "structured_rand", "random_0", "random_1", "random_2"]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))

    ax = axes[0]
    ax.plot(ust["mean_trade_distance_km"], ust["interstate_gini"], "-", color=GRID, lw=6,
            solid_capstyle="round", zorder=1)
    ax.plot(ust["mean_trade_distance_km"], ust["interstate_gini"], "o", color=MUTED,
            ms=4, zorder=2)
    for _, r in ust.iterrows():
        if r["gamma"] in (0.1, 1.0, 3.0):
            ax.annotate(rf"$\gamma={r['gamma']:g}$", (r["mean_trade_distance_km"],
                        r["interstate_gini"]), xytext=(-36, 3),
                        textcoords="offset points", fontsize=8, color=MUTED)
    for nm in order:
        r = sst[sst.name == nm].iloc[0]
        ax.plot(r["mean_trade_distance_km"], r["interstate_gini"], "D",
                color=C["Leontief"], ms=6, zorder=3)
        ax.annotate(NICE[nm], (r["mean_trade_distance_km"], r["interstate_gini"]),
                    xytext=(5, -9), textcoords="offset points", fontsize=7.5,
                    color=C["Leontief"])
    style(ax, "a  Heterogeneous frictions on the uniform locus",
          xlabel="mean inter-state shipping distance (km)",
          ylabel="Gini of state-pair flows")
    ax.plot([], [], "-", color=GRID, lw=5, label=r"uniform $\gamma$ sweep")
    ax.plot([], [], "D", color=C["Leontief"], ms=5, label="commodity-specific")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    x = np.arange(len(order))
    w = 0.26
    for k, m in enumerate(MODELS):
        ref = uni[(uni.model == m) & (uni.gamma == 1.0)].iloc[0]
        d = scen[scen.model == m].set_index("name").reindex(order)
        ax.bar(x + (k - 1) * w, 100 * (d["TOTAL"] / ref["TOTAL"] - 1), w * 0.92,
               color=C[m], label=m)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([NICE[o] for o in order], rotation=22, ha="right", fontsize=8)
    ax.set_ylim(-1.5, 1.5)
    style(ax, "b  Total loss, per scenario", xlabel=None,
          ylabel=r"% deviation from $\gamma=1$")
    ax.legend(loc="upper right", fontsize=8, ncol=3)

    ax = axes[2]
    for k, m in enumerate(MODELS):
        ref = uni[(uni.model == m) & (uni.gamma == 1.0)].iloc[0]
        d = scen[scen.model == m].set_index("name").reindex(order)
        ax.bar(x + (k - 1) * w, 100 * (d["other_US"] / ref["other_US"] - 1), w * 0.92,
               color=C[m], label=m)
        lo = uni[uni.model == m]["other_US"] / ref["other_US"] - 1
    env_lo = min(100 * (uni[uni.model == m]["other_US"].min()
                        / uni[(uni.model == m) & (uni.gamma == 1.0)]["other_US"].iloc[0] - 1)
                 for m in MODELS)
    env_hi = max(100 * (uni[uni.model == m]["other_US"].max()
                        / uni[(uni.model == m) & (uni.gamma == 1.0)]["other_US"].iloc[0] - 1)
                 for m in MODELS)
    ax.axhspan(env_lo, env_hi, color=GRID, alpha=0.55, zorder=0,
               label=r"envelope of the uniform sweep")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([NICE[o] for o in order], rotation=22, ha="right", fontsize=8)
    style(ax, "c  Spillover to the other 50 regions", xlabel=None,
          ylabel=r"% deviation from $\gamma=1$")
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout(w_pad=2.4)
    fig.savefig(FIGDIR / "gamma_per_sector.png")
    plt.close(fig)
    print("wrote", FIGDIR / "gamma_per_sector.png")


if __name__ == "__main__":
    main()
