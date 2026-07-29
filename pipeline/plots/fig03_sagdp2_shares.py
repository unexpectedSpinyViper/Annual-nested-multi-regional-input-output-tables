"""Figure — allocation of the international flows across states.

**a**, The production share matrix ``S`` for the reference year, sectors as rows and
states as columns, states sorted by overall share, logarithmic colour scale.
Concentrated sectors appear as strong rows; diversified services spread across all
states. The row of the sector that has no sub-national counterpart carries the
fallback share ``theta``, the cross-sector mean of ``S``.

**b**, Per-sector Pearson correlation across the fifty-one states between ``S`` and the
state distribution of sub-national value added — the two candidate allocators of the
world-to-United-States flows.

Writes ``figures/fig3_sagdp2_shares.png``.

Run from anywhere:  python pipeline/plots/fig03_sagdp2_shares.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from _figpaths import FIG_DIR
from plot_source_diff_figure import SHORT, WINDC_HARM
import nest_v31 as N

YEAR = 2017
C_S, C_VA = "#3577a8", "#bd5e2c"
INK, MUTED = "#1a1a1a", "#5b5b5b"


def load():
    npz = np.load(WINDC_HARM / f"IOT_{YEAR}_harmonized.npz", allow_pickle=True)
    states = sorted(set(l.split("_")[0] for l in npz["index_labels"]))
    wd_secs = [str(s) for s in npz["proposed_sectors"]]

    oecd = pd.read_parquet(N.find_oecd_file(YEAR))
    oecd_secs = [r.split("_", 1)[1] for r in oecd.index if r.startswith("USA_")]

    shares = N.build_shares_pivot()
    S = N.get_share_matrix(YEAR, states, oecd_secs, shares)      # (state, sector)

    # the competing allocator: the state distribution of sub-national value added
    va = np.asarray(npz["VA"], dtype=float).reshape(len(states), len(wd_secs))
    VA = np.zeros_like(S)
    for wi, ws in enumerate(wd_secs):
        if ws in oecd_secs:
            VA[:, oecd_secs.index(ws)] = va[:, wi]
    col = VA.sum(0, keepdims=True)
    VA = VA / np.where(col > 0, col, 1.0)
    return states, oecd_secs, S, VA, wd_secs


def main():
    states, secs, S, VA, wd_secs = load()
    no_sub = [j for j, s in enumerate(secs) if s not in wd_secs]
    print(f"{len(states)} states x {len(secs)} sectors | "
          f"sector(s) with no sub-national counterpart: {[secs[j] for j in no_sub]}")

    order_st = np.argsort(S.mean(1))[::-1]                       # states by overall share
    M = S[order_st].T                                            # (sector, state)
    labels = [SHORT.get(s, s) for s in secs]

    fig = plt.figure(figsize=(13.2, 9.4))
    axA = fig.add_axes([.175, .400, .720, .545])
    axB = fig.add_axes([.175, .062, .720, .238])
    cax = fig.add_axes([.905, .430, .011, .420])

    # ---- panel a: the share matrix -------------------------------------------
    lo = max(M[M > 0].min(), 1e-6)
    im = axA.imshow(np.where(M > 0, M, np.nan), aspect="auto", cmap="magma_r",
                    norm=LogNorm(vmin=lo, vmax=M.max()), interpolation="nearest")
    axA.set_xticks(range(len(states)))
    axA.set_xticklabels([states[i] for i in order_st], rotation=90, fontsize=6.2)
    axA.set_yticks(range(len(secs)))
    axA.set_yticklabels(labels, fontsize=6.2)
    axA.tick_params(length=1.5, pad=1.5)
    axA.set_xlabel("state, sorted by overall share", fontsize=8.5, labelpad=3)
    axA.set_title("a   the production share matrix $S$, "
                  f"{YEAR}", fontsize=10.5, fontweight="bold", loc="left", pad=6)
    for j in no_sub:                                  # the fallback row
        axA.annotate("fallback share $\\theta$", xy=(len(states) - .3, j),
                     xytext=(len(states) + 2.6, j), fontsize=6.8, color=C_VA,
                     va="center", ha="left", annotation_clip=False,
                     arrowprops=dict(arrowstyle="-", color=C_VA, lw=.7))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("state share of United States\ngross product in the sector",
                 fontsize=8, labelpad=4)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(.5)

    # ---- panel b: how far apart are the two candidate allocators? -------------
    r = np.full(len(secs), np.nan)
    for j in range(len(secs)):
        a, b = S[:, j], VA[:, j]
        if a.std() > 0 and b.std() > 0:
            r[j] = np.corrcoef(a, b)[0, 1]
    ok = ~np.isnan(r)
    o = [j for j in np.argsort(np.where(ok, r, np.inf)) if ok[j]]
    print(f"per-sector Pearson r(S, VA): median {np.nanmedian(r):.3f}  "
          f"min {np.nanmin(r):.3f} ({labels[int(np.nanargmin(r))]})  "
          f"below 0.9: {int((r[ok] < .9).sum())} of {int(ok.sum())}")

    x = np.arange(len(o))
    axB.bar(x, r[o], .68, color=[C_VA if r[j] < .9 else C_S for j in o])
    axB.axhline(.9, color=INK, lw=.8, ls="--", alpha=.55)
    axB.text(len(o) - .4, .905, "0.9", fontsize=7, color=INK, va="bottom", ha="right")
    axB.set_xticks(x)
    axB.set_xticklabels([labels[j] for j in o], rotation=90, fontsize=6.2)
    axB.set_ylim(0, 1.03)
    axB.set_xlim(-.8, len(o) - .2)
    axB.set_ylabel("Pearson $r$ across the\n51 states", fontsize=8.5)
    axB.set_title("b   agreement between the two candidate allocators, per sector: "
                  "production share $S$ against sub-national value added",
                  fontsize=10.5, fontweight="bold", loc="left", pad=6)
    axB.grid(axis="y", ls=":", alpha=.45)
    axB.tick_params(labelsize=7.5)
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)

    out = FIG_DIR / "fig3_sagdp2_shares.png"
    fig.savefig(out, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    main()
