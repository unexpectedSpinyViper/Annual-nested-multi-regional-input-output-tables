"""Figure — closure of the state cost accounts.

**a**, Distribution of the pre-closure cost-identity residual across the 1 887 state
columns of the reference year, as a percentage of column output. Mass concentrates
near zero, with a thin negative tail: those are the columns the clip eventually acts
on.

**b**, Share-based value-added prior against delivered value added, logarithmic
scales. Points on the floor are the clipped columns, whose value added the residual
rule sets to zero and whose shortfall the tax row absorbs.

Re-nests the reference year twice, open and closed, so it needs the harmonised
sub-national block and the global table. Run on a compute node: about 24 GB.

Writes ``figures/fig5_balance.png``.

Run from anywhere:  python pipeline/plots/fig05_balance.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _figpaths import FIG_DIR
import nest_v31 as N

YEAR = 2017
C_MAIN, C_CLIP = "#3577a8", "#bd5e2c"
INK, MUTED = "#1a1a1a", "#5b5b5b"


def main():
    print("nesting", YEAR, "without the closure ...", flush=True)
    shares = N.build_shares_pivot()
    gdp = N.build_gdp_share_pivot()
    openp = N.nest_year(YEAR, shares, gdp, N.VA_SOURCE, N.F_SOURCE, close=False)
    closed, info = N.close_state_columns(openp)

    mask = info["state_col_mask"]
    resid = info["residual_before"][mask]        # cost-identity gap, M$
    va_prior = info["va_prior"][mask]
    va_deliv = info["va_delivered"][mask]

    sec_cols = [c for c in openp.columns
                if "_" in c and c != "OUT" and c.split("_", 1)[1] not in N.FD_CATS]
    out_row = openp.loc["OUT", sec_cols].values.astype(float)[mask]
    pct = 100 * resid / np.where(out_row > 0, out_row, np.nan)

    clipped = info["clipped_mask"][mask]
    print(f"state columns            : {mask.sum():,}")
    print(f"clipped                  : {info['n_va_clipped']} "
          f"({100*info['n_va_clipped']/mask.sum():.1f}%)")
    print(f"tax row absorbs          : {info['tls_absorbed']/1e3:,.1f} bn USD")
    print(f"max residual after       : {info['max_residual_after']:.3g} M USD")
    print(f"residual, %% of output   : p1 {np.nanpercentile(pct,1):.2f}  "
          f"median {np.nanmedian(pct):.3f}  p99 {np.nanpercentile(pct,99):.2f}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.6, 5.2))

    # ---- panel a --------------------------------------------------------------
    lo, hi = np.nanpercentile(pct, [.2, 99.8])
    span = max(abs(lo), abs(hi))
    bins = np.linspace(-span, span, 121)
    axA.hist(pct[~np.isnan(pct)], bins=bins, color=C_MAIN, alpha=.85, linewidth=0)
    axA.axvline(0, color=INK, lw=.8)
    axA.annotate(f"{info['n_va_clipped']} columns of {int(mask.sum()):,} are clipped:\n"
                 f"the identity asks for negative value added",
                 xy=(-span * .82, axA.get_ylim()[1] * .30),
                 xytext=(-span * .95, axA.get_ylim()[1] * .62), fontsize=8,
                 color=C_CLIP, va="center",
                 arrowprops=dict(arrowstyle="->", color=C_CLIP, lw=.8,
                                 connectionstyle="arc3,rad=.2"))
    axA.set_yscale("log")
    axA.set_xlabel("pre-closure cost-identity residual, per cent of column output",
                   fontsize=9)
    axA.set_ylabel("state columns (log scale)", fontsize=9)
    axA.set_title("a   the state columns do not close spontaneously",
                  fontsize=10.5, fontweight="bold", loc="left", pad=6)
    axA.grid(ls=":", alpha=.4); axA.tick_params(labelsize=8)
    for s in ("top", "right"):
        axA.spines[s].set_visible(False)

    # ---- panel b --------------------------------------------------------------
    floor = max(va_deliv[va_deliv > 0].min() / 3, 1e-3)
    yy = np.where(va_deliv > 0, va_deliv, floor)
    xx = np.maximum(va_prior, floor)
    axB.scatter(xx[~clipped], yy[~clipped], s=5, alpha=.30, color=C_MAIN,
                linewidths=0, label="closed on value added")
    axB.scatter(xx[clipped], yy[clipped], s=18, color=C_CLIP, linewidths=0,
                label=f"clipped to zero ({int(clipped.sum())})", zorder=4)
    lim = [floor, max(xx.max(), yy.max()) * 1.4]
    axB.plot(lim, lim, color=INK, lw=.8, ls="--", alpha=.6, label="prior = delivered")
    axB.axhline(floor, color=C_CLIP, lw=.6, ls=":", alpha=.8)
    axB.text(lim[1], floor * 1.15, "floor: value added clipped to zero", fontsize=7,
             color=C_CLIP, ha="right", va="bottom")
    axB.set_xscale("log"); axB.set_yscale("log")
    axB.set_xlim(lim); axB.set_ylim(lim)
    axB.set_xlabel("share-based value-added prior (million USD)", fontsize=9)
    axB.set_ylabel("delivered value added (million USD)", fontsize=9)
    axB.set_title("b   what the residual rule changes", fontsize=10.5,
                  fontweight="bold", loc="left", pad=6)
    axB.legend(frameon=False, fontsize=8, loc="upper left")
    axB.grid(ls=":", alpha=.4); axB.tick_params(labelsize=8)
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)

    plt.tight_layout()
    out = FIG_DIR / "fig5_balance.png"
    fig.savefig(out, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    main()
