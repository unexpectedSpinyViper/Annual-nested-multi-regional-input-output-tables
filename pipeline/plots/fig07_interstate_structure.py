"""Figure — the reconstructed sub-national table.

**a**, Calibrated inter-state intermediate shipments over all commodities, logarithmic
scale; the row and column bands of the large state economies and the dark diagonal of
observed intra-state deliveries.

**b**, Distance decay of the 2 550 calibrated state-pair flows. The ordinary least
squares slope is the *effective* distance elasticity of the table after the
biproportional calibration of the ``gamma = 1`` prior — it is a property of the
delivered table, not the friction exponent that was imposed.

**c**, Industry-account residual per sector before and after the row-preserving column
calibration (``ras_table``), logarithmic scale, twenty-five largest shown.

Rebuilds the intra-US table for the reference year, so it needs the WiNDC GDX and a
licensed GAMS install. Run on a compute node: about 30 GB.

Writes ``figures/interstate_structure.png``.

Run from anywhere:  python pipeline/plots/fig07_interstate_structure.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from _figpaths import FIG_DIR
import gamma_sweep as gs

YEAR = 2017
GAMMA = 1.0                       # the delivered friction exponent
K_SECTORS = 25
C_BEFORE, C_AFTER = "#b9c6d2", "#3577a8"
C_FIT = "#bd5e2c"
INK, MUTED = "#1a1a1a", "#5b5b5b"


def industry_residual(table):
    """|ys0 - (intermediate cost + imported inputs + VA + taxes)| per (region, sector)."""
    Z, ys0 = table["Z"], table["ys0"]
    used = Z.sum(axis=0) + table["M_interm"] + table["VA"] + table["taxes"]
    return np.abs(ys0 - used)


def main():
    gs.setup(verbose=True)
    n, S = gs.n, gs.S
    states, sectors = list(gs.regions), list(gs.sectors)

    print("building the intra-US table ...", flush=True)
    before = gs.build_table(YEAR, gamma_trade=GAMMA, gamma_margin=GAMMA)
    after = gs.ras_table(before, YEAR)

    res_b = industry_residual(before).reshape(n, S).sum(0)      # per sector, over states
    res_a = industry_residual(after).reshape(n, S).sum(0)
    print(f"industry-account residual, total: {res_b.sum():,.0f} -> {res_a.sum():,.0f} bn$")

    # inter-state intermediate shipments, all commodities
    Zst = after["Z"].reshape(n, S, n, S).sum(axis=(1, 3))        # (origin, destination)
    off = ~np.eye(n, dtype=bool)
    D = gs.D_np

    fig = plt.figure(figsize=(14.6, 6.4))
    axA = fig.add_axes([.055, .135, .295, .760])
    axB = fig.add_axes([.435, .135, .245, .760])
    axC = fig.add_axes([.760, .135, .225, .760])
    cax = fig.add_axes([.357, .215, .009, .600])

    # ---- panel a -------------------------------------------------------------
    pos = Zst[Zst > 0]
    im = axA.imshow(np.where(Zst > 0, Zst, np.nan), cmap="magma_r",
                    norm=LogNorm(vmin=max(pos.min(), 1e-4), vmax=Zst.max()),
                    interpolation="nearest")
    axA.set_xticks(range(n)); axA.set_xticklabels(states, rotation=90, fontsize=4.6)
    axA.set_yticks(range(n)); axA.set_yticklabels(states, fontsize=4.6)
    axA.tick_params(length=1.2, pad=1)
    axA.set_xlabel("destination state", fontsize=8.5, labelpad=2)
    axA.set_ylabel("origin state", fontsize=8.5, labelpad=2)
    axA.set_title("a   inter-state intermediate shipments",
                  fontsize=10.5, fontweight="bold", loc="left", pad=6)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("billion USD", fontsize=8, labelpad=3)
    cb.ax.tick_params(labelsize=7); cb.outline.set_linewidth(.5)

    # ---- panel b: distance decay --------------------------------------------
    d = D[off]; f = Zst[off]
    keep = (d > 0) & (f > 0)
    x, y = np.log10(d[keep]), np.log10(f[keep])
    slope, intercept = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    print(f"distance decay: slope {slope:.3f}  r {r:.3f}  n {keep.sum()}")

    axB.scatter(10 ** x, 10 ** y, s=4, alpha=.30, color=C_AFTER, linewidths=0)
    xs = np.linspace(x.min(), x.max(), 50)
    axB.plot(10 ** xs, 10 ** (intercept + slope * xs), color=C_FIT, lw=1.6,
             label=f"OLS slope ${slope:.2f}$   ($r={r:.2f}$)")
    axB.set_xscale("log"); axB.set_yscale("log")
    axB.set_xlabel("great-circle distance between economic centroids (km)", fontsize=8.5)
    axB.set_ylabel("state-pair intermediate shipments (billion USD)", fontsize=8.5)
    axB.set_title(f"b   distance decay, {int(keep.sum()):,} state pairs",
                  fontsize=10.5, fontweight="bold", loc="left", pad=6)
    axB.legend(frameon=False, fontsize=8, loc="lower left")
    axB.grid(ls=":", alpha=.4); axB.tick_params(labelsize=8)
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)

    # ---- panel c: the column calibration closes the industry accounts --------
    top = np.argsort(res_b)[::-1][:K_SECTORS][::-1]
    yy = np.arange(len(top))
    floor = max(min(res_a[res_a > 0].min(), res_b[res_b > 0].min()) / 2, 1e-8)
    axC.barh(yy + .20, np.maximum(res_b[top], floor), .40, color=C_BEFORE,
             label="before calibration")
    axC.barh(yy - .20, np.maximum(res_a[top], floor), .40, color=C_AFTER,
             label="after calibration")
    axC.set_xscale("log")
    axC.set_yticks(yy); axC.set_yticklabels([sectors[i] for i in top], fontsize=6)
    axC.set_xlabel("industry-account residual (billion USD, log scale)", fontsize=8.5)
    axC.set_title("c   residual per sector, 25 largest",
                  fontsize=10.5, fontweight="bold", loc="left", pad=6)
    axC.legend(frameon=False, fontsize=7.5, loc="lower right")
    axC.grid(axis="x", ls=":", alpha=.4); axC.tick_params(labelsize=8)
    for s in ("top", "right"):
        axC.spines[s].set_visible(False)

    out = FIG_DIR / "interstate_structure.png"
    fig.savefig(out, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    main()
