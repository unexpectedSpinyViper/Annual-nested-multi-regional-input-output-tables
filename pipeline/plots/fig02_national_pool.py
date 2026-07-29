"""Figure — diagnostics on the sub-national source: the national commodity pool.

**a**, National pool marginals per commodity, for the fifteen commodities with the
largest export-to-import imbalance. The bilateral reconstruction of stage 2 is a
doubly constrained problem: for each commodity the national pool shipped out of the
states (``xn0``) must equal the national pool absorbed by them. With direct
absorption alone as the import marginal (``nd0``) the margin-providing commodities
are structurally unbalanced and the problem is infeasible for them; adding margin
absorption (``nm0``) closes the pool exactly for all seventy-one commodities.

**b**, Similarity between the state-share distributions of direct pool absorption and
of margin absorption, per margin-providing commodity — cosine similarity of the two
51-vectors of state shares. Both distributions are shared out of national totals with
the same absorption key in the source, so the agreement is a property of the
compilation method and not an independent measurement.

Reads the WiNDC GDX (a licensed GAMS install is required, see data/raw/DOWNLOAD.md).

Writes ``figures/national-pool marginal per commodity.png``.

Run from anywhere:  python pipeline/plots/fig02_national_pool.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _figpaths import FIG_DIR
import gamma_sweep as gs

YEAR = 2017
K_WORST = 15                 # commodities shown in panel a
SKIP = 0.50                  # the feasibility guard of reconstruct_bilateral_3

C_DIR, C_MAR = "#3577a8", "#bd5e2c"
INK, MUTED, GRID = "#1a1a1a", "#5b5b5b", "#d8d8d8"


def pool_marginals(year=YEAR):
    """Per commodity: national pool supply, direct absorption, margin absorption."""
    gs.setup(verbose=False)
    data = gs.load_year_data(year)
    xn0, nd0 = data["xn0_"], data["nd0_"]                       # (region, commodity)
    nm0 = (gs.params["nm0_"][gs.params["nm0_"]["yr"] == gs._yr(year)]
           .groupby(["r", "g"])["value"].sum().unstack("g")
           .reindex(index=gs.regions, columns=gs.sectors, fill_value=0.0)
           .fillna(0.0).values)
    return xn0, nd0, nm0


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan


def main():
    xn0, nd0, nm0 = pool_marginals()
    sectors = list(gs.sectors)

    X = xn0.sum(0)                      # national pool shipped out, per commodity
    M_dir = nd0.sum(0)                  # absorbed as direct demand
    M_mar = nm0.sum(0)                  # absorbed as margin supply
    M_tot = M_dir + M_mar

    with np.errstate(divide="ignore", invalid="ignore"):
        imb_dir = np.where(X > 0, np.abs(X - M_dir) / X, np.nan)
        imb_tot = np.where(X > 0, np.abs(X - M_tot) / X, np.nan)

    live = X > 0
    n_infeasible = int(np.nansum(imb_dir[live] > SKIP))
    print(f"commodities with a live pool          : {int(live.sum())}")
    print(f"  direct absorption only, |imb| > 50% : {n_infeasible}")
    print(f"  with margin absorption, max |imb|   : {np.nanmax(imb_tot[live]):.3e}")

    order = np.argsort(np.where(np.isnan(imb_dir), -1, imb_dir))[::-1][:K_WORST][::-1]
    y = np.arange(len(order))

    fig = plt.figure(figsize=(12.4, 8.2))
    axA1 = fig.add_axes([.150, .455, .335, .495])
    axA2 = fig.add_axes([.560, .455, .335, .495])
    axB = fig.add_axes([.150, .075, .745, .265])

    # ---- panel a: the two import marginals against the export marginal --------
    for ax, M, title, col in (
            (axA1, M_dir, "direct absorption alone,  $M_g=\\sum_s nd0_{sg}$", C_DIR),
            (axA2, M_tot, "with margin absorption,  $M_g=\\sum_s (nd0+nm0)_{sg}$", C_MAR)):
        ax.barh(y + .20, X[order], .40, color="#b9c6d2",
                label="pool shipped out  $X_g$")
        ax.barh(y - .20, M[order], .40, color=col, label="pool absorbed  $M_g$")
        ax.set_yticks(y)
        ax.set_yticklabels([sectors[i] for i in order] if ax is axA1 else [], fontsize=7)
        ax.set_xlabel("billion USD  (WiNDC commodity codes on the axis)", fontsize=8.5)
        ax.set_title(title, fontsize=9.5, loc="left", pad=6)
        ax.grid(axis="x", ls=":", alpha=.45)
        ax.tick_params(labelsize=8)
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # the imbalance actually reached, printed against each bar pair
    for i, g in enumerate(order):
        axA1.text(max(X[g], M_dir[g]) * 1.02, i, f"{100*imb_dir[g]:.0f}%",
                  va="center", fontsize=6.5, color=C_DIR)
        axA2.text(max(X[g], M_tot[g]) * 1.02, i, f"{100*imb_tot[g]:.1e}",
                  va="center", fontsize=6.5, color=C_MAR)
    axA1.axvline(0, color=MUTED, lw=.6)
    axA2.axvline(0, color=MUTED, lw=.6)
    fig.text(.150, .975, "a", fontsize=12, fontweight="bold", color=INK)
    fig.text(.150, .375, "b", fontsize=12, fontweight="bold", color=INK)

    # ---- panel b: do the two absorption keys distribute alike across states? --
    marg = [i for i in range(len(sectors)) if nm0[:, i].sum() > 0 and nd0[:, i].sum() > 0]
    sim = np.array([cosine(nd0[:, i] / nd0[:, i].sum(), nm0[:, i] / nm0[:, i].sum())
                    for i in marg])
    o = np.argsort(sim)
    names = [sectors[marg[i]] for i in o]
    vals = sim[o]
    print(f"\nmargin-providing commodities with a non-empty comparison: {len(marg)}")
    for nme, v in zip(names, vals):
        print(f"  {nme:<12s} {v:.3f}")

    xb = np.arange(len(vals))
    axB.bar(xb, vals, .62, color=[C_MAR if v < .85 else C_DIR for v in vals])
    axB.axhline(.85, color=INK, lw=.8, ls="--", alpha=.6)
    axB.text(len(vals) - .4, .855, "0.85", fontsize=7, color=INK, va="bottom", ha="right")
    axB.set_xticks(xb)
    axB.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    axB.set_ylim(0, 1.04)
    axB.set_ylabel("cosine similarity of the\nstate-share distributions", fontsize=8.5)
    axB.set_title("direct pool absorption against margin absorption, "
                  "per margin-providing commodity", fontsize=9.5, loc="left", pad=6)
    axB.grid(axis="y", ls=":", alpha=.45)
    axB.tick_params(labelsize=8)
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)

    out = FIG_DIR / "national-pool marginal per commodity.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("\nwrote", out)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    main()
