"""Figures and tables for the harmonisation step (H1-H6), 2017.

Writes, to figures/ at the repository root:

  oecd_windc_blocks_comparison.png        block-by-block scatter of the two sources
  frobenius_relative_divergence_oecd_windc.png   divergence to the OECD target, before/after
  vector_multipliers_harmonization.png    the per-sector multipliers sigma^V
  Z_multipliers_harmonization.png         the cell multipliers sigma^Z

plus ``harmonisation_multipliers.csv``, the multiplier table of the manuscript as data,
and prints every figure quoted in the Technical Validation.

Run from anywhere:  python pipeline/plots/plot_harmonization_figures.py
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

from plot_source_diff_figure import (ROOT, YEAR, WINDC_SCALE, FD_CATS, WINDC_RAW,
                                     WINDC_HARM, PAPER_FIG, SHORT, ORDER,
                                     pick_oecd_file, C_NEG, C_POS)

C_GREY = "#b9c6d2"
BLOCKS = [("VA", "value added"), ("TLS", "taxes less subsidies on products"),
          ("EX", "exports"), ("M", "intermediate imports"),
          ("F", "final demand"), ("Z", "intermediate cells")]


# ─────────────────────────────────────────────────────────────────────── data
def agg_npz(path, order):
    z = np.load(path, allow_pickle=True)
    s = [str(x) for x in z["proposed_sectors"]]
    n, nr = len(s), z["Z"].shape[0] // len(s)
    idx = [s.index(k) for k in order]
    one = lambda k: z[k].reshape(nr, n)[:, idx].sum(0) * WINDC_SCALE
    d = dict(
        Z=z["Z"].reshape(nr, n, nr, n)[:, idx][:, :, :, idx].sum((0, 2)) * WINDC_SCALE,
        F=z["F"].reshape(nr, n, -1)[:, idx, :].sum((0, 2)) * WINDC_SCALE,
        VA=one("VA"), EX=one("EX"), M=one("M_interm"))
    d["TLS"] = one("TLS") if "TLS" in z.files else one("taxes")
    for k in ("tls_int", "tls_fd", "tariff", "tax_prod"):
        if k in z.files:
            d[k] = one(k)
    return d


def oecd_blocks(order):
    f = pick_oecd_file(YEAR, order)
    sch = pq.read_schema(f)
    usa = [f"USA_{s}" for s in order]
    fd = [c for c in sch.names if c.startswith("USA_") and c.split("_", 1)[1] in FD_CATS]
    idx = [c for c in (sch.pandas_metadata or {}).get("index_columns", [])
           if isinstance(c, str)]
    d = pq.read_table(f, columns=usa + fd + ["OUT"] + idx).to_pandas()
    Z = d.loc[usa, usa].values.astype(float)
    F = d.loc[usa, fd].values.astype(float).sum(1)
    wrow = [r for r in d.index if "_" in r and len(r.split("_")[0]) == 3
            and not r.startswith("USA_")]
    out = dict(Z=Z, F=F, VA=d.loc["VA", usa].values.astype(float),
               TLS=d.loc["TLS", usa].values.astype(float),
               M=d.loc[wrow, usa].values.astype(float).sum(0))
    out["EX"] = d.loc[usa, "OUT"].values.astype(float) - Z.sum(1) - F
    out["TLS_fd"] = d.loc["TLS", fd].values.astype(float)
    return out


def load_all():
    W = agg_npz(WINDC_RAW / f"IOT_{YEAR}.npz", ORDER)
    H = agg_npz(WINDC_HARM / f"IOT_{YEAR}_harmonized.npz", ORDER)
    O = oecd_blocks(ORDER)
    tr = np.load(WINDC_HARM / f"transform_{YEAR}.npz", allow_pickle=True)
    secs_tr = [str(s) for s in tr["secs_common"]]
    p = [secs_tr.index(s) for s in ORDER]                     # transform -> display
    sf = {k: tr[f"sf_{k}"][p] for k in ("VA", "EX", "M", "F")}
    sfZ = tr["sf_Z"][np.ix_(p, p)]
    return W, H, O, sf, sfZ


div = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ───────────────────────────────────────────────────── 1. block-by-block scatter
def fig_blocks(W, O):
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.8))
    labs = [SHORT[s] for s in ORDER]
    for ax, (blk, name) in zip(axes.ravel(), BLOCKS):
        x = O[blk].ravel().astype(float)
        y = W[blk].ravel().astype(float)
        pos = (x > 0) & (y > 0)
        r = np.corrcoef(np.log10(x[pos]), np.log10(y[pos]))[0, 1]
        lo = max(min(x[pos].min(), y[pos].min()), 1e-1)
        hi = max(x[pos].max(), y[pos].max())
        g = np.array([lo * .7, hi * 1.4])
        ax.fill_between(g, g * .8, g * 1.2, color=C_NEG, alpha=.12, zorder=0,
                        label="$\\pm 20\\%$")
        ax.plot(g, g, color=C_NEG, lw=1.2, ls="--", zorder=1, label="$y = x$")
        ax.scatter(x[pos], y[pos], s=14 if blk == "Z" else 26, c=C_POS,
                   alpha=.55 if blk == "Z" else .8, edgecolor="none", zorder=2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(*g); ax.set_ylim(*g)
        ax.set_aspect("equal", adjustable="box")
        # label the sectors that fall outside +-25%, at most five, staggered
        if blk != "Z":
            dev = np.abs(np.log10(np.where(pos, y, 1) / np.where(pos, x, 1)))
            dev[~pos] = 0
            cand = [i for i in np.argsort(dev)[::-1][:5] if dev[i] > np.log10(1.25)]
            offs = [(6, 4), (6, -10), (-6, 5), (-6, -11), (6, 12)]
            for rank, i in enumerate(cand):
                dx, dy = offs[rank % len(offs)]
                ax.annotate(labs[i], (x[i], y[i]), textcoords="offset points",
                            xytext=(dx, dy), fontsize=6.2, color="#7a3c19",
                            ha="left" if dx > 0 else "right")
        nneg = int((~pos).sum())
        note = f"$r$ = {r:.3f}\n$\\Sigma$W/$\\Sigma$O = {y.sum()/x.sum():.2f}"
        if nneg:
            note += f"\n{nneg} non-positive pair{'s' if nneg > 1 else ''} hidden"
        ax.text(.035, .965, note, transform=ax.transAxes, va="top", ha="left",
                fontsize=7.5, linespacing=1.4)
        ax.set_title(f"{blk} --- {name}", fontsize=10, fontweight="bold", loc="left")
        ax.set_xlabel("OECD ICIO (M USD, log)", fontsize=8)
        ax.set_ylabel("WiNDC aggregate (M USD, log)", fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.grid(ls=":", alpha=.35)
    axes[0, 0].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.suptitle("United States block, 2017, thirty-six common sectors: the two sources "
                 "account by account", fontsize=12.5, fontweight="bold", y=.985)
    plt.tight_layout(rect=(0, 0, 1, .975))
    save(fig, "oecd_windc_blocks_comparison.png", 220)


# ──────────────────────────────────────── 2. divergence before / after, per block
def fig_divergence(W, H, O):
    keys = ["Z", "F", "M", "VA", "EX", "TLS"]
    before = [div(W[k], O[k]) for k in keys]
    after = [div(H[k], O[k]) for k in keys]
    floor = 1e-17
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    y = np.arange(len(keys))[::-1]
    ax.barh(y + .19, np.maximum(before, floor), .36, color=C_POS,
            label="before harmonisation (raw WiNDC aggregate)")
    ax.barh(y - .19, np.maximum(after, floor), .36, color=C_NEG,
            label="after harmonisation (delivered)")
    ax.set_xscale("log")
    ax.set_xlim(floor, 4e4)
    notes = {"Z": "cells never fitted: H4 targets the 36 row\nand 36 column totals only",
             "TLS": "passed through unscaled at H6: the two tax\naccounts are not the "
                    "same object (see text)",
             "EX": "residual of 121 M USD on real estate,\n0.006 per cent of the account"}
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(b * 1.5, y[i] + .19, f"{b:.3f}", va="center", fontsize=8, color="#7a3c19")
        txt = f"{a:.3f}" if a > 1e-3 else f"{a:.0e}"
        ax.text(max(a, floor) * 1.5, y[i] - .19, txt, va="center", fontsize=8,
                color="#23506f")
        if keys[i] in notes:
            ax.text(2.2e2, y[i], notes[keys[i]], va="center", fontsize=7,
                    color="#4a4a4a", linespacing=1.3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{k}\n{n}" for k, n in
                        [(k, dict(BLOCKS)[k]) for k in keys]], fontsize=8)
    ax.axvline(1, color="k", lw=.7, ls=":", alpha=.6)
    ax.axvline(1.4e2, color="k", lw=.5, alpha=.35)
    ax.set_xticks([1e-16, 1e-12, 1e-8, 1e-4, 1e0])
    ax.set_xlabel("relative Frobenius divergence to the OECD target, "
                  "$\\|w-o\\|_F/\\|o\\|_F$ (log scale)", fontsize=9)
    ax.set_title("What the harmonisation closes, and what it leaves open",
                 fontsize=11, fontweight="bold", loc="left", pad=24)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.005),
              ncol=2)
    ax.grid(axis="x", ls=":", alpha=.4)
    plt.tight_layout()
    save(fig, "frobenius_relative_divergence_oecd_windc.png", 220)
    return keys, before, after


# ─────────────────────────────────────────────────── 3. the vector multipliers
def fig_vector_mult(sf, W, O):
    fig, ax = plt.subplots(1, 2, figsize=(15.5, 5.6),
                           gridspec_kw=dict(width_ratios=[1.65, 1]))
    labs = [SHORT[s] for s in ORDER]
    x = np.arange(len(ORDER))
    style = dict(VA=("o", "#3577a8"), EX=("s", "#bd5e2c"),
                 M=("^", "#4c9a52"), F=("D", "#8a5aa8"))
    # A multiplier that H5 overrides by a uniform injection is never applied to the
    # delivered table, and is therefore not plotted: this is the case for every sector
    # whose sub-national base is zero while the target is positive.  In 2017 exactly one
    # such value exists, the final demand of oil and gas extraction.
    dropped = {k: (np.abs(W[k]) < 1e-6) & (np.abs(O[k]) > 1e-6) for k in style}
    for k, (mk, c) in style.items():
        v = np.where(dropped[k], np.nan, sf[k])
        ax[0].plot(x, v, mk, ms=3.4, color=c, label=f"$\\sigma^{{{k}}}$", alpha=.9,
                   lw=.7, ls="-")
    ax[0].axhline(1, color="k", lw=.8, ls="--")
    ax[0].axhspan(.5, 2, color="k", alpha=.05, lw=0)
    ax[0].set_yscale("log")
    ax[0].set_ylim(6e-2, 3.4e2)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labs, rotation=90, fontsize=6)
    ax[0].set_ylabel("multiplier $\\sigma^{V}_{j}=V^{O}_{j}/\\bar V_{j}$ (log)", fontsize=9)
    ax[0].set_title("a   per-sector multipliers of the vector accounts",
                    fontsize=10.5, fontweight="bold", loc="left")
    ax[0].legend(frameon=False, ncol=4, fontsize=9, loc="lower left")
    ax[0].grid(ls=":", alpha=.35)
    ax[0].tick_params(labelsize=7.5)
    for k, (_, c) in style.items():
        for i in np.where(dropped[k])[0]:
            ax[0].axvline(i, color=c, lw=.7, ls=":", alpha=.8)
            ax[0].annotate(f"no $\\sigma^{{{k}}}$ here: the sub-national base is zero,\n"
                           "so H5 injects the target instead of scaling it",
                           xy=(i + .4, 1.4e2), xytext=(i + 3.0, 1.9e2), fontsize=6.8,
                           color=c, va="center", ha="left",
                           arrowprops=dict(arrowstyle="->", color=c, lw=.8))

    # right: multiplier against the share of the OECD account the sector carries
    for k, (mk, c) in style.items():
        w = O[k] / O[k].sum()
        keep = ~dropped[k]
        ax[1].scatter(np.clip(w[keep], 1e-6, None), sf[k][keep], s=22, marker=mk,
                      color=c, alpha=.8, edgecolor="none", label=f"$\\sigma^{{{k}}}$")
    ax[1].axhline(1, color="k", lw=.8, ls="--")
    ax[1].axhspan(.5, 2, color="k", alpha=.05, lw=0)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_ylim(6e-2, 3.4e2)
    ax[1].set_xlabel("share of the OECD account carried by the sector (log)", fontsize=9)
    ax[1].set_ylabel("multiplier (log)", fontsize=9)
    ax[1].set_title("b   the large multipliers are the service exports",
                    fontsize=10.5, fontweight="bold", loc="left")
    ax[1].legend(frameon=False, fontsize=8, loc="lower left", ncol=2)
    ax[1].grid(ls=":", alpha=.35)
    ax[1].tick_params(labelsize=7.5)
    offs = [(6, 4), (6, -10), (-6, 5), (-7, -10)]
    seen = 0
    for k in ("EX", "F"):
        for i in np.argsort(np.where(dropped[k], -1, sf[k]))[::-1][:4]:
            if sf[k][i] < 3:
                continue
            dx, dy = offs[seen % len(offs)]; seen += 1
            ax[1].annotate(SHORT[ORDER[i]], (max(O[k][i] / O[k].sum(), 1e-6), sf[k][i]),
                           textcoords="offset points", xytext=(dx, dy), fontsize=6.2,
                           color="#7a3c19", ha="left" if dx > 0 else "right")
    plt.tight_layout()
    save(fig, "vector_multipliers_harmonization.png", 220)


# ───────────────────────────────────────────────────── 4. the cell multipliers
def fig_Z_mult(sfZ, O):
    Zo = O["Z"]
    iE = ORDER.index("Education and public administration")
    pos = sfZ > 0
    gov = np.zeros_like(pos); gov[iE, :] = True; gov &= pos
    rest = pos & ~gov
    C_GOV = "#8a5aa8"

    fig, ax = plt.subplots(1, 2, figsize=(14.2, 5.2))
    bins = np.logspace(np.log10(sfZ[pos].min()), np.log10(sfZ[pos].max()), 46)
    ax[0].hist([sfZ[rest], sfZ[gov]], bins=bins, stacked=True, color=[C_NEG, C_GOV],
               alpha=.95, label=["all other rows",
                                 "supply row of education and\npublic administration"])
    ax[0].set_xscale("log")
    p5, med, p95 = np.percentile(sfZ[rest], [5, 50, 95])
    for q, c, ls in [(med, C_POS, "-"), (p5, "k", ":"), (p95, "k", ":")]:
        ax[0].axvline(q, color=c, lw=1.2, ls=ls)
    ax[0].axvline(1, color="k", lw=.9, ls="--")
    top = ax[0].get_ylim()[1]
    ax[0].text(med * 1.07, top * .96, f"median {med:.2f}", fontsize=8, color=C_POS)
    ax[0].text(p5 * .93, top * .60, f"p5\n{p5:.2f}", fontsize=7.5, ha="right")
    ax[0].text(p95 * 1.08, top * .60, f"p95\n{p95:.2f}", fontsize=7.5)
    ax[0].set_xlabel("$\\sigma^{Z}_{jj\'}$, positive cells (log)", fontsize=9)
    ax[0].set_ylabel("number of cells", fontsize=9)
    ax[0].set_title(f"a   {int(pos.sum())} positive cells, {int((sfZ==0).sum())} "
                    "structural zeros left at zero", fontsize=10.5, fontweight="bold",
                    loc="left")
    ax[0].legend(frameon=False, fontsize=7.5, loc="upper right")
    ax[0].grid(ls=":", alpha=.35)
    ax[0].tick_params(labelsize=8)

    ax[1].scatter(np.clip(Zo[rest], 1e-2, None), sfZ[rest], s=9, c=C_POS, alpha=.45,
                  edgecolor="none", label="all other rows")
    ax[1].scatter(np.clip(Zo[gov], 1e-2, None), sfZ[gov], s=17, c=C_GOV, alpha=.85,
                  edgecolor="none",
                  label="education and public administration, as a supplier")
    ax[1].axhline(1, color="k", lw=.9, ls="--")
    ax[1].axhspan(.5, 2, color="k", alpha=.06, lw=0)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("target cell of the OECD block, M USD (log)", fontsize=9)
    ax[1].set_ylabel("$\\sigma^{Z}_{jj\'}$ (log)", fontsize=9)
    ax[1].set_title(f"b   {((sfZ[rest]>=.5)&(sfZ[rest]<=2)).mean():.0%} of the "
                    "non-government cells lie within a factor of two",
                    fontsize=10.5, fontweight="bold", loc="left")
    jmin = int(np.argmin(np.where(gov, Zo, np.inf).ravel()))
    ax[1].annotate("one row, rescaled by 12 on average:\nthe WiNDC accounts deliver 1.7 "
                   "per cent\nof this bucket\'s recorded deliveries to\nintermediate use, the "
                   "OECD table 16.3 per cent",
                   xy=(float(Zo.ravel()[jmin]), float(sfZ.ravel()[jmin])),
                   xytext=(1.1, 3.4), fontsize=7, color=C_GOV, va="center", ha="left",
                   arrowprops=dict(arrowstyle="->", color=C_GOV, lw=.8,
                                   connectionstyle="arc3,rad=-.2"))
    ax[1].legend(frameon=False, fontsize=7.5, loc="lower left")
    ax[1].grid(ls=":", alpha=.35)
    ax[1].tick_params(labelsize=8)
    plt.tight_layout()
    save(fig, "Z_multipliers_harmonization.png", 220)


def save(fig, name, dpi):
    fig.savefig(PAPER_FIG / name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("wrote", PAPER_FIG / name)


# ─────────────────────────────────────────────────────────── numbers and table
def report(W, H, O, sf, sfZ):
    print("\n===== sigma^V summary =====")
    for k in ("VA", "EX", "M", "F"):
        v = sf[k]
        fin = v[v < 1e6]
        print(f" {k:3s} p5={np.percentile(fin,5):7.3f} med={np.median(fin):7.3f} "
              f"p95={np.percentile(fin,95):8.3f} min={fin.min():.3f} max={fin.max():.2f} "
              f"| in [0.5,2]: {int(((v>=.5)&(v<=2)).sum())}/{len(v)} "
              f"| account weight of the sectors outside [0.5,2]: "
              f"{O[k][(v<.5)|(v>2)].sum()/O[k].sum():.1%}")
    pos = sfZ[sfZ > 0]
    print(f" Z   p5={np.percentile(pos,5):.3f} med={np.median(pos):.3f} "
          f"p95={np.percentile(pos,95):.3f} min={pos.min():.3f} max={pos.max():.2f} "
          f"| in [0.5,2]: {int(((pos>=.5)&(pos<=2)).sum())}/{pos.size} "
          f"| zeros {int((sfZ==0).sum())}")

    # The multiplier table of the manuscript, as data. Typesetting is left to the
    # manuscript: nothing here emits markup.
    import csv
    rows = [{"sector": SHORT[s],
             "oecd_Z_row_total_bn": round(O["Z"][i, :].sum() / 1e3, 1),
             "sigma_V": round(float(sf["VA"][i]), 3),
             "sigma_EX": round(float(sf["EX"][i]), 3),
             "sigma_M": round(float(sf["M"][i]), 3),
             # a sub-national sector with no counterpart in the global block has an
             # unbounded final-demand multiplier; it is reported as empty, not as a number
             "sigma_F": "" if sf["F"][i] > 1e6 else round(float(sf["F"][i]), 3)}
            for i, s in enumerate(ORDER)]
    out = PAPER_FIG / "harmonisation_multipliers.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\n===== multiplier table -> {out.name} ({len(rows)} sectors) =====")
    for r in rows[:5]:
        print("  " + "  ".join(f"{k}={v}" for k, v in r.items()))
    print("  ...")
    print("\n===== tax decomposition (M USD) =====")
    print(f" OECD TLS on industry columns      {O['TLS'].sum():12,.0f}")
    print(f" OECD TLS on final-demand columns  {O['TLS_fd'].sum():12,.0f}")
    print(f" WiNDC TLS total                   {W['TLS'].sum():12,.0f}")
    for k in ("tls_int", "tls_fd", "tariff", "tax_prod"):
        if k in W:
            print(f"   {k:10s}                      {W[k].sum():12,.0f}")
    print(f" div(TLS)  {div(W['TLS'],O['TLS']):.3f}   "
          f"div(tls_int) {div(W['tls_int'],O['TLS']):.3f}")
    print(f" negative WiNDC TLS sectors: "
          f"{[(SHORT[ORDER[i]], round(W['TLS'][i]/1e3,1)) for i in np.where(W['TLS']<0)[0]]}")


def main():
    W, H, O, sf, sfZ = load_all()
    fig_blocks(W, O)
    fig_divergence(W, H, O)
    fig_vector_mult(sf, W, O)
    fig_Z_mult(sfZ, O)
    report(W, H, O, sf, sfZ)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    main()
