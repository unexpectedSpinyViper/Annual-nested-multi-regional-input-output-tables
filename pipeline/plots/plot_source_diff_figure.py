"""Figure: structural distance between the OECD USA block and the WiNDC US block.

Produces `fig_source_structural_distance.png` (manuscript) and a companion
`figS_source_diff_variants.png` (Supplementary) that shows the twelve candidate
weightings of the difference heatmap that were screened before the delivered one
was chosen.

The delivered heatmap weights each cell difference by the total of the
intermediate block, so that a cell is dark in proportion to the amount of the
block's mass it displaces; relative differences computed cell by cell are not
used as the colour, because they saturate on cells that carry no mass.

Run from anywhere:  python pipeline/plots/plot_source_diff_figure.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, TwoSlopeNorm
import pyarrow.parquet as pq

from _figpaths import ROOT, IOT_INTERIM, FIG_DIR

# ─────────────────────────────────────────────────────────────── configuration
YEAR = 2017
WINDC_VERSION = "v3.1_RAS"
WINDC_SCALE = 1000.0                       # WiNDC Bn$ -> OECD M$
FD_CATS = ["DPABR", "GFCF", "GGFC", "HFCE", "INVNT", "NPISH"]

IOT_USA = IOT_INTERIM / "IOT_USA"
WINDC_RAW = IOT_USA / f"grav_fric_{WINDC_VERSION}_aggregated"
WINDC_HARM = IOT_USA / f"grav_fric_{WINDC_VERSION}_harmonized"
OECD_AGG = IOT_INTERIM / "OCDE ICIO aggregated"
PAPER_FIG = FIG_DIR

# short labels and the display order (primary, manufacturing, trade/transport,
# services), which groups the sectors the way an input-output table is usually read
SHORT = {
    "Agriculture, hunting, forestry, fishing and related": "Agriculture & fishing",
    "Mining, except oil & gas": "Mining exc. oil & gas",
    "Oil and gas extraction": "Oil & gas extraction",
    "Support activities for mining": "Mining support",
    "Manufacture of food and beverage and tobacco products": "Food, bev. & tobacco",
    "Manufacture of textiles, wearing apparel, leather and related products":
        "Textiles & apparel",
    "Manufacture of wood and of products of wood and cork": "Wood products",
    "Manufacture of paper and paper products": "Paper products",
    "Manufacture of coke and refined petroleum products": "Coke & ref. petroleum",
    "Manufacture of chemicals and chemical products, pharmaceutical":
        "Chemicals & pharma",
    "Manufacture of rubber and plastic products": "Rubber & plastics",
    "Manufacture of other non-metallic mineral products": "Non-metallic minerals",
    "primary metals": "Primary metals",
    "Manufacture of fabricated metal products": "Fabricated metal",
    "Manufacture of computer, electronic and optical products": "Computer & electronics",
    "Manufacture of electrical equipment": "Electrical equipment",
    "Manufacture of machinery and equipment n.e.c. ": "Machinery n.e.c.",
    "Manufacture of motor vehicles, trailers and semi-trailers": "Motor vehicles",
    "Manufacture of other transport equipment": "Other transport equip.",
    "Manufacture of furniture": "Furniture & other mfg",
    "Electricity, gas, steam and air conditioning supply, and wasterwater":
        "Electricity, gas & water",
    "Construction": "Construction",
    "Other services + wholesale/retail": "Other services & trade",
    "Land transport and transport via pipelines": "Land transp. & pipelines",
    "Water transport": "Water transport",
    "Air transport": "Air transport",
    "Warehousing and support activities for transportation": "Warehousing & transp. supp.",
    "Accommodation and food service activities": "Accommodation & food",
    "broadcasting, telecommunications, data processing, publishing, information "
    "services, motion picture, video, television": "Information & telecom",
    "Financial and insurance activities": "Finance & insurance",
    "Real estate activities": "Real estate",
    "Professional, scientific and technical activities": "Professional & technical",
    "Administrative and support service activities": "Administrative & support",
    "Education and public administration": "Education & public admin.",
    "Human health and social work activities": "Health & social work",
    "Arts, entertainment and recreation activities": "Arts & recreation",
}
ORDER = list(SHORT.keys())
GROUP_BREAKS = [4, 20, 27]                 # after primary, manufacturing, transport
GROUP_NAMES = ["primary", "manufacturing", "utilities, construction,\ntrade & transport",
               "services"]

C_NEG, C_POS = "#3577a8", "#bd5e2c"


# ───────────────────────────────────────────────────────────────────── loading
def oecd_usa_sectors(f):
    return [c.split("_", 1)[1] for c in pq.read_schema(f).names
            if c.startswith("USA_") and c.split("_", 1)[1] not in FD_CATS]


def pick_oecd_file(year, wd_secs):
    cands = list(OECD_AGG.rglob(f"{year}_*.parquet"))
    if not cands:
        raise FileNotFoundError(f"no OECD file for {year}")
    return max(cands, key=lambda f: len(set(oecd_usa_sectors(f)) & set(wd_secs)))


def windc_Z(npz, sect_order):
    """Sum the state x state WiNDC table over states -> national US block, M$."""
    secs = [str(s) for s in npz["proposed_sectors"]]
    nr = npz["Z"].shape[0] // len(secs)
    idx = [secs.index(s) for s in sect_order]
    Z4 = npz["Z"].reshape(nr, len(secs), nr, len(secs))
    return Z4[:, idx, :, :][:, :, :, idx].sum((0, 2)) * WINDC_SCALE


def oecd_Z(year, sect_order):
    f = pick_oecd_file(year, sect_order)
    cols = [f"USA_{s}" for s in sect_order]
    schema = pq.read_schema(f)
    idx = [c for c in (schema.pandas_metadata or {}).get("index_columns", [])
           if isinstance(c, str)]
    df = pq.read_table(f, columns=cols + idx).to_pandas()
    return df.loc[cols, cols].values.astype(float), f.relative_to(ROOT)


def load(year=YEAR):
    npz_raw = np.load(WINDC_RAW / f"IOT_{year}.npz", allow_pickle=True)
    npz_harm = np.load(WINDC_HARM / f"IOT_{year}_harmonized.npz", allow_pickle=True)
    sectors = [str(s) for s in npz_raw["proposed_sectors"]]
    missing = set(ORDER) ^ set(sectors)
    if missing:
        raise RuntimeError(f"sector list mismatch: {missing}")
    Zw = windc_Z(npz_raw, ORDER)
    Zh = windc_Z(npz_harm, ORDER)
    Zo, src = oecd_Z(year, ORDER)
    return Zo, Zw, Zh, src


# ────────────────────────────────────────────────────────────────── statistics
def stats(W, O):
    D, aD, tot = W - O, np.abs(W - O), O.sum()
    order = np.argsort(aD.ravel())[::-1]
    cum = np.cumsum(aD.ravel()[order]) / aD.sum()
    rel = D / O
    return dict(
        div=np.linalg.norm(D) / np.linalg.norm(O),
        L1=aD.sum() / tot, misplaced=aD.sum() / tot / 2, absD=aD.sum(),
        n50=int(np.searchsorted(cum, .50) + 1), cum=cum,
        top20=cum[19], top100=cum[99], maxcell=aD.max(),
        mass_in={t: O[np.abs(rel) <= t].sum() / tot for t in (.10, .25, .50, 1.0)},
        cells_in={t: int((np.abs(rel) <= t).sum()) for t in (.10, .25, .50, 1.0)},
        rel=rel, D=D)


def mass_curve(W, O, grid):
    """Share of the OECD block mass held by cells whose |relative deviation| <= x."""
    rel = np.abs((W - O) / O).ravel()
    m = O.ravel()
    return np.array([m[rel <= x].sum() / m.sum() for x in grid])


# ─────────────────────────────────────────────────────────────────── the panel
def heat(ax, M, norm, cmap="RdBu_r", labels=True, fs=4.6):
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")
    n = M.shape[0]
    for b in GROUP_BREAKS:
        ax.axhline(b - .5, color="k", lw=.6, alpha=.55)
        ax.axvline(b - .5, color="k", lw=.6, alpha=.55)
    labs = [SHORT[s] for s in ORDER]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labs if labels else [], rotation=90, fontsize=fs)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labs if labels else [], fontsize=fs)
    ax.tick_params(length=1.5, pad=1)
    for s in ax.spines.values():
        s.set_linewidth(.6)
    return im


def main():
    Zo, Zw, Zh, src = load()
    n = len(ORDER)
    tot = Zo.sum()
    sw, sh = stats(Zw, Zo), stats(Zh, Zo)
    print(f"OECD source : {src}")
    print(f"raw       div={sw['div']:.3f}  L1={sw['L1']:.3f}  misplaced={sw['misplaced']:.3f}"
          f"  |D|={sw['absD']/1e3:,.0f} bn$")
    print(f"delivered div={sh['div']:.3f}  L1={sh['L1']:.3f}  misplaced={sh['misplaced']:.3f}"
          f"  |D|={sh['absD']/1e3:,.0f} bn$")

    # difference in per mille of the intermediate block total
    Praw, Pdel = 1e3 * (Zw - Zo) / tot, 1e3 * (Zh - Zo) / tot
    vmax = float(np.abs(Praw).max())
    norm = SymLogNorm(linthresh=.2, linscale=.55, vmin=-vmax, vmax=vmax, base=10)
    ticks = [-20, -5, -1, -.2, 0, .2, 1, 5, 20]

    fig = plt.figure(figsize=(13.6, 9.2))
    #   explicit positions: the two maps are exactly square, so that the shared
    #   colour bar and the lower row can be placed without colliding with the
    #   rotated tick labels
    axA = fig.add_axes([.075, .400, .380, .5617])
    axB = fig.add_axes([.505, .400, .380, .5617])
    axC = fig.add_axes([.075, .050, .345, .190])
    axD = fig.add_axes([.565, .050, .345, .190])

    heat(axA, Praw, norm)
    imB = heat(axB, Pdel, norm)
    axB.set_yticklabels([])
    axA.set_title(f"a   WiNDC (raw) $-$ OECD,   $d_F$ = {sw['div']:.3f}",
                  fontsize=10.5, fontweight="bold", loc="left", pad=6)
    axB.set_title(f"b   delivered (harmonised) $-$ OECD,   $d_F$ = {sh['div']:.3f}",
                  fontsize=10.5, fontweight="bold", loc="left", pad=6)
    axA.set_ylabel("supplying sector", fontsize=8.5, labelpad=2)

    cax = fig.add_axes([.902, .490, .011, .360])
    cb = fig.colorbar(imB, cax=cax, orientation="vertical", ticks=ticks)
    cb.set_label("cell difference, per mille of the\nUnited States intermediate block "
                 "(13.7 tn USD)", fontsize=8, labelpad=4)
    cb.ax.set_yticklabels([f"{t:+g}" if t else "0" for t in ticks], fontsize=7.5)
    cb.outline.set_linewidth(.5)

    # ---- panel c: block mass held by cells agreeing within a given tolerance
    grid = np.logspace(np.log10(.01), np.log10(20), 260)
    axC.semilogx(100 * grid, 100 * mass_curve(Zw, Zo, grid), color=C_POS, lw=1.8,
                 label="WiNDC (raw) $-$ OECD")
    axC.semilogx(100 * grid, 100 * mass_curve(Zh, Zo, grid), color=C_NEG, lw=1.8,
                 label="delivered $-$ OECD")
    for t, c, s in [(.25, "k", "-"), (1.0, "k", ":")]:
        axC.axvline(100 * t, color=c, lw=.7, ls=s, alpha=.5)
    axC.annotate(f"{100*sh['mass_in'][.25]:.0f}% of the mass",
                 xy=(25, 100 * sh["mass_in"][.25]), xytext=(1.25, 92), fontsize=8,
                 color=C_NEG, va="center",
                 arrowprops=dict(arrowstyle="-", color=C_NEG, lw=.7,
                                 connectionstyle="arc3,rad=-.15"))
    axC.annotate(f"{100*sw['mass_in'][.25]:.0f}%", xy=(25, 100 * sw["mass_in"][.25]),
                 xytext=(1.25, 78), fontsize=8, color=C_POS, va="center",
                 arrowprops=dict(arrowstyle="-", color=C_POS, lw=.7,
                                 connectionstyle="arc3,rad=-.15"))
    axC.set_xlabel("tolerance on the cell relative deviation (%, log scale)",
                   fontsize=8.5)
    axC.set_ylabel("share of the OECD block mass\nheld by cells within the tolerance (%)",
                   fontsize=8.5)
    axC.set_title("c   most of the mass sits in cells that agree closely",
                  fontsize=10.5, fontweight="bold", loc="left", pad=6)
    axC.set_ylim(0, 101)
    axC.set_xticks([1, 10, 25, 100, 1000])
    axC.set_xticklabels(["1", "10", "25", "100", "1000"], fontsize=8)
    axC.tick_params(labelsize=8)
    axC.grid(ls=":", alpha=.4)
    axC.legend(frameon=False, fontsize=8, loc="lower right")

    # ---- panel d: per-sector contribution to the total absolute discrepancy
    aD = np.abs(Zh - Zo)
    contrib = (aD.sum(1) + aD.sum(0)) / (2 * aD.sum())
    share = (Zo.sum(1) + Zo.sum(0)) / (2 * tot)
    ownrel = (aD.sum(1) + aD.sum(0)) / (Zo.sum(1) + Zo.sum(0))
    k = 12
    top = np.argsort(contrib)[::-1][:k][::-1]
    y = np.arange(k)
    axD.barh(y + .19, 100 * contrib[top], .38, color=C_POS,
             label="share of the total discrepancy")
    axD.barh(y - .19, 100 * share[top], .38, color="#b9c6d2",
             label="share of the block")
    for i, s in enumerate(top):
        axD.text(100 * contrib[s] + .35, y[i] + .19, f"{100*ownrel[s]:.0f}%",
                 va="center", fontsize=6.5, color="#7a3c19")
    axD.set_yticks(y)
    axD.set_yticklabels([SHORT[ORDER[s]] for s in top], fontsize=7)
    axD.set_xlabel("per cent of the intermediate block / of the total discrepancy",
                   fontsize=8.5)
    axD.set_title("d   where the residual discrepancy sits, by sector",
                  fontsize=10.5, fontweight="bold", loc="left", pad=6)
    axD.tick_params(labelsize=8)
    axD.grid(axis="x", ls=":", alpha=.4)
    axD.legend(frameon=False, fontsize=7.5, loc="lower right")
    axD.text(.985, .40, "labels: sector's own\nrelative misfit", transform=axD.transAxes,
             ha="right", fontsize=6.5, color="#7a3c19")

    out = PAPER_FIG / "fig_source_structural_distance.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)

    # ---- companion: the twelve screened weightings ------------------------
    variants_figure(Zo, Zh)


def variants_figure(O, W):
    D, tot = W - O, O.sum()
    ro, co, eps = O.sum(1), O.sum(0), 1e-12
    V = {
        "a  absolute difference (M USD)": D,
        "b  relative to the OECD cell": np.where(O > 10, D / (O + eps), np.nan),
        "c  relative to the OECD row total": D / ro[:, None],
        "d  relative to the OECD column total": D / co[None, :],
        "e  per mille of the block total  (retained)": 1e3 * D / tot,
        "f  share difference $W/\\Sigma W-O/\\Sigma O$ (pp)":
            100 * (W / W.sum() - O / O.sum()),
        "g  symmetric $2D/(W+O)$": np.where(W + O > 0, 2 * D / (W + O + eps), np.nan),
        "h  relative difference $\\times\\sqrt{\\rm cell\\ mass\\ share}$":
            np.where(O > 0, D / (O + eps), 0) * np.sqrt(O / tot),
        "i  signed share of the total |D| (%)": 100 * D / np.abs(D).sum(),
        "j  $\\log_2(W/O)$, cells above 0.01% of the block":
            np.where((O > 1e-4 * tot) & (W > 0), np.log2((W + eps) / (O + eps)), np.nan),
        "k  relative to $\\sqrt{\\rm row\\times col}$": D / np.sqrt(ro[:, None] * co[None, :]),
        "l  relative to the larger of row and column": D / np.maximum(ro[:, None],
                                                                     co[None, :]),
    }
    fig, axes = plt.subplots(3, 4, figsize=(21, 16))
    for ax, (name, M) in zip(axes.ravel(), V.items()):
        fin = M[np.isfinite(M)]
        v = float(np.percentile(np.abs(fin), 97)) or 1.0
        im = heat(ax, M, TwoSlopeNorm(vmin=-v, vcenter=0, vmax=v), labels=True, fs=3.2)
        ax.set_title(name, fontsize=8.5, fontweight="bold")
        cb = plt.colorbar(im, ax=ax, shrink=.72, pad=.02)
        cb.ax.tick_params(labelsize=6)
    fig.suptitle("Candidate weightings of the difference heatmap, delivered block "
                 "$-$ OECD, 2017 (colour clipped at the 97th percentile of $|$value$|$)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = PAPER_FIG / "figS_source_diff_variants.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "axes.titlesize": 10, "text.usetex": False})
    main()
