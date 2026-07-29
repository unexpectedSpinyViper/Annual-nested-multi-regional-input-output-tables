"""Figure 1 — workflow of the construction.

Schematic, no data input: input datasets on the left, the five processing stages
and the closure step as a vertical sequence in the centre, the delivered series at
the bottom. A double border marks a step whose output is checked against an
accounting identity, the check being stated on the right. The dashed line marks the
global table, used twice: once as one of the two sources aggregated onto the common
classification, once as the frame onto which the sub-national table is harmonised
and into which it is nested.

Writes ``figures/figure1_workflow.png``.

Run from anywhere:  python pipeline/plots/fig01_workflow.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _figpaths import FIG_DIR

# Palette: one hue per role, checked for separation under the three common forms of
# colour-vision deficiency.
C_INPUT = "#b9c6d2"      # third-party inputs
C_STAGE = "#dce6ee"      # processing stages
C_GLOBAL = "#f0dcc8"     # the global table, which enters twice
C_OUT = "#c5d8c5"        # delivered series
INK, MUTED = "#1a1a1a", "#5b5b5b"

# ── the content ───────────────────────────────────────────────────────────────
INPUTS = [
    ("WiNDC sub-national\naccounts (GDX)", "v4.1, 51 regions x 71 commodities"),
    ("OECD ICIO\nworld table", "2025 edition, 1995-2022"),
    ("BEA SAGDP2\nstate GDP by sector", "production share $S$"),
    ("BEA CAGDP2 + Census\ncounty centroids", "GDP-weighted economic centroids"),
]

# (label, detail, checked-against-an-identity)
STAGES = [
    ("1  Densification and identity verification",
     "sub-national accounts completed; SAM identities re-derived", True),
    ("2  Bilateral reconstruction",
     "two-layer (trade + margin) doubly constrained gravity, $d^{-\\gamma}$, RAS", True),
    ("3  Allocation to uses and assembly",
     "flows routed to intermediate and final use; intra-US table assembled", True),
    ("4  Common sector classification",
     "both sources aggregated onto the 37 common sectors", False),
    ("5  Harmonisation to the global United States block",
     "account-by-account multipliers $\\sigma^{V}$, $\\sigma^{Z}$", True),
    ("6  Nesting and residual closure",
     "world$\\leftrightarrow$US flows split by $S$ and $\\Theta$; state columns closed on VA", True),
]

CHECKS = [
    "zero-profit, market clearance\nand income balance",
    "national commodity pool closes\nexactly, all 71 commodities",
    "row and column balance of the\nintra-US table",
    "",
    "relative Frobenius divergence\nto each OECD account",
    "row and column balance;\nnon-US block bit-identical",
]

OUTPUT = ("26 annual nested MRIOTs   $\\bf{nested\\_mriot\\_<year>.parquet}$, 1997-2022",
          "4 850 x 5 634 — 51 US states x 37 sectors alongside 76 national economies")


def box(ax, x, y, w, h, text, sub, face, double=False, dashed=False, fs=8.2):
    """One rounded box; a double border marks an identity-checked step."""
    style = "round,pad=0.012,rounding_size=0.014"
    if double:                                   # outer line of the double border
        ax.add_patch(FancyBboxPatch((x - .006, y - .006), w + .012, h + .012,
                                    boxstyle=style, linewidth=.9, edgecolor=INK,
                                    facecolor="none", zorder=2))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=.9,
                                edgecolor=INK, facecolor=face, zorder=3,
                                linestyle=(0, (4, 2)) if dashed else "solid"))
    ax.text(x + w / 2, y + h * (.62 if sub else .5), text, ha="center", va="center",
            fontsize=fs, color=INK, zorder=4, linespacing=1.25)
    if sub:
        ax.text(x + w / 2, y + h * .245, sub, ha="center", va="center", fontsize=6.4,
                color=MUTED, zorder=4, linespacing=1.3)


def arrow(ax, p0, p1, dashed=False, rad=0.0, lw=.9):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=9, linewidth=lw, color=MUTED,
        linestyle=(0, (4, 2)) if dashed else "solid", zorder=1,
        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=4))


def main():
    fig, ax = plt.subplots(figsize=(11.2, 9.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # geometry
    x_in, w_in = .015, .215                 # inputs column
    x_st, w_st = .295, .430                 # stages column
    x_ck = .755                             # checks column
    top, h_st, gap = .900, .083, .050       # stage boxes
    h_in = .078

    # ---- centre column: the five stages and the closure ----------------------
    ys = [top - i * (h_st + gap) for i in range(len(STAGES))]
    for (label, detail, checked), y in zip(STAGES, ys):
        box(ax, x_st, y, w_st, h_st, label, detail, C_STAGE, double=checked, fs=8.6)
    for y0, y1 in zip(ys[:-1], ys[1:]):
        arrow(ax, (x_st + w_st / 2, y0), (x_st + w_st / 2, y1 + h_st))

    # ---- left column: the third-party inputs ---------------------------------
    y_in = [.860, .752, .545, .432]
    for (label, detail), y in zip(INPUTS, y_in):
        dashed = label.startswith("OECD")
        box(ax, x_in, y, w_in, h_in, label, detail,
            C_GLOBAL if dashed else C_INPUT, dashed=dashed, fs=8.2)

    # WiNDC -> stage 1 ; centroids -> stage 2 ; SAGDP2 -> stage 6
    arrow(ax, (x_in + w_in, y_in[0] + h_in / 2), (x_st, ys[0] + h_st * .55))
    arrow(ax, (x_in + w_in, y_in[3] + h_in / 2), (x_st, ys[1] + h_st * .45), rad=.10)
    arrow(ax, (x_in + w_in, y_in[2] + h_in / 2), (x_st, ys[5] + h_st * .62), rad=-.16)

    # the global table enters twice: aggregation (4) and harmonisation/nesting (5, 6)
    arrow(ax, (x_in + w_in, y_in[1] + h_in / 2), (x_st, ys[3] + h_st * .55), dashed=True)
    arrow(ax, (x_in + w_in * .5, y_in[1]), (x_st, ys[4] + h_st * .5), dashed=True, rad=-.30)

    # ---- right column: the identity checked at each stage --------------------
    for (label, _, checked), y, chk in zip(STAGES, ys, CHECKS):
        if not checked:
            continue
        ax.annotate("", xy=(x_ck - .012, y + h_st / 2), xytext=(x_st + w_st, y + h_st / 2),
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=.7, ls=":"))
        ax.text(x_ck, y + h_st / 2, chk, ha="left", va="center", fontsize=6.6,
                color=MUTED, linespacing=1.35)

    # ---- bottom: the delivered series ----------------------------------------
    y_out = ys[-1] - .105
    box(ax, x_st - .075, y_out, w_st + .150, .072, OUTPUT[0], OUTPUT[1], C_OUT, fs=8.6)
    arrow(ax, (x_st + w_st / 2, ys[-1]), (x_st + w_st / 2, y_out + .072), lw=1.2)

    # ---- legend ---------------------------------------------------------------
    ax.text(x_in, .265, "Legend", fontsize=8, fontweight="bold", color=INK)
    box(ax, x_in, .208, .026, .030, "", None, C_STAGE, double=True)
    ax.text(x_in + .058, .223, "output checked against\nan accounting identity",
            fontsize=6.6, va="center", color=MUTED, linespacing=1.35)
    box(ax, x_in, .150, .026, .030, "", None, C_GLOBAL, dashed=True)
    ax.text(x_in + .058, .165, "the global table, used\nas source and as frame",
            fontsize=6.6, va="center", color=MUTED, linespacing=1.35)
    ax.text(x_in, .100, "All monetary values are millions of current United States dollars.",
            fontsize=6.6, color=MUTED)

    out = FIG_DIR / "figure1_workflow.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    main()
