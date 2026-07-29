"""Figure — layout of a delivered file.

**a**, The whole delivered table on a logarithmic scale of the absolute value in
millions of dollars: the world-by-world block at the top left is the aggregated
global source, carried over unchanged; the state rows and columns carry the
disaggregated United States; the final demand block and the output column close the
accounts.

**b**, The intra-United-States block, 1 887 x 1 887, organised as 51 x 51 state tiles
of 37 x 37 sectors. The dark diagonal band is intra-state trade; the row and column
bands of the large state economies reflect the estimated bilateral structure.

Reads one delivered file. Run on a compute node: about 16 GB.

Writes ``figures/fig1_nested_layout.png``.

Run from anywhere:  python pipeline/plots/fig04_nested_layout.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from _figpaths import FIG_DIR, NESTED_DIR

YEAR = 2017
FD_CATS = {"DPABR", "GFCF", "GGFC", "HFCE", "INVNT", "NPISH"}
EXTRA_ROWS = ["OUT", "TLS", "VA"]
INK, MUTED = "#1a1a1a", "#5b5b5b"
C_MARK = "#bd5e2c"


def is_state(l):
    return "_" in l and len(l.split("_")[0]) == 2


def is_world(l):
    return "_" in l and len(l.split("_")[0]) == 3


def block_edges(ax, x, y, w, h, label, fs=7.5):
    """Outline one block and label it inside its top-left corner, on a white patch
    so the label stays legible over the densest part of the map."""
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor=C_MARK,
                               lw=1.1, zorder=5))
    ax.text(x + w * .02, y + h * .045, label, color=C_MARK, fontsize=fs,
            ha="left", va="top", zorder=6,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=.82))


def main():
    path = NESTED_DIR / f"nested_mriot_{YEAR}.parquet"
    print("reading", path)
    df = pd.read_parquet(path)
    print("shape", df.shape)

    rows = list(df.index)
    cols = list(df.columns)
    ind_rows = [r for r in rows if r not in EXTRA_ROWS]
    n_world_r = sum(1 for r in ind_rows if is_world(r))
    n_state_r = sum(1 for r in ind_rows if is_state(r))
    world_sec = [c for c in cols if is_world(c) and c.split("_", 1)[1] not in FD_CATS]
    state_sec = [c for c in cols if is_state(c) and c.split("_", 1)[1] not in FD_CATS]
    world_fd = [c for c in cols if is_world(c) and c.split("_", 1)[1] in FD_CATS]
    state_fd = [c for c in cols if is_state(c) and c.split("_", 1)[1] in FD_CATS]
    n_ws, n_ss, n_wf, n_sf = map(len, (world_sec, state_sec, world_fd, state_fd))
    print(f"world rows {n_world_r} | state rows {n_state_r} | "
          f"cols: world sec {n_ws}, state sec {n_ss}, world fd {n_wf}, state fd {n_sf}")

    M = np.abs(df.values.astype(np.float32))
    pos = M[M > 0]
    norm = LogNorm(vmin=max(np.percentile(pos, 1), 1e-3), vmax=pos.max())

    fig = plt.figure(figsize=(15.0, 7.4))
    axA = fig.add_axes([.045, .100, .460, .820])
    axB = fig.add_axes([.560, .100, .360, .820])
    caxA = fig.add_axes([.512, .250, .009, .520])
    caxB = fig.add_axes([.928, .250, .009, .520])

    # ---- panel a: the whole file ---------------------------------------------
    imA = axA.imshow(np.where(M > 0, M, np.nan), cmap="magma_r", norm=norm,
                     aspect="auto", interpolation="nearest")
    nr, nc = M.shape
    for x in (n_ws, n_ws + n_ss, n_ws + n_ss + n_wf, n_ws + n_ss + n_wf + n_sf):
        axA.axvline(x - .5, color=INK, lw=.7, alpha=.65)
    for y in (n_world_r, n_world_r + n_state_r):
        axA.axhline(y - .5, color=INK, lw=.7, alpha=.65)
    block_edges(axA, -.5, -.5, n_ws, n_world_r, "world x world")
    block_edges(axA, n_ws - .5, n_world_r - .5, n_ss, n_state_r, "intra-US  (panel b)")
    block_edges(axA, n_ws + n_ss - .5, -.5, n_wf + n_sf, nr, "final demand")
    # column and row groups are named outside the axes: five of them are narrow and
    # would collide as tick labels
    axA.set_xticks([]); axA.set_yticks([])
    groups = [(n_ws / 2, "world sectors"), (n_ws + n_ss / 2, "state sectors"),
              (n_ws + n_ss + n_wf / 2, "world\nfinal demand"),
              (n_ws + n_ss + n_wf + n_sf / 2, "state\nfinal demand"),
              (nc - 1, "OUT")]
    for xc, lab in groups:
        axA.text(xc / nc, -.012, lab, transform=axA.transAxes, ha="center", va="top",
                 fontsize=7.2, color=MUTED, linespacing=1.25)
    for yc, lab in ((n_world_r / 2, "world\nsectors"),
                    (n_world_r + n_state_r / 2, "state\nsectors"),
                    (nr - 1.5, "OUT/TLS/VA")):
        axA.text(-.012, 1 - yc / nr, lab, transform=axA.transAxes, ha="right",
                 va="center", fontsize=7.2, color=MUTED, linespacing=1.25)
    axA.set_title(f"a   the delivered file, {YEAR}   ({nr:,} x {nc:,})",
                  fontsize=10.5, fontweight="bold", loc="left", pad=8)
    cbA = fig.colorbar(imA, cax=caxA)
    cbA.set_label("|value|, million USD", fontsize=8, labelpad=3)
    cbA.ax.tick_params(labelsize=7); cbA.outline.set_linewidth(.5)

    # ---- panel b: the intra-US block ----------------------------------------
    B = df.loc[[r for r in ind_rows if is_state(r)], state_sec].values.astype(np.float32)
    posB = B[B > 0]
    imB = axB.imshow(np.where(B > 0, B, np.nan), cmap="magma_r",
                     norm=LogNorm(vmin=max(np.percentile(posB, 1), 1e-3),
                                  vmax=posB.max()),
                     aspect="equal", interpolation="nearest")
    states = sorted({c.split("_")[0] for c in state_sec})
    n_sec = len(state_sec) // len(states)
    for k in range(1, len(states)):
        axB.axhline(k * n_sec - .5, color=INK, lw=.25, alpha=.30)
        axB.axvline(k * n_sec - .5, color=INK, lw=.25, alpha=.30)
    ticks = [k * n_sec + n_sec / 2 for k in range(len(states))]
    axB.set_xticks(ticks); axB.set_xticklabels(states, rotation=90, fontsize=4.6)
    axB.set_yticks(ticks); axB.set_yticklabels(states, fontsize=4.6)
    axB.tick_params(length=1.2, pad=1)
    axB.set_title(f"b   the intra-United-States block   "
                  f"({B.shape[0]:,} x {B.shape[1]:,} = 51 x 51 tiles of {n_sec} sectors)",
                  fontsize=10.5, fontweight="bold", loc="left", pad=8)
    cbB = fig.colorbar(imB, cax=caxB)
    cbB.set_label("|value|, million USD", fontsize=8, labelpad=3)
    cbB.ax.tick_params(labelsize=7); cbB.outline.set_linewidth(.5)

    out = FIG_DIR / "fig1_nested_layout.png"
    fig.savefig(out, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    main()
