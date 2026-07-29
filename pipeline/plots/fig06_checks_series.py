"""Figure — the accounting checks across the delivered series.

**a**, Balance residuals of the twenty-six delivered files, logarithmic scale. World
rows and columns sit at the native residual level of the published global source,
which the construction carries over rather than corrects; state rows and columns
balance at machine precision.

**b**, Weight of the fifty-one-region block in world gross output across the series.

Streams the 26 delivered files one at a time. Run on a compute node: about 24 GB.

Writes ``figures/fig6_checks_series.png`` and the per-year table
``figures/checks_series.csv``.

Run from anywhere:  python pipeline/plots/fig06_checks_series.py
"""
import gc

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _figpaths import FIG_DIR, NESTED_DIR

FD_CATS = {"DPABR", "GFCF", "GGFC", "HFCE", "INVNT", "NPISH"}
EXTRA_ROWS = ["OUT", "TLS", "VA"]
TOL_WORLD = 200.0                      # M$, the tolerance of the Technical Validation

C_WROW, C_WCOL = "#bd5e2c", "#e0a878"
C_SROW, C_SCOL = "#3577a8", "#7fa8c9"
INK, MUTED = "#1a1a1a", "#5b5b5b"


def is_state(l):
    return "_" in l and len(l.split("_")[0]) == 2


def is_world(l):
    return "_" in l and len(l.split("_")[0]) == 3


def one_year(path):
    df = pd.read_parquet(path)
    rows = list(df.index)
    cols = list(df.columns)
    ind_rows = [r for r in rows if r not in EXTRA_ROWS]
    w_rows = [r for r in ind_rows if is_world(r)]
    s_rows = [r for r in ind_rows if is_state(r)]
    sec_cols = [c for c in cols if "_" in c and c != "OUT"
                and c.split("_", 1)[1] not in FD_CATS]
    fd_cols = [c for c in cols if "_" in c and c.split("_", 1)[1] in FD_CATS]
    w_sec = [c for c in sec_cols if is_world(c)]
    s_sec = [c for c in sec_cols if is_state(c)]

    Z = df.loc[ind_rows, sec_cols].values.astype(float)
    F = df.loc[ind_rows, fd_cols].values.astype(float)
    out_col = df.loc[ind_rows, "OUT"].values.astype(float)
    out_row = df.loc["OUT", sec_cols].values.astype(float)
    va = df.loc["VA", sec_cols].values.astype(float)
    tls = df.loc["TLS", sec_cols].values.astype(float)

    row_err = np.abs(Z.sum(1) + F.sum(1) - out_col)
    col_err = np.abs(Z.sum(0) + va + tls - out_row)
    r_is_w = np.array([is_world(r) for r in ind_rows])
    c_is_w = np.array([is_world(c) for c in sec_cols])

    s_out = out_row[~c_is_w].sum()
    w_out = out_row.sum()
    res = dict(
        world_row=row_err[r_is_w].max(), state_row=row_err[~r_is_w].max(),
        world_col=col_err[c_is_w].max(), state_col=col_err[~c_is_w].max(),
        us_share=100 * s_out / w_out,
        world_output=w_out, us_output=s_out,
        n_world_rows=len(w_rows), n_state_rows=len(s_rows),
        n_world_cols=len(w_sec), n_state_cols=len(s_sec))
    del df, Z, F
    gc.collect()
    return res


def main():
    files = sorted(NESTED_DIR.glob("nested_mriot_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no delivered file under {NESTED_DIR}")
    rows = []
    for f in files:
        year = int(f.stem.split("_")[-1])
        r = one_year(f)
        r["year"] = year
        rows.append(r)
        print(f"[{year}] world row {r['world_row']:10.3f}  state row {r['state_row']:.2e}"
              f"  world col {r['world_col']:10.3f}  state col {r['state_col']:.2e}"
              f"  US share {r['us_share']:.2f}%", flush=True)

    T = pd.DataFrame(rows).set_index("year").sort_index()
    T.to_csv(FIG_DIR / "checks_series.csv")

    over = T.index[(T["world_row"] > TOL_WORLD) | (T["world_col"] > TOL_WORLD)].tolist()
    print(f"\nyears above the {TOL_WORLD:.0f} M$ world tolerance: {over}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # ---- panel a --------------------------------------------------------------
    floor = 1e-12
    for col, c, m, lab in ((("world_row"), C_WROW, "o", "world rows"),
                           (("world_col"), C_WCOL, "s", "world columns"),
                           (("state_row"), C_SROW, "o", "state rows"),
                           (("state_col"), C_SCOL, "s", "state columns")):
        axA.plot(T.index, np.maximum(T[col], floor), marker=m, ms=3.4, lw=1.1,
                 color=c, label=lab)
    axA.axhline(TOL_WORLD, color=INK, lw=.9, ls="--", alpha=.65)
    axA.text(T.index[-1], TOL_WORLD * 1.25, f"{TOL_WORLD:.0f} M USD tolerance",
             fontsize=7.5, color=INK, ha="right", va="bottom")
    for y in over:
        axA.annotate(str(y), xy=(y, max(T.loc[y, "world_row"], T.loc[y, "world_col"])),
                     xytext=(0, 7), textcoords="offset points", fontsize=6.8,
                     color=C_WROW, ha="center")
    axA.set_yscale("log")
    axA.set_xlabel("year", fontsize=9)
    axA.set_ylabel("largest absolute balance residual (million USD)", fontsize=9)
    axA.set_title("a   row and column balance of the delivered files",
                  fontsize=10.5, fontweight="bold", loc="left", pad=6)
    axA.legend(frameon=False, fontsize=8, ncol=2, loc="center left")
    axA.grid(ls=":", alpha=.4); axA.tick_params(labelsize=8)
    for s in ("top", "right"):
        axA.spines[s].set_visible(False)

    # ---- panel b --------------------------------------------------------------
    axB.plot(T.index, T["us_share"], marker="o", ms=3.6, lw=1.4, color=C_SROW)
    axB.fill_between(T.index, T["us_share"], color=C_SROW, alpha=.12)
    axB.set_xlabel("year", fontsize=9)
    axB.set_ylabel("share of world gross output (%)", fontsize=9)
    axB.set_title("b   weight of the fifty-one-region block",
                  fontsize=10.5, fontweight="bold", loc="left", pad=6)
    axB.grid(ls=":", alpha=.4); axB.tick_params(labelsize=8)
    lo, hi = T["us_share"].min(), T["us_share"].max()
    axB.set_ylim(lo - .08 * (hi - lo) - .2, hi + .18 * (hi - lo) + .2)
    axB.annotate(f"{T['us_share'].iloc[-1]:.1f}%  ({T.index[-1]})",
                 xy=(T.index[-1], T["us_share"].iloc[-1]), xytext=(-6, 8),
                 textcoords="offset points", fontsize=8, color=C_SROW, ha="right")
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)

    plt.tight_layout()
    out = FIG_DIR / "fig6_checks_series.png"
    fig.savefig(out, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)
    print("wrote", FIG_DIR / "checks_series.csv")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    main()
