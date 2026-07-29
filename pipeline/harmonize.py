"""Harmonize WinDC state-level IOT to match OECD US aggregates.

Steps:
  1. Aggregate WinDC over states.
  2. Scale vector components (VA, EX, M_interm, F) to match OECD US totals.
  3. RAS the aggregated Z matrix to match OECD row+col sums.
  4. Apply all national-level adjustments uniformly to the state-level data
     (preserves inter-state proportions).

Taxes are KEPT AS-IS (the WinDC tax blocks — taxes, tax_prod, tariff, tls_int,
tls_fd — and the effective rates ta0/tm0/ty0 are carried through unchanged,
only restricted to the common sectors). They are NOT scaled to OECD TLS and
NOT moved into VA: per the project decision, WinDC taxes are retained.

The output is a state-level WinDC table whose sum over states equals the
OECD US block exactly for VA, EX, M, F, and Z row/col sums, while the tax
blocks remain at their native WinDC values.
"""
from paths import ROOT
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
CONC_PATH  = ROOT / "data/raw/correspondence/windc_oecd_concordance_handmade_cleaned_correspondence.csv"
OECD_ROOT  = ROOT / "data/interim/IOT/OCDE ICIO aggregated"
IOT_ROOT   = ROOT / "data/interim/IOT/IOT_USA"
# WiNDC aggregated build to harmonize. Single knob: change WINDC_VERSION (e.g.
# "v3", "v3.1_RAS") to point at grav_fric_<WINDC_VERSION>_aggregated. Importers
# (the notebook) can override WINDC_VERSION / WINDC_AGG before use.
WINDC_VERSION = "v3.1_RAS"
WINDC_AGG  = IOT_ROOT / f"grav_fric_{WINDC_VERSION}_aggregated"
FD_CATS    = {"DPABR","GFCF","GGFC","HFCE","INVNT","NPISH"}
EXTRA      = {"OUT","TLS","VA"}
OECD_SCALE = 1e-3
# WinDC tax blocks kept as-is (only restricted to common sectors, never scaled):
TAX_AMOUNT_KEYS = ["taxes", "tax_prod", "tariff", "tls_int", "tls_fd"]
TAX_RATE_KEYS   = ["ta0", "tm0", "ty0"]

# ════════════════════════════════════════════════════════════════════════════
#  RAS (biproportional fitting)
# ════════════════════════════════════════════════════════════════════════════
def ras(Z_prior, target_row, target_col, max_iter=2000, tol=1e-9):
    Z = np.where(Z_prior < 0, 0, Z_prior.astype(float).copy())
    tot = (target_row.sum() + target_col.sum()) / 2
    target_row = target_row * (tot / target_row.sum())
    target_col = target_col * (tot / target_col.sum())
    for it in range(max_iter):
        rs = Z.sum(axis=1)
        Z *= np.where(rs > 1e-12, target_row / rs, 1.0)[:, np.newaxis]
        cs = Z.sum(axis=0)
        Z *= np.where(cs > 1e-12, target_col / cs, 1.0)[np.newaxis, :]
        err = max(np.abs(Z.sum(1) - target_row).max(),
                  np.abs(Z.sum(0) - target_col).max())
        if err < tol:
            return Z, {"iters": it+1, "err": err}
    return Z, {"iters": max_iter, "err": err}


def seed_zero_cells(Z, frac, prior=None):
    """Lift exact-zero cells of Z with a small positive floor so RAS can
    redistribute flow into them (RAS is multiplicative and can never fill a
    structural zero, which forces neighbouring cells to absorb the margin and
    distorts the fit). The floor is `frac` × mean positive cell of Z.

    If `prior` (e.g. the OECD Z block) is given the floor is shaped by it
    (zeros where the prior is large get more seed); otherwise the seed is flat
    background noise. Returns (seeded Z, n_cells_seeded)."""
    Z = Z.astype(float).copy()
    zero = Z <= 0
    n = int(zero.sum())
    if n == 0 or frac <= 0:
        return Z, 0
    scale = Z[Z > 0].mean() if (Z > 0).any() else 1.0
    floor = frac * scale
    if prior is not None:
        p  = np.clip(prior.astype(float), 0, None)
        pm = p[p > 0].mean() if (p > 0).any() else 1.0
        seed = np.where(p > 0, floor * p / pm, floor)
    else:
        seed = np.full_like(Z, floor)
    Z[zero] = seed[zero]
    return Z, n

# ════════════════════════════════════════════════════════════════════════════
#  Load OECD targets for a given year (restricted to common sectors)
# ════════════════════════════════════════════════════════════════════════════
def load_oecd_targets(year, secs_common, secs_oe_all, oe_idx):
    """Returns OECD US aggregate targets restricted to common sectors."""
    for f in OECD_ROOT.rglob(f"{year}_*.parquet"):
        fn = f; break
    dfo = pd.read_parquet(fn)

    usa_sec_rows = [f"USA_{s}" for s in secs_oe_all if f"USA_{s}" in dfo.index]
    usa_sec_cols = [f"USA_{s}" for s in secs_oe_all if f"USA_{s}" in dfo.columns]
    nusa_sec_rows = [r for r in dfo.index   if r not in EXTRA and "_" in r
                     and not r.startswith("USA_") and r.split("_",1)[1] not in FD_CATS]
    nusa_sec_cols = [c for c in dfo.columns if "_" in c and not c.startswith("USA_")
                     and c.split("_",1)[1] not in FD_CATS]
    nusa_fd_cols  = [c for c in dfo.columns if "_" in c and not c.startswith("USA_")
                     and c.split("_",1)[1] in FD_CATS]
    usa_fd_cols   = [c for c in dfo.columns if c.startswith("USA_")
                     and c.split("_",1)[1] in FD_CATS]

    # Z (intermediate USA→USA only, common sectors)
    Z_us_oe = (dfo.loc[usa_sec_rows, usa_sec_cols].values.astype(float)
               [np.ix_(oe_idx, oe_idx)] * OECD_SCALE)
    # VA, TLS, OUT
    VA_oe = dfo.loc["VA",  usa_sec_cols].values.astype(float) * OECD_SCALE
    TX_oe = dfo.loc["TLS", usa_sec_cols].values.astype(float) * OECD_SCALE
    OUT_oe = dfo.loc[usa_sec_rows, "OUT"].values.astype(float) * OECD_SCALE
    # M_interm (non-USA → USA intermediate)
    M_oe = (dfo.loc[nusa_sec_rows, usa_sec_cols]
              .values.astype(float).sum(axis=0)) * OECD_SCALE
    # FD domestic (USA → USA FD)
    F_oe = (dfo.loc[usa_sec_rows, usa_fd_cols]
              .values.astype(float).sum(axis=1)) * OECD_SCALE
    # Exports (USA → non-USA intermediate + non-USA FD)
    EX_oe = (dfo.loc[usa_sec_rows, nusa_sec_cols + nusa_fd_cols]
               .values.astype(float).sum(axis=1)) * OECD_SCALE

    return {
        "Z":   Z_us_oe,                          # (n_sc, n_sc)
        "VA":  VA_oe[oe_idx],
        "TX":  TX_oe[oe_idx],
        "M":   M_oe[oe_idx],
        "F":   F_oe[oe_idx],
        "EX":  EX_oe[oe_idx],
        "OUT": OUT_oe[oe_idx],
    }

# ════════════════════════════════════════════════════════════════════════════
#  Main harmonization function (reusable)
# ════════════════════════════════════════════════════════════════════════════
def harmonize_windc_to_oecd(npz_wd, oecd_targets, wd_idx, secs_common,
                              return_transform=True, seed_zeros=0.0,
                              seed_prior=None, balancer=None):
    """Harmonize a WinDC state-level IOT to match OECD US aggregates.

    Parameters
    ----------
    npz_wd : dict-like (np.load result)
        WinDC aggregated to proposed sectors, with keys:
        Z (n_r*n_sw, n_r*n_sw), F (n_r*n_sw, n_fd), VA/EX/M_interm/taxes (n_r*n_sw,),
        regions (n_r,), proposed_sectors (n_sw,).
    oecd_targets : dict
        US aggregate targets restricted to common sectors (output of load_oecd_targets).
    wd_idx : list[int]
        Indices to restrict WinDC sectors to the common set.
    secs_common : list[str]
        Common sector names (n_sc).
    return_transform : bool
        If True, return the transformation parameters in addition to the harmonized table.
    seed_zeros : float
        Approach 2. If > 0, the exact-zero cells of the aggregated Z are seeded
        with a small floor (= seed_zeros × mean positive Z cell) BEFORE the RAS,
        so the biproportional fit can place flow into cells WinDC has at 0 but
        OECD does not (12.9% of cells in 2017). 0 disables it (approach 1).
        At state level a seeded aggregate cell is injected uniformly across all
        state pairs. Typical value: 1e-3 to 1e-2.
    seed_prior : np.ndarray or None
        Optional (n_sc, n_sc) prior shaping the seed (e.g. oecd_targets["Z"]).
        None ⇒ flat background seed (neutral).

    Returns
    -------
    out_state : dict with harmonized state-level arrays:
        Z (n_r, n_sc, n_r, n_sc), F (n_r, n_sc, n_fd),
        VA/EX/M_interm/taxes (n_r, n_sc),
        regions (n_r,), proposed_sectors (n_sc,)
    transform : dict (if return_transform=True) with the per-sector multipliers
    """
    sectors_wd = list(npz_wd["proposed_sectors"])
    regions    = list(dict.fromkeys([str(r) for r in npz_wd["regions"]]))
    n_r  = len(regions)
    n_sw = len(sectors_wd)
    n_sc = len(secs_common)

    # ── Reshape raw WinDC to (n_r, n_sw, ...) ─────────────────────────────
    Z_st_full  = npz_wd["Z"].reshape(n_r, n_sw, n_r, n_sw)
    F_st_full  = npz_wd["F"].reshape(n_r, n_sw, -1)
    VA_st_full = npz_wd["VA"].reshape(n_r, n_sw)
    EX_st_full = npz_wd["EX"].reshape(n_r, n_sw)
    MI_st_full = npz_wd["M_interm"].reshape(n_r, n_sw) if "M_interm" in npz_wd.files \
                 else np.zeros((n_r, n_sw))
    TX_st_full = npz_wd["taxes"].reshape(n_r, n_sw)    if "taxes"    in npz_wd.files \
                 else np.zeros((n_r, n_sw))

    # ── Restrict to common sectors on both axes ────────────────────────────
    Z_st  = Z_st_full[:, wd_idx, :, :][:, :, :, wd_idx]    # n_r × n_sc × n_r × n_sc
    F_st  = F_st_full[:, wd_idx, :]
    VA_st = VA_st_full[:, wd_idx]
    EX_st = EX_st_full[:, wd_idx]
    MI_st = MI_st_full[:, wd_idx]

    # Tax blocks (amounts + effective rates) — kept as-is, only restricted to
    # the common sectors. Never scaled, never moved into VA.
    tax_st = {}
    for k in TAX_AMOUNT_KEYS + TAX_RATE_KEYS:
        if k in npz_wd.files:
            tax_st[k] = npz_wd[k].reshape(n_r, n_sw)[:, wd_idx]   # n_r × n_sc

    # ── Aggregate over states ──────────────────────────────────────────────
    Z_us  = Z_st.sum(axis=(0, 2))
    F_us  = F_st.sum(axis=(0, 2))
    VA_us = VA_st.sum(axis=0)
    EX_us = EX_st.sum(axis=0)
    MI_us = MI_st.sum(axis=0)

    # ──────────────────────────────────────────────────────────────────────
    # STEP 1: Per-sector scaling factors for vector components
    #         (taxes are NOT scaled — kept at native WinDC values)
    # ──────────────────────────────────────────────────────────────────────
    eps = 1e-9
    sf_VA = oecd_targets["VA"] / np.where(np.abs(VA_us) > eps, VA_us, eps)
    sf_EX = oecd_targets["EX"] / np.where(np.abs(EX_us) > eps, EX_us, eps)
    sf_M  = oecd_targets["M"]  / np.where(np.abs(MI_us) > eps, MI_us, eps)
    sf_F  = oecd_targets["F"]  / np.where(np.abs(F_us)  > eps, F_us,  eps)

    # ──────────────────────────────────────────────────────────────────────
    # STEP 2: RAS the aggregated Z to match OECD row+col sums
    #         (approach 2: seed zero cells first so RAS can fill them)
    # ──────────────────────────────────────────────────────────────────────
    #         A `balancer(Z_prior, row_target, col_target, oecd_Z)` callable can
    #         replace the default seed+RAS (e.g. cross-entropy on an OECD-blended
    #         prior — see balance_methods.py). It must return (X, info) with X
    #         matching the row/col targets; the state-level transposition below is
    #         method-agnostic (multiplicative on positive cells, uniform injection
    #         on cells WinDC has at zero).
    Z_target_row = oecd_targets["Z"].sum(axis=1)
    Z_target_col = oecd_targets["Z"].sum(axis=0)
    if balancer is None:
        Z_seed, n_seeded = seed_zero_cells(Z_us, seed_zeros, seed_prior)
        Z_us_harm, ras_info = ras(Z_seed, Z_target_row, Z_target_col)
    else:
        Z_us_harm, ras_info = balancer(Z_us, Z_target_row, Z_target_col,
                                       oecd_targets["Z"])
        n_seeded = int(((Z_us <= eps) & (Z_us_harm > eps)).sum())
    sf_Z = np.where(Z_us > eps, Z_us_harm / np.where(Z_us > eps, Z_us, eps), 0.0)
    seeded_mask = (Z_us <= eps) & (Z_us_harm > eps)   # cells lifted from zero

    # ──────────────────────────────────────────────────────────────────────
    # STEP 3: Apply scaling to state-level data (preserves inter-state proportions)
    # ──────────────────────────────────────────────────────────────────────
    Z_st_h  = Z_st  * sf_Z [np.newaxis, :, np.newaxis, :]
    # Seeded aggregate cells (WinDC=0): spread uniformly across all state pairs
    # so the state-level sum reproduces the RAS-filled aggregate value.
    if n_seeded:
        n_pairs = n_r * n_r
        ii, jj = np.where(seeded_mask)
        Z_st_h[:, ii, :, jj] = (Z_us_harm[ii, jj] / n_pairs)[:, np.newaxis, np.newaxis]
    VA_st_h = VA_st * sf_VA[np.newaxis, :]
    EX_st_h = EX_st * sf_EX[np.newaxis, :]
    MI_st_h = MI_st * sf_M [np.newaxis, :]
    F_st_h  = F_st  * sf_F [np.newaxis, :, np.newaxis]
    # Tax amounts pass through unchanged; rates pass through unchanged.
    tax_st_h = {k: v.copy() for k, v in tax_st.items()}

    # Inject missing FD where WD has zero but OECD has a positive target
    # (distribute uniformly across states × FD categories)
    n_fd_cats = F_st.shape[2]
    inject_F  = (np.abs(F_us) < eps) & (np.abs(oecd_targets["F"]) > eps)
    for sc in np.where(inject_F)[0]:
        F_st_h[:, sc, :] = oecd_targets["F"][sc] / (n_r * n_fd_cats)
    # Same logic for the other vector components, just in case
    inject_VA = (np.abs(VA_us) < eps) & (np.abs(oecd_targets["VA"]) > eps)
    for sc in np.where(inject_VA)[0]:
        VA_st_h[:, sc] = oecd_targets["VA"][sc] / n_r
    inject_EX = (np.abs(EX_us) < eps) & (np.abs(oecd_targets["EX"]) > eps)
    for sc in np.where(inject_EX)[0]:
        EX_st_h[:, sc] = oecd_targets["EX"][sc] / n_r
    inject_M  = (np.abs(MI_us) < eps) & (np.abs(oecd_targets["M"]) > eps)
    for sc in np.where(inject_M)[0]:
        MI_st_h[:, sc] = oecd_targets["M"][sc] / n_r

    # Flat (n_r*n_sc, ...) view — backward compatible with the aggregated WinDC format
    index_labels = np.array([f"{r}_{s}" for r in regions for s in secs_common])
    out_state = {
        "Z":         Z_st_h.reshape(n_r * n_sc, n_r * n_sc),
        "F":         F_st_h.reshape(n_r * n_sc, -1),
        "VA":        VA_st_h.reshape(-1),
        "EX":        EX_st_h.reshape(-1),
        "M_interm":  MI_st_h.reshape(-1),
        "regions":   np.array(regions),
        "proposed_sectors": np.array(secs_common),
        "index_labels":     index_labels,
    }
    # WinDC tax blocks (amounts + rates) carried through unchanged.
    for k, v in tax_st_h.items():
        out_state[k] = v.reshape(-1)
    if not return_transform:
        return out_state
    transform = {
        "sf_Z":     sf_Z,
        "sf_VA":    sf_VA,
        "sf_EX":    sf_EX,
        "sf_M":     sf_M,
        "sf_F":     sf_F,
        "ras_info": ras_info,
        "n_seeded": n_seeded,
        "seeded_mask": seeded_mask,
    }
    return out_state, transform

# ════════════════════════════════════════════════════════════════════════════
#  Test on year 2000
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    YEAR = 2017

    # ── Resolve common sectors (mirror the notebook setup) ────────────────
    conc = pd.read_csv(CONC_PATH)
    d0 = np.load(next(WINDC_AGG.glob("*.npz")), allow_pickle=True)
    secs_wd = list(dict.fromkeys(str(s) for s in d0["proposed_sectors"]))
    dfo0 = pd.read_parquet(next(OECD_ROOT.rglob(f"{YEAR}_*.parquet")))
    secs_oe_all = [c.split("_",1)[1] for c in dfo0.columns
                    if c.startswith("USA_") and c.split("_",1)[1] not in FD_CATS]
    secs_common = [s for s in secs_wd if s in set(secs_oe_all)]
    n_sc = len(secs_common)
    wd_idx = [secs_wd.index(s)     for s in secs_common]
    oe_idx = [secs_oe_all.index(s) for s in secs_common]
    print(f"Common sectors: {n_sc}")

    # ── Load data ─────────────────────────────────────────────────────────
    oecd_t = load_oecd_targets(YEAR, secs_common, secs_oe_all, oe_idx)
    npz_wd = np.load(WINDC_AGG / f"IOT_{YEAR}.npz", allow_pickle=True)

    # ── Run harmonization ─────────────────────────────────────────────────
    print(f"\nRunning harmonization for {YEAR}...")
    out, trans = harmonize_windc_to_oecd(npz_wd, oecd_t, wd_idx, secs_common)
    print(f"  RAS converged in {trans['ras_info']['iters']} iters "
          f"(err = {trans['ras_info']['err']:.2e})")

    # ── Verification: aggregate match ─────────────────────────────────────
    print(f"\n══ AGGREGATE MATCH VERIFICATION (Bn$) ══")
    n_r  = len(out["regions"])
    n_sc = len(out["proposed_sectors"])
    # Reshape flat → 4D for verification
    Z_h_us = out["Z"].reshape(n_r, n_sc, n_r, n_sc).sum(axis=(0, 2))
    F_h_us = out["F"].reshape(n_r, n_sc, -1).sum(axis=(0, 2))
    VA_h_us = out["VA"].reshape(n_r, n_sc).sum(axis=0)
    EX_h_us = out["EX"].reshape(n_r, n_sc).sum(axis=0)
    M_h_us  = out["M_interm"].reshape(n_r, n_sc).sum(axis=0)

    print(f"{'Component':<10}  {'WD harmonized':>14}  {'OECD target':>14}  {'max sector err':>14}")
    for name, wd_v, oe_v in [
        ("Z.row",   Z_h_us.sum(1),  oecd_t["Z"].sum(1)),
        ("Z.col",   Z_h_us.sum(0),  oecd_t["Z"].sum(0)),
        ("VA",      VA_h_us,        oecd_t["VA"]),
        ("EX",      EX_h_us,        oecd_t["EX"]),
        ("M_interm",M_h_us,         oecd_t["M"]),
        ("F",       F_h_us,         oecd_t["F"]),
    ]:
        err = np.abs(wd_v - oe_v).max()
        rel = err / np.abs(oe_v).max() * 100
        print(f"{name:<10}  {wd_v.sum():>14.1f}  {oe_v.sum():>14.1f}  "
              f"{err:>10.4f}  ({rel:.4f}%)")

    # Taxes are intentionally NOT matched to OECD — kept at WinDC values.
    TX_h_us = out["taxes"].reshape(n_r, n_sc).sum(axis=0)
    TX_wd_orig = (npz_wd["taxes"].reshape(n_r, len(secs_wd))[:, wd_idx]).sum(axis=0)
    print(f"{'TX (kept)':<10}  {TX_h_us.sum():>14.1f}  {oecd_t['TX'].sum():>14.1f}  "
          f"(unchanged vs WinDC: max delta = {np.abs(TX_h_us - TX_wd_orig).max():.2e})")

    # ── Verification: state-level structure preserved ─────────────────────
    print(f"\n══ STATE-LEVEL CONSISTENCY CHECKS ══")
    Z_h_4D = out["Z"].reshape(n_r, n_sc, n_r, n_sc)
    VA_h_2D = out["VA"].reshape(n_r, n_sc)
    print(f"  Z (state sum vs US harm):  max err = {np.abs(Z_h_4D.sum(axis=(0,2)) - Z_h_us).max():.2e}")
    print(f"  VA (state sum vs US harm): max err = {np.abs(VA_h_2D.sum(0) - VA_h_us).max():.2e}")

    # Inter-state proportions preservation — spot check on largest cell
    Z_st_orig = npz_wd["Z"].reshape(len(out['regions']), len(secs_wd),
                                      len(out['regions']), len(secs_wd))
    Z_st_orig = Z_st_orig[:, wd_idx, :, :][:, :, :, wd_idx]
    i = int(np.unravel_index(np.argmax(Z_st_orig.sum(axis=(0,2))), Z_h_us.shape)[0])
    j = int(np.unravel_index(np.argmax(Z_st_orig.sum(axis=(0,2))), Z_h_us.shape)[1])
    orig_props = Z_st_orig[:, i, :, j] / Z_st_orig[:, i, :, j].sum()
    new_props  = Z_h_4D[:, i, :, j]    / Z_h_4D[:, i, :, j].sum()
    print(f"  Inter-state proportion preservation (largest cell i={i}, j={j}):")
    print(f"    max abs diff in proportions = {np.abs(orig_props - new_props).max():.2e}")

    # ── Save harmonized result + transformation params (flat format = backward compat) ──
    OUT_DIR = IOT_ROOT / f"grav_fric_{WINDC_VERSION}_harmonized"
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    out_path = OUT_DIR / f"IOT_{YEAR}_harmonized.npz"
    np.savez_compressed(out_path, **out)
    print(f"\nHarmonized table saved to {out_path}")
    print(f"  keys: {sorted(out.keys())}")
    print(f"  shape Z: {out['Z'].shape}  (n_r*n_sc, n_r*n_sc) — flat for backward compat")

    # Save transformation params for reuse
    trans_path = OUT_DIR / f"transform_{YEAR}.npz"
    np.savez_compressed(trans_path, **trans, secs_common=np.array(secs_common))
    print(f"Transformation params saved to {trans_path}")
