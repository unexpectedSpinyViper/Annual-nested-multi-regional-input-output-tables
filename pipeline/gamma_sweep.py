"""Gamma sweep — full intra-US MRIOT pipeline as a function of the gravity exponent.

Runs the complete production pipeline for a chosen gravity friction exponent gamma:

    build (v3.1) -> column RAS (v3.1_RAS) -> VA convention
      -> sectoral aggregation -> harmonization to OECD -> nesting into OECD ICIO

For each gamma the standard artifacts are written to disk under version-tagged
directories (``grav_fric_gamma<g>_RAS`` / ``..._RAS_aggregated`` /
``..._RAS_harmonized`` and ``nested_mriot_gamma<g>``), exactly mirroring the
layout of the canonical ``v3.1_RAS`` build so the downstream notebooks keep
working unchanged.

The build code is the v3 construction (``reconstruct_bilateral_3`` + the md0
margin chain), identical to ``pipeline/v3_construction.ipynb``;
the aggregation mapping is the one validated in ``aggregation.ipynb``; the
harmonization and nesting steps reuse ``harmonize.py`` and ``nest_v31.py``
verbatim. Only ``gamma`` changes between runs (gamma_trade = gamma_margin = gamma).
"""
from __future__ import annotations

from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

import harmonize          # reused as-is (load_oecd_targets, harmonize_windc_to_oecd)
import nest_v31           # reused as-is (build_shares_pivot, nest_year)

# ── Paths ──────────────────────────────────────────────────────────────────────
from paths import ROOT
GDX_PATH  = ROOT / "data/raw/GTAPWiNDC/data/core/WiNDCdatabase.gdx"
IOT_ROOT  = ROOT / "data/interim/IOT/IOT_USA"
CONC_PATH = ROOT / ("data/raw/correspondence/"
                    "windc_oecd_concordance_handmade_cleaned_correspondence.csv")
OECD_AGG  = ROOT / "data/interim/IOT/OCDE ICIO aggregated"
CENTROIDS = ROOT / "data/interim/economic_centroids.csv"

FD_CATS = {"DPABR", "GFCF", "GGFC", "HFCE", "INVNT", "NPISH"}

# Module-level state populated by setup() (mirrors the notebook's global style)
params: dict | None = None
regions: list | None = None
sectors: list | None = None
sector_to_idx: dict | None = None
region_to_idx: dict | None = None
D_np: np.ndarray | None = None
n: int = 0
S: int = 0

# IO blocks loaded per year inside build_table
NAMES = ['dd0_', 'nd0_', 'xn0_', 'xd0_', 'x0_', 'm0_',
         'cd0_', 'i0_', 'g0_', 'ld0_', 'kd0_', 'ty0_']

# Reference points of the gravity distance matrix: the GDP-weighted economic
# centroids of step 02, the same file the delivered build reads. Loaded lazily by
# setup() so that importing this module does not require the file to exist.


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlam = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def setup(verbose=True):
    """Load the WiNDC GDX once and populate module-level state (params, regions,
    sectors, distance matrix). Idempotent: re-running is a no-op."""
    global params, regions, sectors, sector_to_idx, region_to_idx, D_np, n, S
    if params is not None:
        return
    from gdx2py import GdxFile
    from gdx2py.gams import GAMSParameter
    import gamspy_base

    gdx = GdxFile(str(GDX_PATH), gams_dir=gamspy_base.directory)
    params = {}
    for name, obj in gdx:
        if isinstance(obj, GAMSParameter):
            s = obj.to_pandas()
            if s is not None and len(s) > 0:
                df = s.reset_index()
                df.columns = list(df.columns[:-1]) + ['value']
                params[name] = df

    # The GDX stores the year as a label, i.e. a string. Normalise it once here so a
    # caller passing the int 2017 selects the same rows as one passing "2017"; the
    # comparisons below all go through _yr(). Without this the year filter silently
    # matches nothing and every loader returns a table of zeros.
    for _df in params.values():
        if 'yr' in _df.columns:
            _df['yr'] = _df['yr'].astype(str)

    regions = sorted(set(params['xn0_']['r'].unique()) | set(params['nd0_']['r'].unique()))
    sectors = sorted(params['xn0_']['g'].unique())           # EXCLUDED = {} (keep fen/sle)
    region_to_idx = {r: i for i, r in enumerate(regions)}
    sector_to_idx = {s: i for i, s in enumerate(sectors)}
    n, S = len(regions), len(sectors)

    if not CENTROIDS.exists():
        raise FileNotFoundError(
            f"{CENTROIDS} not found -- run 02_economic_centroids.ipynb first. The sweep "
            f"must use the same reference points as the delivered series.")
    _cen = pd.read_csv(CENTROIDS)
    coords = {r.abbr: (r.lat, r.lon) for r in _cen.itertuples(index=False)}
    missing = [r for r in regions if r not in coords]
    if missing:
        raise KeyError(f"no economic centroid for {missing}")

    D = pd.DataFrame(index=regions, columns=regions, dtype=float)
    for i, j in product(regions, regions):
        D.loc[i, j] = np.nan if i == j else _haversine(*coords[i], *coords[j])
    D_np = D.values.copy()
    if verbose:
        print(f"setup: {n} regions x {S} sectors | "
              f"distance {D_np[~np.isnan(D_np)].min():.0f}-{D_np[~np.isnan(D_np)].max():.0f} km")


# ════════════════════════════════════════════════════════════════════════════
#  Loaders (verbatim from v3_construction.ipynb)
# ════════════════════════════════════════════════════════════════════════════
def _yr(year):
    """The GDX year label matching ``year`` (int or str)."""
    return str(year)


def load_matrix(param_name, year):
    df = params[param_name]
    dim = 'g' if 'g' in df.columns else 's'
    return (df[df['yr'] == _yr(year)]
            .groupby(['r', dim])['value'].sum().unstack(dim)
            .reindex(index=regions, columns=sectors, fill_value=0.0)
            .fillna(0.0).values)


def load_year_data(year):
    mats = {name: load_matrix(name, year) for name in NAMES}
    id0_tensor = (params['id0_'][params['id0_']['yr'] == _yr(year)]
                  .groupby(['r', 'g', 's'])['value'].sum().unstack('s')
                  .reindex(pd.MultiIndex.from_product([regions, sectors], names=['r', 'g']),
                           fill_value=0.0)
                  .reindex(columns=sectors, fill_value=0.0)
                  .fillna(0.0).values.reshape(n, S, S))
    return {**mats, 'id0': id0_tensor}


# ════════════════════════════════════════════════════════════════════════════
#  RAS helpers (verbatim from v3_construction.ipynb)
# ════════════════════════════════════════════════════════════════════════════
def ras_robust(seed, X, M, max_iter=2000, tol=1e-8):
    T = seed.copy().astype(float)
    r = X.values if hasattr(X, 'values') else X
    c = M.values if hasattr(M, 'values') else M
    initial_err = max(np.abs(T.sum(axis=1) - r).max(), np.abs(T.sum(axis=0) - c).max())
    for it in range(max_iter):
        rs = T.sum(axis=1); rs[rs == 0] = 1
        T *= (r / rs)[:, None]
        cs = T.sum(axis=0); cs[cs == 0] = 1
        T *= (c / cs)[None, :]
        err = max(np.abs(T.sum(axis=1) - r).max(), np.abs(T.sum(axis=0) - c).max())
        if err < tol:
            return T, True, err, it + 1, initial_err
    return T, False, err, max_iter, initial_err


def ras_2(Z0, u, v, max_iter=300, tol=1e-6):
    Z = Z0.copy().astype(float)
    for _ in range(max_iter):
        rs = Z.sum(1); Z *= np.divide(u, rs, out=np.ones_like(u), where=rs > 1e-12)[:, None]
        cs = Z.sum(0); Z *= np.divide(v, cs, out=np.ones_like(v), where=cs > 1e-12)[None, :]
        if max(np.abs(Z.sum(1) - u).max(), np.abs(Z.sum(0) - v).max()) < tol:
            break
    return Z


# ════════════════════════════════════════════════════════════════════════════
#  Bilateral reconstruction + margin routing (verbatim from v3_construction.ipynb)
# ════════════════════════════════════════════════════════════════════════════
def _as_sector_gamma(gamma):
    """Normalize ``gamma`` to a ``{sector: value}`` dict over the build sectors.

    Accepts a **scalar** (broadcast to every sector — the historical behaviour),
    a ``{sector: value}`` **mapping** (missing sectors default to 1.0), or an
    **array-like** aligned with the module-level ``sectors`` order.
    """
    if np.isscalar(gamma):
        return {g: float(gamma) for g in sectors}
    if isinstance(gamma, dict):
        return {g: float(gamma.get(g, 1.0)) for g in sectors}
    arr = np.asarray(gamma, dtype=float).ravel()
    if arr.shape != (len(sectors),):
        raise ValueError(f"per-sector gamma must have length {len(sectors)} "
                         f"(one per build sector), got {arr.shape}")
    return {g: float(arr[i]) for i, g in enumerate(sectors)}


def reconstruct_bilateral_3(xn0_mat, nd0_mat, nm0_mat, D_np, gamma_trade=1.0,
                            gamma_margin=1.0, imbalance_skip=0.50):
    # gamma_trade / gamma_margin may be scalar or per-sector (vector/dict); the
    # friction d_ij^{-gamma} is therefore resolved per sector and memoized by
    # value, so a scalar gamma reproduces the previous build bit-for-bit.
    gt = _as_sector_gamma(gamma_trade)
    gm = _as_sector_gamma(gamma_margin)
    _fric_cache = {}

    def friction(gamma):
        if gamma not in _fric_cache:
            with np.errstate(divide='ignore', invalid='ignore'):
                _fric_cache[gamma] = np.where(np.isnan(D_np), 0.0, D_np ** (-gamma))
        return _fric_cache[gamma]

    def _ras_one_flow(X_row, M_col, friction):
        tX, tM = X_row.sum(), M_col.sum()
        if tX < 1e-10 or tM < 1e-10:
            return np.zeros((n, n)), True, 0.0, 0, 0.0
        seed = np.outer(X_row.values, M_col.values) * friction
        seed += 1e-8 * np.outer(X_row.values / tX, M_col.values / tM)
        np.fill_diagonal(seed, 0.0)
        return ras_robust(seed, X_row, M_col)

    def _status(converged, err, tol_soft=1e-6):
        if converged:      return 'ok'
        if err < tol_soft: return 'ok_soft'
        return 'FAILED'

    T_trade_all, T_margin_all, log_ras = {}, {}, []
    for g in sectors:
        g_i = sector_to_idx[g]
        X_g = pd.Series(xn0_mat[:, g_i], index=regions)
        nd_g = pd.Series(nd0_mat[:, g_i], index=regions)
        nm_g = pd.Series(nm0_mat[:, g_i], index=regions)
        total_X, total_nd, total_nm = X_g.sum(), nd_g.sum(), nm_g.sum()
        total_M = total_nd + total_nm
        imbalance = abs(total_X - total_M) / total_X if total_X else np.inf
        if total_M < 1e-10 or imbalance > imbalance_skip:
            T_trade_all[g] = np.zeros((n, n)); T_margin_all[g] = np.zeros((n, n))
            log_ras.append({'sector': g, 'status': 'skipped_imbalance', 'err': imbalance, 'iters': 0})
            continue
        X_trade = X_g * (total_nd / total_X)
        X_margin = X_g * (total_nm / total_X)
        T_trade, c_t, err_t, it_t, _ = _ras_one_flow(X_trade, nd_g, friction(gt[g]))
        T_margin, c_m, err_m, it_m, _ = _ras_one_flow(X_margin, nm_g, friction(gm[g]))
        T_trade_all[g] = T_trade; T_margin_all[g] = T_margin
        log_ras.append({'sector': g, 'status_trade': _status(c_t, err_t),
                        'status_margin': _status(c_m, err_m), 'err_trade': err_t,
                        'err_margin': err_m, 'nd_total': total_nd, 'nm_total': total_nm})
    return T_trade_all, T_margin_all, pd.DataFrame(log_ras)


def build_margin_tensors(year, margins=('trd', 'trn')):
    M = len(margins); margins = list(margins)
    nm0_rgm = (params['nm0_'][params['nm0_']['yr'] == _yr(year)]
               .groupby(['r', 'g', 'm'])['value'].sum()
               .reindex(pd.MultiIndex.from_product([regions, sectors, margins],
                                                   names=['r', 'g', 'm']), fill_value=0.0)
               .values.reshape(n, S, M))
    md0_rmg = (params['md0_'][params['md0_']['yr'] == _yr(year)]
               .groupby(['r', 'm', 'g'])['value'].sum()
               .reindex(pd.MultiIndex.from_product([regions, margins, sectors],
                                                   names=['r', 'm', 'g']), fill_value=0.0)
               .values.reshape(n, M, S))
    return nm0_rgm, md0_rmg, margins


def compute_use_shares_3(id0_df, cd0_mat, i0_mat, g0_mat, nm0_rgm, md0_rmg):
    total_demand = id0_df.sum(axis=2) + cd0_mat + i0_mat + g0_mat
    safe = np.where(total_demand < 1e-10, 1.0, total_demand)
    d_int = id0_df / safe[:, :, None]
    dC, dI, dG = cd0_mat / safe, i0_mat / safe, g0_mat / safe
    total_inputs = id0_df.sum(axis=1)
    row_sum = total_inputs.sum(axis=1, keepdims=True)
    fallback = total_inputs / np.where(row_sum < 1e-10, 1.0, row_sum)
    mask_zero = (total_demand < 1e-10)[:, :, None]
    d_int = np.where(mask_zero, fallback[:, None, :], d_int)

    w = nm0_rgm.astype(float)
    w_sum = w.sum(axis=2, keepdims=True)
    w = np.divide(w, w_sum, out=np.zeros_like(w), where=w_sum > 1e-12)
    tot_md = md0_rmg.sum(axis=2)
    safe_md = np.where(tot_md < 1e-12, 1.0, tot_md)
    bf_int = np.einsum('rmh,rhs->rms', md0_rmg, d_int) / safe_md[:, :, None]
    bf_C = (md0_rmg * dC[:, None, :]).sum(axis=2) / safe_md
    bf_I = (md0_rmg * dI[:, None, :]).sum(axis=2) / safe_md
    bf_G = (md0_rmg * dG[:, None, :]).sum(axis=2) / safe_md
    m_int = np.einsum('rgm,rms->rgs', w, bf_int)
    mC = np.einsum('rgm,rm->rg', w, bf_C)
    mI = np.einsum('rgm,rm->rg', w, bf_I)
    mG = np.einsum('rgm,rm->rg', w, bf_G)
    return d_int, dC, dI, dG, m_int, mC, mI, mG


def build_Z_3(dd0_mat, dm0_mat, T_dir, T_mar, ush_dir, ush_mar):
    Z = np.zeros((n, S, n, S))
    for r in range(n):
        Z[r, :, r, :] += dd0_mat[r, :, None] * ush_dir[r, :, :]
        Z[r, :, r, :] += dm0_mat[r, :, None] * ush_mar[r, :, :]
    for gi, g in enumerate(sectors):
        Z[:, gi, :, :] += T_dir[g][:, :, None] * ush_dir[:, gi, :][None, :, :]
        Z[:, gi, :, :] += T_mar[g][:, :, None] * ush_mar[:, gi, :][None, :, :]
    return Z.reshape(n * S, n * S)


def build_F_3(dd0_mat, dm0_mat, T_dir, T_mar, dC, dI, dG, mC, mI, mG):
    F = np.zeros((n, S, n, 3))
    for r in range(n):
        F[r, :, r, 0] += dd0_mat[r, :] * dC[r, :] + dm0_mat[r, :] * mC[r, :]
        F[r, :, r, 1] += dd0_mat[r, :] * dI[r, :] + dm0_mat[r, :] * mI[r, :]
        F[r, :, r, 2] += dd0_mat[r, :] * dG[r, :] + dm0_mat[r, :] * mG[r, :]
    for gi, g in enumerate(sectors):
        F[:, gi, :, 0] += T_dir[g] * dC[:, gi][None, :] + T_mar[g] * mC[:, gi][None, :]
        F[:, gi, :, 1] += T_dir[g] * dI[:, gi][None, :] + T_mar[g] * mI[:, gi][None, :]
        F[:, gi, :, 2] += T_dir[g] * dG[:, gi][None, :] + T_mar[g] * mG[:, gi][None, :]
    return F.reshape(n * S, n * 3)


# ════════════════════════════════════════════════════════════════════════════
#  Build one year (v3.1) -> RAS (v3.1_RAS) -> VA convention
# ════════════════════════════════════════════════════════════════════════════
def build_table(year, gamma_trade=1.0, gamma_margin=1.0):
    """Build the intra-US MRIOT for one year (grav_fric v3.1 schema)."""
    year = type(params['ys0_']['yr'].iloc[0])(year)   # match the GDX 'yr' dtype
    data = load_year_data(year)
    dd0_mat, nd0_mat = data['dd0_'], data['nd0_']
    xn0_mat, xd0_mat, x0_mat = data['xn0_'], data['xd0_'], data['x0_']
    m0_mat, cd0_mat, i0_mat, g0_mat = data['m0_'], data['cd0_'], data['i0_'], data['g0_']
    ld0_mat, kd0_mat, id0_df = data['ld0_'], data['kd0_'], data['id0']

    nm0_mat = (params['nm0_'][params['nm0_']['yr'] == _yr(year)]
               .groupby(['r', 'g'])['value'].sum().unstack('g')
               .reindex(index=regions, columns=sectors, fill_value=0.0).fillna(0.0).values)

    T_dir, T_mar, log = reconstruct_bilateral_3(
        xn0_mat, nd0_mat, nm0_mat, D_np, gamma_trade=gamma_trade, gamma_margin=gamma_margin)

    dm0_mat = xd0_mat - dd0_mat
    nm0_rgm, md0_rmg, _ = build_margin_tensors(year)
    d_int, dC, dI, dG, m_int, mC, mI, mG = compute_use_shares_3(
        id0_df, cd0_mat, i0_mat, g0_mat, nm0_rgm, md0_rmg)

    Z = build_Z_3(dd0_mat, dm0_mat, T_dir, T_mar, d_int, m_int)
    F = build_F_3(dd0_mat, dm0_mat, T_dir, T_mar, dC, dI, dG, mC, mI, mG)
    VA = (ld0_mat + kd0_mat).reshape(n * S)
    EX = x0_mat.reshape(n * S)

    ys0_mat = (params['ys0_'][params['ys0_']['yr'] == _yr(year)].groupby(['r', 's'])['value'].sum()
               .unstack('s').reindex(index=regions, columns=sectors, fill_value=0.0)
               .fillna(0.0).values)
    ty0_mat = load_matrix('ty0_', year)
    tm0_mat = load_matrix('tm0_', year)
    ta0_mat = load_matrix('ta0_', year)

    M = m0_mat.reshape(n * S)
    M_interm = (m0_mat[:, :, None] * d_int).sum(axis=1).reshape(n * S)

    interm_use = Z.reshape(n, S, n, S).sum(axis=(0, 3)).T
    tax_prod = ty0_mat * ys0_mat
    tariff = tm0_mat * m0_mat
    tls_int = ta0_mat * interm_use
    tls_fd = ta0_mat * (cd0_mat + i0_mat + g0_mat)
    taxes = (tax_prod + tariff + tls_int + tls_fd).reshape(n * S)

    return dict(
        Z=Z, F=F, VA=VA, EX=EX, M=M, M_interm=M_interm, taxes=taxes,
        tax_prod=tax_prod.reshape(n * S), tariff=tariff.reshape(n * S),
        tls_int=tls_int.reshape(n * S), tls_fd=tls_fd.reshape(n * S),
        ta0=ta0_mat, tm0=tm0_mat, ty0=ty0_mat,
        ys0=ys0_mat.reshape(n * S),
        regions=np.array(regions), sectors=np.array(sectors))


def ras_table(table, year):
    """Column RAS the v3.1 table onto ys0 (preserving row sums); recompute the
    Z-dependent tax block (tls_int) and combined taxes. -> v3.1_RAS schema."""
    out = {k: v for k, v in table.items()}
    Z = out['Z']
    ys0 = out['ys0']
    u = Z.sum(axis=1)
    v = (ys0 - (out['M_interm'] + out['VA'] + out['taxes'])).clip(0)
    v = v * u.sum() / max(v.sum(), 1e-10)
    Z_ras = ras_2(Z, u, v)
    out['Z'] = Z_ras
    interm_use = Z_ras.reshape(n, S, n, S).sum(axis=(0, 3)).T
    out['tls_int'] = (out['ta0'] * interm_use).reshape(n * S)
    out['taxes'] = out['tax_prod'] + out['tariff'] + out['tls_int'] + out['tls_fd']
    return out


def apply_va_convention(d):
    """OECD/SNA: move production tax into gross VA; tax row becomes TLS (taxes less
    subsidies on products = tariff + tls_int + tls_fd). Output total unchanged."""
    d = {k: v for k, v in d.items()}
    if 'va_convention' in d:
        return d
    d['VA'] = d['VA'] + d['tax_prod']
    d['TLS'] = d['tariff'] + d['tls_int'] + d['tls_fd']
    d['taxes'] = d['TLS'].copy()
    d['va_convention'] = np.array(True)
    return d


def build_v31_ras(year, gamma):
    """Full single-state build for one gamma: v3.1 -> RAS -> VA convention.
    gamma_trade = gamma_margin = gamma (scalar, or a per-sector vector/dict).
    Returns the v3.1_RAS-schema dict."""
    table = build_table(year, gamma_trade=gamma, gamma_margin=gamma)
    table = ras_table(table, year)
    table = apply_va_convention(table)
    table.pop('ys0', None)        # not part of the saved schema
    return table


# ════════════════════════════════════════════════════════════════════════════
#  Sectoral aggregation (mapping + aggregate_windc, from aggregation.ipynb)
# ════════════════════════════════════════════════════════════════════════════
def build_windc_to_proposed():
    """Authoritative WiNDC->proposed-sector mapping (concordance + 5 validated fixes)."""
    conc = pd.read_csv(CONC_PATH)
    w = conc.drop_duplicates("windc_sector").set_index("windc_sector")["proposed sector"].to_dict()
    o = conc.drop_duplicates("oecd_code").set_index("oecd_code")["proposed sector"].to_dict()
    # (1) rnt -> Administrative & support
    w["rnt"] = "Administrative and support service activities"
    # (2) mmf -> Furniture/other-manuf
    w["mmf"] = "Manufacture of furniture"
    # (3) Education + Public administration merged
    for d in (w, o):
        for k, val in list(d.items()):
            if val in ("Education", "Public administration and defence"):
                d[k] = "Education and public administration"
    # (4) Other services + Wholesale/retail merged
    for d in (w, o):
        for k, val in list(d.items()):
            if val in ("Other service activities", "Wholesale and retail trade, stores"):
                d[k] = "Other services + wholesale/retail"
    # (5) fen + OECD H53 -> Warehousing/transport support
    w["fen"] = w["wrh"]
    o["H53"] = w["wrh"]
    return w


def aggregate_windc(table, windc_to_proposed):
    """Aggregate a WiNDC IOT (dict of flat arrays) to the proposed sectors."""
    regions_l = table["regions"].tolist()
    sectors_l = table["sectors"].tolist()
    proposed = [windc_to_proposed.get(s, s) for s in sectors_l]
    full = pd.MultiIndex.from_tuples([(r, p) for r in regions_l for p in proposed],
                                     names=["region", "proposed_sector"])

    Z_agg = pd.DataFrame(table["Z"], index=full, columns=full).groupby(level=["region", "proposed_sector"]).sum()
    Z_agg = Z_agg.T.groupby(level=["region", "proposed_sector"]).sum().T

    def agg_rows(arr): return pd.Series(arr, index=full).groupby(level=["region", "proposed_sector"]).sum()
    def agg_mat(arr):  return pd.DataFrame(arr, index=full).groupby(level=["region", "proposed_sector"]).sum()

    out = {"Z": Z_agg, "F": agg_mat(table["F"])}
    for k in ("VA", "EX", "M", "M_interm", "taxes", "TLS",
              "tax_prod", "tariff", "tls_int", "tls_fd"):
        if k in table:
            out[k] = agg_rows(table[k])

    def _sdiv(a, b): return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=np.abs(b) > 1e-9)
    if all(k in table for k in ("ta0", "tm0", "ty0", "tax_prod", "tariff", "tls_int", "tls_fd")):
        ty0_flat = np.asarray(table["ty0"]).reshape(-1)
        ta0_flat = np.asarray(table["ta0"]).reshape(-1)
        ys0_base = agg_rows(_sdiv(table["tax_prod"], ty0_flat))
        abs_base = agg_rows(_sdiv(table["tls_int"] + table["tls_fd"], ta0_flat))
        out["ty0"] = pd.Series(_sdiv(out["tax_prod"].values, ys0_base.values), index=out["tax_prod"].index)
        out["tm0"] = pd.Series(_sdiv(out["tariff"].values, out["M"].values), index=out["tariff"].index)
        out["ta0"] = pd.Series(_sdiv((out["tls_int"] + out["tls_fd"]).values, abs_base.values), index=out["tls_int"].index)
    if "va_convention" in table:
        out["va_convention"] = np.asarray(table["va_convention"])

    regions_u = list(dict.fromkeys(r for r, _ in Z_agg.index))
    sectors_u = list(dict.fromkeys(p for _, p in Z_agg.index))
    payload = {k: (v.values if hasattr(v, "values") else np.asarray(v)) for k, v in out.items()}
    payload.update(regions=np.array(regions_u), proposed_sectors=np.array(sectors_u),
                   index_labels=np.array([f"{r}_{p}" for r, p in Z_agg.index]))
    return payload


# ════════════════════════════════════════════════════════════════════════════
#  Harmonization (reuse harmonize.py)
# ════════════════════════════════════════════════════════════════════════════
def _resolve_common_sectors(year, agg_payload):
    secs_wd = list(dict.fromkeys(str(s) for s in agg_payload["proposed_sectors"]))
    dfo0 = pd.read_parquet(next((OECD_AGG).rglob(f"{year}_*.parquet")))
    secs_oe_all = [c.split("_", 1)[1] for c in dfo0.columns
                   if c.startswith("USA_") and c.split("_", 1)[1] not in harmonize.FD_CATS]
    secs_common = [s for s in secs_wd if s in set(secs_oe_all)]
    wd_idx = [secs_wd.index(s) for s in secs_common]
    oe_idx = [secs_oe_all.index(s) for s in secs_common]
    return secs_common, secs_oe_all, wd_idx, oe_idx


class _NpzLike(dict):
    """Minimal stand-in for an np.load result: adds a `.files` attribute."""
    @property
    def files(self):
        return list(self.keys())


def harmonize_payload(year, agg_payload):
    """Harmonize an aggregated WiNDC payload to OECD US aggregates (harmonize.py)."""
    secs_common, secs_oe_all, wd_idx, oe_idx = _resolve_common_sectors(year, agg_payload)
    oecd_t = harmonize.load_oecd_targets(year, secs_common, secs_oe_all, oe_idx)
    out, trans = harmonize.harmonize_windc_to_oecd(
        _NpzLike(agg_payload), oecd_t, wd_idx, secs_common)
    return out, trans


# ════════════════════════════════════════════════════════════════════════════
#  Full per-gamma pipeline (writes the standard artifacts to disk)
# ════════════════════════════════════════════════════════════════════════════
def gamma_tag(gamma, label=None):
    """Version tag for a gamma. Scalar -> 'gamma0.1' / 'gamma1.0'; a per-sector
    vector has no single value, so an explicit ``label`` is used (e.g. 'economic')
    and falls back to a short hash of the vector when none is given."""
    if label is not None:
        return f"gamma_{label}"
    if np.isscalar(gamma):
        return f"gamma{gamma:g}"
    arr = np.asarray(gamma, dtype=float).ravel()
    h = hex(abs(hash(arr.round(6).tobytes())) % (16 ** 6))[2:]
    return f"gammavec_{h}"


_GDP_PIVOT = None


def _gdp_pivot():
    """State share of all-industry gross state product, the destination allocator of the
    government final-demand category. Built once and reused across the sweep."""
    global _GDP_PIVOT
    if _GDP_PIVOT is None:
        _GDP_PIVOT = nest_v31.build_gdp_share_pivot()
    return _GDP_PIVOT


def run_pipeline_for_gamma(year, gamma, shares_pivot, save=False, verbose=True, label=None):
    """build -> RAS -> VA conv -> aggregate -> harmonize -> nest, for one gamma.

    ``gamma`` may be a **scalar** (gamma_trade = gamma_margin = gamma, applied to
    every sector) or a **per-sector vector/dict** aligned with ``gs.sectors`` — in
    the latter case pass ``label`` to name the scenario (used for the on-disk tag).

    Returns a dict with the nested DataFrame and the intermediate payloads, all
    held in memory. Nesting requires the harmonized table on disk, so it is
    written to a throwaway temp dir (under /tmp, off the home quota) and removed
    right after.

    save=False (default): nothing is persisted to the project tree — the five
    gamma runs together would be ~1.2 GB. Set save=True to additionally write the
    standard npz/parquet artifacts under version-tagged directories mirroring the
    canonical v3.1_RAS layout (needs free quota).
    """
    import tempfile, shutil
    setup(verbose=False)
    tag = gamma_tag(gamma, label=label)
    if verbose:
        print(f"[{tag}] build (v3.1 -> RAS -> VA conv) ...", flush=True)
    table = build_v31_ras(year, gamma)

    w2p = build_windc_to_proposed()
    agg = aggregate_windc(table, w2p)
    if verbose:
        print(f"[{tag}] aggregated -> {len(agg['proposed_sectors'])} proposed sectors", flush=True)

    harm, trans = harmonize_payload(year, agg)
    if verbose:
        print(f"[{tag}] harmonized: RAS {trans['ras_info']['iters']} iters "
              f"(err {trans['ras_info']['err']:.1e})", flush=True)

    # Nest: nest_year reads the harmonized npz from disk -> write it to a temp dir.
    tmp = Path(tempfile.mkdtemp(prefix=f"harm_{tag}_"))
    try:
        np.savez_compressed(tmp / f"IOT_{year}_harmonized.npz", **harm)
        nest_v31.WINDC_HARM = tmp
        nested = nest_v31.nest_year(year, shares_pivot, _gdp_pivot(),
                                    nest_v31.VA_SOURCE, nest_v31.F_SOURCE)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if save:
        (IOT_ROOT / f"grav_fric_{tag}_RAS").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(IOT_ROOT / f"grav_fric_{tag}_RAS" / f"IOT_{year}.npz", **table)
        agg_dir = IOT_ROOT / f"grav_fric_{tag}_RAS_aggregated"; agg_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(agg_dir / f"IOT_{year}.npz", **agg)
        harm_dir = IOT_ROOT / f"grav_fric_{tag}_RAS_harmonized"; harm_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(harm_dir / f"IOT_{year}_harmonized.npz", **harm)
        out_root = ROOT / f"data/interim/IOT/nested_mriot_{tag}"; out_root.mkdir(parents=True, exist_ok=True)
        nested.to_parquet(out_root / f"nested_mriot_{year}.parquet",
                          engine="fastparquet", compression="gzip")

    if verbose:
        print(f"[{tag}] nested table shape {nested.shape}", flush=True)
    return {"gamma": gamma, "label": label, "tag": tag, "table": table, "agg": agg,
            "harm": harm, "trans": trans, "nested": nested}


# ════════════════════════════════════════════════════════════════════════════
#  Static impact models on the nested table (Leontief / IIM / Ghosh)
#  Mirrors pipeline/static_impact_models.ipynb.
# ════════════════════════════════════════════════════════════════════════════
EXTRA_ROWS = {"OUT", "TLS", "VA"}


def load_nested_blocks(nested):
    """Extract (Z, f, x, v, sectors, region masks) from a nested-table DataFrame."""
    def _is_fd(c):  return "_" in c and c.split("_", 1)[1] in FD_CATS
    def _is_sec(c): return "_" in c and c.split("_", 1)[1] not in FD_CATS

    secs = [r for r in nested.index if r not in EXTRA_ROWS]
    fd_cols = [c for c in nested.columns if _is_fd(c)]
    assert secs == [c for c in nested.columns if _is_sec(c)], "row/col order mismatch"

    Z = nested.loc[secs, secs].to_numpy(float)
    f = nested.loc[secs, fd_cols].to_numpy(float).sum(1)
    x = nested.loc[secs, "OUT"].to_numpy(float)
    v = (nested.loc["VA", secs] + nested.loc["TLS", secs]).to_numpy(float)

    prefix = np.array([s.split("_")[0] for s in secs])
    plen = np.array([len(p) for p in prefix])
    return {"sectors": secs, "prefix": prefix, "Z": Z, "f": f, "x": x, "v": v,
            "mask_us": plen == 2, "mask_world": plen == 3}


def compute_impacts(blocks, shock_region="NY", theta=0.30):
    """Leontief (demand), IIM (inoperability) and Ghosh (supply) shocks on the
    nested table. Returns per-sector impact vectors + a spillover summary DataFrame."""
    Z, f, x, v = blocks["Z"], blocks["f"], blocks["x"], blocks["v"]
    secs, prefix = blocks["sectors"], blocks["prefix"]
    nn = len(secs)
    mask_shock = prefix == shock_region
    mask_us, mask_world = blocks["mask_us"], blocks["mask_world"]

    with np.errstate(divide="ignore", invalid="ignore"):
        A = np.where(x[None, :] > 0, Z / x[None, :], 0.0)
        B = np.where(x[:, None] > 0, Z / x[:, None], 0.0)
    L = np.linalg.inv(np.eye(nn) - A)
    G = np.linalg.inv(np.eye(nn) - B)

    # Leontief: final demand to the shocked state drops by theta
    df = np.zeros(nn); df[mask_shock] = -theta * f[mask_shock]
    dx_leontief = L @ df

    # IIM: inoperability c=theta injected in the shocked state -> economic loss
    c = np.zeros(nn); c[mask_shock] = theta
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(x > 0, (L @ (x * c)) / x, 0.0)
    q = np.clip(q, 0.0, 1.0)
    iim_loss = -(q * x)

    # Ghosh: primary inputs of the shocked state drop by theta -> downstream loss
    dv = np.zeros(nn); dv[mask_shock] = -theta * v[mask_shock]
    dx_ghosh = G.T @ dv

    comp = pd.DataFrame({"Leontief": dx_leontief, "IIM": iim_loss, "Ghosh": dx_ghosh},
                        index=pd.Index(secs, name="sector"))
    summary = pd.DataFrame({
        f"{shock_region}_direct":  comp[mask_shock].sum(),
        "other_US_spillover":      comp[mask_us & ~mask_shock].sum(),
        "world_spillover":         comp[mask_world].sum(),
        "TOTAL":                   comp.sum(),
    }).T
    repro = {"leontief_err": float(np.abs(L @ f - x).max()),
             "ghosh_err": float(np.abs(G.T @ v - x).max())}
    return {"comp": comp, "summary": summary, "repro": repro,
            "L": L, "G": G, "A": A}


# ════════════════════════════════════════════════════════════════════════════
#  Structural diagnostics on the intra-US Z block (as a function of gamma)
# ════════════════════════════════════════════════════════════════════════════
def structural_metrics(table):
    """Structural indicators of one v3.1_RAS state-level build (dict from
    build_v31_ras). All computed on the n_states x n_states bilateral picture."""
    Zst = table["Z"].reshape(n, S, n, S)
    # interstate (off-diagonal-state) vs intrastate (same state) intermediate flows
    same = np.zeros((n, n), bool); np.fill_diagonal(same, True)
    Z_rs = Zst.sum(axis=(1, 3))                       # (state, state) total interm flow
    intra = Z_rs[same].sum()
    inter = Z_rs[~same].sum()
    total = intra + inter

    # mean shipping distance of interstate intermediate trade (flow-weighted)
    w = Z_rs.copy(); np.fill_diagonal(w, 0.0)
    dmask = ~np.isnan(D_np)
    mean_dist = (w[dmask] * D_np[dmask]).sum() / max(w[dmask].sum(), 1e-12)

    # spatial concentration of interstate flows (Gini on off-diagonal cells)
    off = w[~same]
    gini = _gini(off)
    return {
        "intra_share": intra / total,
        "inter_share": inter / total,
        "mean_trade_distance_km": mean_dist,
        "interstate_gini": gini,
        "Z_rs": Z_rs,
    }


def _gini(a):
    a = np.sort(np.asarray(a, float).ravel())
    if a.sum() <= 0:
        return np.nan
    nloc = a.size
    return (2 * np.arange(1, nloc + 1) - nloc - 1).dot(a) / (nloc * a.sum())
