"""Reduced nesting — v3.1_RAS harmonized WiNDC nested into the OECD ICIO MRIOT.

Trimmed down from nesting.ipynb to only what the v3.1_RAS harmonized table needs:
  * harmonized WiNDC sectors (36) are a strict subset of the OECD USA sectors (37);
    the only OECD-extra sector is "T". GFE and the separate "Wholesale and retail
    trade" sector are gone (already merged upstream), so the old GFE/T zero-fill and
    the "Wholesale and retail trade" -> "...stores" merge are no longer needed.
  * "T" is the single OECD sector with no WiNDC counterpart -> zero-filled in Z_ss
    and given the total-GDP share proxy in the state-share matrix.

Pipeline: SAGDP2 state GDP shares -> proportionality split of OECD world<->USA flows
to states -> WiNDC harmonized state<->state block -> assemble -> save -> diagnose.
"""
from pathlib import Path
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
from paths import ROOT
WINDC_VER   = "v3.1_RAS"
OECD_AGG    = ROOT / "data/interim/IOT/OCDE ICIO aggregated"
WINDC_HARM  = ROOT / f"data/interim/IOT/IOT_USA/grav_fric_{WINDC_VER}_harmonized"
SAGDP2_PATH = ROOT / "data/raw/BEA/SAGDP/SAGDP2__ALL_AREAS_1997_2025.csv"
OUT_ROOT    = ROOT / f"data/interim/IOT/nested_mriot_{WINDC_VER}"
OUT_ROOT.mkdir(exist_ok=True, parents=True)

YEAR      = 2017
VA_SOURCE = "oecd_shares"   # "oecd_shares" | "windc"

# Destination allocator of final demand -- Theta of the data descriptor, eq. (Theta).
#   "subnational" : DELIVERED. Sub-national state x category final-demand shares for the
#                   household and investment categories, state share of gross state product
#                   for government. Chosen against external benchmarks; see the Technical
#                   Validation, "Destination allocator of final demand".
#   "breadth"     : the breadth share theta of eq. (theta), used before 2026-07. Kept so the
#                   earlier series can be reproduced.
#   "windc_block" : the WiNDC final-demand block itself, levels included, not just its shares.
F_SOURCE  = "subnational"

FD_CATS     = ["DPABR", "GFCF", "GGFC", "HFCE", "INVNT", "NPISH"]
EXTRA_ROWS  = ["OUT", "TLS", "VA"]
WINDC_SCALE = 1000.0   # WiNDC harmonized in Bn$, OECD in M$

# Grouping of the three sub-national final-demand categories onto the six of the global
# table. The WiNDC F block stores its columns in the order C, I, G.
WCAT_OF_FD  = {"DPABR": 0, "HFCE": 0, "NPISH": 0,    # C, household consumption (cd0)
               "GFCF":  1, "INVNT": 1,               # I, investment (i0)
               "GGFC":  2}                           # G, government (g0)

# Sectors with no SAGDP2 counterpart -> use the total-GDP share proxy
NO_SAGDP_SECTORS = {"Federal government enterprises (GFE)", "T"}

# SAGDP2 LineCode -> proposed sector (only sectors present in the harmonized build)
PROPOSED_TO_SAGDP2 = {
    "Accommodation and food service activities": [80, 81],
    "Administrative and support service activities": [66],
    "Agriculture, hunting, forestry, fishing and related": [4, 5],
    "Air transport": [37],
    "Arts, entertainment and recreation activities": [77, 78],
    "Construction": [11],
    "Education and public administration": [69, 84, 85, 86],
    "Electricity, gas, steam and air conditioning supply, and wasterwater": [10, 67],
    "Financial and insurance activities": [52, 53, 54, 55],
    "Human health and social work activities": [71, 72, 73, 74],
    "Land transport and transport via pipelines": [38, 40, 41, 42],
    "Manufacture of chemicals and chemical products, pharmaceutical": [32],
    "Manufacture of coke and refined petroleum products": [31],
    "Manufacture of computer, electronic and optical products": [19],
    "Manufacture of electrical equipment": [20],
    "Manufacture of fabricated metal products": [17],
    "Manufacture of food and beverage and tobacco products": [26],
    "Manufacture of furniture": [23],
    "Manufacture of machinery and equipment n.e.c. ": [18, 24],
    "Manufacture of motor vehicles, trailers and semi-trailers": [21],
    "Manufacture of other non-metallic mineral products": [15],
    "Manufacture of other transport equipment": [22],
    "Manufacture of paper and paper products": [29, 30],
    "Manufacture of rubber and plastic products": [33],
    "Manufacture of textiles, wearing apparel, leather and related products": [27, 28],
    "Manufacture of wood and of products of wood and cork": [14],
    "Mining, except oil & gas": [8],
    "Oil and gas extraction": [7],
    # merged sector in the harmonized build: other services + wholesale/retail
    "Other services + wholesale/retail": [58, 82, 34, 35],
    "Professional, scientific and technical activities": [61, 63, 64],
    "Real estate activities": [57],
    "Support activities for mining": [9],
    "Warehousing and support activities for transportation": [43, 44],
    "Water transport": [39],
    "broadcasting, telecommunications, data processing, publishing, information services, motion picture, video, television": [46, 47, 48, 49, 62],
    "primary metals": [16],
}

STATE_NAME_TO_ABBR = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'District of Columbia': 'DC', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI',
    'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
    'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY',
}


# ── 1. SAGDP2 state shares (share of each state in US sectoral GDP, per year) ──
def build_shares_pivot():
    df = pd.read_csv(SAGDP2_PATH)
    df["GeoName"] = df["GeoName"].str.strip()
    df["GeoFIPS"] = df["GeoFIPS"].str.strip().str.strip('"')
    year_cols = [c for c in df.columns if c.isdigit() and int(c) <= 2024]
    fips = pd.to_numeric(df["GeoFIPS"], errors="coerce")
    df = df[fips.between(1000, 57000)].copy()                # states only
    df[year_cols] = df[year_cols].apply(pd.to_numeric, errors="coerce")

    line_to_sector = {lc: sec for sec, lines in PROPOSED_TO_SAGDP2.items() for lc in lines}
    df = df[df["LineCode"].isin(line_to_sector)].copy()
    df["proposed_sector"] = df["LineCode"].map(line_to_sector)

    wide = (df.groupby(["GeoName", "proposed_sector"])[year_cols]
              .sum(min_count=1).reset_index())
    long = wide.melt(id_vars=["GeoName", "proposed_sector"], value_vars=year_cols,
                     var_name="year", value_name="gdp")
    long["year"] = long["year"].astype(int)
    long["share"] = (long.groupby(["proposed_sector", "year"])["gdp"]
                         .transform(lambda x: x / x.sum())).fillna(0)
    long["abbr"] = long["GeoName"].map(STATE_NAME_TO_ABBR)
    return (long.dropna(subset=["abbr"])
                .set_index(["abbr", "proposed_sector", "year"])["share"])


def build_gdp_share_pivot():
    """State share of all-industry gross state product, per year (SAGDP2 line code 1).
    This is the destination allocator of the government final-demand category."""
    df = pd.read_csv(SAGDP2_PATH)
    df["GeoName"] = df["GeoName"].str.strip()
    df["GeoFIPS"] = df["GeoFIPS"].str.strip().str.strip('"')
    year_cols = [c for c in df.columns if c.isdigit() and int(c) <= 2024]
    fips = pd.to_numeric(df["GeoFIPS"], errors="coerce")
    df = df[fips.between(1000, 57000) & (df["LineCode"] == 1)].copy()   # states, all industries
    df[year_cols] = df[year_cols].apply(pd.to_numeric, errors="coerce")
    df["abbr"] = df["GeoName"].map(STATE_NAME_TO_ABBR)
    long = (df.dropna(subset=["abbr"])
              .melt(id_vars=["abbr"], value_vars=year_cols, var_name="year", value_name="gdp"))
    long["year"] = long["year"].astype(int)
    long["share"] = long.groupby("year")["gdp"].transform(lambda x: x / x.sum())
    return long.set_index(["abbr", "year"])["share"]


def get_share_matrix(year, states, oecd_secs, shares_pivot):
    """S[state, sector] in [0,1], columns sum to 1. Sectors absent from SAGDP2
    ("T", GFE) get the total-GDP share proxy (mean share across sectors)."""
    S = np.zeros((len(states), len(oecd_secs)))
    for ji, sec in enumerate(oecd_secs):
        if sec in NO_SAGDP_SECTORS:
            continue
        for si, st in enumerate(states):
            key = (st, sec, year)
            if key in shares_pivot.index:
                S[si, ji] = shares_pivot[key]
    col = S.sum(0, keepdims=True); col[col == 0] = 1
    S /= col
    total_share = S.mean(1); total_share /= total_share.sum()
    for ji, sec in enumerate(oecd_secs):
        if sec in NO_SAGDP_SECTORS:
            S[:, ji] = total_share
    return S


# ── 1b. Destination allocator of final demand, Theta of eq. (Theta) ───────────
def subnational_fd_shares(npz, states, wd_secs):
    """psi[state, c] for c in (C, I, G): the state share of the national total of each
    sub-national final-demand category, summed over origin states and goods.

    Taken from the harmonised block, so the category totals are already those of the
    global table. Harmonisation scales one sector uniformly across states, which leaves
    the shares within a sector untouched and moves the sector-aggregated shares only
    through the change in composition."""
    n_s, n_wd = len(states), len(wd_secs)
    dest = npz["F"].reshape(n_s, n_wd, n_s, 3).sum((0, 1))       # (state, category)
    tot = dest.sum(0, keepdims=True)
    return dest / np.where(tot > 0, tot, 1.0)


def fd_allocator(mode, npz, states, wd_secs, gdp_share, breadth_share):
    """Theta[state, fd_cat], one column per OECD final-demand category, each summing to
    one over states so that the national total of the global table is conserved exactly.

    mode "subnational" is the delivered specification: the sub-national state x category
    shares for household consumption and investment, the state share of gross state
    product for government. mode "breadth" reproduces the earlier series, which applied
    the single breadth share to all six categories."""
    n_s = len(states)
    if mode == "breadth":
        return np.repeat(breadth_share[:, None], len(FD_CATS), axis=1)

    psi = subnational_fd_shares(npz, states, wd_secs)            # (n_s, 3) on C, I, G
    theta = np.zeros((n_s, len(FD_CATS)))
    for k, cat in enumerate(FD_CATS):
        theta[:, k] = gdp_share if cat == "GGFC" else psi[:, WCAT_OF_FD[cat]]
    bad = np.abs(theta.sum(0) - 1.0) > 1e-9
    if bad.any():
        raise ValueError(f"allocator columns do not sum to one: "
                         f"{[FD_CATS[k] for k in np.where(bad)[0]]}")
    return theta


# ── 2. Proportionality split (USA aggregate -> states by share) ───────────────
def expand_cols_by_share(block, S):       # (R, J),(St,J) -> (R, St*J)
    return (block[:, None, :] * S[None, :, :]).reshape(block.shape[0], -1)

def expand_rows_by_share(block, S):       # (I, C),(St,I) -> (St*I, C)
    return (block[None, :, :] * S[:, :, None]).reshape(-1, block.shape[1])


# ── 3. WiNDC (36 sec) -> OECD (37 sec) re-indexing; "T" stays zero ────────────
def _wd_to_oc(wd_secs, oecd_secs):
    return {wi: oecd_secs.index(ws) for wi, ws in enumerate(wd_secs) if ws in oecd_secs}

def windc_Z_to_oecd(Z_wd, wd_secs, oecd_secs, n_s):
    n_wd, n_oc = len(wd_secs), len(oecd_secs)
    m = _wd_to_oc(wd_secs, oecd_secs)
    out = np.zeros((n_s * n_oc, n_s * n_oc))
    for sr in range(n_s):
        for sc in range(n_s):
            sub = Z_wd[sr*n_wd:(sr+1)*n_wd, sc*n_wd:(sc+1)*n_wd]
            for wr, orr in m.items():
                for wc, occ in m.items():
                    out[sr*n_oc + orr, sc*n_oc + occ] += sub[wr, wc]
    return out

def windc_1d_to_oecd(arr, wd_secs, oecd_secs, n_s):
    n_wd, n_oc = len(wd_secs), len(oecd_secs)
    m = _wd_to_oc(wd_secs, oecd_secs)
    out = np.zeros(n_s * n_oc)
    for s in range(n_s):
        for wi, oc in m.items():
            out[s*n_oc + oc] += arr[s*n_wd + wi]
    return out

def windc_F_to_oecd(F_wd, wd_secs, oecd_secs, n_s, F_uu):
    """WiNDC F (n_s*n_wd, n_s*3) -> (n_s*n_oc, n_s*6). WiNDC FD: C/I/G; split into
    OECD's 6 categories with sector-level proportions taken from OECD USA F_uu."""
    n_wd, n_oc, n_fd = len(wd_secs), len(oecd_secs), len(FD_CATS)
    m = _wd_to_oc(wd_secs, oecd_secs)
    F_row = np.zeros((n_s * n_oc, n_s * 3))
    for s in range(n_s):
        for wi, oc in m.items():
            F_row[s*n_oc + oc, :] += F_wd[s*n_wd + wi, :]
    # FD groups: C->{DPABR,HFCE,NPISH}, I->{GFCF,INVNT}, G->{GGFC}
    C = F_uu[:, 0] + F_uu[:, 3] + F_uu[:, 5]
    I = F_uu[:, 1] + F_uu[:, 4]
    ratio = np.zeros((n_oc, 3, 6))
    ratio[:, 0, 0] = np.where(C > 0, F_uu[:, 0] / C, 0.0)
    ratio[:, 0, 3] = np.where(C > 0, F_uu[:, 3] / C, 1.0)
    ratio[:, 0, 5] = np.where(C > 0, F_uu[:, 5] / C, 0.0)
    ratio[:, 1, 1] = np.where(I > 0, F_uu[:, 1] / I, 0.5)
    ratio[:, 1, 4] = np.where(I > 0, F_uu[:, 4] / I, 0.5)
    ratio[:, 2, 2] = 1.0
    rows_sec = np.arange(n_s * n_oc) % n_oc
    F_3d = F_row.reshape(n_s * n_oc, n_s, 3)
    out = np.einsum("rsc,rck->rsk", F_3d, ratio[rows_sec])
    return out.reshape(n_s * n_oc, n_s * n_fd)


# ── 3b. Residual closure of the state cost accounts ───────────────────────────
def close_state_columns(df, clip_va=True):
    """Close every state column through the value-added row, in place on a copy.

    Implements equations (closureVA) and (closureTLS) of the data descriptor:

        VA[s,j]  = ( OUT[s,j] - z_in[s,j] - TLS0[s,j] )_+
        TLS[s,j] = TLS0[s,j] + min(0, OUT[s,j] - z_in[s,j] - TLS0[s,j])

    with ``z_in`` the realised intermediate input bill of the column. The state
    columns mix three provenances -- world inputs from the share expansion, intra-US
    inputs from the harmonised sub-national block, and tax and value-added priors
    from the shares -- so they do not close spontaneously. No intermediate or final
    demand cell is touched, hence no row total moves.

    Returns ``(df_closed, info)``; ``info["residual_before"]`` is the pre-closure
    cost-identity residual over the state columns, the quantity plotted in the
    Technical Validation.
    """
    ind_rows = [r for r in df.index if r not in EXTRA_ROWS]
    sec_cols = [c for c in df.columns
                if "_" in c and c != "OUT" and c.split("_", 1)[1] not in FD_CATS]
    state_col = np.array([len(c.split("_")[0]) == 2 for c in sec_cols])

    out = df.copy()
    Z_col = out.loc[ind_rows, sec_cols].values.astype(float).sum(0)
    TLS0 = out.loc["TLS", sec_cols].values.astype(float)
    VA0 = out.loc["VA", sec_cols].values.astype(float)
    OUT_row = out.loc["OUT", sec_cols].values.astype(float)

    residual = OUT_row - Z_col - TLS0          # the VA the identity calls for
    VA_new, TLS_new = VA0.copy(), TLS0.copy()

    if clip_va:
        neg = (residual < 0) & state_col
        TLS_new[neg] = OUT_row[neg] - Z_col[neg]   # the tax row absorbs the shortfall
        VA_new[state_col] = np.maximum(residual, 0.0)[state_col]
    else:
        neg = np.zeros_like(state_col)
        VA_new[state_col] = residual[state_col]

    out.loc["VA", sec_cols] = VA_new
    out.loc["TLS", sec_cols] = TLS_new

    after = (out.loc[ind_rows, sec_cols].values.astype(float).sum(0)
             + VA_new + TLS_new - OUT_row)
    info = {
        "n_state_columns": int(state_col.sum()),
        "n_va_clipped": int(neg.sum()),
        "clipped_mask": neg,
        "tls_absorbed": float((TLS_new - TLS0)[neg].sum()),
        "residual_before": residual - VA0,     # cost-identity gap, per column
        "va_prior": VA0,
        "va_delivered": VA_new,
        "state_col_mask": state_col,
        "max_residual_after": float(np.abs(after[state_col]).max()),
    }
    return out, info


def find_oecd_file(year):
    for folder in OECD_AGG.iterdir():
        if folder.is_dir():
            for f in folder.glob(f"{year}_*.parquet"):
                return f
    raise FileNotFoundError(f"OECD {year} not found")


# ── 4. Nesting for one year ───────────────────────────────────────────────────
def nest_year(year, shares_pivot, gdp_pivot, va_source, f_source, close=True):
    df_oecd = pd.read_parquet(find_oecd_file(year))
    npz = np.load(WINDC_HARM / f"IOT_{year}_harmonized.npz", allow_pickle=True)
    states  = sorted(set(l.split("_")[0] for l in npz["index_labels"]))
    wd_secs = list(npz["proposed_sectors"])

    oecd_secs = [r.split("_", 1)[1] for r in df_oecd.index if r.startswith("USA_")]
    n_oc, n_s = len(oecd_secs), len(states)

    rows, cols = list(df_oecd.index), list(df_oecd.columns)
    usa_row   = [i for i, r in enumerate(rows) if r.startswith("USA_")]
    world_row = [i for i, r in enumerate(rows)
                 if "_" in r and not r.startswith("USA_") and len(r.split("_")[0]) == 3]
    extra_row = [i for i, r in enumerate(rows) if r in EXTRA_ROWS]
    usa_sec_c = [j for j, c in enumerate(cols) if c.startswith("USA_") and c.split("_", 1)[1] in oecd_secs]
    usa_fd_c  = [j for j, c in enumerate(cols) if c.startswith("USA_") and c.split("_", 1)[1] in FD_CATS]
    world_sec_c = [j for j, c in enumerate(cols)
                   if "_" in c and not c.startswith("USA_") and len(c.split("_")[0]) == 3
                   and c.split("_", 1)[1] in oecd_secs]
    world_fd_c  = [j for j, c in enumerate(cols)
                   if "_" in c and not c.startswith("USA_") and len(c.split("_")[0]) == 3
                   and c.split("_", 1)[1] in FD_CATS]
    out_c = [j for j, c in enumerate(cols) if c == "OUT"]

    # The final-demand allocator is category-specific, so its columns must follow the
    # order in which the final-demand columns were actually selected above, not the
    # order of FD_CATS. Guard rather than assume: a different vintage of the global
    # table would otherwise silently allocate one category with another's shares.
    usa_fd_order = [cols[j].split("_", 1)[1] for j in usa_fd_c]
    if usa_fd_order != FD_CATS:
        raise ValueError(f"unexpected final-demand column order {usa_fd_order}, "
                         f"expected {FD_CATS}")

    mat = df_oecd.values.astype(float)
    Z_wu = mat[np.ix_(world_row, usa_sec_c)]
    Z_uw = mat[np.ix_(usa_row,   world_sec_c)]
    Z_ww = mat[np.ix_(world_row, world_sec_c)]
    F_wu = mat[np.ix_(world_row, usa_fd_c)]
    F_uw = mat[np.ix_(usa_row,   world_fd_c)]
    F_uu = mat[np.ix_(usa_row,   usa_fd_c)]
    F_ww = mat[np.ix_(world_row, world_fd_c)]
    E_wu = mat[np.ix_(extra_row, usa_sec_c)]
    E_ww = mat[np.ix_(extra_row, world_sec_c)]
    E_uw_fd = mat[np.ix_(extra_row, world_fd_c)]
    E_uu_fd = mat[np.ix_(extra_row, usa_fd_c)]
    OUT_w = mat[np.ix_(world_row, out_c)]
    OUT_e = mat[np.ix_(extra_row, out_c)]

    S = get_share_matrix(year, states, oecd_secs, shares_pivot)
    # Cross-sector mean of S. Two distinct uses, do not conflate them: it fills the
    # column of S of a sector with no SAGDP2 counterpart (done inside get_share_matrix,
    # eq. (theta)), and it is the legacy final-demand allocator kept under
    # f_source="breadth".
    total_share = S.mean(1); total_share /= total_share.sum()

    gdp_share = np.array([gdp_pivot[(st, year)] for st in states])
    Theta = fd_allocator(f_source, npz, states, wd_secs, gdp_share, total_share)

    # Z blocks
    Z_ws = expand_cols_by_share(Z_wu, S)
    Z_sw = expand_rows_by_share(Z_uw, S)
    Z_ss = windc_Z_to_oecd(npz["Z"] * WINDC_SCALE, wd_secs, oecd_secs, n_s)

    # F blocks. Theta sets the destination state of the two blocks whose final demand
    # terminates in the United States, the imported one and the intra-US one; the export
    # block is split on its state of origin by S and never sees the allocator.
    F_ws = (F_wu[:, None, :] * Theta[None, :, :]).reshape(F_wu.shape[0], n_s * len(FD_CATS))
    F_sw = expand_rows_by_share(F_uw, S)
    if f_source == "windc_block":
        F_ss = windc_F_to_oecd(npz["F"] * WINDC_SCALE, wd_secs, oecd_secs, n_s, F_uu)
    else:
        tmp = (F_uu[None, :, :] * S[:, :, None]).reshape(n_s * n_oc, len(FD_CATS))
        F_ss = (tmp[:, None, :] * Theta[None, :, :]).reshape(n_s * n_oc, n_s * len(FD_CATS))

    # extra rows for state columns
    E_ws = expand_cols_by_share(E_wu, S)
    if va_source == "windc":
        E_ws[EXTRA_ROWS.index("VA"), :] = windc_1d_to_oecd(npz["VA"] * WINDC_SCALE, wd_secs, oecd_secs, n_s)
    E_sw_fd = (E_uu_fd[:, None, :] * Theta[None, :, :]).reshape(len(EXTRA_ROWS), n_s * len(FD_CATS))

    # assemble
    Z_full = np.vstack([np.hstack([Z_ww, Z_ws]), np.hstack([Z_sw, Z_ss])])
    F_full = np.vstack([np.hstack([F_ww, F_ws]), np.hstack([F_sw, F_ss])])
    bottom = np.hstack([Z_sw, Z_ss]); F_bottom = np.hstack([F_sw, F_ss])
    OUT_s = (bottom.sum(1) + F_bottom.sum(1))[:, None]   # real row sum -> row balance
    OUT_full = np.vstack([OUT_w, OUT_s])
    E_ws[0, :] = OUT_s.flatten()                          # sync OUT extra row

    ind_block = np.hstack([Z_full, F_full, OUT_full])
    ext_block = np.hstack([np.hstack([E_ww, E_ws]), np.hstack([E_uw_fd, E_sw_fd]), OUT_e])
    full_mat  = np.vstack([ind_block, ext_block])

    world_row_lbl = [rows[i] for i in world_row]
    state_row_lbl = [f"{s}_{sec}" for s in states for sec in oecd_secs]
    row_labels    = world_row_lbl + state_row_lbl + EXTRA_ROWS
    world_sec_lbl = [cols[j] for j in world_sec_c]
    state_sec_lbl = [f"{s}_{sec}" for s in states for sec in oecd_secs]
    world_fd_lbl  = [cols[j] for j in world_fd_c]
    state_fd_lbl  = [f"{s}_{fd}" for s in states for fd in FD_CATS]
    col_labels    = world_sec_lbl + state_sec_lbl + world_fd_lbl + state_fd_lbl + ["OUT"]

    nested = pd.DataFrame(full_mat, index=row_labels, columns=col_labels)
    if close:
        nested, _ = close_state_columns(nested)
    return nested


if __name__ == "__main__":
    print(f"Nesting {YEAR}  (WiNDC {WINDC_VER}, VA={VA_SOURCE}, F={F_SOURCE})")
    shares_pivot = build_shares_pivot()
    gdp_pivot    = build_gdp_share_pivot()
    nested = nest_year(YEAR, shares_pivot, gdp_pivot, VA_SOURCE, F_SOURCE)
    out_path = OUT_ROOT / f"nested_mriot_{YEAR}.parquet"
    nested.to_parquet(out_path, engine="fastparquet", compression="gzip")
    print(f"  shape {nested.shape}  ->  saved {out_path}")
