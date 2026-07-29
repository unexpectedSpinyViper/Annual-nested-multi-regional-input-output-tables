"""
Comparison of the four candidate distance matrices used by the gravity seed.

The gravity seed of the intra-US MRIOT (pipeline.ipynb / v3_construction.ipynb, cell
"State capital coordinates for distance matrix") needs one reference point per region.
Four candidates are compared here, for the 51 WiNDC regions (50 states + DC):

  cap   political      state capital                      (the CAPITALS dict of the pipeline)
  geo   geometric      land-area-weighted mean of county internal points (Census Gazetteer 2020)
  pop   demographic    Census 2020 mean center of population (state file)
  gdp   economic       county-GDP-weighted mean of county population centroids (BEA CAGDP2)

For each variant a 51x51 great-circle (haversine, R = 6371 km) distance matrix is built,
and the 6 pairs of matrices are compared on the 1275 unordered off-diagonal pairs.
Alaska and Hawaii inflate every distance, so every statistic is also reported for the
CONUS subset (49 regions, 1176 pairs).

Outputs (written to figures/ at the repository root):
  centroids_four_variants.csv          the four reference points per region
  distance_matrix_comparison.csv       pairwise summary statistics
  fig_distance_variants.png / .pdf     3-panel figure for the Supplementary

Data are cached in CACHE_DIR; delete it to force a re-download.
"""

from __future__ import annotations

import io
import os
import zipfile
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from _figpaths import FIG_DIR, ROOT

HERE = FIG_DIR
# Census downloads are cached here; set DISTANCE_VARIANTS_CACHE to move it, for
# instance to node-local scratch on an HPC. Delete the directory to re-download.
CACHE_DIR = Path(
    os.environ.get("DISTANCE_VARIANTS_CACHE", ROOT / "data/interim/distance_variants_cache")
)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

GAZETTEER_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/"
    "2020_Gaz_counties_national.zip"
)
# 2023 vintage: needed only for the Connecticut planning regions, which replaced the
# CT counties in 2022 and are the geography BEA now reports county GDP on.
GAZETTEER_2023_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/"
    "2023_Gaz_counties_national.zip"
)
CENPOP_STATE_URL = "https://www2.census.gov/geo/docs/reference/cenpop2020/CenPop2020_Mean_ST.txt"
CENPOP_COUNTY_URL = (
    "https://www2.census.gov/geo/docs/reference/cenpop2020/county/CenPop2020_Mean_CO.txt"
)
BEA_CAGDP2_URL = "https://apps.bea.gov/regional/zip/CAGDP2.zip"

HEADERS = {"User-Agent": "Mozilla/5.0"}

# Exactly the dict used by the delivered build (pipeline.ipynb cell 13).
CAPITALS = {
    "AL": (32.36, -86.30), "AK": (58.30, -134.42), "AZ": (33.45, -112.07),
    "AR": (34.74, -92.33), "CA": (38.56, -121.47), "CO": (39.73, -104.98),
    "CT": (41.77, -72.68), "DE": (39.16, -75.52), "DC": (38.91, -77.01),
    "FL": (30.44, -84.28), "GA": (33.76, -84.39), "HI": (21.31, -157.82),
    "ID": (43.61, -116.20), "IL": (39.78, -89.65), "IN": (39.79, -86.15),
    "IA": (41.59, -93.62), "KS": (39.04, -95.69), "KY": (38.20, -84.87),
    "LA": (30.45, -91.14), "ME": (44.32, -69.77), "MD": (38.97, -76.50),
    "MA": (42.36, -71.06), "MI": (42.73, -84.55), "MN": (44.95, -93.10),
    "MS": (32.32, -90.21), "MO": (38.57, -92.17), "MT": (46.60, -112.02),
    "NE": (40.81, -96.68), "NV": (39.16, -119.75), "NH": (43.22, -71.55),
    "NJ": (40.22, -74.77), "NM": (35.68, -105.96), "NY": (42.66, -73.80),
    "NC": (35.77, -78.64), "ND": (46.81, -100.78), "OH": (39.96, -83.00),
    "OK": (35.48, -97.53), "OR": (44.93, -123.03), "PA": (40.27, -76.88),
    "RI": (41.82, -71.42), "SC": (34.00, -81.03), "SD": (44.37, -100.34),
    "TN": (36.17, -86.78), "TX": (30.27, -97.74), "UT": (40.78, -111.89),
    "VT": (44.26, -72.58), "VA": (37.54, -77.44), "WA": (47.04, -122.90),
    "WV": (38.35, -81.63), "WI": (43.07, -89.39), "WY": (41.14, -104.82),
}

# FIPS -> USPS, restricted to the 51 WiNDC regions.
FIPS_TO_ABBR = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT",
    "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL",
    "18": "IN", "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD",
    "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT", "31": "NE",
    "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA", "54": "WV",
    "55": "WI", "56": "WY",
}

# BEA does not report GDP separately for most Virginia independent cities (nor for
# Kalawao HI): they are merged with the surrounding county into a "combination area"
# whose FIPS has no Census counterpart. 44 % of Virginia's GDP sits in these rows, so
# they cannot be dropped -- each one is anchored on the population-weighted centroid of
# its constituent counties. Constituents are checked against the Census names at run time.
BEA_COMBINATION_AREAS = {
    "15901": ["15009", "15005"],                     # Maui + Kalawao
    "51901": ["51003", "51540"],                     # Albemarle + Charlottesville
    "51903": ["51005", "51580"],                     # Alleghany + Covington
    "51907": ["51015", "51790", "51820"],            # Augusta, Staunton + Waynesboro
    "51911": ["51031", "51680"],                     # Campbell + Lynchburg
    "51913": ["51035", "51640"],                     # Carroll + Galax
    "51918": ["51053", "51570", "51730"],            # Dinwiddie, Colonial Heights + Petersburg
    "51919": ["51059", "51600", "51610"],            # Fairfax, Fairfax City + Falls Church
    "51921": ["51069", "51840"],                     # Frederick + Winchester
    "51923": ["51081", "51595"],                     # Greensville + Emporia
    "51929": ["51089", "51690"],                     # Henry + Martinsville
    "51931": ["51095", "51830"],                     # James City + Williamsburg
    "51933": ["51121", "51750"],                     # Montgomery + Radford
    "51939": ["51143", "51590"],                     # Pittsylvania + Danville
    "51941": ["51149", "51670"],                     # Prince George + Hopewell
    "51942": ["51153", "51683", "51685"],            # Prince William, Manassas + Manassas Park
    "51944": ["51161", "51775"],                     # Roanoke + Salem
    "51945": ["51163", "51530", "51678"],            # Rockbridge, Buena Vista + Lexington
    "51947": ["51165", "51660"],                     # Rockingham + Harrisonburg
    "51949": ["51175", "51620"],                     # Southampton + Franklin
    "51951": ["51177", "51630"],                     # Spotsylvania + Fredericksburg
    "51953": ["51191", "51520"],                     # Washington + Bristol
    "51955": ["51195", "51720"],                     # Wise + Norton
    "51958": ["51199", "51735"],                     # York + Poquoson
}

REGIONS = sorted(CAPITALS)          # 51 regions, alphabetical -- same set as the build
NON_CONUS = ("AK", "HI")
VARIANTS = ["cap", "geo", "pop", "gdp"]
VARIANT_LABEL = {
    "cap": "capital (political)",
    "geo": "geometric centroid",
    "pop": "population-weighted",
    "gdp": "GDP-weighted",
}


# --------------------------------------------------------------------------- I/O
def fetch(url: str) -> bytes:
    """Download `url`, caching the payload under CACHE_DIR."""
    path = CACHE_DIR / url.rsplit("/", 1)[-1]
    if not path.exists():
        resp = requests.get(url, timeout=120, headers=HEADERS)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return path.read_bytes()


def read_gazetteer(url: str) -> pd.DataFrame:
    """County-level Gazetteer file: GEOID, land area (m2) and internal point."""
    with zipfile.ZipFile(io.BytesIO(fetch(url))) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".txt"))
        with zf.open(name) as fh:
            df = pd.read_csv(fh, sep="\t", dtype={"GEOID": str}, encoding="latin-1")
    df.columns = [c.strip() for c in df.columns]
    df["fips"] = df["GEOID"].str.zfill(5)
    df["state_fips"] = df["fips"].str[:2]
    df["abbr"] = df["state_fips"].map(FIPS_TO_ABBR)
    return df.dropna(subset=["abbr"])


def geometric_centroids() -> pd.DataFrame:
    """Land-area-weighted mean of the county internal points (Census Gazetteer 2020).

    The centroid of a union of polygons is the area-weighted mean of the sub-polygon
    centroids; internal points stand in for county centroids, which keeps the estimate
    within a few km of a shapefile centroid without requiring geopandas.
    """
    df = read_gazetteer(GAZETTEER_URL)
    grp = df.groupby("abbr")
    out = pd.DataFrame(
        {
            "geo_lat": grp.apply(
                lambda d: np.average(d["INTPTLAT"], weights=d["ALAND"]), include_groups=False
            ),
            "geo_lon": grp.apply(
                lambda d: np.average(d["INTPTLONG"], weights=d["ALAND"]), include_groups=False
            ),
            "land_area_km2": grp["ALAND"].sum() / 1e6,
        }
    )
    return out.reset_index()


def population_centroids() -> pd.DataFrame:
    """Census 2020 mean center of population, state file (block-level anchors)."""
    df = pd.read_csv(io.BytesIO(fetch(CENPOP_STATE_URL)), encoding="utf-8-sig",
                     dtype={"STATEFP": str})
    df.columns = [c.strip().upper() for c in df.columns]
    df["abbr"] = df["STATEFP"].str.zfill(2).map(FIPS_TO_ABBR)
    df = df.dropna(subset=["abbr"])
    return df.rename(columns={"LATITUDE": "pop_lat", "LONGITUDE": "pop_lon",
                              "POPULATION": "population"})[
        ["abbr", "population", "pop_lat", "pop_lon"]
    ]


def gdp_centroids() -> pd.DataFrame:
    """County-GDP-weighted mean of the county population centroids (BEA CAGDP2)."""
    co = pd.read_csv(io.BytesIO(fetch(CENPOP_COUNTY_URL)), encoding="utf-8-sig",
                     dtype={"STATEFP": str, "COUNTYFP": str})
    co.columns = [c.strip().upper() for c in co.columns]
    co["fips"] = co["STATEFP"].str.zfill(2) + co["COUNTYFP"].str.zfill(3)
    co = co.rename(columns={"LATITUDE": "county_lat", "LONGITUDE": "county_lon"})

    with zipfile.ZipFile(io.BytesIO(fetch(BEA_CAGDP2_URL))) as zf:
        name = next(n for n in zf.namelist() if n.upper().endswith(".CSV"))
        with zf.open(name) as fh:
            raw = pd.read_csv(fh, encoding="latin-1", low_memory=False)
    raw.columns = [c.strip() for c in raw.columns]
    raw["GeoFIPS"] = raw["GeoFIPS"].astype(str).str.replace('"', "").str.strip().str.zfill(5)
    total = raw[raw["LineCode"] == 1].copy()               # all-industry total
    year_cols = [c for c in total.columns if c.strip().isdigit()]
    latest = max(year_cols, key=int)
    total["gdp"] = pd.to_numeric(total[latest], errors="coerce")
    county = total[~total["GeoFIPS"].str.endswith("000")][["GeoFIPS", "gdp"]]
    county = county.rename(columns={"GeoFIPS": "fips"}).dropna()

    # BEA carries both the pre-2022 CT counties (09001-09015, empty in recent years) and
    # the planning regions that replaced them (09110-09190). Keep one of the two.
    ct_new = county["fips"].str.match(r"091[1-9]0$")
    if ct_new.any():
        county = county[~county["fips"].str.match(r"090[0-9][0-9]$") | ct_new]

    # Anchors: 2020 county population centroids, completed by Gazetteer internal points
    # for the geographies Census has not published a 2020 population centroid for (CT).
    anchors = co[["fips", "county_lat", "county_lon"]]
    gaz = read_gazetteer(GAZETTEER_2023_URL)
    extra = gaz[~gaz["fips"].isin(anchors["fips"])][["fips", "INTPTLAT", "INTPTLONG"]]
    extra = extra.rename(columns={"INTPTLAT": "county_lat", "INTPTLONG": "county_lon"})
    anchors = pd.concat([anchors, extra], ignore_index=True)

    # ... and by the population-weighted centroid of each BEA combination area.
    pop_co = co.set_index("fips")
    combos = []
    for combo, parts in BEA_COMBINATION_AREAS.items():
        unknown = [p for p in parts if p not in pop_co.index]
        if unknown:
            raise RuntimeError(f"combination area {combo}: unknown constituents {unknown}")
        d = pop_co.loc[parts]
        combos.append({
            "fips": combo,
            "county_lat": np.average(d["county_lat"], weights=d["POPULATION"]),
            "county_lon": np.average(d["county_lon"], weights=d["POPULATION"]),
        })
    anchors = pd.concat([anchors, pd.DataFrame(combos)], ignore_index=True)
    # The constituents are also reported individually by Census; BEA reports only the
    # combination, so no double counting can occur on the GDP side.

    m = anchors.merge(county, on="fips")
    unmatched = set(county["fips"]) - set(m["fips"])
    if unmatched:
        lost = county[county["fips"].isin(unmatched)]["gdp"].sum() / county["gdp"].sum()
        print(f"[warn] {len(unmatched)} BEA counties without an anchor "
              f"({100 * lost:.2f} % of US county GDP): {sorted(unmatched)[:10]}")
    m["abbr"] = m["fips"].str[:2].map(FIPS_TO_ABBR)
    m = m.dropna(subset=["abbr"])
    grp = m.groupby("abbr")
    out = pd.DataFrame(
        {
            "gdp_lat": grp.apply(
                lambda d: np.average(d["county_lat"], weights=d["gdp"]), include_groups=False
            ),
            "gdp_lon": grp.apply(
                lambda d: np.average(d["county_lon"], weights=d["gdp"]), include_groups=False
            ),
            "gdp_total": grp["gdp"].sum(),
            "n_counties": grp.size(),
        }
    )
    out.attrs["latest_year"] = latest
    return out.reset_index()


# ----------------------------------------------------------------------- geometry
def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km, same constants as the pipeline."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlam = np.radians(np.asarray(lat2) - np.asarray(lat1)), np.radians(
        np.asarray(lon2) - np.asarray(lon1)
    )
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def distance_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Full n x n haversine matrix (zeros on the diagonal)."""
    return haversine(lat[:, None], lon[:, None], lat[None, :], lon[None, :])


def compare(Da: np.ndarray, Db: np.ndarray, mask: np.ndarray) -> dict:
    """Statistics of Db - Da over the unordered pairs selected by `mask`."""
    iu = np.triu_indices_from(Da, k=1)
    keep = mask[iu[0]] & mask[iu[1]]
    a, b = Da[iu][keep], Db[iu][keep]
    d = np.abs(b - a)
    rel = d / a
    return {
        "n_pairs": int(keep.sum()),
        "mean_abs_km": d.mean(),
        "median_abs_km": np.median(d),
        "p95_abs_km": np.percentile(d, 95),
        "max_abs_km": d.max(),
        "mean_rel_pct": 100 * rel.mean(),
        "median_rel_pct": 100 * np.median(rel),
        "p95_rel_pct": 100 * np.percentile(rel, 95),
        "max_rel_pct": 100 * rel.max(),
        "corr": np.corrcoef(a, b)[0, 1],
        "mean_signed_km": (b - a).mean(),
    }


# --------------------------------------------------------------------------- main
def main() -> None:
    pts = pd.DataFrame({"abbr": REGIONS})
    pts["cap_lat"] = pts["abbr"].map(lambda r: CAPITALS[r][0])
    pts["cap_lon"] = pts["abbr"].map(lambda r: CAPITALS[r][1])

    gdp = gdp_centroids()
    pts = (
        pts.merge(geometric_centroids(), on="abbr", how="left")
        .merge(population_centroids(), on="abbr", how="left")
        .merge(gdp, on="abbr", how="left")
    )
    missing = pts[pts.isna().any(axis=1)]["abbr"].tolist()
    if missing:
        raise RuntimeError(f"regions without a full set of reference points: {missing}")

    # Displacement of each candidate point relative to the delivered one (capitals).
    for v in VARIANTS[1:]:
        pts[f"d_cap_{v}_km"] = haversine(
            pts["cap_lat"], pts["cap_lon"], pts[f"{v}_lat"], pts[f"{v}_lon"]
        )
    pts["d_pop_gdp_km"] = haversine(
        pts["pop_lat"], pts["pop_lon"], pts["gdp_lat"], pts["gdp_lon"]
    )
    pts["d_geo_gdp_km"] = haversine(
        pts["geo_lat"], pts["geo_lon"], pts["gdp_lat"], pts["gdp_lon"]
    )

    D = {v: distance_matrix(pts[f"{v}_lat"].values, pts[f"{v}_lon"].values) for v in VARIANTS}

    conus = ~pts["abbr"].isin(NON_CONUS).values
    all_mask = np.ones(len(pts), dtype=bool)

    rows = []
    for scope, mask in [("all51", all_mask), ("conus49", conus)]:
        for a, b in combinations(VARIANTS, 2):
            rows.append({"scope": scope, "ref": a, "alt": b, **compare(D[a], D[b], mask)})
    comp = pd.DataFrame(rows)

    pts.to_csv(HERE / "centroids_four_variants.csv", index=False)
    comp.to_csv(HERE / "distance_matrix_comparison.csv", index=False)

    # ---------------------------------------------------------------- console report
    pd.set_option("display.width", 200)
    print(f"BEA CAGDP2 reference year: {gdp.attrs['latest_year']}")
    print(f"regions: {len(pts)}  ({len(pts) - len(NON_CONUS)} CONUS+DC)\n")

    print("=== point displacement relative to the state capital (km) ===")
    for v in VARIANTS[1:]:
        c = pts[f"d_cap_{v}_km"]
        cc = pts.loc[conus, f"d_cap_{v}_km"]
        worst = pts.loc[c.idxmax(), "abbr"]
        print(f"  capital -> {VARIANT_LABEL[v]:<22} mean {c.mean():6.1f} | median "
              f"{c.median():6.1f} | max {c.max():7.1f} ({worst}) | CONUS mean {cc.mean():6.1f}")
    for lbl, col in [("pop  -> gdp", "d_pop_gdp_km"), ("geo  -> gdp", "d_geo_gdp_km")]:
        c = pts[col]
        print(f"  {lbl:<33} mean {c.mean():6.1f} | median {c.median():6.1f} | "
              f"max {c.max():7.1f} ({pts.loc[c.idxmax(), 'abbr']})")

    print("\n=== distance-matrix comparison ===")
    show = ["scope", "ref", "alt", "n_pairs", "mean_abs_km", "median_abs_km", "max_abs_km",
            "mean_rel_pct", "median_rel_pct", "p95_rel_pct", "max_rel_pct", "corr"]
    print(comp[show].round(3).to_string(index=False))

    # ------------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(12.5, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.28)
    colors = {"cap": "#D62728", "geo": "#7F7F7F", "pop": "#1F77B4", "gdp": "#2CA02C"}
    markers = {"cap": "*", "geo": "s", "pop": "o", "gdp": "D"}

    # (a) the four reference points, CONUS only (AK/HI would flatten the map)
    ax = fig.add_subplot(gs[0])
    sub = pts[conus]
    for _, r in sub.iterrows():
        ax.plot([r.cap_lon, r.geo_lon, r.pop_lon, r.gdp_lon, r.cap_lon],
                [r.cap_lat, r.geo_lat, r.pop_lat, r.gdp_lat, r.cap_lat],
                color="0.75", lw=0.5, zorder=1)
    # pop and gdp nearly coincide: draw gdp last and smaller so both stay visible.
    sizes = {"cap": 46, "geo": 26, "pop": 30, "gdp": 13}
    for v in ["cap", "geo", "pop", "gdp"]:
        ax.scatter(sub[f"{v}_lon"], sub[f"{v}_lat"], s=sizes[v], c=colors[v],
                   marker=markers[v], lw=0.3, edgecolor="white",
                   label=VARIANT_LABEL[v], zorder=2)
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("latitude (deg)")
    ax.set_title("(a) Four reference points per state (CONUS + DC)", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower left", framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.4)

    # (b) delivered distances vs the alternatives, pair by pair
    ax = fig.add_subplot(gs[1])
    iu = np.triu_indices(len(pts), k=1)
    keep = conus[iu[0]] & conus[iu[1]]
    x = D["cap"][iu][keep]
    for v in ["geo", "pop", "gdp"]:
        ax.scatter(x, D[v][iu][keep], s=3, alpha=0.35, c=colors[v], lw=0,
                   label=VARIANT_LABEL[v])
    lim = [0, 1.05 * x.max()]
    ax.plot(lim, lim, color="k", lw=0.8, ls="--")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("$d_{ij}$, capitals (km)")
    ax.set_ylabel("$d_{ij}$, alternative (km)")
    ax.set_title("(b) Pairwise distances, CONUS + DC", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left", markerscale=3, framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.4)

    # (c) distribution of the relative deviation from the delivered matrix
    ax = fig.add_subplot(gs[2])
    for v in ["geo", "pop", "gdp"]:
        rel = 100 * np.abs(D[v][iu][keep] - x) / x
        ax.hist(rel, bins=np.linspace(0, 30, 61), histtype="step", lw=1.4,
                color=colors[v], label=f"{VARIANT_LABEL[v]} (mean {rel.mean():.1f}%)")
    rel_pg = 100 * np.abs(D["gdp"][iu][keep] - D["pop"][iu][keep]) / D["pop"][iu][keep]
    ax.hist(rel_pg, bins=np.linspace(0, 30, 61), histtype="step", lw=1.4, ls=":",
            color="#9467BD", label=f"pop vs GDP (mean {rel_pg.mean():.1f}%)")
    ax.set_xlabel(r"$|d^{alt}_{ij}-d^{ref}_{ij}|\,/\,d^{ref}_{ij}$  (%)")
    ax.set_ylabel("number of state pairs")
    ax.set_title("(c) Relative deviation, CONUS + DC", fontsize=10)
    ax.legend(fontsize=7, framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.4)

    fig.suptitle("Candidate distance matrices for the gravity seed", fontsize=11.5, y=1.0)
    fig.savefig(HERE / "fig_distance_variants.png", dpi=200, bbox_inches="tight")
    fig.savefig(HERE / "fig_distance_variants.pdf", bbox_inches="tight")
    print(f"\n[OK] wrote centroids_four_variants.csv, distance_matrix_comparison.csv, "
          f"fig_distance_variants.png/.pdf in {HERE}")


if __name__ == "__main__":
    main()
