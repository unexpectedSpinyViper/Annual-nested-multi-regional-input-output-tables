"""GDP-weighted economic centroids of the 51 WiNDC regions (50 states + DC).

These are the **delivered reference points** of the gravity distance matrix used by
the bilateral reconstruction (step 11/12). Each region's centroid is the mean of its
county population centroids weighted by county GDP (BEA CAGDP2, all-industry total).
By default the latest year available in the CAGDP2 file is used, giving one static
distance matrix for the whole 1997-2022 series, exactly as the earlier capital-based
prior did; pass ``reference_year`` to pin another year.

Two BEA geography quirks are handled explicitly:
  * Connecticut planning regions replaced the CT counties in 2022 and are the geography
    BEA now reports county GDP on; their anchors come from the 2023 Census Gazetteer.
  * BEA "combination areas" merge most Virginia independent cities (and Maui+Kalawao)
    with a neighbouring county; each is anchored on the population-weighted centroid of
    its constituent counties, so ~44 % of Virginia's GDP is not lost.

Inputs (see data/raw/DOWNLOAD.md):
  data/raw/census/CenPop2020_Mean_CO.txt          2020 county population centroids
  data/raw/BEA/CAGDP2.zip                          county GDP (all-industry total)
  data/raw/census/2023_Gaz_counties_national.zip   Gazetteer 2023 (CT planning regions)

Output (written by ``main`` / 02_economic_centroids.ipynb):
  data/interim/economic_centroids.csv   columns: abbr, lat, lon, gdp_total, n_counties
"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

from paths import ROOT

CENPOP_COUNTY  = ROOT / "data/raw/census/CenPop2020_Mean_CO.txt"
CAGDP2_ZIP     = ROOT / "data/raw/BEA/CAGDP2.zip"
GAZETTEER_2023 = ROOT / "data/raw/census/2023_Gaz_counties_national.zip"

OUT_CSV = ROOT / "data/interim/economic_centroids.csv"

# FIPS -> USPS, restricted to the 51 WiNDC regions (50 states + DC).
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

# BEA does not report GDP separately for most Virginia independent cities (nor Kalawao,
# HI): each is merged with a surrounding county into a "combination area" whose FIPS has
# no Census counterpart. 44 % of Virginia's GDP sits in these rows, so they cannot be
# dropped -- each is anchored on the population-weighted centroid of its constituents.
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


def _read_gazetteer(path) -> pd.DataFrame:
    """County-level Census Gazetteer: GEOID + internal point (INTPTLAT/INTPTLONG)."""
    with zipfile.ZipFile(path) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".txt"))
        with zf.open(name) as fh:
            df = pd.read_csv(fh, sep="\t", dtype={"GEOID": str}, encoding="latin-1")
    df.columns = [c.strip() for c in df.columns]
    df["fips"] = df["GEOID"].str.zfill(5)
    return df


def economic_centroids(reference_year: int | None = None) -> pd.DataFrame:
    """Return per-region GDP-weighted economic centroids (abbr, lat, lon, ...)."""
    # -- 1. county population centroids (geographic anchors) --------------------
    co = pd.read_csv(CENPOP_COUNTY, encoding="utf-8-sig",
                     dtype={"STATEFP": str, "COUNTYFP": str})
    co.columns = [c.strip().upper() for c in co.columns]
    co["fips"] = co["STATEFP"].str.zfill(2) + co["COUNTYFP"].str.zfill(3)
    co = co.rename(columns={"LATITUDE": "county_lat", "LONGITUDE": "county_lon"})

    # -- 2. county GDP (BEA CAGDP2, all-industry total) -------------------------
    with zipfile.ZipFile(CAGDP2_ZIP) as zf:
        name = next((n for n in zf.namelist()
                     if "ALL_AREAS" in n.upper() and n.upper().endswith(".CSV")), None)
        if name is None:
            name = next(n for n in zf.namelist() if n.upper().endswith(".CSV"))
        with zf.open(name) as fh:
            raw = pd.read_csv(fh, encoding="latin-1", low_memory=False)
    raw.columns = [c.strip() for c in raw.columns]
    raw["GeoFIPS"] = raw["GeoFIPS"].astype(str).str.replace('"', "").str.strip().str.zfill(5)
    total = raw[raw["LineCode"] == 1].copy()                    # all-industry total
    year_cols = [c for c in total.columns if c.strip().isdigit()]
    year = str(reference_year) if reference_year is not None else max(year_cols, key=int)
    if year not in year_cols:
        raise ValueError(f"year {year} not in CAGDP2 (available {min(year_cols)}-{max(year_cols)})")
    total["gdp"] = pd.to_numeric(total[year], errors="coerce")
    county = (total[~total["GeoFIPS"].str.endswith("000")][["GeoFIPS", "gdp"]]
              .rename(columns={"GeoFIPS": "fips"}).dropna())

    # keep the CT planning regions (091x0), drop the retired CT counties (090xx)
    ct_new = county["fips"].str.match(r"091[1-9]0$")
    if ct_new.any():
        county = county[~county["fips"].str.match(r"090[0-9][0-9]$") | ct_new]

    # -- 3. anchors: pop centroids + Gazetteer 2023 (CT) + combination areas ----
    anchors = co[["fips", "county_lat", "county_lon"]]
    gaz = _read_gazetteer(GAZETTEER_2023)
    extra = (gaz[~gaz["fips"].isin(anchors["fips"])][["fips", "INTPTLAT", "INTPTLONG"]]
             .rename(columns={"INTPTLAT": "county_lat", "INTPTLONG": "county_lon"}))
    anchors = pd.concat([anchors, extra], ignore_index=True)

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

    # -- 4. GDP-weight the county anchors within each region --------------------
    m = anchors.merge(county, on="fips")
    unmatched = set(county["fips"]) - set(m["fips"])
    if unmatched:
        lost = county[county["fips"].isin(unmatched)]["gdp"].sum() / county["gdp"].sum()
        print(f"[warn] {len(unmatched)} BEA counties without an anchor "
              f"({100 * lost:.2f} % of US county GDP): {sorted(unmatched)[:10]}")
    m["abbr"] = m["fips"].str[:2].map(FIPS_TO_ABBR)
    m = m.dropna(subset=["abbr"])
    grp = m.groupby("abbr")
    out = pd.DataFrame({
        "lat": grp.apply(lambda d: np.average(d["county_lat"], weights=d["gdp"]),
                         include_groups=False),
        "lon": grp.apply(lambda d: np.average(d["county_lon"], weights=d["gdp"]),
                         include_groups=False),
        "gdp_total": grp["gdp"].sum(),
        "n_counties": grp.size(),
    }).reset_index()
    out.attrs["reference_year"] = year
    return out


def main() -> None:
    out = economic_centroids()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"[OK] {len(out)} regions | CAGDP2 reference year {out.attrs['reference_year']} "
          f"-> {OUT_CSV}")
    if len(out) != 51:
        raise RuntimeError(f"expected 51 regions (50 states + DC), got {len(out)}: "
                           f"{sorted(out['abbr'])}")


if __name__ == "__main__":
    main()
