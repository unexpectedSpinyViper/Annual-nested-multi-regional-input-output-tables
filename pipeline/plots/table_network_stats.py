"""Table — centrality of the regions of a delivered file.

Collapses the delivered table to its region-by-region intermediate network, 51 states
alongside the national economies of the global source, and reports the standard
centrality measures of the Data Overview: weighted in- and out-strength, eigenvector
centrality, PageRank, and betweenness.

Conventions, stated because centrality is convention-dependent:
  * nodes are regions; the weight of the arc r -> r' is the intermediate flow summed
    over supplying and using sectors, in millions of dollars;
  * self-loops (intra-regional deliveries) are removed, so strength measures trade;
  * eigenvector centrality and PageRank are computed on the weighted digraph, the
    first in the Euclidean normalisation returned by the eigenproblem, the second
    summing to one;
  * betweenness uses distance = 1/weight, so that a heavy arc is a short one, and the
    pair-count normalisation of the standard definition.

Writes ``figures/network_stats_<year>.csv`` and prints the LaTeX-free table body.

Run from anywhere:  python pipeline/plots/table_network_stats.py [--year 2017] [--top 11]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import networkx as nx

from _figpaths import FIG_DIR, NESTED_DIR

FD_CATS = {"DPABR", "GFCF", "GGFC", "HFCE", "INVNT", "NPISH"}
EXTRA_ROWS = ["OUT", "TLS", "VA"]

# The delivered files carry ISO-3 codes for countries and USPS codes for states; the
# Data Overview names the large ones in full.
NAMES = {
    "CHN": "China", "JPN": "Japan", "KOR": "Korea", "DEU": "Germany", "IND": "India",
    "FRA": "France", "MEX": "Mexico", "GBR": "United Kingdom", "CAN": "Canada",
    "ITA": "Italy", "BRA": "Brazil", "RUS": "Russia", "ESP": "Spain", "NLD": "Netherlands",
    "CA": "California", "TX": "Texas", "NY": "New York", "FL": "Florida",
    "IL": "Illinois", "PA": "Pennsylvania", "OH": "Ohio", "GA": "Georgia",
    "NC": "North Carolina", "MI": "Michigan", "NJ": "New Jersey", "WA": "Washington",
}


def region_network(year):
    """Region-by-region intermediate flows of a delivered file, self-loops removed."""
    df = pd.read_parquet(NESTED_DIR / f"nested_mriot_{year}.parquet")
    rows = [r for r in df.index if r not in EXTRA_ROWS]
    cols = [c for c in df.columns
            if "_" in c and c != "OUT" and c.split("_", 1)[1] not in FD_CATS]
    Z = df.loc[rows, cols].values.astype(float)
    r_reg = np.array([r.split("_")[0] for r in rows])
    c_reg = np.array([c.split("_")[0] for c in cols])
    regions = sorted(set(r_reg) | set(c_reg))
    idx = {r: i for i, r in enumerate(regions)}
    M = np.zeros((len(regions), len(regions)))
    ri = np.array([idx[r] for r in r_reg])
    ci = np.array([idx[c] for c in c_reg])
    np.add.at(M, (ri[:, None], ci[None, :]), Z)
    np.fill_diagonal(M, 0.0)
    del df, Z
    return regions, M


def centralities(regions, M):
    G = nx.from_numpy_array(M, create_using=nx.DiGraph)
    G = nx.relabel_nodes(G, {i: r for i, r in enumerate(regions)})

    eig = nx.eigenvector_centrality_numpy(G, weight="weight")
    pr = nx.pagerank(G, weight="weight")
    # betweenness on a length metric: a heavy arc is a short one
    L = G.copy()
    for _, _, d in L.edges(data=True):
        d["length"] = 1.0 / d["weight"] if d["weight"] > 0 else np.inf
    btw = nx.betweenness_centrality(L, weight="length", normalized=True)

    return pd.DataFrame({
        "region": regions,
        "in_strength_M": M.sum(0),
        "out_strength_M": M.sum(1),
        "eigenvector": [eig[r] for r in regions],
        "pagerank": [pr[r] for r in regions],
        "betweenness": [btw[r] for r in regions],
    }).set_index("region")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=2017)
    ap.add_argument("--top", type=int, default=11)
    args = ap.parse_args()

    regions, M = region_network(args.year)
    T = centralities(regions, M).sort_values("eigenvector", ascending=False)
    T.insert(0, "name", [NAMES.get(r, r) for r in T.index])

    out = FIG_DIR / f"network_stats_{args.year}.csv"
    T.to_csv(out)
    print(f"{len(regions)} regions | wrote {out}\n")

    print(f"{'Region':<16}{'in-strength':>14}{'out-strength':>14}"
          f"{'eigenvector':>13}{'PageRank':>10}{'betweenness':>13}")
    for r, row in T.head(args.top).iterrows():
        print(f"{row['name']:<16}{row.in_strength_M:14,.0f}{row.out_strength_M:14,.0f}"
              f"{row.eigenvector:13.3f}{row.pagerank:10.4f}{row.betweenness:13.3f}")


if __name__ == "__main__":
    main()
