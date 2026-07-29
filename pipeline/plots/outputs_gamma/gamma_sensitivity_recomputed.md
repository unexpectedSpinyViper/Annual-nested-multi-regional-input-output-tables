# γ-sensitivity table, recomputed on the delivered distance matrix

Re-run 2026-07-28 (`validation/gamma_pipeline_comparison.ipynb`, job 14907747) after
`gamma_sweep.py` was switched from **state capitals** to the **GDP-weighted economic
centroids** that the delivered series has been using since step 02. The published table had
been produced on the capital-based matrix, so it did not describe the series it was testing.

Reference year 2017, shock = New York, −30 %.

## Table (replaces the γ-sensitivity table of the Technical Validation)

| γ | mean trade distance (km) | inter-state Gini | Leontief total | Leontief spillover | Inoperability spillover | Ghosh spillover |
|---|---|---|---|---|---|---|
| 0.1 | 1,746 | 0.638 | −745,563 | −184,459 | −350,279 | −254,575 |
| 0.5 | 1,559 | 0.655 | −746,050 | −183,860 | −348,649 | −253,236 |
| 1.0 | 1,316 | 0.698 | −746,959 | −181,498 | −343,324 | −249,464 |
| 1.5 | 1,117 | 0.753 | −747,989 | −178,006 | −335,894 | −244,356 |
| 3.0 |   833 | 0.861 | −750,030 | −170,609 | −320,732 | −233,955 |

Published values, for comparison (capital-based matrix):

| γ | mean trade distance | Gini | Leontief total | Leontief spill. | Inop. spill. | Ghosh spill. |
|---|---|---|---|---|---|---|
| 0.1 | 1,753 | 0.638 | −745,540 | −184,455 | −350,286 | −254,601 |
| 0.5 | 1,555 | 0.656 | −745,883 | −183,975 | −348,966 | −253,573 |
| 1.0 | 1,296 | 0.702 | −746,543 | −182,148 | −344,786 | −250,674 |
| 1.5 | 1,093 | 0.757 | −747,383 | −179,371 | −338,768 | −246,485 |
| 3.0 |   805 | 0.865 | −749,495 | −172,720 | −324,826 | −236,395 |

The two agree to $0.003\%$ on the aggregate response at $\gamma = 0.1$ and to $1.3\%$ on the
spillovers at $\gamma = 3$: the reference points matter more the stronger the friction, which
is what one would expect, and the difference stays an order of magnitude below the effect the
table is measuring.

## Sentences of the surrounding prose that change

| statement | published | recomputed |
|---|---|---|
| mean distance over the sweep | falls from 1,753 to 805 km | falls from **1,746 to 833 km** |
| inter-state Gini | rises from 0.64 to 0.86 | **0.638 to 0.861** (unchanged to 2 d.p.) |
| intra- vs inter-state split | 45.1 / 54.9 %, invariant in γ | **45.06–45.07 / 54.93–54.94 %**, invariant |
| aggregate response varies by | +0.5 % Leontief, −0.2 % inoperability, −0.04 % Ghosh | **+0.60 %, −0.22 %, +0.01 %** |
| spillover moves by | −6.4 % to −7.3 % | **−7.5 % to −8.4 %** |
| relative $L_1$ to the γ=1 build | below 1.1 % | **below 1.09 %** — the claim stands as written |
| commodity-specific scenarios | intra-state share pinned at 45.06–45.07 %, responses within ±2 % | **unchanged**, and every heterogeneous case still falls inside the envelope of the uniform sweep |

## Invariance to the final-demand allocator

The same re-run also carries the new destination allocator $\Theta$. It leaves this table
**exactly** unchanged, and that is structural rather than lucky: the shock is applied to the
rows of the shocked state, and the row total of the final-demand block is
$\sum_{s'} F^{n}_{(s,i),(s',c)} = F^{O}_{\mathrm{USA}_i,c}\,S_{s,i}\sum_{s'}\Theta^{c}_{s'}
= F^{O}_{\mathrm{USA}_i,c}\,S_{s,i}$, since $\Theta$ sums to one over destination states. The
allocator integrates out of every row-side statistic. It would move a table built on a shock
to a state's *final demand as a destination*, which this one is not.

Worth stating in the descriptor: it means the γ-sensitivity and the choice of allocator are
independent questions, and neither result contaminates the other.
