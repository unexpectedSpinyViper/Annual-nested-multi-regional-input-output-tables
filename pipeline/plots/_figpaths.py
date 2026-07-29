"""Shared paths for the scripts that draw the figures of the data descriptor.

Every script in this directory writes to a single output directory, ``figures/``
at the repository root, under the exact file name the manuscript expects, so the
LaTeX source can point at it unchanged.

Importing this module also puts ``pipeline/`` on ``sys.path``, so the figure
scripts can ``import nest_v31`` / ``gamma_sweep`` / ``harmonize`` whether or not
the package has been installed with ``pip install -e .``.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # pipeline/plots
_PIPELINE = _HERE.parent                         # pipeline

try:                                             # after `pip install -e .`
    from paths import ROOT
except ModuleNotFoundError:                      # plain checkout, no install
    ROOT = _HERE.parents[1]
    sys.path.insert(0, str(ROOT / "src"))
    from paths import ROOT

for _p in (str(_PIPELINE), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

IOT_INTERIM = ROOT / "data/interim/IOT"
NESTED_DIR = IOT_INTERIM / "nested_mriot_v3.1_RAS"

# All manuscript figures land here, under the names used by the .tex source.
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

__all__ = ["ROOT", "IOT_INTERIM", "NESTED_DIR", "FIG_DIR"]
