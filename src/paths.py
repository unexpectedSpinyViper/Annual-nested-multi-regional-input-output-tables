"""Single source of truth for the repository root.

Import ``ROOT`` and build every path relative to it, so the project runs
unchanged on any machine after ``pip install -e .``::

    from paths import ROOT
    df = pd.read_parquet(ROOT / "data/final/nested_mriot_2017.parquet")

Installed as a top-level module via ``pyproject.toml`` (``py-modules = ["paths"]``),
so the import resolves from any notebook regardless of the working directory.
"""
from pathlib import Path

# src/paths.py -> parents[0] = src/, parents[1] = repository root
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

__all__ = ["ROOT", "DATA"]
