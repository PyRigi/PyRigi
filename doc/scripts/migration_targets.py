"""
Target files for docstring migration (method-style → function-style).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TARGET_FILES = [
    ROOT / "pyrigi/graph/_general.py",
    ROOT / "pyrigi/graph/_export/export.py",
    ROOT / "pyrigi/graph/_sparsity/sparsity.py",
    ROOT / "pyrigi/graph/_constructions/constructions.py",
    ROOT / "pyrigi/graph/_constructions/extensions.py",
    ROOT / "pyrigi/graph/_other/apex.py",
    ROOT / "pyrigi/graph/_other/separating_set.py",
    ROOT / "pyrigi/graph/_rigidity/generic.py",
    ROOT / "pyrigi/graph/_rigidity/redundant.py",
    ROOT / "pyrigi/graph/_rigidity/matroidal.py",
    ROOT / "pyrigi/graph/_rigidity/global_.py",
]
