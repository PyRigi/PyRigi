"""
Target files for docstring migration (method-style → function-style).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

GRAPH_TARGET_FILES = [
    ROOT / "pyrigi/graph/_general.py",
    ROOT / "pyrigi/graph/_export/export.py",
    ROOT / "pyrigi/graph/_sparsity/sparsity.py",
    ROOT / "pyrigi/graph/_constructions/constructions.py",
    ROOT / "pyrigi/graph/_constructions/extensions.py",
    ROOT / "pyrigi/graph/_flexibility/nac/facade.py",
    ROOT / "pyrigi/graph/_other/apex.py",
    ROOT / "pyrigi/graph/_other/separating_set.py",
    ROOT / "pyrigi/graph/_rigidity/generic.py",
    ROOT / "pyrigi/graph/_rigidity/global_.py",
    ROOT / "pyrigi/graph/_rigidity/matroidal.py",
    ROOT / "pyrigi/graph/_rigidity/realization_counting.py",
    ROOT / "pyrigi/graph/_rigidity/redundant.py",
]

FRAMEWORK_TARGET_FILES = [
    ROOT / "pyrigi/framework/_general.py",
    ROOT / "pyrigi/framework/_export/export.py",
    ROOT / "pyrigi/framework/_rigidity/infinitesimal.py",
    ROOT / "pyrigi/framework/_rigidity/stress.py",
    ROOT / "pyrigi/framework/_rigidity/matroidal.py",
    ROOT / "pyrigi/framework/_rigidity/redundant.py",
    ROOT / "pyrigi/framework/_rigidity/second_order.py",
    ROOT / "pyrigi/framework/_plot/plot.py",
    ROOT / "pyrigi/framework/_transformations/transformations.py",
]

CLASS_TO_TARGET_FILES = {
    "Graph": GRAPH_TARGET_FILES,
    "Framework": FRAMEWORK_TARGET_FILES,
}
