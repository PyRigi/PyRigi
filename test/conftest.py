"""Graph-class parametrization for ``--graph-class`` CLI option.

Test modules that define a module-level ``graph_class`` variable are
automatically parametrized so that each test in the module runs once per
selected graph class. Modules without ``graph_class`` are unaffected.

Example usage::

    pytest test/graph/                        # default: graph_class = nx.Graph only
    pytest test/graph/ --graph-class=pyrigi   # pyrigi.Graph only
    pytest test/graph/ --graph-class=both     # both, i.e., run twice
"""

import networkx as nx
import pytest

from pyrigi.graph import Graph as PyRigiGraph

_GRAPH_IDS = {
    nx.Graph: "nx.Graph",
    PyRigiGraph: "pyrigi.Graph",
}


def pytest_addoption(parser):
    """Register the ``--graph-class`` command-line option."""
    parser.addoption(
        "--graph-class",
        action="store",
        default="nx",
        choices=["nx", "pyrigi", "both"],
        help="Graph class to test: nx (default), pyrigi, or both.",
    )


def pytest_generate_tests(metafunc):
    """Parametrize tests whose module defines ``graph_class``.

    For each test function whose module has a ``graph_class`` attribute,
    this hook injects the ``_graph_class`` fixture parametrized with the
    graph classes selected by ``--graph-class``. Modules without
    ``graph_class`` are left untouched.
    """
    option = metafunc.config.getoption("--graph-class")
    if not hasattr(metafunc.module, "graph_class"):
        return
    if option == "both":
        classes = [nx.Graph, PyRigiGraph]
    elif option == "pyrigi":
        classes = [PyRigiGraph]
    else:
        classes = [nx.Graph]
    metafunc.parametrize(
        "_graph_class",
        classes,
        indirect=True,
        ids=[_GRAPH_IDS[c] for c in classes],
    )


@pytest.fixture(autouse=True)
def _graph_class(request, monkeypatch):
    """Patch the calling module's ``graph_class`` variable.

    Uses ``request.param`` (supplied by :func:`pytest_generate_tests` via
    ``indirect=True``) to set the module-level ``graph_class`` attribute
    of the test's own module through ``monkeypatch``, ensuring that
    ``graph_class(...)`` in the test body constructs the correct graph type.
    """
    param = getattr(request, "param", None)
    if param is None:
        return
    mod = getattr(request, "module", None)
    if mod is not None and hasattr(mod, "graph_class"):
        monkeypatch.setattr(mod, "graph_class", request.param)
