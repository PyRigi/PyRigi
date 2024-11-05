import pytest
from pyrigi.graph import Graph
import pyrigi.graphDB as graphs


# can be run with pytest -m large


def pytest_addoption(parser):
    parser.addoption(
        "--run-large", action="store_true", default=False, help="run large tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "large: mark a test as large")


@pytest.mark.parametrize(
    "graph",
    [
        graphs.Complete(2),
        graphs.Complete(3),
        graphs.Complete(4),
        graphs.CompleteBipartite(3, 3),
        graphs.CompleteBipartite(3, 4),
        graphs.CompleteBipartite(4, 4),
        graphs.Diamond(),
        graphs.K33plusEdge(),
        graphs.ThreePrism(),
        graphs.ThreePrismPlusEdge(),
    ],
)


@pytest.mark.large
def test_rigid_in_d2(graph):
    assert graph.is_rigid(dim=2, combinatorial=True)
    assert graph.is_rigid(dim=2, combinatorial=False)


