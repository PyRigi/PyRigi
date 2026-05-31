"""
Proxy functions used by _BadWrappers.
Each is a separate callable so that _find_patch_target can
patch them independently.
"""


class _BadWrappersBase(object):
    def __init__(self):
        self.order = 1


def missing_kwarg_param(graph: _BadWrappersBase, x: int, label: str = "a") -> str:
    return f"{label}:{x} (graph has {graph.order} nodes)"


def missing_positional_param(graph: _BadWrappersBase, x: int, label: str = "a") -> str:
    return f"{label}:{x} (graph has {graph.order} nodes)"


def wrong_value(graph: _BadWrappersBase, x: int, label: str = "a") -> str:
    return f"{label}:{x} (graph has {graph.order} nodes)"


def instance_not_first(graph: _BadWrappersBase, x: int, label: str = "a") -> str:
    return f"{label}:{x} (graph has {graph.order} nodes)"


def different_instance_first(
    graph: _BadWrappersBase, x: _BadWrappersBase, label: str = "a"
) -> str:
    return f"{label}:{x} (graph has {graph.order} nodes)"


def proxy_not_called(graph: _BadWrappersBase, x: int, label: str = "a") -> str:
    return f"{label}:{x} (graph has {graph.order} nodes)"


def extra_kwarg(graph: _BadWrappersBase, x: int, label: str = "a") -> str:
    return f"{label}:{x} (graph has {graph.order} nodes)"


def extra_positional(graph: _BadWrappersBase, x: int, label: str = "a") -> str:
    return f"{label}:{x} (graph has {graph.order} nodes)"


def correct_wrapping(graph: _BadWrappersBase, x: int, label: str = "a") -> str:
    return f"{label}:{x} (graph has {graph.order} nodes)"


def different_function(graph: _BadWrappersBase, x: int, label: str = "a") -> str:
    return f"{label}:{x} (graph has {graph.order} nodes)"


def different_function2(graph: _BadWrappersBase, x: int, label: str = "a") -> str:
    return f"{label}:{x} (graph has {graph.order} nodes)"
