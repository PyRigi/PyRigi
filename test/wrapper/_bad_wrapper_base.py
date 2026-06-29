"""
A proxy class is defined for meta-testing of wrapping together with proxy functions.

Each function is a separate callable so that _find_patch_target can
patch them independently.
"""


class _BadWrapperBase(object):
    def __init__(self):
        self.order = 1


def missing_kwarg_param(graph: _BadWrapperBase, x: int, label: str = "a") -> str:
    return f"{label}:{x} (graph has {graph.order} nodes)"


def missing_positional_param(graph: _BadWrapperBase, x: int, label: str = "a") -> str:
    return f"{label}:{x} (graph has {graph.order} nodes)"


def wrong_value(graph: _BadWrapperBase, x: int, label: str = "a") -> str:
    return f"{label}:{x} (graph has {graph.order} nodes)"


def instance_not_first(graph: _BadWrapperBase, x: int, label: str = "a") -> str:
    return f"{label}:{x} (graph has {graph.order} nodes)"


def different_instance_first(
    graph: _BadWrapperBase, x: _BadWrapperBase, label: str = "a"
) -> str:
    return f"{label}:{x} (graph has {graph.order} nodes)"


def proxy_not_called(graph: _BadWrapperBase, x: int, label: str = "a") -> str:
    return f"{label}:{x} (graph has {graph.order} nodes)"


def extra_kwarg(graph: _BadWrapperBase, x: int, label: str = "a") -> str:
    return f"{label}:{x} (graph has {graph.order} nodes)"


def extra_positional(graph: _BadWrapperBase, x: int, label: str = "a") -> str:
    return f"{label}:{x} (graph has {graph.order} nodes)"


def correct_wrapping(graph: _BadWrapperBase, x: int, label: str = "a") -> str:
    return f"{label}:{x} (graph has {graph.order} nodes)"


def different_function(graph: _BadWrapperBase, x: int, label: str = "a") -> str:
    return f"{label}:{x} (graph has {graph.order} nodes)"


def different_function2(graph: _BadWrapperBase, x: int, label: str = "a") -> str:
    return f"{label}:{x} (graph has {graph.order} nodes)"


def missing_kwargs(graph: _BadWrapperBase, x: int, label: str = "a", **kwargs) -> str:
    return f"{label}:{x} (graph has {graph.order} nodes: {kwargs.keys()})"


def switched_positional(graph: _BadWrapperBase, x: int, y: int) -> str:
    return f"{y}:{x} (graph has {graph.order} nodes)"
