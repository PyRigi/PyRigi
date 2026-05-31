import inspect
import sys
from typing import Any, Callable, Type

import networkx as nx

from pyrigi import Framework
from pyrigi.graph import Graph

from test.wrapper._proxies import _BadWrappersBase
from test.wrapper._wrappers import _BadWrappers

_MISSING = object()


def _find_patch_target(method: Callable, wrapped_func: Callable) -> tuple:
    """
    Determine the module and attribute name to patch so that
    ``patch.object(module, name)`` replaces ``wrapped_func`` at the
    call-site used by ``method``.

    Returns a ``(module, name)`` pair suitable for ``patch.object``.
    """
    method_module = sys.modules[method.__module__]

    for attr_name, val in vars(method_module).items():
        if val is wrapped_func:
            return method_module, attr_name
        if (
            inspect.ismodule(val)
            and getattr(val, wrapped_func.__name__, None) is wrapped_func
        ):
            return val, wrapped_func.__name__

    # Fall back to the module where the proxy function is defined
    return sys.modules[wrapped_func.__module__], wrapped_func.__name__


def _assert_name_matches(cls: Type, attr_name: str, proxy_func_name: str) -> None:
    assert attr_name == proxy_func_name, (
        f"{cls.__name__}.{attr_name} is decorated with @copy_doc({proxy_func_name}) "
        f"but the method name differs from the proxy function name"
    )


def _assert_params_forwarded(
    cls: Type,
    attr_name: str,
    proxy_param_names: list,
    mock_args: dict,
    mock_func: Any,
) -> None:
    """
    Assert correct parameter forwarding in both directions:

    1. The instance is passed as the first positional argument.
    2. Every entry in ``mock_args`` arrives at the proxy with the same value
       (checks both keyword and positional forwarding).
    3. No unexpected arguments arrive at the proxy beyond what ``mock_args``
       sent (reverse check).

    ``proxy_param_names`` is the ordered list of the proxy's parameter names
    (including the first graph/framework parameter) and is used to resolve
    arguments that were forwarded positionally rather than by keyword.
    """
    call_args = mock_func.call_args

    # First positional arg must be the instance itself
    first_arg = call_args[0][0]
    if cls == _BadWrappers:
        assert isinstance(first_arg, _BadWrappersBase), (
            f"{cls.__name__}.{attr_name} didn't pass "
            f"the _BadWrappers instance as first arg."
        )
    elif cls == Graph:
        assert isinstance(
            first_arg, nx.Graph
        ), f"{cls.__name__}.{attr_name} didn't pass the Graph instance as first arg."
    elif cls == Framework:
        assert isinstance(
            first_arg, Framework
        ), f"{cls.__name__}.{attr_name} didn't pass the Framework instance as first arg."
    else:
        raise NotImplementedError(
            f"_assert_params_forwarded: unsupported class {cls.__name__}"
        )

    # Every other parameter must be forwarded with the same value.
    # Some wrappers forward args positionally (e.g. rescale, is_critically_k_edge_apex),
    # so fall back to positional lookup when a key is absent from call_args.kwargs.
    for param_name, expected_value in mock_args.items():
        actual_value = call_args.kwargs.get(param_name, _MISSING)
        if actual_value is _MISSING and len(call_args[0]) > 1:
            try:
                idx = proxy_param_names.index(param_name)
                if idx < len(call_args[0]):
                    actual_value = call_args[0][idx]
            except ValueError:
                pass

        assert actual_value == expected_value, (
            f"{cls.__name__}.{attr_name} didn't forward '{param_name}' correctly. "
            f"Expected {expected_value}, got {actual_value}"
        )

    # Reverse check: no unexpected arguments must arrive at the proxy.
    forwarded_names: set[str] = set(call_args.kwargs.keys())
    for i, _ in enumerate(call_args[0][1:], start=1):
        if i < len(proxy_param_names):
            forwarded_names.add(proxy_param_names[i])

    # Extra positionals beyond the proxy's named params are always unexpected.
    extra_positional_count = max(0, len(call_args[0]) - len(proxy_param_names))
    assert extra_positional_count == 0, (
        f"{cls.__name__}.{attr_name} forwarded {extra_positional_count} unexpected "
        f"positional argument(s) to the proxy"
    )

    unexpected = forwarded_names - set(mock_args.keys())
    assert not unexpected, (
        f"{cls.__name__}.{attr_name} forwarded unexpected argument(s) to the proxy: "
        f"{unexpected}"
    )
