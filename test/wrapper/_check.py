import inspect
import sys
from typing import Any, Callable, Type
from unittest.mock import patch

import networkx as nx

from pyrigi import Framework
from pyrigi.framework.base import FrameworkBase
from pyrigi.graph import Graph

from test.wrapper._bad_wrapper_base import _BadWrapperBase
from test.wrapper._bad_wrapper import _BadWrapper

_MISSING = object()


def _find_patch_target(method: Callable, wrapped_func: Callable) -> tuple:
    """
    Determine the module and attribute name to patch so that
    ``patch.object(module, name)`` replaces ``wrapped_func`` at the
    call-site used by ``method``.

    Return a ``(module, name)`` pair suitable for ``patch.object``.
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


def _invoke_wrapper(
    test_instance: Any,
    attr_name: str,
    method: Callable,
    mock_args: dict,
) -> tuple:
    """
    Patch the proxy for ``method``, call ``attr_name(**mock_args)`` on
    ``test_instance``, and return ``(proxy_was_called, call_args, exception)``.

    ``call_args`` persists after the patch context exits.
    ``exception`` is the first exception raised during the call or generator
    consumption, or ``None`` if everything ran cleanly.
    """
    patch_module, patch_name = _find_patch_target(method, method._wrapped_func)
    with patch.object(patch_module, patch_name) as mock_func:
        exc = None
        result = None
        try:
            result = getattr(test_instance, attr_name)(**mock_args)
        except Exception as e:
            exc = e

        if exc is None and inspect.isgeneratorfunction(method):
            try:
                list(result)
            except Exception as e:
                exc = e

        return mock_func.called, mock_func.call_args, exc


def _check_name_matches(
    cls: Type, attr_name: str, proxy_func_name: str
) -> tuple[bool, str]:
    """Return ``(True, "")`` if the method name matches the proxy function name."""
    if attr_name != proxy_func_name:
        return False, (
            f"{cls.__name__}.{attr_name} is decorated with @copy_doc({proxy_func_name}) "
            f"but the method name differs from the proxy function name"
        )
    return True, ""


def _check_first_arg(cls: Type, attr_name: str, call_args: Any) -> tuple[bool, str]:
    """Check that the first positional argument to the proxy is the correct instance."""
    first_arg = call_args[0][0]
    if cls == _BadWrapper:
        expected_type, label = _BadWrapperBase, "_BadWrapper"
    elif cls == Graph:
        expected_type, label = nx.Graph, "Graph"
    elif cls == Framework:
        expected_type, label = FrameworkBase, "Framework"
    else:
        raise NotImplementedError(f"_check_first_arg: unsupported class {cls.__name__}")
    if not isinstance(first_arg, expected_type):
        return False, (
            f"{cls.__name__}.{attr_name} didn't pass the {label} instance as first arg."
        )
    return True, ""


def _check_param_values(
    cls: Type,
    attr_name: str,
    proxy_param_names: list,
    mock_args: dict,
    call_args: Any,
) -> tuple[bool, str]:
    """Check that every entry in mock_args was forwarded with the same value.

    Falls back to positional lookup for wrappers that forward some args
    positionally (e.g. ``rescale``, ``is_critically_k_edge_apex``).
    """
    for param_name, expected_value in mock_args.items():
        actual_value = call_args.kwargs.get(param_name, _MISSING)
        if actual_value is _MISSING and len(call_args[0]) > 1:
            try:
                idx = proxy_param_names.index(param_name)
                if idx < len(call_args[0]):
                    actual_value = call_args[0][idx]
            except ValueError:
                pass

        if actual_value != expected_value:
            return False, (
                f"{cls.__name__}.{attr_name} didn't forward '{param_name}' "
                f"correctly. Expected {expected_value}, got {actual_value}"
            )
    return True, ""


def _check_no_extra_args(
    cls: Type,
    attr_name: str,
    proxy_param_names: list,
    mock_args: dict,
    call_args: Any,
) -> tuple[bool, str]:
    """Check that no unexpected arguments arrived at the proxy (reverse check)."""
    forwarded_names: set[str] = set(call_args.kwargs.keys())
    for _, param_name in zip(call_args[0][1:], proxy_param_names[1:]):
        forwarded_names.add(param_name)

    extra_positional_count = len(call_args[0]) - len(proxy_param_names)
    if extra_positional_count > 0:
        return False, (
            f"{cls.__name__}.{attr_name} forwarded {extra_positional_count} "
            f"unexpected positional argument(s) to the proxy."
        )

    unexpected = forwarded_names - set(mock_args.keys())
    if unexpected:
        return False, (
            f"{cls.__name__}.{attr_name} forwarded unexpected argument(s) "
            f"to the proxy: {unexpected}."
        )
    return True, ""


def _check_params_forwarded(
    cls: Type,
    attr_name: str,
    proxy_param_names: list,
    mock_args: dict,
    call_args: Any,
) -> tuple[bool, str]:
    """
    Check correct parameter forwarding in both directions.

    1. The instance is passed as the first positional argument.
    2. Every entry in ``mock_args`` arrives at the proxy with the same value
       (checks both keyword and positional forwarding).
    3. No unexpected arguments arrive at the proxy beyond what ``mock_args``
       sent (reverse check).

    ``proxy_param_names`` is the ordered list of the proxy's parameter names
    (including the first graph/framework parameter) and is used to resolve
    arguments that were forwarded positionally rather than by keyword.

    Return ``(True, "")`` on success or ``(False, message)`` on the first
    failed check.
    """
    ok, msg = _check_first_arg(cls, attr_name, call_args)
    if not ok:
        return False, msg
    ok, msg = _check_param_values(
        cls, attr_name, proxy_param_names, mock_args, call_args
    )
    if not ok:
        return False, msg
    return _check_no_extra_args(cls, attr_name, proxy_param_names, mock_args, call_args)
