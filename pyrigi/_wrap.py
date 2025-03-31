# Based on https://github.com/Tinche/tightwrap/blob/main/src/tightwrap/__init__.py

import functools
from inspect import Signature, _empty
from typing import Any, Callable, TypeVar, cast
import sys
from types import GetSetDescriptorType, ModuleType

from typing import ParamSpec

P = ParamSpec("P")
T = TypeVar("T")
R = TypeVar("R")

Annotations = dict[str, Any]
Globals = dict[str, Any]
Locals = dict[str, Any]
GetAnnotationsResults = tuple[Annotations, Globals, Locals]


def _get_annotations(
    obj: Callable[..., Any],
    remove_params: list[str],
) -> GetAnnotationsResults:
    # Copied from https://github.com/python/cpython/blob/3.12/Lib/inspect.py#L176-L288

    obj_globals: Any
    obj_locals: Any
    unwrap: Any

    if isinstance(obj, type):
        obj_dict = getattr(obj, "__dict__", None)

        if obj_dict and hasattr(obj_dict, "get"):
            ann = obj_dict.get("__annotations__", None)
            if isinstance(ann, GetSetDescriptorType):
                ann = None
        else:
            ann = None

        obj_globals = None
        module_name = getattr(obj, "__module__", None)

        if module_name:
            module = sys.modules.get(module_name, None)

            if module:
                obj_globals = getattr(module, "__dict__", None)

        obj_locals = dict(vars(obj))
        unwrap = obj

    elif isinstance(obj, ModuleType):
        ann = getattr(obj, "__annotations__", None)
        obj_globals = getattr(obj, "__dict__")
        obj_locals = None
        unwrap = None

    elif callable(obj):
        ann = getattr(obj, "__annotations__", None)
        obj_globals = getattr(obj, "__globals__", None)
        obj_locals = None
        unwrap = obj

    else:
        raise TypeError(f"{obj!r} is not a module, class, or callable.")

    if ann is None:
        return cast(GetAnnotationsResults, ({}, obj_globals, obj_locals))

    if not isinstance(ann, dict):
        raise ValueError(f"{obj!r}.__annotations__ is neither a dict nor None")

    if not ann:
        return cast(GetAnnotationsResults, ({}, obj_globals, obj_locals))

    if unwrap is not None:
        while True:
            if hasattr(unwrap, "__wrapped__"):
                unwrap = unwrap.__wrapped__
                continue
            if isinstance(unwrap, functools.partial):
                unwrap = unwrap.func
                continue
            break
        if hasattr(unwrap, "__globals__"):
            obj_globals = unwrap.__globals__

    return_value = {
        key: _eval_if_necessary(value, obj_globals, obj_locals)
        for key, value in cast(dict[str, Any], ann).items()
        if key not in remove_params
    }

    return cast(GetAnnotationsResults, (return_value, obj_globals, obj_locals))


def _eval_if_necessary(source: Any, globals: Globals, locals: Locals) -> Any:
    if not isinstance(source, str):
        return source

    return eval(source, globals, locals)


def _get_resolved_signature(
    fn: Callable[..., Any],
    remove_params: list[str] = [],
) -> Signature:
    signature = Signature.from_callable(fn)
    evaluated_annotations, fn_globals, fn_locals = _get_annotations(fn, remove_params)

    filtered_parameters = [
        (n, p) for n, p in signature.parameters.items() if n not in remove_params
    ]

    for name, parameter in filtered_parameters:
        parameter._annotation = evaluated_annotations.get(name, _empty)  # type: ignore

    signature = signature.replace(
        parameters=[p for _, p in filtered_parameters],
    )

    new_return_annotation = _eval_if_necessary(
        signature.return_annotation, fn_globals, fn_locals
    )
    signature._return_annotation = new_return_annotation  # type: ignore

    return signature


def wraps(
    source_func: Callable[P, Any],
    remove_params: list[str] = ["graph"],
) -> Callable[[Callable[..., R]], Callable[P, R]]:
    """
    Apply ``functools.wraps``
    while updating signatures and docs from ``source_func``.

    Parameters
    ----------
    source_func:
        Function from which signature and documentation is copied.
    remove_params:
        Parameters to remove from the ``source_func`` signature.

    Note
    ----
    Other approaches were also tried. This is the only one working for
    doc generation, pyright and PyCharm.
    The other approach tried is based on idea that decorator
    directly returns the ``source_func``.
    For that configuration PyCharm does not work.
    """

    def wrapper(orig_func: Callable[..., R]) -> Callable[P, R]:
        res = functools.wraps(source_func)(orig_func)
        res.__signature__ = _get_resolved_signature(  # type: ignore
            source_func,
            remove_params,
        )

        return cast(Callable[P, R], res)

    return wrapper
