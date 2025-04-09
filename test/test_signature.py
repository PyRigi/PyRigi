# Based on https://github.com/Tinche/tightwrap/blob/main/src/tightwrap/__init__.py
# which is released under
# Apache License
# Version 2.0, January 2004
# http://www.apache.org/licenses/

import inspect
from inspect import Parameter, Signature, _empty
from typing import Any, Callable, Type, TypeVar, cast, ParamSpec
from types import GetSetDescriptorType, ModuleType
import sys
import functools
import pytest

from pyrigi.graph import Graph

P = ParamSpec("P")
T = TypeVar("T")

Annotations = dict[str, Any]
Globals = dict[str, Any]
Locals = dict[str, Any]
GetAnnotationsResults = tuple[Annotations, Globals, Locals]


def _get_annotations(
    obj: Callable[..., Any],
    ignored_params: list[str],
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
        if key not in ignored_params
    }

    return cast(GetAnnotationsResults, (return_value, obj_globals, obj_locals))


def _eval_if_necessary(source: Any, globals: Globals, locals: Locals) -> Any:
    if not isinstance(source, str):
        return source

    return eval(source, globals, locals)


def _get_resolved_signature(
    fn: Callable[..., Any],
    convert_first_to_self: bool,
    ignored_params: list[str],
) -> Signature:
    signature = Signature.from_callable(fn)

    # We don't care what the first parameters type is as we replace it with
    # self anyway. Also, often it cannot be resolved as its type is set by a string.
    if convert_first_to_self:
        ignored_params.append(next(iter(signature.parameters)))

    # Get annotations like return types
    evaluated_annotations, fn_globals, fn_locals = _get_annotations(fn, ignored_params)

    new_parameters: list[Parameter] = []

    # Replace the first argument by self and add annotations to the rest
    params_iter = iter(signature.parameters.items())
    if convert_first_to_self:
        next(params_iter)
        new_parameters.append(Parameter("self", Parameter.POSITIONAL_OR_KEYWORD))

    # Copy the other annotations (with string names resolved)
    for name, parameter in params_iter:
        new_parameters.append(parameter)
        if name not in ignored_params:
            parameter._annotation = (  # type: ignore
                evaluated_annotations.get(name, _empty)
            )

    new_return_annotation = _eval_if_necessary(
        signature.return_annotation, fn_globals, fn_locals
    )

    signature = signature.replace(
        parameters=new_parameters,
        return_annotation=new_return_annotation,  # type: ignore
    )

    return signature


def _assert_same_sign(method: Callable[..., T], func: Callable[..., T]) -> None:
    """
    Make sure both provided callable objects share the same signature except
    for the first parameter.

    The intended use is to check if a method and
    a function replacing it have the same signature.
    """
    sgn_method = _get_resolved_signature(
        method, convert_first_to_self=True, ignored_params=[]
    )
    sgn_func = _get_resolved_signature(
        func, convert_first_to_self=True, ignored_params=[]
    )

    if sgn_method.return_annotation != sgn_func.return_annotation:
        raise TypeError(
            f"""
                Method's return type does not match the one of proxy function
                method  [{method.__name__}]={sgn_method.return_annotation}
                function[{func.__name__}]={sgn_func.return_annotation}
                """.strip()
        )

    params_method = list(sgn_method.parameters.values())
    params_func = list(sgn_func.parameters.values())
    if params_method[1:] != params_func[1:]:
        raise TypeError(
            f"""
            Method's parameters signature does not match the one of proxy function
            method  [{method.__name__}]={params_method[1:]}
            function[{func.__name__}]={params_func[1:]}
            """.strip()
        )


@pytest.mark.parametrize(("cls"), [Graph])
def test_signature_graph(cls: Type):
    """
    Test that all methods have the same signature as the proxy functions
    replacing them. See :func:`pyrigi._wrap.copy_doc`.
    """

    # make sure the tests are actually run
    any_checked = False

    for _, method in inspect.getmembers(cls):
        wrapped_func = getattr(method, "_wrapped_func", None)
        if wrapped_func is None:
            continue

        _assert_same_sign(method, wrapped_func)
        any_checked = True

    assert any_checked
