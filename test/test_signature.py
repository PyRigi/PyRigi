# Based on https://github.com/Tinche/tightwrap/blob/main/src/tightwrap/__init__.py
# which is released under
# Apache License
# Version 2.0, January 2004
# http://www.apache.org/licenses/

import functools
import inspect
import sys
from inspect import Parameter, Signature, _empty
from types import GetSetDescriptorType, ModuleType
from typing import Any, Callable, ParamSpec, Type, TypeVar, cast, get_args, get_origin

import pytest

from pyrigi import Framework
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

    def is_subtype(sub_type, super_type):
        if get_origin(sub_type) != get_origin(super_type):
            return False
        sub_args, super_args = get_args(sub_type), get_args(super_type)
        if not sub_args or not super_args:
            return False
        if len(sub_args) != len(super_args):
            return False
        for sub_arg, super_arg in zip(sub_args, super_args):
            if sub_arg == super_arg:
                pass
            elif is_subtype(sub_arg, super_arg):
                pass
            elif issubclass(sub_arg, super_arg) and sub_arg == Graph:
                pass
            else:
                return False
        return True

    meth_return_ann = sgn_method.return_annotation
    funct_return_ann = sgn_func.return_annotation

    try:
        if meth_return_ann != funct_return_ann:
            if meth_return_ann == Graph and issubclass(
                meth_return_ann, funct_return_ann
            ):
                pass
            elif is_subtype(meth_return_ann, funct_return_ann):
                pass
            else:
                raise TypeError(
                    f"""
                    The return type of the method does not match
                    the one of the proxy function:
                    method  [{method.__name__}]={meth_return_ann}
                    function[{func.__name__}]={funct_return_ann}
                    """.strip()
                )
    except TypeError as e:
        raise TypeError(
            f"""There is a problem with the return type of
                        the method and the proxy function:
                        method  [{method.__name__}]={meth_return_ann}
                        function[{func.__name__}]={funct_return_ann}
                        """.strip()
        ) from e

    params_method = list(sgn_method.parameters.values())
    params_func = list(sgn_func.parameters.values())
    if params_method[1:] != params_func[1:]:
        raise TypeError(
            f"""
            The parameters signature of the method does not match
            the one of the proxy function:
            method  [{method.__name__}]={params_method[1:]}
            function[{func.__name__}]={params_func[1:]}
            """.strip()
        )


@pytest.mark.parametrize(("cls"), [Graph, Framework])
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
