# Based on https://github.com/Tinche/tightwrap/blob/main/src/tightwrap/__init__.py

from inspect import Parameter, Signature, _empty
from typing import Any, Callable, TypeVar, cast
import sys
from types import GetSetDescriptorType, ModuleType
from typing import ParamSpec
import os
import functools

P = ParamSpec("P")
T = TypeVar("T")
R = TypeVar("R")

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

    # Get's annotations like return types
    evaluated_annotations, fn_globals, fn_locals = _get_annotations(fn, ignored_params)

    new_parameters: list[Parameter] = []

    # Replaces the first argument by self and adds annotations to the rest
    params_iter = iter(signature.parameters.items())
    if convert_first_to_self:
        next(params_iter)
        new_parameters.append(Parameter("self", Parameter.POSITIONAL_OR_KEYWORD))

    # Copies the other annotations (with string names resolved)
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


@functools.cache
def _is_pytest() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ


def _assert_same_sign(method: Callable[..., T], func: Callable[..., T]) -> None:
    """
    Make sure both the callable objects provided share the same signature except
    for the first parameter. The intended use is to check if a method and
    a function replacing it have the same signature.
    """
    sgn_method = _get_resolved_signature(
        method, convert_first_to_self=True, ignored_params=[]
    )
    sgn_func = _get_resolved_signature(
        func, convert_first_to_self=True, ignored_params=[]
    )

    if sgn_method.return_annotation != sgn_func.return_annotation:
        print("Method's return type does not match the one of proxy function")
        print(f"method[{method.__name__}]={sgn_method.return_annotation}")
        print(f"function[{func.__name__}]={sgn_func.return_annotation}")
    assert sgn_method.return_annotation == sgn_func.return_annotation

    params_method = list(sgn_method.parameters.values())
    params_func = list(sgn_func.parameters.values())
    if params_method[1:] != params_func[1:]:
        print("Method's parameters signature does not match the one of proxy function")
        print(f"method[{method.__name__}]={params_method[1:]}")
        print(f"function[{func.__name__}]={params_func[1:]}")
    assert params_method[1:] == params_func[1:]


def proxy_call(
    proxy_func: Callable[P, T],
) -> Callable[[Callable[..., T]], Callable[P, T]]:
    """
    Call the provided function instead of the decorated method.
    Intended use is to replace a method's implementation with
    an another function's implementation.

    Note
    ----
    From listed bellow all the integrations work just fine:
    Docs are correctly generated, help() gives correct doc string,
    ? works i Jupyter notebook, pyright provides correct hints.
    Only PyCharm missed doc string and finds correct signature only
    because the signature has to be copied exactly to the calling class.
    """

    def wrapped(method: Callable[..., T]) -> Callable[P, T]:
        if _is_pytest:
            _assert_same_sign(method, proxy_func)
        return proxy_func

    return wrapped
