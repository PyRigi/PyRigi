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
from unittest.mock import Mock, patch

import networkx as nx
import pytest

from pyrigi import Framework
from pyrigi.graph import Graph

from test.wrapper._wrappers import _BadWrappers
from test.wrapper._proxies import _BadWrappersBase

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


def _assert_same_sign(  # noqa: C901
    method: Callable[..., T], func: Callable[..., T]
) -> None:
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
            elif issubclass(sub_arg, super_arg) and sub_arg in [Graph, Framework]:
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
    for par_method, par_func in zip(params_method, params_func):
        if par_method != par_func:
            if (
                par_method.name == par_func.name
                and par_method.annotation in [Graph, Framework]
                and issubclass(par_method.annotation, par_func.annotation)
            ):
                pass
            elif par_method.name == par_func.name and is_subtype(
                par_method.annotation, par_func.annotation
            ):
                pass
            else:
                raise TypeError(
                    f"""
                    The parameters signature of the method does not match
                    the one of the proxy function:
                    method  [{method.__name__}]={params_method[1:]}
                    function[{func.__name__}]={params_func[1:]}
                    """.strip()
                )


@pytest.mark.parametrize(("cls"), [Graph, Framework, _BadWrappers])
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
    Assert that ``mock_func`` was called with the graph/framework instance
    as the first positional argument and all entries of ``mock_args``
    forwarded unchanged.

    ``proxy_param_names`` is the ordered list of the proxy's parameter names
    """
    call_args = mock_func.call_args

    # First positional arg must be the instance itself
    first_arg = call_args[0][0]
    if cls == _BadWrappers:
        assert isinstance(
            first_arg, _BadWrappersBase
        ), f"{cls.__name__}.{attr_name} didn't pass "
        "the _BadWrappers instance as first arg."
    elif cls == Graph:
        assert isinstance(
            first_arg, nx.Graph
        ), f"{cls.__name__}.{attr_name} didn't pass the Graph instance as first arg."
    else:
        assert isinstance(first_arg, Framework), (
            f"{cls.__name__}.{attr_name} didn't pass "
            f"the Framework instance as first arg."
        )

    # Every other parameter must be forwarded with the same value.
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


@pytest.mark.parametrize(
    ("cls", "expect_pass", "test_instance"),
    [
        (Graph, True, Graph([(0, 1), (1, 2)])),
        (
            Framework,
            True,
            Framework(
                Graph([(0, 1), (1, 2)]),
                {0: (0, 0), 1: (1, 0), 2: (0, 1)},
            ),
        ),
        (_BadWrappers, False, _BadWrappers()),
    ],
)
def test_wrapper_parameter_forwarding(cls, expect_pass, test_instance):
    """
    Test that all @copy_doc wrapper methods correctly forward parameters
    to their underlying proxy functions, and that every class of
    forwarding mistake is detected.
    """
    any_checked = False

    for attr_name, method in inspect.getmembers(cls):
        wrapped_func = getattr(method, "_wrapped_func", None)
        if wrapped_func is None or not callable(method):
            continue

        sig = inspect.signature(wrapped_func)
        proxy_param_names = list(sig.parameters.keys())

        # Exclude VAR_KEYWORD/VAR_POSITIONAL; probe **kwargs with a sentinel key.
        mock_args = {
            name: Mock(name=name)
            for name, param in list(sig.parameters.items())[1:]
            if param.kind
            not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
        }
        if any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        ):
            mock_args["_test_kwarg"] = Mock(name="_test_kwarg")

        patch_module, patch_name = _find_patch_target(method, wrapped_func)

        with patch.object(patch_module, patch_name) as mock_func:
            try:
                result = getattr(test_instance, attr_name)(**mock_args)
            except Exception as e:
                if expect_pass:
                    pytest.fail(
                        f"{cls.__name__}.{attr_name} raised an "
                        f"unexpected exception while forwarding parameters — "
                        f"the wrapper may be modifying an argument before "
                        f"passing it: {type(e).__name__}: {e}"
                    )

            # If the wrapper returns a generator, consume it to trigger the call
            if hasattr(result, "__iter__") and hasattr(result, "__next__"):
                try:
                    list(result)
                except Exception:
                    pass

            if not expect_pass:
                if not mock_func.called:
                    continue  # proxy was never invoked
                with pytest.raises(AssertionError) as error:
                    _assert_name_matches(cls, attr_name, wrapped_func.__name__)
                    _assert_params_forwarded(
                        cls, attr_name, proxy_param_names, mock_args, mock_func
                    )
                print("Did not raise error: ", attr_name, error)
            else:
                assert mock_func.called, (
                    f"{cls.__name__}.{attr_name} didn't call "
                    f"{wrapped_func.__name__}"
                )
                _assert_name_matches(cls, attr_name, wrapped_func.__name__)
                _assert_params_forwarded(
                    cls, attr_name, proxy_param_names, mock_args, mock_func
                )

        any_checked = True

    assert any_checked


# def test_temp():
#     with pytest.raises(TypeError):
#         pass
