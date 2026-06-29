import inspect

import pytest
from unittest.mock import Mock

from pyrigi import Framework
from pyrigi.graph import Graph

from test.wrapper._check import (
    _check_name_matches,
    _check_params_forwarded,
    _invoke_wrapper,
)
from test.wrapper._bad_wrapper import _BadWrapper


def _build_mock_args(sig: inspect.Signature) -> dict:
    """Build mock sentinels for every named param except the first (graph/framework).

    VAR_KEYWORD/VAR_POSITIONAL are excluded; a ``_test_kwarg`` probe is added
    when **kwargs is present to verify the catch-all is forwarded.
    """
    mock_args = {
        name: Mock(name=name)
        for name, param in list(sig.parameters.items())[1:]
        if param.kind
        not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    }
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        mock_args["_test_kwarg"] = Mock(name="_test_kwarg")
    return mock_args


@pytest.mark.parametrize(
    "cls,test_instance",
    [
        (Graph, Graph([(0, 1), (1, 2)])),
        (
            Framework,
            Framework(
                Graph([(0, 1), (1, 2)]),
                {0: (0, 0), 1: (1, 0), 2: (0, 1)},
            ),
        ),
    ],
)
def test_wrapper_parameter_forwarding(cls, test_instance):
    """
    Test that all ``@copy_doc`` wrapper methods on ``Graph`` and ``Framework`` correctly
    forward parameters to their underlying proxy functions.
    """
    any_checked = False

    for attr_name, method in inspect.getmembers(cls):
        wrapped_func = getattr(method, "_wrapped_func", None)
        if wrapped_func is None:
            continue

        sig = inspect.signature(wrapped_func)
        proxy_param_names = list(sig.parameters.keys())
        mock_args = _build_mock_args(sig)

        called, call_args, exc = _invoke_wrapper(
            test_instance, attr_name, method, mock_args
        )

        if exc is not None:
            pytest.fail(
                f"{cls.__name__}.{attr_name} raised an unexpected exception "
                f"while forwarding parameters — the wrapper may be modifying "
                f"an argument before passing it: {type(exc).__name__}: {exc}"
            )

        assert called, f"{cls.__name__}.{attr_name} didn't call {wrapped_func.__name__}"

        name_ok, name_msg = _check_name_matches(cls, attr_name, wrapped_func.__name__)
        assert name_ok, name_msg

        params_ok, params_msg = _check_params_forwarded(
            cls, attr_name, proxy_param_names, mock_args, call_args
        )
        assert params_ok, params_msg

        any_checked = True

    assert any_checked


@pytest.mark.parametrize(
    "attr_name, proxy_reaches_mock",
    [
        ("missing_kwarg_param", True),  # proxy called - missing param detected
        ("missing_positional_param", True),  # proxy called - missing param detected
        ("wrong_value", False),  # Mock()+1 raises TypeError before proxy
        ("instance_not_first", True),  # proxy called - wrong first arg detected
        ("different_instance_first", True),  # proxy called - wrong first arg detected
        ("proxy_not_called", False),  # wrapper never calls the proxy
        ("different_function", False),  # wrapper calls different_function2 instead
        ("extra_kwarg", True),  # proxy called - unexpected kwarg detected
        ("extra_positional", True),  # proxy called - extra positional detected
        ("function_named_differently", True),  # proxy called - name mismatch detected
        ("missing_kwargs", True),  # proxy called - kwargs not forwarded
        ("switched_positional", True),  # proxy called - positional arguments switched
    ],
)
def test_bad_wrapper_detected(attr_name, proxy_reaches_mock):
    """
    Test that each known forwarding mistake in _BadWrapper is caught
    by the check helpers.

    ``proxy_reaches_mock`` documents whether the proxy mock is expected to be
    called at all. When False, the bad behaviour manifests before the proxy is
    reached (exception or wrong function called); when True, the proxy is called
    but the check helpers catch the forwarding mistake.
    """
    test_instance = _BadWrapper()
    method = getattr(_BadWrapper, attr_name)
    wrapped_func = method._wrapped_func

    sig = inspect.signature(wrapped_func)
    proxy_param_names = list(sig.parameters.keys())
    mock_args = _build_mock_args(sig)

    called, call_args, _ = _invoke_wrapper(test_instance, attr_name, method, mock_args)

    assert called == proxy_reaches_mock, (
        f"_BadWrapper.{attr_name}: expected proxy "
        f"{'to be' if proxy_reaches_mock else 'not to be'} called"
    )

    if proxy_reaches_mock:
        name_ok, _ = _check_name_matches(_BadWrapper, attr_name, wrapped_func.__name__)
        params_ok, _ = _check_params_forwarded(
            _BadWrapper, attr_name, proxy_param_names, mock_args, call_args
        )
        assert not (name_ok and params_ok), (
            f"_BadWrapper.{attr_name}: bad wrapper was not detected "
            f"by the check helpers"
        )
