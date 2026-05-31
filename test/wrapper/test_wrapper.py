import inspect

import pytest
from unittest.mock import Mock, patch

from pyrigi import Framework
from pyrigi.graph import Graph

from test.wrapper._helpers import (
    _assert_name_matches,
    _assert_params_forwarded,
    _find_patch_target,
)
from test.wrapper._wrappers import _BadWrappers


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
    Test that all @copy_doc wrapper methods on Graph and Framework correctly
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

        patch_module, patch_name = _find_patch_target(method, wrapped_func)

        with patch.object(patch_module, patch_name) as mock_func:
            try:
                result = getattr(test_instance, attr_name)(**mock_args)
            except Exception as e:
                pytest.fail(
                    f"{cls.__name__}.{attr_name} raised an unexpected exception "
                    f"while forwarding parameters — the wrapper may be modifying "
                    f"an argument before passing it: {type(e).__name__}: {e}"
                )

            if inspect.isgeneratorfunction(method):
                list(result)

            assert (
                mock_func.called
            ), f"{cls.__name__}.{attr_name} didn't call {wrapped_func.__name__}"
            _assert_name_matches(cls, attr_name, wrapped_func.__name__)
            _assert_params_forwarded(
                cls, attr_name, proxy_param_names, mock_args, mock_func
            )

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
    ],
)
def test_bad_wrapper_detected(attr_name, proxy_reaches_mock):
    """
    Test that each known forwarding mistake in _BadWrappers is caught
    by the assertion helpers.

    ``proxy_reaches_mock`` documents whether the proxy mock is expected to be
    called at all. When False, the bad behaviour manifests before the proxy is
    reached (exception or wrong function called); when True, the proxy is called
    but the assertion helpers catch the forwarding mistake.
    """
    test_instance = _BadWrappers()
    method = getattr(_BadWrappers, attr_name)
    wrapped_func = method._wrapped_func

    sig = inspect.signature(wrapped_func)
    proxy_param_names = list(sig.parameters.keys())
    mock_args = _build_mock_args(sig)

    patch_module, patch_name = _find_patch_target(method, wrapped_func)

    with patch.object(patch_module, patch_name) as mock_func:
        result = None
        try:
            result = getattr(test_instance, attr_name)(**mock_args)
        except Exception:
            pass

        if inspect.isgeneratorfunction(method):
            try:
                list(result)
            except Exception:
                pass

        assert mock_func.called == proxy_reaches_mock, (
            f"_BadWrappers.{attr_name}: expected proxy "
            f"{'to be' if proxy_reaches_mock else 'not to be'} called"
        )

        if proxy_reaches_mock:
            with pytest.raises(AssertionError):
                _assert_name_matches(_BadWrappers, attr_name, wrapped_func.__name__)
                _assert_params_forwarded(
                    _BadWrappers, attr_name, proxy_param_names, mock_args, mock_func
                )
