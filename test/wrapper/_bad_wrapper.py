"""
Deliberately broken ``@copy_doc`` wrappers used by test_wrapper.py to verify
that every class of forwarding mistake is detected by the assertion helpers.
"""

from __future__ import annotations

from pyrigi._utils._doc import copy_doc

from test.wrapper._bad_wrapper_base import (
    different_function,
    different_function2,
    different_instance_first,
    extra_kwarg,
    extra_positional,
    instance_not_first,
    missing_kwarg_param,
    missing_kwargs,
    missing_positional_param,
    proxy_not_called,
    wrong_value,
)
from test.wrapper._bad_wrapper_base import _BadWrapperBase


class _BadWrapper(_BadWrapperBase):
    # a keyword parameter is not forwarded to the proxy
    @copy_doc(missing_kwarg_param)
    def missing_kwarg_param(self, x: int, label: str = "a") -> str:  # noqa: U100
        return missing_kwarg_param(self, x)

    # a positional parameter is not forwarded to the proxy
    @copy_doc(missing_positional_param)
    def missing_positional_param(self, x: int, label: str = "a") -> str:  # noqa: U100
        return missing_positional_param(self, label="b")

    # a parameter is modified before forwarding
    @copy_doc(wrong_value)
    def wrong_value(self, x: int, label: str = "a") -> str:
        return wrong_value(self, x + 1, label)

    # the instance is not passed as first positional arg
    @copy_doc(instance_not_first)
    def instance_not_first(self, x: int, label: str = "a") -> str:
        return instance_not_first(x, label, self)

    # a different instance is passed as first positional arg
    @copy_doc(different_instance_first)
    def different_instance_first(self, x: _BadWrapperBase, label: str = "a") -> str:
        return different_instance_first(x, self, label)

    # the proxy function is never invoked
    @copy_doc(proxy_not_called)
    def proxy_not_called(self, x: int, label: str = "a") -> str:
        return f"{x}:{label}"

    # an unexpected keyword argument is passed to the proxy
    @copy_doc(extra_kwarg)
    def extra_kwarg(self, x: int, label: str = "a") -> str:
        return extra_kwarg(self, x, label, extra=True)

    # an unexpected positional argument is passed to the proxy
    @copy_doc(extra_positional)
    def extra_positional(self, x: int, label: str = "a") -> str:
        return extra_positional(self, x, label, 42)

    # a different proxy function is called than the one from @copy_doc
    @copy_doc(different_function)
    def different_function(self, x: int, label: str = "a") -> str:
        return different_function2(self, x, label)

    # the wrapper method name differs from the proxy function name
    @copy_doc(different_function)
    def function_named_differently(self, x: int, label: str = "a") -> str:
        return different_function(self, x, label)

    # missing kwargs
    @copy_doc(missing_kwargs)
    def missing_kwargs(self, x: int, label: str = "a", **kwargs) -> str:  # noqa: U100
        return missing_kwargs(self, x, label)
