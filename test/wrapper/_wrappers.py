"""
Deliberately broken @copy_doc wrappers used by test_signature.py.
"""

from __future__ import annotations

from pyrigi._utils._doc import copy_doc

# from pyrigi.graph import Graph

from test.wrapper._proxies import (
    extra_kwarg,
    extra_positional,
    instance_not_first,
    missing_param,
    proxy_not_called,
    wrong_value,
    # correct_wrapping,
)
from test.wrapper._proxies import _BadWrappersBase

# ---------------------------------------------------------------------------
# Each method shares its name with a dedicated proxy function in
# test/wrapper/_proxies.py and illustrates exactly one wrong
# parameter-forwarding pattern.
# ---------------------------------------------------------------------------
#
# Patterns:
#   1. missing_param      — a parameter is not forwarded to the proxy
#   2. wrong_value        — a parameter is modified before forwarding
#   3. instance_not_first — the instance is not first positional arg
#   4. proxy_not_called   — the proxy function is never invoked
#   5. extra_kwarg        — an unexpected keyword argument is passed
#   6. extra_positional   — an unexpected positional argument is passed


class _BadWrappers(_BadWrappersBase):
    @copy_doc(missing_param)
    def missing_param(self, x: int, label: str = "a") -> str:  # noqa: U100
        return missing_param(self, x)

    @copy_doc(wrong_value)
    def wrong_value(self, x: int, label: str = "a") -> str:
        return wrong_value(self, x + 1, label)

    @copy_doc(instance_not_first)
    def instance_not_first(self, x: int, label: str = "a") -> str:
        return instance_not_first(x, label, self)

    @copy_doc(proxy_not_called)
    def proxy_not_called(self, x: int, label: str = "a") -> str:
        return f"{x}:{label}"

    @copy_doc(extra_kwarg)
    def extra_kwarg(self, x: int, label: str = "a") -> str:
        return extra_kwarg(self, x, label, extra=True)

    @copy_doc(extra_positional)
    def extra_positional(self, x: int, label: str = "a") -> str:
        return extra_positional(self, x, label, 42)

    # @copy_doc(correct_wrapping)
    # def correct_wrapping(self, x: int, label: str = "a") -> str:
    #     return extra_positional(self, x, label)
