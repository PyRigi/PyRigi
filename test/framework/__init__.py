from pyrigi import Framework
from pyrigi.framework.base import FrameworkBase


def _to_FrameworkBase(framework: Framework) -> FrameworkBase:
    return FrameworkBase(framework._graph, framework._realization)
