# flake8: noqa

"""
Module related to the :prf:ref:`NAC-coloring <def-nac>` search.

This implementation is based on https://github.com/Lastaapps/bc_thesis_code.
Before you try to optimize this code,
please check out the original implementation first.
"""

from pyrigi.graph._flexibility.nac.check import is_NAC_coloring
from pyrigi.graph._flexibility.nac.facade import NAC_colorings
from pyrigi.graph._flexibility.nac.mono_classes import MonoClassType, find_mono_classes
