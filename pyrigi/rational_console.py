"""
Python console in which integer literals are SymPy integers.

In this console, divisions of integer literals give exact rational numbers,
e.g., ``1/3`` is ``Rational(1, 3)`` instead of the float ``0.333...``.

The console is started from a Python or IPython session by

.. code-block:: python

    from pyrigi.rational_console import run_rational_console
    run_rational_console(globals())

Passing ``globals()`` makes the variables of the calling session available
in the console and keeps the variables defined in the console afterwards.
The console is left with Ctrl-D (Ctrl-Z followed by Enter on Windows),
which returns to the calling session.
"""

import code

from sympy.interactive.session import int_to_Integer


class RationalConsole(code.InteractiveConsole):
    """
    Interactive console that wraps integer literals into ``sympy.Integer``.

    Each input is transformed by :func:`sympy.interactive.session.int_to_Integer`
    before being executed, so ``1/3`` is executed as ``Integer(1)/Integer(3)``.
    Hence, ``Integer`` must be defined in the namespace of the console,
    which is done in the constructor.

    Parameters
    ----------
    namespace:
        The namespace in which the code is executed.
        If ``None``, a new namespace is used.
        Pass ``globals()`` to share the namespace with the calling code.
    """

    def __init__(self, namespace: dict = None):
        super().__init__(namespace if namespace is not None else {})
        self.push("from sympy import Integer")

    def runsource(
        self, source: str, filename: str = "<input>", symbol: str = "single"
    ) -> bool:
        """
        Transform integer literals of the source and execute it.

        If the source cannot be transformed, e.g., because it is incomplete
        (an open bracket), it is passed on unchanged.
        """
        try:
            source = int_to_Integer(source)
        except Exception:
            pass
        return super().runsource(source, filename, symbol)


def run_rational_console(namespace: dict = None) -> None:
    """
    Start an interactive :class:`RationalConsole`.

    Parameters
    ----------
    namespace:
        See :class:`RationalConsole`.
    """
    RationalConsole(namespace).interact(
        banner=(
            "Python console with exact division of integer literals "
            "(1/3 is Rational(1, 3)).\n"
            "Available: Integer. Exit with Ctrl-D."
        ),
        exitmsg="",
    )
