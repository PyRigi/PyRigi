"""
Module for standard input checks.
"""

import math


def dimension(dim: int) -> None:
    """
    Checks whether an input dimension is a positive integer and
    raises an error otherwise.
    """
    if not isinstance(dim, int):
        raise TypeError(f"The dimension needs to be an integer, but is {type(dim)}!")
    elif dim < 1:
        raise ValueError(
            f"The dimension needs to be positive, but is {dim}!"
        )


def dimension_for_algorithm(dim: int, possible: list, algorithm: str = "") -> None:
    """
    Checks whether an input dimension is a member of list and
    raises an error otherwise.
    """
    if dim not in possible:
        raise ValueError(
            "For " + algorithm + f" the dimension needs to be in {possible},"
            f"but is {dim}!"
        )


def integrality_and_range(
    n: int, name: str = "number n", min_n: int = 0, max_n: int = math.inf
) -> None:
    """
    Checks whether an input parameter n is an integer in a certain range and
    raises an error otherwise.
    """
    if not isinstance(n, int):
        raise TypeError("The " + name + f" has to be an integer, but is {type(n)}!")
    if n < min_n or n > max_n:
        raise ValueError(
            "The " + name + f" has to be an integer in [{min_n},{max_n}], but is {n}!"
        )


def greater_equal(val1: int, val2: int, name1: str, name2: str = "") -> None:
    """
    Checks whether an input parameter val1 is greater than or equal to val2 and
    raises an error otherwise.
    """
    if name2 == "":
        str2 = ""
    else:
        str2 = name2 + ", i.e. "
    if val1 < val2:
        raise ValueError(
            "The " + name1 + " needs to be "
            "greater than or equal to " + str2 + f"{val2}, "
            f"but is {val1}!"
        )


def greater(val1: int, val2: int, name1: str, name2: str = "") -> None:
    """
    Checks whether an input parameter val1 is greater than or equal to val2 and
    raises an error otherwise.
    """
    if name2 == "":
        str2 = ""
    else:
        str2 = name2 + ", i.e. "
    if val1 <= val2:
        raise ValueError(
            "The " + name1 + " needs to be "
            "greater than " + str2 + f"{val2}, "
            f"but is {val1}!"
        )



def pebble_values(K: int, L: int) -> None:
    """
    Check if K and L satisfy the pebble conditions K > 0 and 0 <= L < 2K.
    """
    # Check that K and L are integers and range
    integrality_and_range(K, "K", min_n=1)
    integrality_and_range(L, "L", min_n=0)

    # Check the conditions on relation
    greater(2 * K, L, "value 2*K", "L")
