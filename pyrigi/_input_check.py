"""
Module for standard input checks.
"""

import math


def dimension(dim: int) -> None:
    """
    Check whether an input dimension is a positive integer and
    raise an error otherwise.
    """
    if not isinstance(dim, int):
        raise TypeError(f"The dimension needs to be an integer, but is {type(dim)}!")
    elif dim < 1:
        raise ValueError(f"The dimension needs to be positive, but is {dim}!")


def dimension_for_algorithm(dim: int, possible: list, algorithm: str = "") -> None:
    """
    Check whether an input dimension is a member of list and
    raise an error otherwise.

    Parameters
    ----------
    dim:
        Dimension to be checked
    possible:
        Values that are allowed
    algorithm:
        Name of the algorithm for the error message
    """
    if dim not in possible:
        if len(possible) == 1:
            in_str = str(possible[0])
        else:
            in_str = f"in {possible}"
        raise ValueError(
            "For `" + algorithm + "` the dimension needs to be " + in_str + ", "
            f"but is {dim}!"
        )


def integrality_and_range(
    n: int, name: str = "number n", min_n: int = 0, max_n: int = math.inf
) -> None:
    """
    Check whether an input parameter n is an integer in a certain range and
    raise an error otherwise.

    Parameters
    ----------
    n:
        Value to be checked
    name:
        Name of the parameter
    min_n:
        Lower limit for the value
    max_n:
        Upper limit for the value
    """
    if not isinstance(n, int):
        raise TypeError("The " + name + f" has to be an integer, but is {type(n)}!")
    if n < min_n or n > max_n:
        raise ValueError(
            "The " + name + f" has to be an integer in [{min_n},{max_n}], but is {n}!"
        )


def greater_equal(val1: int, val2: int, name1: str, name2: str = "") -> None:
    """
    Check whether an input parameter val1 is greater than or equal to val2 and
    raise an error otherwise.

    Parameters
    ----------
    val1:
        Value that shall be greater/equal
    val2:
        Value that shall be smaller
    name1:
        Name of the parameter val1
    name2:
        Name of the parameter val2
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
    Check whether an input parameter val1 is greater than or equal to val2 and
    raise an error otherwise.

    Parameters
    ----------
    val1:
        Value that shall be greater
    val2:
        Value that shall be smaller/equal
    name1:
        Name of the parameter val1
    name2:
        Name of the parameter val2
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


def smaller_equal(val1: int, val2: int, name1: str, name2: str = "") -> None:
    """
    Check whether an input parameter val1 is greater than or equal to val2 and
    raise an error otherwise.

    Parameters
    ----------
    val1:
        Value that shall be smaller/equal
    val2:
        Value that shall be greater
    name1:
        Name of the parameter val1
    name2:
        Name of the parameter val2
    """
    if name2 == "":
        str2 = ""
    else:
        str2 = name2 + ", i.e. "
    if val1 > val2:
        raise ValueError(
            "The " + name1 + " needs to be "
            "smaller than or equal to " + str2 + f"{val2}, "
            f"but is {val1}!"
        )


def pebble_values(K: int, L: int) -> None:
    """
    Check if K and L satisfy the pebble conditions K > 0 and 0 <= L < 2K and
    raise an error otherwise.
    """
    # Check that K and L are integers and range
    integrality_and_range(K, "K", min_n=1)
    integrality_and_range(L, "L", min_n=0)

    # Check the conditions on relation
    greater(2 * K, L, "value 2*K", "L")
