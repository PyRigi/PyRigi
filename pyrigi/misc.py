"""
Module for miscellaneous functions.
"""

import math


def doc_category(category):
    def decorator_doc_category(func):
        setattr(func, "_doc_category", category)
        return func

    return decorator_doc_category


def generate_category_tables(cls, tabs, cat_order=[], include_all=False):
    categories = {}
    for func in dir(cls):
        if callable(getattr(cls, func)) and func[:2] != "__":
            f = getattr(cls, func)
            if hasattr(f, "_doc_category"):
                category = f._doc_category
                if category in categories:
                    categories[category].append(func)
                else:
                    categories[category] = [func]
            elif include_all:
                if "Not categorized" in categories:
                    categories["Not categorized"].append(func)
                else:
                    categories["Not categorized"] = [func]

    for category in categories:
        if category not in cat_order:
            cat_order.append(category)

    res = "Methods\n-------\n"
    for category, functions in sorted(
        categories.items(), key=lambda t: cat_order.index(t[0])
    ):
        res += f"**{category}**"
        res += "\n\n.. autosummary::\n\n    "
        res += "\n    ".join(functions)
        res += "\n\n"
    indent = "".join(["    " for _ in range(tabs)])
    return ("\n" + indent).join(res.splitlines())


def check_integrality_and_range(
    n: int, name: str = "number n", min_n: int = 0, max_n: int = math.inf
):
    if not isinstance(n, int):
        raise TypeError("The " + name + f" has to be an integer, not {type(n)}.")
    if n < min_n or n > max_n:
        raise ValueError(
            "The " + name + f" has to be an integer in [{min_n},{max_n}], not {n}."
        )
