import os
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")

try:
    from IPython.core.magic import register_cell_magic

    @register_cell_magic
    def skip_execution(line, cell):  # noqa: U100
        print(
            "This cell was marked to be skipped (probably due to long execution time)."
        )
        print("Remove the cell magic `%%skip_execution` to run it.")
        return

except NameError:

    def skip_execution():
        pass


def copy_doc(
    proxy_func: Callable[P, T],
) -> Callable[[Callable[..., T]], Callable[P, T]]:
    """
    Copy the docstring from the provided function.

    In tests, it also ensures that the signatures match.
    """

    def wrapped(method: Callable[..., T]) -> Callable[P, T]:
        method.__doc__ = proxy_func.__doc__
        method._wrapped_func = proxy_func
        return method

    return wrapped


# ==================== functions for generating package structure ===================


def _contains_py_files(path: str):
    """Check if a directory or its subdirectories contain .py files."""
    for _, _, files in os.walk(path):
        if any(file.endswith(".py") for file in files):
            return True
    return False


def _get_comment_for_file(
    file_path: str, root_dir: str, comment_dict: dict[str, dict[str, str]]
) -> str:
    """
    Resolve a comment for a given file path using nested comment dict.

    Parameters
    ----------
    file_path:
        Full file path.
    root_dir:
        Base directory.
    comment_dict:
        Nested comment dictionary.
    """
    rel_path = os.path.relpath(file_path, root_dir)
    parts = rel_path.split(os.sep)

    if len(parts) == 1:
        return comment_dict.get(".", {}).get(parts[0], "")
    elif parts[0] in comment_dict:
        return comment_dict[parts[0]].get(parts[-1], "")
    return ""


def generate_myst_tree(
    root_path: str,
    comments: dict[str, dict[str, str]] = None,
    indent: int = 0,
    base_path: str = None,
    show_line_numbers: bool = False,
) -> str:
    """
    Generate MyST-compatible Markdown tree showing only ``.py`` files, folders first.

    Parameters
    ----------
    root_path:
        Current directory to process.
    comments:
        Nested dict ``{folder: {file: comment}}``.
    indent:
        Indentation level.
    base_path:
        Root directory for relative paths.
    show_line_numbers:
        Whether to show line numbers.
    """
    if comments is None:
        comments = {}
    if base_path is None:
        base_path = root_path

    output = []
    entries = sorted(os.listdir(root_path))

    # Separate folders and Python files
    folders = [
        e
        for e in entries
        if os.path.isdir(os.path.join(root_path, e))
        and _contains_py_files(os.path.join(root_path, e))
    ]
    py_files = [
        e
        for e in entries
        if e.endswith(".py") and os.path.isfile(os.path.join(root_path, e))
    ]

    # List folders first
    for folder in folders:
        folder_path = os.path.join(root_path, folder)
        prefix = "    " * indent
        output.append(f"{prefix}{folder}/")
        sub_tree = generate_myst_tree(
            folder_path,
            comments,
            indent + 1,
            base_path,
            show_line_numbers=show_line_numbers,
        )
        output.append(sub_tree)

    # Then list .py files
    for file in py_files:
        file_path = os.path.join(root_path, file)
        if show_line_numbers:
            with open(file_path, "r") as f:
                n = len(f.readlines())

        prefix_and_file = "    " * indent + f"{file} "
        comment = _get_comment_for_file(file_path, base_path, comments)
        num_dots = 80 - len(prefix_and_file) - len(comment)
        comment_str = "." * num_dots + f" {comment}" if comment else ""
        output.append(
            prefix_and_file
            + comment_str
            + (f" ({n:4d} lines)" if show_line_numbers else "")
        )

    return "\n".join(output)


# ======================functions for generating tables of methods ===============


def doc_category(category):
    """
    Decorator for doc categories.
    """

    def decorator_doc_category(func):
        setattr(func, "_doc_category", category)
        return func

    return decorator_doc_category


def generate_category_tables(
    cls, tabs, cat_order=None, include_all=False, add_attributes=True
) -> str:
    """
    Generate a formatted string that categorizes methods of a given class.

    Parameters
    ----------
    cls:
        A class.
    tabs:
        The number of indentation levels that are applied to the output.
    cat_order:
        Optional list specifying the order in which categories appear
        in the output.
    include_all:
        Optional boolean determining whether methods without a specific category
        should be included.
    add_attributes:
        Optional boolean determining whether the public attributes should
        be listed among attribute getters.
    """
    if cat_order is None:
        cat_order = []
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
        elif isinstance(getattr(cls, func), property) and add_attributes:
            category = "Attribute getters"
            if category in categories:
                categories[category].append(func)
            else:
                categories[category] = [func]

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
