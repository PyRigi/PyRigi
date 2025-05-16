# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

from sphinx.application import Sphinx

import pyrigi.misc._input_check as _input_check
from pyrigi import Framework

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "PyRigi"
copyright = "2024, The PyRigi Developers"
author = "The PyRigi Developers"

# The short X.Y version
version = "1.0"
# The full version, including alpha/beta/rc tags
release = "1.0.2"


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.doctest",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_proof",
    "myst_nb",
    "sphinxcontrib.bibtex",
    "sphinx_math_dollar",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_tippy",
]
coverage_modules = ["pyrigi"]
coverage_statistics_to_stdout = True
coverage_show_missing_items = True

bibtex_bibfiles = ["refs.bib"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "networkx": ("https://networkx.org/documentation/stable", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_custom_sections = ["Definitions", "Methods", "Suggested Improvements"]

autodoc_type_aliases = {
    "Vertex": ":type:`~pyrigi.data_type.Vertex`",
    "Edge": ":type:`~pyrigi.data_type.Edge`",
    "DirectedEdge": ":type:`~pyrigi.data_type.DirectedEdge`",
    "Point": ":type:`~pyrigi.data_type.Point`",
    "Number": ":type:`~pyrigi.data_type.Number`",
    "Stress": ":type:`~pyrigi.data_type.Stress`",
    "InfFlex": ":type:`~pyrigi.data_type.InfFlex`",
    "Inf": ":type:`~pyrigi.data_type.Inf`",
}
napoleon_attr_annotations = True

autodoc_typehints = "description"

autosummary_generate = True
numpydoc_show_inherited_class_members = False


myst_enable_extensions = [
    "amsmath",
    #     "attrs_inline",
    "colon_fence",
    #     "deflist",
    "dollarmath",
    #     "fieldlist",
    #     "html_admonition",
    #     "html_image",
    #     "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    #     "tasklist",
]


myst_substitutions = {
    "pyrigi_crossref": "**`PyRigi`**:",
    "references": "*References*:",
}

myst_heading_anchors = 3

nb_execution_mode = "cache"
nb_execution_raise_on_error = True
nb_execution_show_tb = True
nb_execution_timeout = 120

tippy_enable_mathjax = True
tippy_props = {
    "theme": "light",
}
tippy_enable_doitips = False
tippy_skip_anchor_classes = ("headerlink", "next-page")
tippy_anchor_parent_selector = "div.content"
tippy_props = {
    "placement": "top",
    "maxWidth": 500,
    "interactive": False,
    "duration": [200, 100],
    "delay": [800, 500],
}

mathjax3_config = {
    "tex": {
        "inlineMath": [["\\(", "\\)"]],
        "displayMath": [["\\[", "\\]"]],
        "macros": {
            "RR": "{\\mathbb{R}}",  # real numbers
            "CC": "{\\mathbb{C}}",  # complex numbers
            "QQ": "{\\mathbb{Q}}",  # rational numbers
            "ZZ": "{\\mathbb{Z}}",  # integers
            "NN": "{\\mathbb{N}}",  # natural numbers (including 0)
            "PP": "{\\mathbb{P}}",  # projective space
            "KK": "{\\mathbb{K}}",  # a field
        },
    }
}
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = [".md"]

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "userguide/tutorials/*.ipynb",
    "notebooks/*.ipynb",
    "userguide/*.ipynb",
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

# html_logo = "../assets/logo_nofont.png"
html_theme_options = {
    "sidebar_hide_name": False,
    "light_logo": "logo_nofont.png",
    "dark_logo": "logo_nofont_dark.png",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/pyrigi/pyrigi",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,  # noqa: E501
            "class": "",
        },
    ],
    "top_of_page_buttons": ["view"],
    "source_repository": "https://github.com/pyrigi/pyrigi",
    "source_branch": "main",
    "source_directory": "doc/",
}

html_title = "PyRigi " + version


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["tippy.css", "thm.css"]


# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "PyRigidoc"


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    "preamble": r"""
    \newcommand{\RR}{\mathbb{R}}    % real numbers
    \newcommand{\CC}{\mathbb{C}}    % complex numbers
    \newcommand{\QQ}{\mathbb{Q}}    % rational numbers
    \newcommand{\ZZ}{\mathbb{Z}}    % integers
    \newcommand{\NN}{\mathbb{N}}    % natural numbers (including 0)
    \newcommand{\PP}{\mathbb{P}}    % projective space
    \newcommand{\KK}{\mathbb{K}}    % a field
    """,
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "pyrigi.tex",
        "PyRigi Documentation",
        "The PyRigi Developers",
        "manual",
    ),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "PyRigi", "PyRigi Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "pyrigi",
        "PyRigi Documentation",
        author,
        "pyrigi",
        "One line description of project.",
        "Miscellaneous",
    ),
]


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]


# -- Extension configuration -------------------------------------------------

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Local Sphinx extensions -------------------------------------------------


def setup(app: Sphinx):
    """Add functions to the Sphinx setup."""
    from myst_parser._docs import MystLexer

    app.add_lexer("myst", MystLexer)


# -----create the documentation of input checks

input_check_str = ""
for cls in [Framework]:
    methods = [
        method
        for method in dir(cls)
        if method.startswith("_input_check_") and callable(getattr(cls, method))
    ]
    input_check_str += (
        f"""
Input check methods of {cls.__name__}
======================={''.join(["=" for _ in range(len(cls.__name__))])}

.. automethod:: {cls.__module__}.{cls.__name__}."""
        + f"""

.. automethod:: {cls.__module__}.{cls.__name__}.""".join(
            methods
        )
        + "\n\n"
    )

input_check_str += """

General input check functions
=============================


"""

_input_check.__doc__ = input_check_str


# ----------generate module structure with comments------------------------


def contains_py_files(path: str):
    """Check if a directory or its subdirectories contain .py files."""
    for _, _, files in os.walk(path):
        if any(file.endswith(".py") for file in files):
            return True
    return False


def get_comment_for_file(
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
        and contains_py_files(os.path.join(root_path, e))
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
        sub_tree = generate_myst_tree(folder_path, comments, indent + 1, base_path)
        output.append(sub_tree)

    # Then list .py files
    for file in py_files:
        file_path = os.path.join(root_path, file)
        prefix_and_file = "    " * indent + f"{file} "
        comment = get_comment_for_file(file_path, base_path, comments)
        num_dots = 80 - len(prefix_and_file) - len(comment)
        comment_str = "." * num_dots + f" {comment}" if comment else ""
        output.append(prefix_and_file + comment_str)

    return "\n".join(output)


# Nested comment dictionary
comments = {
    ".": {
        "data_type.py": "definitions of data types",
        "exception.py": "definitions of exceptions",
        "warning.py": "definitions of warnings",
        "graphDB.py": "database of graphs",
        "frameworkDB.py": "database of frameworks",
        "plot_style.py": "implementation of Plotstyle(2D/3D)",
    },
    "graph": {
        "extensions.py": "functions for k-extensions",
        "generic.py": "functions for generic rigidity",
        "global_.py": "functions for global rigidity",
        "matroidal.py": "functions for generic rigidity matroid",
        "redundant.py": "functions for redundant rigidity",
        "_pebble_digraph.py": "implementation of PebbleDigraph",
        "sparsity.py": "functions for (k,l)-sparsity",
        "_input_check.py": "input checks for Graph",
        "apex.py": "functions for apex graphs",
        "graph.py": "implementation of Graph",
        "separating_set.py": "functions for (stable) separating sets",
    },
    "graph_drawer": {
        "graph_drawer.py": "implementation of GraphDrawer",
    },
    "framework": {
        "base.py": "implementation of FrameworkBase",
        "framework.py": "implementation of Framework",
    },
}

tree_output = generate_myst_tree("../pyrigi", comments)
with open("development/howto/pyrigi_structure.txt", "w") as file:
    file.write(tree_output)
