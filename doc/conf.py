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
sys.path.insert(0, os.path.abspath('..'))

from sphinx.application import Sphinx

# -- Project information -----------------------------------------------------

project = u'PyRigi'
copyright = u'2024, The PyRigi Developers'
author = u'The PyRigi Developers'

# The short X.Y version
version = u''
# The full version, including alpha/beta/rc tags
release = u''


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    "sphinx.ext.intersphinx",
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.doctest',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx_proof',
    'myst_parser',
    'sphinxcontrib.bibtex',
    'sphinx_math_dollar',
    "sphinx_copybutton",
    "sphinx_design",
]

bibtex_bibfiles = ['refs.bib']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
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
napoleon_custom_sections = ['Definitions']

autodoc_type_aliases = {
    "Vertex": "Vertex",
    "Edge": "Edge",
    }
napoleon_attr_annotations = True

autodoc_typehints = 'description'

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
  "pyrigi" : "**`PyRigi`**",
  "pyrigi_crossref" : "{{pyrigi}}:",
  "references" : "*References*:" ,
}

myst_heading_anchors = 3

mathjax3_config = {
  "tex": {
    "inlineMath": [['\\(', '\\)']],
    "displayMath": [["\\[", "\\]"]],
    "macros": {                 
            "RR": '{\\mathbb{R}}',  # real numbers
            "CC": '{\\mathbb{C}}',  # complex numbers
            "QQ": '{\\mathbb{Q}}',  # rational numbers
            "ZZ": '{\\mathbb{Z}}',  # integers
            "NN": '{\\mathbb{N}}',  # natural numbers (including 0)
            "PP": '{\\mathbb{P}}',  # projective space
            "KK": '{\\mathbb{K}}',  # a field
            }    
  }
}
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [u'_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

# html_logo = "../assets/logo_nofont.png"
html_theme_options = {
    "sidebar_hide_name": False,
    "light_logo": "logo_nofont.png",
    "dark_logo": "logo_nofont_dark.png",
    "announcement": "<em>The package has not reached a stable version yet!</em>",
}

html_title = "PyRigi"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

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
htmlhelp_basename = 'PyRigidoc'


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
    'preamble': r"""
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
    (master_doc, 'pyrigi.tex', u'PyRigi Documentation',
     u'The PyRigi Developers', 'manual'),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'PyRigi', u'PyRigi Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'pyrigi', u'PyRigi Documentation',
     author, 'pyrigi', 'One line description of project.',
     'Miscellaneous'),
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
epub_exclude_files = ['search.html']


# -- Extension configuration -------------------------------------------------

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Local Sphinx extensions -------------------------------------------------


def setup(app: Sphinx):
    """Add functions to the Sphinx setup."""
    from myst_parser._docs import MystLexer
    app.add_lexer("myst", MystLexer)

