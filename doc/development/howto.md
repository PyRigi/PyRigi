# How to contribute


## Gitflow

## Code




## Documentation

We aim to have the package well-documented.
Since the main purpose of the package is mathematical research,
it requires rigorous definitions.
Hence, an essential part of the documentation is the
[mathematical background](#definitions).

The documentation is generated from docstrings using [Sphinx](https://www.sphinx-doc.org).
We use the theme [Furo](https://github.com/pradyunsg/furo).
The docstrings are formatted according to [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) and parsed using [napoleon](https://sphinxcontrib-napoleon.readthedocs.io/)
to retrieve the information from type hinting.
The other documentation source files are written in [MyST](https://myst-parser.readthedocs.io/).
The following extensions are used:
 - [`myst-parser`](https://myst-parser.readthedocs.io) - parsing MyST syntax;
 - [`sphinx-math-dollar`](https://www.sympy.org/sphinx-math-dollar/) - dollar signs to typeset inline and displayed math expressions;
 - [`sphinx-proof`](https://sphinx-proof.readthedocs.io) - mathematical environments (definitions, theorems,...);
 - [`sphinxcontrib-bibtex`](https://sphinxcontrib-bibtex.readthedocs.io) - bibliography using `.bib` file;
 - [`sphinx-copybutton`](https://sphinx-copybutton.readthedocs.io) - button to copy code easily;
 - [`sphinx-design`](https://sphinx-design.readthedocs.io) - fancy features like tabs.

These can be installed by
```
pip install sphinxcontrib-napoleon myst-parser sphinx-math-dollar sphinx-proof sphinxcontrib-bibtex sphinx-copybutton sphinx-design
```

To compile, run
```
make html
```
in the folder `doc`.

New `latex` commands can be created by modifying `mathjax3_config` in `doc/conf.py`.


### Auto-build

https://github.com/sphinx-doc/sphinx-autobuild

pip install sphinx-autobuild

in the doc folder:

sphinx-autobuild . _build/html --open-browser



