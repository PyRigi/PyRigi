# How to contribute


## Gitflow

## Code

### PEP8

We follow [PEP8](https://peps.python.org/pep-0008/) indications regarding Python code format.
To check whether the code is PEP8-compliant, one can use [`pycodestyle`](https://pycodestyle.pycqa.org).
There are tools that format code according to PEP8 indications, as for example [`autopep8`](https://pypi.org/project/autopep8/), which we reccommend     to run as
```
autopep8 --in-place --aggressive --aggressive <filename>
```
(as suggested in the documentation) to modify the files in place.
Another, more stringent, tool is [`black`](https://black.readthedocs.io).

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

To compile, runSphinx
```
make html
```
in the folder `doc`.

New `latex` commands can be created by modifying `mathjax3_config` in `doc/conf.py`.


### Auto-build

If you want that the documentation folder is kept watched and documentation is automatically rebuilt once a change is detected, you can use the Sphinx extension [`sphinx-autobuild`](https://github.com/sphinx-doc/sphinx-autobuild), which can be installed via
```
pip install sphinx-autobuild
```
At this point, run
```
sphinx-autobuild . _build/html --open-browser
```
in the `doc` folder.


