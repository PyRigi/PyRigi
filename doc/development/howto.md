# How to contribute


## Gitflow

## Code




## Documentation

We aim to have the package well-documented.
Since the main purpose of the package is mathematical research,
it requires rigorous definitions.
Hence, an essential part of the documentation is the 
[mathematical background](#definitions).

The documentation is generated from docstrings using `sphinx`.
We use the theme [Furo](https://github.com/pradyunsg/furo).
The docstrings are formatted according to [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) and parsed using [napoleon](https://sphinxcontrib-napoleon.readthedocs.io/)
to retrieve the information from type hinting.
The other documentation source files are written in [MyST](https://myst-parser.readthedocs.io/). 
The following extensions are used:
 - `myst-parser` - parsing MyST syntax;
 - `sphinx-math-dollar` - dollar signs to typeset inline and displayed math expressions;
 - `sphinx_proof` - mathematical environments (definitions, theorems,...);
 - `sphinxcontrib.bibtex` - bibliography using `.bib` file;
 - `sphinx_copybutton` - button to copy code easily;
 - `sphinx_design` - fancy features like tabs.
 
These can be installed by
```
pip install sphinxcontrib-napoleon TBA
```

:::{todo}
Complete the list, add links.
:::

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



