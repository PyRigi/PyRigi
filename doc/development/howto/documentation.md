(documentation)=
# Documentation

We aim to have the package well-documented.
Since the main purpose of the package is mathematical research,
it requires rigorous definitions.
Hence, an essential part of the documentation is the
[mathematical background](#definitions).

The documentation is generated from docstrings using [Sphinx](https://www.sphinx-doc.org).
We use the theme [Furo](https://github.com/pradyunsg/furo).
The docstrings are written in [reST](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html),
formatted according to [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html)
and parsed using [napoleon](https://sphinxcontrib-napoleon.readthedocs.io/)
to retrieve the information from type hinting.
The other documentation source files are written in [MyST](https://myst-parser.readthedocs.io/)
(see also the [cheatsheet](cheatsheet)).
The following extensions are used:
 - [`myst-nb`](https://myst-nb.readthedocs.io/) - parsing MyST syntax and Jupyter notebooks;
 - [`sphinx-math-dollar`](https://www.sympy.org/sphinx-math-dollar/) - dollar signs to typeset inline and displayed math expressions;
 - [`sphinx-proof`](https://sphinx-proof.readthedocs.io) - mathematical environments (definitions, theorems,...);
 - [`sphinxcontrib-bibtex`](https://sphinxcontrib-bibtex.readthedocs.io) - bibliography using `.bib` file;
 - [`sphinx-copybutton`](https://sphinx-copybutton.readthedocs.io) - button to copy code easily;
 - [`sphinx-design`](https://sphinx-design.readthedocs.io) - fancy features like tabs;
 - [`sphinx-tippy`](https://sphinx-tippy.readthedocs.io/en/latest/) - previews of definitions and methods.

These are already installed if you used `poetry install`.

To compile, run Sphinx in the folder `doc` (with Poetry environment activated) by:
```
make html
```
or
```
make latexpdf
```

To clean and remove all the created files, run in the folder `doc`
```
make clean
```
Cleaning is necessary especially to get the documentation updated
after a change in docstrings.

If you do not have `make` installed, run Sphinx in the root folder by
```
sphinx-build -M html doc doc/_build/html
```
To clean and remove all the created files, run in the root folder
```
sphinx-build -M clean doc doc/_build/html
```

## Docstrings

For an example how a docstring should look like,
see for instance the docstring of {class}`.Framework`
or {meth}`.Framework.realization`.
In general, a docstring should contain the following items (in this order):
 - short description (one line, compulsory)
 - longer description (optional)
 - definitions (optional, but recommended): references to related definitions
 - parameters (optional, but recommended): description of parameters, types are added automatically from type hinting
 - examples (highly recommended)
 - notes (optional): implementation details
 - suggested improvements (optional): proposed changes/extensions to the method

To check whether some docstrings are missing, run
```
make html coverage
```

## Auto-build

If you want that the documentation folder is kept watched and documentation is automatically rebuilt once a change is detected (works only for `.md` files, not docstrings), you can use the Sphinx extension [`sphinx-autobuild`](https://github.com/sphinx-doc/sphinx-autobuild).
Run in the `doc` folder (with Poetry environment activated):
```
sphinx-autobuild . _build/html --open-browser
```

## References
We use a bib file to collect the metadata of references (`refs.bib`).
In order to avoid duplicates and missing entries we use a consistent style.
An item is labeled according to the following format
 - 1 author: `SurnameYear`
 - 2 authors: `SurnameSurnameYear`
 - 3 authors: `SurnameSurnameSurnameYear`
 - 4 or more authors: `SurnameSurnameEtAlYear`

where year is in YYYY format. Labels shall not have diacritical signs.
The references shall be sorted in the file by label even though there is automatic sorting for the output.
Within the metadata for one reference we keep the following order:
 - `author` (full names)
 - `title`
 - `journal`/`publisher`
 - `volume`/`address`
 - `number`
 - `pages`
 - `year`
 - `doi`/`eprint` (for published papers the `doi` shall be provided, for preprints an `eprint` identifier is fine)

## Tutorials

The [tutorials](#tutorials) section is generated from Jupyter notebooks;
more precisely, from MyST Markdown mirrors of `.ipynb` notebooks.
This allows versioning the `.md` notebooks in Git without having troubles with the metadata, outputs etc. of `.ipynb` notebooks.
The pairing of `.ipynb` notebooks with MyST `.md` notebooks
is achieved using [Jupytext](https://jupytext.readthedocs.io/en/latest/index.html).

Please, **do not** commit the `.ipynb` to the repository.
You can contact a maintainer if you have a `.ipynb` tutorial
you want to contribute but struggle to get its `.md` version.


In case the Poetry environment is [activated](dependencies), Jupyterlab and Jupytext
can be installed using
```
pip install jupyterlab jupytext
```
After setting the virtual environment in Jupyterlab to the one created
by Poetry, `.md` notebooks can be opened directly.

If the execution of a cell takes long time,
then import the cell magic

```python
from pyrigi.misc import skip_execution
```

and use it to skip a cell in the documentation compilation as follows

```python
%%skip_execution
# code of a cell taking more than a second
```

Both the import and cell magic are removed before the online documentation is compiled,
hence the output is displayed there.
Namely, the goal is to avoid long doc compilation on the `dev` branch,
but to keep it in the online documentation.
