# How to contribute


## Gitflow

We use [Gitflow](https://nvie.com/posts/a-successful-git-branching-model/) (see also [this description](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)) for Pyrigi development.
In a nutshell, this means that there are two prominent branches in Pyrigi's Git repository:

- `main`, which ...
- `dev`, which ...

Collaborators are not allowed to push their Git commits directly to these two branches.
Rather, they should employ _pull requests_.
Say Alice and Bob want to implement feature X in Pyrigi.
These are the tasks to be performed:

1. they branch from `dev`, creating a branch called `feature-X`, and there they develop the intended functionality;
2. once they are done, they push `feature-X` to GitHub and solicit a pull request of `feature-X` into `dev`;
3. the code is checked by the maintainers, who may ask some other collaborator to serve as reviewer; in this process, comments and suggested of change may be sent to Alice and Bob until agreement is reached about the feature and the pull request receives approval;
4. a maintainer merges `feature-X` into `dev`.

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

### Testing

Tests are extremely important to guarantee the realiability of code.
Please create tests for the functionalities that you implement and place them in the `test` folder, within the appropriate file.
Each test should be in the form of a method starting with `test_`.

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
The other documentation source files are written in [MyST](https://myst-parser.readthedocs.io/) 
(see also the [cheatsheet](#cheatsheet)).
The following extensions are used:
 - [`myst-parser`](https://myst-parser.readthedocs.io) - parsing MyST syntax;
 - [`sphinx-math-dollar`](https://www.sympy.org/sphinx-math-dollar/) - dollar signs to typeset inline and displayed math expressions;
 - [`sphinx-proof`](https://sphinx-proof.readthedocs.io) - mathematical environments (definitions, theorems,...);
 - [`sphinxcontrib-bibtex`](https://sphinxcontrib-bibtex.readthedocs.io) - bibliography using `.bib` file;
 - [`sphinx-copybutton`](https://sphinx-copybutton.readthedocs.io) - button to copy code easily;
 - [`sphinx-design`](https://sphinx-design.readthedocs.io) - fancy features like tabs.

These can be installed by running the following command in the folder `doc`:
```
pip install -r requirements.txt
```

To compile, run Sphinx in the folder `doc` by:
```
make html
```

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


