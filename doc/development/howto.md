# How to contribute

```{include} ../../CONTRIBUTING.md
:relative-docs: docs/
:start-after: <!-- start-input -->
:end-before: <!-- end-input -->
```

This page provides basic information to start contributing.

## Git(flow)

We use [Git](https://git-scm.com/) for version control and the project is hosted at [Github](https://github.com/PyRigi/Pyrigi).
We use [Gitflow](https://nvie.com/posts/a-successful-git-branching-model/) (see also [this description](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)) for PyRigi development.
In a nutshell, this means that there are two prominent branches in PyRigi's Git repository:

- `main`, which contains the stable version of the package
- `dev`, which is used for the development.

Collaborators are not allowed to push their Git commits directly to these two branches.
Rather, they should employ _pull requests_.
Say Alice and Bob want to implement feature X in PyRigi.
These are the tasks to be performed:

1. they branch from `dev`, creating a branch called `feature-X`, and there they develop the intended functionality;
2. once they are done, they push `feature-X` to GitHub and solicit a pull request of `feature-X` into `dev`;
3. the code is checked by the maintainers, who may ask some other collaborator to serve as reviewer; in this process, comments and suggested of change may be sent to Alice and Bob until agreement is reached about the feature and the pull request receives approval;
4. a maintainer merges `feature-X` into `dev`.

We propose a few categories for contributing branches:
* _features_: branches to implement new features/improvements to the current status; their name should start by `feature-`
* _documentation_: branches to modify the documentation; their name should start by `doc-`
* _bugs_: branches to solve known bugs; their name should start by `bug-`
* _hotfix_: branches to solve an urgent error; their name should start by `hotfix-`

Once in a while, the maintainers merge the branch `dev` into `main` and create a new release.

## Code

### PEP8

We follow [PEP8](https://peps.python.org/pep-0008/) indications regarding Python code format.

To check whether the code is PEP8-compliant, we strongly suggest to use
[flake8](https://flake8.pycqa.org).
To install it, run
```
pip install flake8
```
To check your code, simply run
```
flake8 .
```
in PyRigi's home folder.
The `.flake8` file in PyRigi's home folder, which specifies `flake8` configuration,
is the same that is used in the automatic tests once a pull request is filed in GitHub.
Therefore, please check your code with `flake8` before performing a pull request.

There are tools that format code according to PEP8 indications.
We **strongly** suggest to use [`black`](https://black.readthedocs.io).
To install it, run
```
pip install black
```
To format your code, run
```
black .
```
in the root folder to modify the files in place.
We suggest to integrate the use of `black` at every commit
as explained at [this page](https://black.readthedocs.io/en/stable/integrations/source_version_control.html) of `black`'s guide.

The lines in the source code can be at most 90 characters long.
The only exceptions are lines in docstrings that might be longer
due to long output in an example or a reference.
To avoid checking them by the automatic tests for pull requests,
use `noqa` like this:
```python
"""
veeeeery long docstring that has to violate the 90 characters limit due to a reference or an example
"""  # noqa: E501
```

### Testing

Tests are extremely important to guarantee the realiability of code.
Please create tests for the functionalities that you implement and place them in the `test` folder, within the appropriate file.
Each test should be in the form of a method starting with `test_`.

Moreover, please add a section `EXAMPLES` in the docstring of the classes and methods that you introduce and provide there examples of the functionalities you implemented.

Please keep in mind that whenever a pull request is opened, all the tests in the `test`folder and in the docstrings are run.
Therefore, before opening a pull request we **strongly advise** to run
```
pytest --doctest-modules
```
in the root folder of PyRigi.
The reason why the examples in the docstrings are tested is to make sure their outputs are valid,
they do **not** replace the tests in the `test` folder.

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
 - [`myst-nb`](https://myst-nb.readthedocs.io/) - parsing MyST syntax and Jupyter notebooks;
 - [`sphinx-math-dollar`](https://www.sympy.org/sphinx-math-dollar/) - dollar signs to typeset inline and displayed math expressions;
 - [`sphinx-proof`](https://sphinx-proof.readthedocs.io) - mathematical environments (definitions, theorems,...);
 - [`sphinxcontrib-bibtex`](https://sphinxcontrib-bibtex.readthedocs.io) - bibliography using `.bib` file;
 - [`sphinx-copybutton`](https://sphinx-copybutton.readthedocs.io) - button to copy code easily;
 - [`sphinx-design`](https://sphinx-design.readthedocs.io) - fancy features like tabs;
 - [`sphinx-tippy`](https://sphinx-tippy.readthedocs.io/en/latest/) - previews of definitions and methods.

These can be installed by running the following command in the folder `doc`:
```
pip install --upgrade -r requirements.txt
```

To compile, run Sphinx in the folder `doc` by:
```
make html
```
or
```
make latexpdf
```


### Auto-build

If you want that the documentation folder is kept watched and documentation is automatically rebuilt once a change is detected (works only for `.md` files, not docstrings), you can use the Sphinx extension [`sphinx-autobuild`](https://github.com/sphinx-doc/sphinx-autobuild), which can be installed via
```
pip install sphinx-autobuild
```
At this point, run in the `doc` folder:
```
sphinx-autobuild . _build/html --open-browser
```
To recompile everything, stop the previous command and run
```
make clean
make html
```
Cleaning is necessary especially to get the documentation updated
after a change in docstrings. 

### Creating tutorials

We appreciate a lot if you can contribute with a notebook that
illustrates how to use PyRigi, describes a rigidity theory problem, accompanies a paper etc.

The [tutorials](#tutorials) section is generated from Jupyter notebooks;
more precisely, from MyST Markdown mirrors of `.ipynb` notebooks.
This allows versioning the `.md` notebooks in Git without having troubles with the metadata, outputs etc. of `.ipynb` notebooks.
The pairing of `.ipynb` notebooks with MyST `.md` notebooks
is achieved using [Jupytext](https://jupytext.readthedocs.io/en/latest/index.html).

Please, **do not** commit the `.ipynb` to the repository.
You can contact a maintainer if you have a `.ipynb` tutorial
you want to contribute but struggle to get its `.md` version.
