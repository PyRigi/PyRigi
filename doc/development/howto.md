# How to contribute

```{include} ../../CONTRIBUTING.md
:relative-docs: docs/
:start-after: <!-- start-input -->
:end-before: <!-- end-input -->
```

This page provides basic information to start contributing.

## Communication

We use a [Zulip chat](https://pyrigi.zulipchat.com) for the communication among contributors.
If you want to get access to it, please send an email to
[this address](mailto:external.dc4f45edef70cb7e0c621ad50377d9f1.show-sender.include-footer@streams.zulipchat.com).
Feel free to ask any questions regarding PyRigi in a Zulip channel.

You can come with your own ideas on what to develop or you can check the channel
[To be implemented](https://pyrigi.zulipchat.com/#narrow/stream/444087-To-be-implemented)
for some suggestions about what the maintainers would be happy to have in PyRigi.

:::{important}
Currently, we prefer the following contributions to the code:
 - resolving an existing TODO (especially adding examples and tests),
 - implementing a method marked as `Waiting for implementation`
   or returning `NotImplementedError`,
 - or adding missing definitions.
:::

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
* _testing_: branches to add tests; their name should start by `test-`
* _refactoring_: branches to refactor the code; their name should start by `refactor-`

Once in a while, the maintainers merge the branch `dev` into `main` and create a new release.
The release numbers follow this scheme:

* MAJOR version: significant functionality extensions yielding possibly incompatible API changes (x+1.y.z)
* MINOR version: new functionality in a backward compatible manner (x.y+1.z)
* PATCH version: backward compatible bug fixes (x.y.z+1).


## Code

(dev-dependencies)=
### Dependencies

We maintain the dependencies of the package using [Poetry](https://python-poetry.org/).
See the [installation instructions](https://python-poetry.org/docs/#installation).

To install the package dependencies including those needed for the development, run
```
poetry install --no-root
```
in the root folder of PyRigi.
Omitting `--no-root` installs also PyRigi itself, so it can be used system-wide.
Poetry installs the dependencies and the package to a virtual environment.
To activate this environment, run `poetry shell`.
You can exit it with `exit` or `Ctrl+D`.

If you want to install dependencies necessary only for the package itself, not for the development, run
```
poetry install --only main
```



### PEP8

We follow [PEP8](https://peps.python.org/pep-0008/) indications regarding Python code format.

To check whether the code is PEP8-compliant, we strongly suggest to use
[flake8](https://flake8.pycqa.org).
To check your code, simply run
```
flake8
```
in PyRigi's home folder (with poetry shell activated).
The `.flake8` file in PyRigi's home folder, which specifies `flake8` configuration,
is the same that is used in the automatic tests once a pull request is filed in GitHub.
Therefore, please check your code with `flake8` before performing a pull request.

There are tools that format code according to PEP8 indications.
We **strongly** suggest to use [`black`](https://black.readthedocs.io).
To format your code, run
```
black .
```
in the root folder (with poetry shell activated) to modify the files in place.
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
Tests can be parametrized, see for instance `test_inf_rigid` in `test_framework.py`.

Moreover, please add a section `EXAMPLES` in the docstring of the classes and methods that you introduce and provide there examples of the functionalities you implemented.

Please keep in mind that whenever a pull request is opened, all the tests in the `test`folder and in the docstrings are run.
Therefore, before opening a pull request we **strongly advise** to run
```
pytest --doctest-modules
```
in the root folder of PyRigi (with poetry shell activated).
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
The docstrings are written in [reST](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html),
formatted according to [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html)
and parsed using [napoleon](https://sphinxcontrib-napoleon.readthedocs.io/)
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

These are already installed if you used `poetry install`.

To compile, run Sphinx in the folder `doc` (with poetry shell activated) by:
```
make html
```
or
```
make latexpdf
```

### Docstrings

For an example how a docstring should look like,
see for instance the docstring of {class}`.Framework`
or {meth}`.Framework.realization`.
In general, a docstring should contain the following items (in this order):
 - short description (one line, compulsory)
 - longer description (optional)
 - list of definitions (optional)
 - parameters description (optional): types are added automatically from type hinting
 - examples (highly recommended)
 - notes (optional): implementation details

### Auto-build

If you want that the documentation folder is kept watched and documentation is automatically rebuilt once a change is detected (works only for `.md` files, not docstrings), you can use the Sphinx extension [`sphinx-autobuild`](https://github.com/sphinx-doc/sphinx-autobuild).
Run in the `doc` folder (with poetry shell activated):
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


In case `poetry shell` is used as described above, Jupyterlab and Jupytext
can be install using
```
pip install jupyterlab jupytext
```
After setting the virtual enviroment in Jupyterlab to the one created
by `poetry shell`, `.md` notebooks can be opened directly.
