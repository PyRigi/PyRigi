
(dependencies)=
# Dependencies

We maintain the dependencies of the package using [Poetry](https://python-poetry.org/).
See the [installation instructions](https://python-poetry.org/docs/#installation).

To install the package dependencies including those needed for the development, run
```
poetry install --no-root --all-extras
```
in the root folder of PyRigi.
Omitting `--no-root` installs also PyRigi itself, so it can be used system-wide.
The option `--all-extras` specifies to install also all optional packages.
To install a specific group of optional packages, use
```
poetry install --extras "extra_name"
```
These are documented in the [Installation Guide](#optional-packages).

Poetry installs the dependencies and the package to a virtual environment.
To activate this environment, run `poetry shell`.
You can exit it with `exit` or `Ctrl+D`.

If you want to install dependencies necessary only for the package itself, not for the development, run
```
poetry install --only main
```

## PEP8

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

## Testing

Tests are extremely important to guarantee the realiability of code.
Please create tests for the functionalities that you implement and place them in the `test` folder, within the appropriate file.
Each test should be in the form of a function starting with `test_`.
Tests can be parametrized, see for instance `test_is_inf_rigid` in `test_framework.py`.

Please, follow this naming convention for tests of a specific function/method:

| function  | test name        | test...
|-----------|------------------|---------------------------------------
| `foo`     | `test_foo`       | the functionality of `foo`
| `foo(dim)`| `test_foo_d1`    | the functionality of `foo` for `dim=1`
| `foo`     | `test_foo_error` | that `foo` raises correct errors
| `is_bar`  | `test_is_bar`    | that property `bar` holds
| `is_bar`  | `test_is_not_bar`| that property `bar` does not hold

There can be also more complex tests, i.e., testing more than a single method;
please, choose a reasonable name for them following the idea above.

Moreover, please add a section `EXAMPLES` in the docstring of the classes and methods that you introduce and provide there examples of the functionalities you implemented.

Please keep in mind that whenever a pull request is opened, all the tests in the `test`folder and in the docstrings are run.
Therefore, before opening a pull request we **strongly advise** to run
```
pytest
```
in the root folder of PyRigi (with poetry shell activated).
The reason why the examples in the docstrings are tested is to make sure their outputs are valid,
they do **not** replace the tests in the `test` folder.
If you do not want to run doctests, run
```
pytest -p no:doctestplus
```
Functionalities requiring optional packages are tested by default;
if you want to skip some specific optional feature(s), run
```
pytest -m "not optional_feature1_name and not optional_feature2_name"
```
See the file `pyproject.toml` for the markers that specify groups of tests relying on optional packages.

We mark tests that take longer time according to the following table:

| marker               | per test | total time | execution
| -------------------- | -------- | ---------- | -------------------
| standard (no marker) | < 0.5s   | < 2 min    | on merge/PR to `dev`
| `slow_main`          | < 10s    | < 15 min   | on merge/PR to `main`
| `long_local`         | > 10s    | hours      | locally when needed

The column `total time` indicates how much time is needed to run all tests with the given marker.
The time limits per tests are approximate: it is better to have a longer standard tests than none.
Also most of the standard tests should be much faster then the indicated limit.

The command `pytest` executes only standard tests.
To include also the tests marked `slow_main`, run
```
pytest -m 'not slow_main or slow_main'
```