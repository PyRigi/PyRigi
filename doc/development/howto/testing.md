(testing)=
# Testing

Tests are extremely important to guarantee the reliability of code.
Please create tests for the functionalities that you implement and
place them in the `test` folder, within the appropriate file
following the [package structure](#pkg_structure).
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

There can be also more complex tests, i.e., testing more than a single function;
please, choose a reasonable name for them following the idea above.

Moreover, please add a section `EXAMPLES` in the docstring of the classes and functions
that you introduce and provide there examples of the functionalities you implemented.

Please keep in mind that whenever a pull request is opened,
all the tests in the `test` folder and in the docstrings are run.
Therefore, before opening a pull request we **strongly advise** to run
```
pytest
```
in the root folder of PyRigi (with Poetry environment [activated](#dependencies-poetry)).
The reason why the examples in the docstrings are tested is to make sure their outputs are valid,
they do **not** replace the tests in the `test` folder.
If you do not want to run doctests, run
```
pytest -p no:doctestplus
```

## Function vs. method testing

As described in [Package Structure](#pkg_structure), most of the functionality of
the classes {class}`.Graph` and {class}`.Framework` is implemented
via functions in various modules, which are then wrapped as methods.
Thanks to the tests described below, only the functions need
to be tested as the wrapping is guaranteed to be correct.

The test `test_signature` in `test/test_signature.py` checks whether the signatures
of the methods and the wrapped function match.
Hereby, "match" means that the parameters are the same, have the same default values,
and have the same type (or inherited type).

The plugin [`flake8-unused-arguments`](https://github.com/nhoad/flake8-unused-arguments)
guarantees that all arguments of each method are indeed used when calling the wrapped
function. This plugin is automatically used (calling `flake8`) once dependencies are
installed [via Poetry](#dependencies-poetry).

In addition, tests that verify correct wrapping are found in`test/wrapper/test_wrapper.py`.
This test suite checks systematically that all
`@copy_doc`-decorated methods of {class}`.Graph` and {class}`.Framework` correctly forward arguments
to the underlying functions. It creates various mock arguments, invokes the method,
and asserts that all parameters are properly passed through, both in name and value.

The suite also includes negative tests using intentionally broken wrappers (see
`test.wrapper._bad_wrapper._BadWrapper`), which verify that common mistakes in wrapping
(such as missing, extra, or reordered arguments, or wrong function calls) are detected
by the test helpers.

## Markers

Functionalities requiring optional packages are tested by default;
if you want to skip some specific optional feature(s), run
```
pytest -m "not slow_main and not long_local and not opt_feature1 and not opt_feature2"
```
See the file `pyproject.toml` for the markers that specify groups of tests relying on [optional packages](#optional-packages).

We mark tests that take longer time according to the following table:

| marker               | per test | total time | execution             | GitHub action timeout |
| -------------------- | -------- | ---------- | ----------------------| ----------------------|
| standard (no marker) | < 0.5s   | < 2 min    | on merge/PR to `dev`  | 5 min                 |
| `slow_main`          | < 10s    | < 15 min   | on merge/PR to `main` | 30 min                |
| `long_local`         | > 10s    | hours      | locally when needed   | -                     |

The column `total time` indicates how much time is needed to run all tests with the given marker.
The time limits per tests are approximate: it is better to have a longer standard tests than none.
Also most of the standard tests should be much faster than the indicated limit.

The command `pytest` executes only standard tests.
To run the standard tests and those marked `slow_main`, run
```
pytest -m 'not long_local'
```

## Coverage

To check the coverage by the unit tests, run
```
pytest --cov=pyrigi test/ --cov-report html -p no:warnings
```
The following command includes also the doctests
```
pytest --doctest-modules --cov=pyrigi test/ pyrigi/ --cov-report html -p no:warnings
```
Suppressing all warnings with `-p no:warnings` solves the issues cause by
filtering `RandomizedAlgorithmWarning`.
