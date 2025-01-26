(testing)=
# Testing

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
