(style-guide)=
# Style Guide

We follow [PEP8](https://peps.python.org/pep-0008/) guidelines concerning the Python coding standard.

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