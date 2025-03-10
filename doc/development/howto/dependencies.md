
(dependencies)=
# Dependencies

We maintain the dependencies of the package using [Poetry](https://python-poetry.org/).
See its [installation instructions](https://python-poetry.org/docs/#installation).

To install the package dependencies including those needed for the development, run
```
poetry install --all-extras
```
in the root folder of PyRigi.
The command installs also `PyRigi` itself, so it can be used system-wide.
Use `--no-root` to install only dependencies without PyRigi itself. 
The option `--all-extras` specifies to install also all optional packages.
To install a specific group of optional packages, use
```
poetry install --extras "extra_name"
```
These are documented in the [Installation Guide](#optional-packages).

Poetry installs the dependencies and the package to a virtual environment.
To activate this environment, use the command
```
eval $(poetry env activate)
```
or consult [Poetry documentation](https://python-poetry.org/docs/managing-environments/#activating-the-environment).
An alternative is to execute command line tools like `poetry run flake8`.

If you want to install dependencies necessary only for the package itself, not for the development, run
```
poetry install --only main
```
