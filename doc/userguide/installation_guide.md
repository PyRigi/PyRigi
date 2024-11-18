(installation-guide)=

# Installation Guide

Depending on your operating system, the installation of the `PyRigi` package differs.

## Linux and Mac

1. Make sure that your Python version is at least 3.10.
2. The command `pip install pyrigi` installs PyRigi on your machine.
3. If not already installed, you can install Jupyter notebooks via the command `pip install jupyterlab`. It can be run by typing `jupyter lab`. Doing so provides a convenient user interface to use the functionality of PyRigi.


## Windows

In this tutorial, we assume a clean Windows without Python installed. In case that you have `pip` already installed, you can skip the first 3 steps. This can be checked using the command `python -m pip -v`. Otherwise, we recommend uninstalling any existing Python installation in the "Add or Delete Programs" settings menu first.

1. Download Python with a Version >= 3.10 from the website https://www.python.org/downloads/.
2. When installing, make sure that you tick the box "add to path variables" on the first installation page. You may need to tick the box that you run the installation as an Administrator as well.
3. As soon as your installation is successful, go to the Windows command line tool by following Windows key > type "cmd" > Enter. You can check if your Python Installation has been added to the environment variables by typing `python -v`. This command opens Python. It can be closed again with the command `exit()`.
4. Install PyRigi by typing `python -m pip install pyrigi`.
5. If you want to use PyRigi in a Jupyter notebook, install Jupyter Labs via the command `python -m pip install jupyterlab`. You can open a Jupyter notebook by typing `jupyter lab`. This provides a convenient user interface for PyRigi.


(optional-packages)=
## Optional packages

For counting the number of realizations of a minimally rigid graph,
the package `lnumber` is necessary. To install `PyRigi` including `lnumber`, run
```
pip install pyrigi[realization-counting]
```
Before installing this package, please read the [Python instructions of the package](https://github.com/jcapco/lnumber).
