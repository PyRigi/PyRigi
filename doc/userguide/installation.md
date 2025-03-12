(installation-guide)=
# Installation

If you are familiar with `pip`, you can install the latest version of PyRigi including the necessary dependencies
(see also [optional packages](#optional-packages)) by
```
pip install pyrigi
```
Otherwise, see how to start using Python and `PyRigi` depending on your [operating system](#OS-instructions).

Alternatively, one can clone/download the package
from [this GitHub repository](https://github.com/pyRigi/PyRigi),
see the branch `dev` for the development version.
Installation for development is done via [Poetry](#dependencies-poetry).

(optional-packages)=
## Optional packages

Some packages are not installed by default.
Here we list the groups of optional packages that may be used to increase the functionalities of PyRigi.

### Realization counting

For counting the number of realizations of a minimally rigid graph,
the package `lnumber` is necessary. To install `PyRigi` including [`lnumber`](https://github.com/jcapco/lnumber), run
```
pip install pyrigi[realization-counting]
```
Before installing this package, please read the [Python instructions of the package](https://github.com/jcapco/lnumber).

### Creation of meshes

To create meshes of bars that can be exported as STL files,
the packages [`trimesh`](https://trimesh.org/) and [`manifold3d`](https://github.com/elalish/manifold) are necessary.
To install `PyRigi` including `trimesh` and `manifold3d`, run
```
pip install pyrigi[meshing]
```

(OS-instructions)=
## Operating systems

### Linux and Mac

1. Make sure that your Python version is at least 3.11.
2. The command `pip install pyrigi` installs PyRigi on your machine.
3. If not already installed, you can install Jupyter notebooks via the command `pip install jupyterlab`. It can be run by typing `jupyter lab`. Doing so provides a convenient user interface to use the functionality of PyRigi.


### Windows

In this tutorial, we assume a clean Windows without Python installed. In case that you have `pip` already installed, you can skip the first 3 steps. This can be checked using the command `python -m pip -v`.

1. Download Python with a version >= 3.11 from the website [https://www.python.org/downloads/](https://www.python.org/downloads/).
2. When installing, make sure that you tick the box "add to path variables" on the first installation page. You may need to tick the box that you run the installation as an Administrator as well.
3. As soon as your installation is successful, go to the Windows command line tool by following Windows key > type "cmd" > Enter. You can check if your Python Installation has been added to the environment variables by typing `python -v`. This command opens Python. It can be closed again with the command `exit()`.
4. Install PyRigi by typing `python -m pip install pyrigi`.
5. If you want to use PyRigi in a Jupyter notebook, install Jupyter Labs via the command `python -m pip install jupyterlab`. You can open a Jupyter notebook by typing `jupyter lab`. This provides a convenient user interface for PyRigi.
