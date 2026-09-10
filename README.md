<p align="center">
<img src="https://raw.githubusercontent.com/PyRigi/PyRigi/refs/heads/main/assets/icon.jpg" alt="Pyrigi Logo" width="260">
</p>

[![PyRigi documentation](https://img.shields.io/badge/PyRigi-Documentation-blue?style=plastic&link=pyrigi.github.io%2FPyRigi%2F%20)](https://pyrigi.github.io/PyRigi/)
[![MIT license](https://img.shields.io/badge/license-MIT-yellow?style=plastic)](LICENSE)
[![python](https://img.shields.io/badge/dynamic/toml?url=https://raw.githubusercontent.com/PyRigi/PyRigi/main/pyproject.toml&query=%24.project.requires-python&label=python&style=plastic&color=blue)](https://www.python.org/)
[![Black code style](https://img.shields.io/badge/code%20style-black-black?style=plastic)](https://github.com/psf/black)
<!-- The package badges read the minimal supported version from pyproject.toml with
the regex "<package>\s*\(?>=\s*([^,")\s]+) ; it has to be percent-encoded below,
since parentheses and backslashes are not allowed in a Markdown link. -->
[![networkx](https://img.shields.io/badge/dynamic/regex?url=https://raw.githubusercontent.com/PyRigi/PyRigi/main/pyproject.toml&search=%22networkx%5Cs*%5C%28%3F%3E%3D%5Cs*%28%5B%5E%2C%22%29%5Cs%5D%2B%29&replace=%3E%3D%241&label=networkx&style=plastic&color=blue)](https://networkx.org/)
[![numpy](https://img.shields.io/badge/dynamic/regex?url=https://raw.githubusercontent.com/PyRigi/PyRigi/main/pyproject.toml&search=%22numpy%5Cs*%5C%28%3F%3E%3D%5Cs*%28%5B%5E%2C%22%29%5Cs%5D%2B%29&replace=%3E%3D%241&label=numpy&style=plastic&color=blue)](https://numpy.org/)
[![sympy](https://img.shields.io/badge/dynamic/regex?url=https://raw.githubusercontent.com/PyRigi/PyRigi/main/pyproject.toml&search=%22sympy%5Cs*%5C%28%3F%3E%3D%5Cs*%28%5B%5E%2C%22%29%5Cs%5D%2B%29&replace=%3E%3D%241&label=sympy&style=plastic&color=blue)](https://www.sympy.org/)


<!-- start-input -->

PyRigi is a Python package for research in rigidity and flexibility of bar-and-joint frameworks.
We aim at providing a tool for investigating combinatorial and geometric questions
such as infinitesimal, global, minimal, or generic rigidity. An article explaining the functionality
and internal structure of PyRigi is freely available [here](https://doi.org/10.1145/3815171).


We use [NetworkX](https://networkx.org/) for graph theory, [SymPy](https://www.sympy.org/)
for symbolic and [NumPy](https://numpy.org/) for numerical computations.
We acknowledge these and all the other open-source projects upon which PyRigi is based.

## Installation and usage

To install the latest stable version of PyRigi, run:
```
pip install pyrigi
```
Once installed, you can start using it with:
```python
from pyrigi import Graph, Framework
```
For more details, we refer to the
[Getting started](https://pyrigi.github.io/PyRigi/userguide/getting_started.html)
guide in the [documentation](https://pyrigi.github.io/PyRigi/).
The development version is available on the `dev` branch
in [this GitHub repository](https://github.com/pyRigi/PyRigi).

## Documentation

The documentation of the latest stable version is available [online](https://pyrigi.github.io/PyRigi/).
For compiling it locally,
see the [development guide](https://pyrigi.github.io/PyRigi/development/howto).

An important part of the documentation is the
[mathematical background](https://pyrigi.github.io/PyRigi/math/rigidity.html).
We specify the outputs of the methods in the package
by providing rigorous mathematical definitions.

## Questions and feature requests

We have a [Zulip chat](https://pyrigi.zulipchat.com),
where you can ask questions or propose new functionality.
If you want to get access to it, please send an email to
[this address](mailto:external.dc4f45edef70cb7e0c621ad50377d9f1.show-sender.include-footer@streams.zulipchat.com).
You can also use the [GitHub Discussions](https://github.com/PyRigi/PyRigi/discussions).

To report bugs or ask for new features, please create an [issue](https://github.com/PyRigi/PyRigi/issues/new/choose).

## Contributing

We appreciate contributions!
Do you have a research result
about rigidity or flexibility of bar-joint frameworks
that could be implemented?
[Let us know](https://github.com/PyRigi/PyRigi/issues/new/choose)!
Or even better, implement it and contribute to the package!

Besides coding, you can also help for instance
by extending the mathematical documentation or
creating tutorials.

If you want to contribute, please,
read the [development guide](https://pyrigi.github.io/PyRigi/development/howto)
and [contact us](mailto:external.dc4f45edef70cb7e0c621ad50377d9f1.show-sender.include-footer@streams.zulipchat.com).

## License

The package is licensed under the [MIT license](https://github.com/PyRigi/PyRigi/blob/main/LICENSE).

## The PyRigi Developers

See the complete [list of contributors](https://pyrigi.github.io/PyRigi/development/contributors.html).

The current maintainers of the project are:

[Matthias Adrian-Himmelmann](https://matthiashimmelmann.github.io/) \
[Matteo Gallet](mailto:matteo.gallet@units.it) \
Georg Grasegger \
[Jan Legerský](https://jan.legersky.cz/)

The decision to create PyRigi was made by the participants of the workshop
[Code of Rigidity](https://www.ricam.oeaw.ac.at/specsem/specsem2024/workshop2/)
(March 11–15, 2024), which was part of the
Special Semester on Rigidity and Flexibility at [RICAM](https://www.oeaw.ac.at/ricam/), Linz, Austria.

## Citing PyRigi

If you would like to cite PyRigi, please use the following reference:

Matthias Adrian-Himmelmann, Matteo Gallet, Georg Grasegger, and Jan Legerský.
*PyRigi – A General-Purpose Python Package for the Rigidity
and Flexibility of Bar-and-Joint Frameworks*,
ACM Transactions on Mathematical Software, 52(3), Article No. 14, 2026.
[doi:10.1145/3815171](https://doi.org/10.1145/3815171).

```
@article{pyrigi,
      title = {{PyRigi -- A General-Purpose Python Package for the Rigidity and Flexibility of Bar-and-Joint Frameworks}},
      author = {Matthias Adrian-Himmelmann and Matteo Gallet and Georg Grasegger and Jan Legerský},
      journal = {ACM Transactions on Mathematical Software},
      volume = {52},
      number = {3},
      pages = {1--22, Article No. 14},
      year = {2026},
      doi = {10.1145/3815171},
}
```
