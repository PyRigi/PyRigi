<p align="center">
<img src="https://raw.githubusercontent.com/PyRigi/PyRigi/refs/heads/main/assets/icon.jpg" alt="Pyrigi Logo" width="260">
</p>

[![PyRigi documentation](https://img.shields.io/badge/PyRigi-Documentation-blue?style=plastic&link=pyrigi.github.io%2FPyRigi%2F%20)](https://pyrigi.github.io/PyRigi/)
[![MIT license](https://img.shields.io/badge/license-MIT-yellow?style=plastic)](LICENSE)
[![Black code style](https://img.shields.io/badge/code%20style-black-black?style=plastic)](https://github.com/psf/black)


<!-- start-input -->

PyRigi is a Python package for research in rigidity and flexibility of bar-and-joint frameworks.
We aim at providing a tool for investigating combinatorial and geometric questions
such as infinitesimal, global, minimal, or generic rigidity. An article explaining the functionality
and internal structure of PyRigi is freely available [here](https://doi.org/10.48550/arXiv.2505.22652).


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

[Matteo Gallet](mailto:matteo.gallet@units.it) \
Georg Grasegger \
[Matthias Himmelmann](https://matthiashimmelmann.github.io/) \
[Jan Legerský](https://jan.legersky.cz/)

The decision to create PyRigi was made by the participants of the workshop
[Code of Rigidity](https://www.ricam.oeaw.ac.at/specsem/specsem2024/workshop2/)
(March 11–15, 2024), which was part of the
Special Semester on Rigidity and Flexibility at [RICAM](https://www.oeaw.ac.at/ricam/), Linz, Austria.

## Citing PyRigi

If you would like to cite PyRigi, please use the following reference:

Matteo Gallet, Georg Grasegger, Matthias Himmelmann, and Jan Legerský. *PyRigi
– a general-purpose Python package for the rigidity and flexibility of bar-and-joint
frameworks*, 2025, preprint. [doi:10.48550/arXiv.2505.22652](https://doi.org/10.48550/arXiv.2505.22652).

```
@misc{pyrigi,
      title = {{PyRigi -- a general-purpose Python package for the rigidity and flexibility of bar-and-joint frameworks}},
      author = {Matteo Gallet and Georg Grasegger and Matthias Himmelmann and Jan Legerský},
      year = {2025},
      doi = {10.48550/arXiv.2505.22652},
      note = {preprint},
}
```
