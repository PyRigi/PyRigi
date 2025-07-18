(input_checks)=
# Input Checks

To unify to a certain extent error messages there are two ways.

## Check the input before actual computations
For checking input type and values for standard data types,
we collected in `pyrigi/_utils/_input_check.py` a set of methods that are often needed,
see the [list below](#general-input-check-functions).
For instance, {func}`pyrigi._utils._input_check.dimension` checks whether
the parameter `dim` is a positive integer and raises an error otherwise.

Checks related to the class {class}`.Graph` are in {mod}`.graph._utils._input_check`.
For instance, {func}`.graph._utils._input_check.no_loop` checks whether
a graph is loop free and raises an error otherwise.

Example:
```python
import pyrigi._utils._input_check as _input_check
import pyrigi.graph._utils._input_check as _graph_input_check

class Graph(nx.Graph):
    ...

    def method(self, dim: int):
        _input_check.dimension(dim)
        _graph_input_check.no_loop(self)
        ...
```

Note that these input checks are in a private module.
However, we do test them.


## Check the input at the end by exclusion
In some cases a method would run a conditional statement
through all possible options of an input parameter.
If the method reaches the end
an error would be raised to tell the user that the option is not supported.
For instance a parameter `algorithm` might have several supported values.

Example:
```python
from pyrigi.exception import NotSupportedValueError

def method(self, algorithm: str = "default"):
    if algorithm == "alg1":
        ...
    elif algorithm == "alg2":
        ...
    else:
        raise NotSupportedValueError(algorithm, "algorithm", self.method)
```

## Input check functions of Graph

```{eval-rst}
.. automodule:: pyrigi.graph._utils._input_check
   :members:
   :show-inheritance:
```

## Input check functions of Framework

See the methods of {class}`~.framework.base.FrameworkBase` starting with `_input_check`.

## General input check functions

```{eval-rst}
.. automodule:: pyrigi._utils._input_check
   :members:
   :show-inheritance:
```
