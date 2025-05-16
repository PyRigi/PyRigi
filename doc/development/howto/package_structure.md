(package_structure)=
# Package structure

From user perspective, PyRigi follows object-oriented design.
The main functionality can be accessed via the methods of {class}`.Graph`, {class}`.Framework`,
or the classes inherited from {class}`.Motion`.
However, in order to have extendable and maintainable code,
most of the algorithms are implemented as functions and wrapped as methods of the corresponding classes.
As such, they are suggested by autocompletion tools once
an instance, like `Graph`, is available and therefore easy to search for and use.
This approach allows one to implement functionality in separate modules according to the topics,
see [below](#overview).

## Graph functionality

Functions implementing graph functionalities accept {class}`networkx.Graph` as the first parameter
and are then wrapped as {class}`pyrigi.Graph<.Graph>` methods.

For example, consider the method {meth}`.Graph.is_rigid`.
In the file `pyrigi/graph/graph.py` it looks like:

```python
from .rigidity import generic as generic_rigidity
class Graph(nx.Graph):
    @copy_doc(generic_rigidity.is_rigid)
    def is_rigid(
        self,
        dim: int = 2,
        algorithm: str = "default",
        use_precomputed_pebble_digraph: bool = False,
        prob: float = 0.0001,
    ) -> bool:
        return generic_rigidity.is_rigid(
            self,
            dim=dim,
            algorithm=algorithm,
            use_precomputed_pebble_digraph=use_precomputed_pebble_digraph,
            prob=prob,
        )
```

As one can see, this method simply calls the function `is_rigid`
located in the file `pyrigi/graph/rigidity/generic.py`.
The decorator `@copy_doc` copies the docstring from the function.
If we now look at the function, we see

```python
def is_rigid(
    graph: nx.Graph,
    dim: int = 2,
    algorithm: str = "default",
    use_precomputed_pebble_digraph: bool = False,
    prob: float = 0.0001,
) -> bool:
    """
    Return whether the graph is ``dim``-rigid.
    ...
    """
    # implementation of the function
```

As one may notice, the parameters of the function `is_rigid` are the same as those
for the method {meth}`.Graph.is_rigid`, except for the first one,
which is of type {class}`networkx.Graph` and is always called `graph`.

Therefore, if new graph functionalities are added,
they should be implemented as a function accepting {class}`networkx.Graph`
and then wrapped as methods of {class}`pyrigi.Graph<.Graph>`.
Since the docstrings are shown as those of methods,
they should be written keeping that in mind,
hence not referring to their first parameter,
but rather to expressions like "the graph" and not as `graph`.
Moreover, examples in docstrings should be methods of {class}`pyrigi.Graph<.Graph>` instances,
and not functions taking {class}`networkx.Graph`.

Regarding type hinting, {class}`networkx.Graph` should be used in the function signature,
while {class}`pyrigi.Graph<.Graph>` should be used in the method signature.
This is needed, for example, when a function/method returns a `Graph`.

## Framework functionality

Similarly to the graph case, the functions implementing framework functionalities
accept {class}`pyrigi.framework.base.FrameworkBase` as the first parameter, called `framework`, and
are wrapped as methods of {class}`pyrigi.Framework<.Framework>`,
which is inherited from {class}`pyrigi.framework.base.FrameworkBase`.

## Correct wrapping checks

The test `test_signature` in `test/test_signature.py` checks whether the signatures
of the methods and the wrapped function match.
Hereby, "match" means that the parameters are the same, have the same default values,
and have the same type (or inherited type).

The plugin [`flake8-unused-arguments`](https://github.com/nhoad/flake8-unused-arguments) guarantees that all arguments of each method are indeed used when calling the wrapped function.
This plugin is automatically used (calling `flake8`) once dependences are installed [via Poetry](#dependencies-poetry).


## Overview

```{literalinclude} ./pyrigi_structure.txt
:language: text
```
