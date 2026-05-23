# Graph

Documentation of the module-level functions that implement the `Graph` class
behaviour. These functions take an `nx.Graph` instance as their
first argument and are called by the corresponding `Graph` methods via
`@copy_doc`. Examples shown here are in **function-call style**.

## Writing examples for `copy_doc`-wrapped functions

Examples in function-style docstrings must follow two rules so `copy_doc` can
invert them to method-style at import time:

1. **No chained instantiation.** Assign the graph to a variable first:
   ```python
   >>> g = graphs.Diamond()
   >>> len(list(all_k_extensions(g, 1, 2, only_non_isomorphic=True)))
   ```
   Chained calls like `graphs.Diamond().all_k_extensions(...)` are not caught
   by the regex and will remain in method-style inside the function docstring.

2. **Outer wrappers are allowed** (`list`, `len`, `type`, `sorted`, etc.), as
   long as no argument to the class method is itself a function call:
   ```python
   >>> type(all_extensions(G))          # OK — no nested call in args
   >>> len(list(all_k_extensions(G, 0)))  # OK — linear args
   ```

:::{toctree}
:maxdepth: 1

generic_rigidity
global_rigidity
matroidal_rigidity
redundant_rigidity
sparsity
pebble_digraph
extensions
constructions
export
general
apex
separating_set
nac
:::
