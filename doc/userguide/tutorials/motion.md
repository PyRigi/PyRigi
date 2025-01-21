---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Continuous Motions

It is possible to create {prf:ref}`continuous motions <def-motion>` of {prf:ref}`frameworks <def-realization>` in PyRigi.

## Parametric Motion

The user can specify a parametric motion using the class {class}`~.ParametricMotion`. 
As an example, consider the 4-cycle. A parametric motion can be specified
using the following sequence of commands:

```{code-cell} ipython3
from pyrigi import graphDB as graphs
from pyrigi.motion import ParametricMotion
import sympy as sp
motion = ParametricMotion(
    graphs.Cycle(4),
    {
        0: ("0", "0"),
        1: ("1", "0"),
        2: ("4 * (t**2 - 2) / (t**2 + 4)", "12 * t / (t**2 + 4)"),
        3: (
            "(t**4 - 13 * t**2 + 4) / (t**4 + 5 * t**2 + 4)",
            "6 * (t**3 - 2 * t) / (t**4 + 5 * t**2 + 4)",
        ),
    },
    [-sp.oo, sp.oo],
)
motion.animate()
```

It is also possible to provide trivial motions in a similar manner.

```{code-cell} ipython3
motion = ParametricMotion(
    graphs.Complete(5),
    {
        0: ("cos(2*pi/5 + t)", "sin(2*pi/5 + t)"),
        1: ("cos(4*pi/5 + t)", "sin(4*pi/5 + t)"),
        2: ("cos(6*pi/5 + t)", "sin(6*pi/5 + t)"),
        3: ("cos(8*pi/5 + t)", "sin(8*pi/5 + t)"),
        4: ("cos(t)", "sin(t)")
    },
    [0, sp.sympify("2*pi")],
)
motion.animate()
```

Internal checks on the edge lengths are in place to ensure that the specified parametric motion
never violates the edge-length equations. 

## Approximate Motion

However, a parametric motion is not always available. If you still want to get an
intuition for how a deformation path looks, it can be numerically approximated using
the class {class}`~.ApproximateMotion`. As an example, consider the complete bipartite graph $K_{4,2}$.
A cyclic motion of $K_{4,2}$ can be approximated using the following code:

```{code-cell} ipython3
from pyrigi.motion import ApproximateMotion
from pyrigi import frameworkDB as frameworks
F = frameworks.CompleteBipartite(2,4)
motion = ApproximateMotion.from_framework(F, 393, chosen_flex=0, step_size=0.15)
motion.animate(duration=10)
```

Currently, only nontrivial motions can be computed in this way.