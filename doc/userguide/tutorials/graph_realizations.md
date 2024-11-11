---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Graph Realizations (3D printing)

+++

This notebook can be downloaded {download}`here <../../notebooks/graph_realizations.ipynb>`.

```{code-cell} ipython3
# The import will work if the package was installed using pip.
from pyrigi import Graph, Framework
```

Method {meth}`.Framework.generate_onshape_parameters_for_3d_print` allows to generate basic input that can be used 
to create a 3D models of a framework in online CAD system Onshape. Onshape is a cloud-based CAD system that can be
used for free of charge for everyone. Academic licenses are available for students and educators and offer 
addition features for no cost.

The pre-prepared model in OnShape can generate bars of a framework that can be easily exported to STL and 3D printed.

We start with a definition of Graph and its Framework.


```{code-cell} ipython3
G = Graph([(0,1), (1,2), (2,3), (0,3)])
F = Framework(G, {0:[0,0], 1:[1,0], 2:[1,'1/2 * sqrt(5)'], 3:[1/2,'4/3']})
F.plot()
```

If a method {meth}`.Framework.generate_onshape_parameters_for_3d_print` is called, it will return a URL to the Onshape
model and a list of lengths that can be used to create bars of the framework. 

```{code-cell} ipython3
url, l = F.generate_onshape_parameters_for_3d_print()
print(url)
print(l)
```

Often, we wish to scale the framework or to adjust the rounding of the values. This can be done 
by following arguments:

```{code-cell} ipython3
url, l = F.generate_onshape_parameters_for_3d_print(scale=20, roundings=2)
print(url)
print(l)
```

These values in milimeters may be used as input for the OnShape model. First, sign in with our account and opend the model
using the URL. Then, copy the model to your workspace.

