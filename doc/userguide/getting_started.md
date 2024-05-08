# Getting started

SOME BASIC INFORMATION


## Installation and usage

We have not reached a stable version yet.
Hence, the current usage is to clone/download the package
from [this GitHub repository](https://github.com/pyRigi/PyRigi).
In the root folder of the package, it can be used by
```python
from pyrigi import Graph, Framework
```
If you are not working in the root folder of the package you may use
```python
import os
import sys
sys.path.insert(0,os.path.abspath("<path_to_pyrigi>"))
from pyrigi import Graph, Framework
```
where `<path_to_pyrigi>` is replaced by the path to the root folder of the package.

In the future, we will enable installation via `pip`.
