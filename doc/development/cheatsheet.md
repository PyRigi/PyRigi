## Cheatsheet

### Cross-references

###### Cross-references to definitions, theorems or literature

Code                 | Output             
-------------------- | -------------------
`` {prf:ref}`def-framework` `` | {prf:ref}`def-framework`
`` {prf:ref}`Framework <def-framework>` `` | {prf:ref}`Framework <def-framework>`
`` {cite:p}`Geiringer1927` `` |  {cite:p}`Geiringer1927` 

###### Cross-references to classes or methods

Code                 | Output             
-------------------- | -------------------
`` {class}`~pyrigi.framework.Framework` `` | {class}`~pyrigi.framework.Framework`
`` {class}`pyrigi.framework.Framework` `` | {class}`pyrigi.framework.Framework`
`` {meth}`pyrigi.framework.Framework.delete_edge` `` | {meth}`pyrigi.framework.Framework.delete_edge`
`` {meth}`~pyrigi.framework.Framework.delete_edge` `` | {meth}`~pyrigi.framework.Framework.delete_edge`
`` {meth}`~.Framework.delete_edge` `` | {meth}`~.Framework.delete_edge`
`` {meth}`.Framework.delete_edge` `` | {meth}`.Framework.delete_edge`


###### Sample definition

````myst
:::{prf:definition} Sample definition
:label: def-sample

Here one can introduce a new _concept_.
The definition can be linked using its label: {prf:ref}`def-sample`.

{{pyrigi_crossref}} {class}`~pyrigi.framework.Framework`
{meth}`~.Framework.underlying_graph`
{meth}`~.Framework.get_realization`
% list of related objects, methods,..., no separating commas

{{references}} {cite:p}`Lee2008`
% list of related references, no separating commas
:::
````

:::{prf:definition} Sample definition
:label: def-sample

Here one can introduce a new _concept_.
The definition can be linked using its label: {prf:ref}`def-sample`.

{{pyrigi_crossref}} {class}`~pyrigi.framework.Framework`
{meth}`~.Framework.underlying_graph`
{meth}`~.Framework.get_realization`
% list of related objects, methods,..., no separating commas

{{references}} {cite:p}`Lee2008`
% list of related references, no separating commas
:::
