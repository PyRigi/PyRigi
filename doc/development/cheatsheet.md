## Cheatsheet

### Cross-references


###### Cross-references to definitions, theorems or literature

:::::{tab-set}

::::{tab-item} MyST (`.md` files)
:sync: myst

:::{csv-table}
`` {prf:ref}`def-framework` ``, {prf:ref}`def-framework`
`` {prf:ref}`Framework <def-framework>` ``, {prf:ref}`Framework <def-framework>`
`` {cite:p}`Geiringer1927` ``, {cite:p}`Geiringer1927` 
:::
::::

::::{tab-item} reST  (docstrings)
:sync: rest

:::{csv-table}
`` :prf:ref:`def-framework` `` , {prf:ref}`def-framework`
`` :prf:ref:`Framework <def-framework>` `` , {prf:ref}`Framework <def-framework>`
`` :cite:p:`Geiringer1927` `` , {cite:p}`Geiringer1927`
:::
::::

:::::



###### Cross-references to classes or methods


:::::{tab-set}

::::{tab-item} MyST (`.md` files)
:sync: myst

:::{csv-table}
`` {class}`~pyrigi.framework.Framework` `` , {class}`~pyrigi.framework.Framework`
`` {class}`pyrigi.framework.Framework` `` , {class}`pyrigi.framework.Framework`
`` {meth}`pyrigi.framework.Framework.delete_edge` `` , {meth}`pyrigi.framework.Framework.delete_edge`
`` {meth}`~pyrigi.framework.Framework.delete_edge` `` , {meth}`~pyrigi.framework.Framework.delete_edge`
`` {meth}`~.Framework.delete_edge` `` , {meth}`~.Framework.delete_edge`
`` {meth}`.Framework.delete_edge` `` , {meth}`.Framework.delete_edge`
:::
::::

::::{tab-item} reST  (docstrings)
:sync: rest

:::{csv-table}
`` :class:`~pyrigi.framework.Framework` `` , {class}`~pyrigi.framework.Framework`
`` :class:`pyrigi.framework.Framework` `` , {class}`pyrigi.framework.Framework`
`` :meth:`pyrigi.framework.Framework.delete_edge` `` , {meth}`pyrigi.framework.Framework.delete_edge`
`` :meth:`~pyrigi.framework.Framework.delete_edge` `` , {meth}`~pyrigi.framework.Framework.delete_edge`
`` :meth:`~.Framework.delete_edge` `` , {meth}`~.Framework.delete_edge`
`` :meth:`.Framework.delete_edge` `` , {meth}`.Framework.delete_edge`
:::
::::

:::::


###### Sample definition

````myst
:::{prf:definition} Sample definition
:label: def-sample

Here one can introduce a new _concept_.
Inline math can be used: $\omega\colon S_0 \rightarrow \RR^{d-1}$, and also display:
\begin{equation*}
 \omega\colon S_0 \rightarrow \RR^{d-1}\,.
\end{equation*} 

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
Inline math can be used: $\omega\colon S_0 \rightarrow \RR^{d-1}$, and also display:
\begin{equation*}
 \omega\colon S_0 \rightarrow \RR^{d-1}\,.
\end{equation*} 

{{pyrigi_crossref}} {class}`~pyrigi.framework.Framework`
{meth}`~.Framework.underlying_graph`
{meth}`~.Framework.get_realization`
% list of related objects, methods,..., no separating commas

{{references}} {cite:p}`Lee2008`
% list of related references, no separating commas
:::

###### Math

See above in the definition example or [MyST documentation](https://myst-parser.readthedocs.io/en/latest/syntax/math.html) for more details.
In the definition environment , `$$ ... $$` does not work so
````latex
\begin{equation*}
 ...
\end{equation*}
````
must used (or an alternative). 