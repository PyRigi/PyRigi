(cheatsheet)=
# Cheatsheet

While `monospace` is done using `` `monospace` `` in MyST,
two ``` `` ``` are needed in reST (docstrings), namely ``` ``variable`` ```.

For detailed overview of MyST syntax, see the [MyST documentation](https://myst-parser.readthedocs.io/en/latest/syntax/typography.html).

## Cross-references

### Cross-references to definitions, theorems or literature

:::::{tab-set}

::::{tab-item} MyST (`.md` files)
:sync: myst

:::{csv-table}
:delim: ;
`` {prf:ref}`def-framework` ``; {prf:ref}`def-framework`
`` {prf:ref}`Framework <def-framework>` ``; {prf:ref}`Framework <def-framework>`
`` {cite:p}`PollaczekGeiringer1927` ``; {cite:p}`PollaczekGeiringer1927`
`` {cite:p}`Laman1970{Thm 5.6}` ``; {cite:p}`Laman1970{Thm 5.6}`
`` {cite:p}`PollaczekGeiringer1927,Laman1970` ``; {cite:p}`PollaczekGeiringer1927,Laman1970` 
:::
::::

::::{tab-item} reST  (docstrings)
:sync: rest

:::{csv-table}
:delim: ;
`` :prf:ref:`def-framework` `` ; {prf:ref}`def-framework`
`` :prf:ref:`Framework <def-framework>` `` ; {prf:ref}`Framework <def-framework>`
`` :cite:p:`PollaczekGeiringer1927` `` ; {cite:p}`PollaczekGeiringer1927`
`` :cite:p:`Laman1970{Thm 5.6}` ``; {cite:p}`Laman1970{Thm 5.6}`
`` :cite:p:`PollaczekGeiringer1927,Laman1970` ``; {cite:p}`PollaczekGeiringer1927,Laman1970` 
:::
::::

:::::



### Cross-references to classes or methods


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
`` {func}`networkx.classes.function.degree` `` , {func}`networkx.classes.function.degree`
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
`` :func:`networkx.classes.function.degree` `` , {func}`networkx.classes.function.degree`
:::
::::

:::::



### Cross-references to pages
For creating a label on a page use ``(label)=`` before a section title.
This label is used together with ``ref`` for internal references, see below.

For references to ``networkx`` we use ``doc`` instead provided by the setting ``intersphinx_mapping``
in ``conf.py``. This is also available for references to ``python`` and ``sympy``.

:::::{tab-set}

::::{tab-item} MyST (`.md` files)
:sync: myst

:::{csv-table}
`` {ref}`installation-guide` ``, {ref}`installation-guide`
`` {ref}`Link to Instalation Guide<installation-guide>` ``, {ref}`Link to Instalation Guide<installation-guide>`
`` {doc}`networkx:reference/drawing` ``, {doc}`networkx:reference/drawing`
:::
::::

::::{tab-item} reST  (docstrings)
:sync: rest

:::{csv-table}
`` :ref:`installation-guide` ``, {ref}`installation-guide`
`` :ref:`Link to Instalation Guide<installation-guide>` ``, {ref}`Link to Instalation Guide<installation-guide>`
`` :doc:`networkx:reference/drawing` ``, {doc}`networkx:reference/drawing`
:::
::::

:::::


### Sample definition

````myst
:::{prf:definition} Sample definition
:label: def-sample

Here one can introduce a new _concept_.
Inline math can be used: $\omega\colon S_0 \rightarrow \RR^{d-1}$, and also display:
\begin{equation*}
 \omega\colon S_0 \rightarrow \RR^{d-1}\,.
\end{equation*} 

{{pyrigi_crossref}} {class}`~pyrigi.framework.Framework`
{meth}`~.Framework.graph`
{meth}`~.Framework.realization`
% list of related objects, methods,..., no separating commas

{{references}} {cite:p}`LeeStreinu2008`
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
{meth}`~.Framework.graph`
{meth}`~.Framework.realization`
% list of related objects, methods,..., no separating commas

{{references}} {cite:p}`LeeStreinu2008`
% list of related references, no separating commas
:::

### Math

See above in the definition example or [MyST documentation](https://myst-parser.readthedocs.io/en/latest/syntax/math.html) for more details.
In the definition environment , `$$ ... $$` does not work so
````latex
\begin{equation*}
 ...
\end{equation*}
````
must used (or an alternative like `align`).

The following `latex` macros can be used
:::{csv-table}
`\RR`, $\RR$ ,   real numbers
`\CC`, $\CC$ ,   complex numbers
`\QQ`, $\QQ$ ,   rational numbers
`\ZZ`, $\ZZ$ ,   integers
`\NN`, $\NN$ ,   natural numbers (including 0)
`\PP`, $\PP$ ,   projective space
`\KK`, $\KK$ ,   a field
:::

New `latex` commands can be created by modifying both `latex_elements` and `mathjax3_config` in `doc/conf.py`.

Docstrings that use `latex` must be raw strings, namely `r""" ... """`.
