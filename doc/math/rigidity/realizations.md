# Realization Counting

## Complex Space

:::{prf:definition} Complex Realization
:label: def-complex-realization

Let $G=(V,E)$ be a simple graph and $d\in\NN$.
A $d$-dimensional _complex realization_ of $G$ is a map $p\colon V\rightarrow \CC^d$.
We denote $p(v)$ also by $p_v$.
:::

:::{prf:definition} Congruent Complex Realizations
:label: def-complex-congruent

Two complex realizations $p,q\in(\CC^d)^V$ are _congruent_ if $p_v = A q_v + b$ for all $v\in V$, where $A$ is a $d\times d$ matrix over $\CC$ with $AA^T=A^TA=I$ and $b\in\CC^d$.
:::

:::{prf:definition} Complex Rigidity Map
:label: def-complex-rigidity-map

Let $G=(V,E)$ be a rigid graph.
We define $f_{G,d}\colon (\CC^d)^V \rightarrow \CC^E$ by $p\mapsto \left(\frac{1}{2}||p_v-p_w||\right)_{vw\in E}$ to be the _complex rigidity map_.
:::

:::{prf:definition} Realization Space
:label: def-complex-realizations-space

Let $G=(V,E)$ be a rigid graph and $p$ a complex realization, $p\in(\CC^d)^V$.
We define $C_d(G,p):=f^{-1}_{G,d}(f_{G,d}(p))/_\sim$ to be the _complex realization space_, where $\sim$ denotes the congruence of complex realizations.
:::

:::{prf:definition} Number of Realizations
:label: def-number-of-realizations

Let $G=(V,E)$ be a $d$-rigid graph.
We define the _number of complex realizations_ by $c_d(G)$, where
\begin{equation*}
    c_d(G):=
    \begin{cases}
        |C_{G,d}(p)| \text{ for some generic } p\in(\CC^d)^V, & \text{if } |V|\geq d+1,\\
        1, & \text{otherwise, i.e. if $G$ is a complete graph with $|V|\leq d$.}
    \end{cases}
\end{equation*}

{{pyrigi_crossref}} {meth}`~.Graph.number_of_realizations()`
:::

The implemented combinatorial algorithm for computing $2\cdot c_2(G)$ for minimally $2$-rigid graphs can be found in {cite:p}`CapcoGalletGraseggerEtAl2018`.
Note that this algorithm does count reflections to be different realizations, while here we do not.

## Complex Sphere

:::{prf:definition} Complex Spherical Realization
:label: def-complex-spherical-realization

Let $G=(V,E)$ be a simple graph and $d\in\NN$.
A $d$-dimensional _complex spherical realization_ of $G$ is a map $p\colon V\rightarrow \mathbb{S}_{\CC}^d := \{x\in\CC^{d+1}\colon ||x||^2 = 1\}$.
:::

:::{prf:definition} Congruent Complex Spherical Realizations
:label: def-complex-spherical-congruent

Two complex spherical realizations $p,q\in(\mathbb{S}_{\CC}^d)^V$ are _congruent_ if $p_v = A q_v$ for all $v\in V$, where $A$ is a $d\times d$ matrix over $\CC$ with $AA^T=A^TA=I$.
:::

:::{prf:definition} Complex Spherical Rigidity Map
:label: def-complex-spherical-rigidity-map

Let $G=(V,E)$ be a rigid graph.
We define $s_{G,d}\colon (\mathbb{S}_{\CC}^d)^V \rightarrow \CC^E$ by $p\mapsto \left(\frac{1}{2}||p_v-p_w||\right)_{vw\in E}$ to be the _complex spherical rigidity map_.
:::

:::{prf:definition} Spherical Realization Space
:label: def-complex-spherical-realizations-space

Let $G=(V,E)$ be a rigid graph and $p$ a complex spherical realization, $p\in(\mathbb{S}_{\CC}^d)^V$.
We define $C_d^{\circ}(G,p):=s^{-1}_{G,d}(s_{G,d}(p))/_\sim$ to be the _complex spherical realization space_, where $\sim$ denotes the congruence of complex spherical realizations.
:::

:::{prf:definition} Number of Spherical Realizations
:label: def-number-of-spherical-realizations

Let $G=(V,E)$ be a $d$-rigid graph.
We define the _number of complex spherical realizations_ by $c_d^{\circ}(G)$, where
\begin{equation*}
    c_d^{\circ}(G):=
    \begin{cases}
        |C^{\circ}_{G,d}(p)| \text{ for some generic } p\in(\mathbb{S}_{\CC}^d)^V, & \text{if } |V|\geq d+1,\\
        1, & \text{otherwise, i.e. if $G$ is a complete graph with $|V|\leq d$.}
    \end{cases}
\end{equation*}

{{pyrigi_crossref}} {meth}`~.Graph.number_of_realizations()`
:::

The implemented combinatorial algorithm for computing $2\cdot c_2^{\circ}(G)$ for minimally $2$-rigid graphs can be found in {cite:p}`GalletGraseggerSchicho2020`.
Note that this algorithm does count reflections to be different realizations, while here we do not.
