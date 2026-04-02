# Realization Counting


Here by $||x||^2$ we denote the extension of the squared Euclidean norm to $\CC^d$ by defining $||x||^2:= \sum_{i=1}^d x_i^2$, where $x_i$ is the $i$-th coordinate of $x$.

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
We define $f_{G,d}\colon (\CC^d)^V \rightarrow \CC^E$ by $p\mapsto \left(\frac{1}{2}||p_v-p_w||^2\right)_{vw\in E}$ to be the _complex rigidity map_.
:::

:::{prf:definition} Realization Space
:label: def-complex-realizations-space

Let $G=(V,E)$ be a $d$-rigid graph and $p$ a complex realization, $p\in(\CC^d)^V$.
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

The implemented combinatorial algorithm for computing $2\cdot c_2(G)$ for minimally $2$-rigid graphs can be found in {cite:p}`CapcoGalletEtAl2018`.
Note that this algorithm does count reflections to be different realizations, while here we do not.
For $2$-rigid graphs that are not minimally $2$-rigid the algorithm from {cite:p}`DewarGraseggerEtAl2025` is used to compute $c_2(G)$ (see also {prf:ref}`thm-realization-rigid-not-3-connected` and {prf:ref}`thm-realization-rigid-3-connected`).

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
We define $s_{G,d}\colon (\mathbb{S}_{\CC}^d)^V \rightarrow \CC^E$ by $p\mapsto \left(\frac{1}{2}||p_v-p_w||^2\right)_{vw\in E}$ to be the _complex spherical rigidity map_.
:::

:::{prf:definition} Spherical Realization Space
:label: def-complex-spherical-realizations-space

Let $G=(V,E)$ be a $d$-rigid graph and $p$ a complex spherical realization, $p\in(\mathbb{S}_{\CC}^d)^V$.
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
For $2$-rigid graphs that are not minimally $2$-rigid the algorithm from {cite:p}`DewarGraseggerEtAl2025` is used to compute $c_2^{\circ}(G)$.

## Theorems


:::{prf:theorem}
:label: thm-sphere-plane-realization

In any positive dimension we have $c_d(G)\leq c_d^{\circ}(G)$.

{{references}} {cite:p}`DewarGrasegger2024{Thm 1.1}`
:::

:::{prf:lemma}
:label: lem-realization-0-extension

Let $G$ be a $d$-{prf:ref}`rigid<def-gen-rigid>` graph and let $G'$ be obtained from $G$ by a $0$-{prf:ref}`extension<def-k-extension>`.
Then $c_d(G')=2c_d(G)$ and $c_d^{\circ}(G')=2c_d^{\circ}(G)$.

{{references}} {cite:p}`DewarGrasegger2024{Lem 7.1}`
:::

:::{prf:theorem}
:label: thm-realization-rigid-not-3-connected

Let $G=(V,E)$ be a 2-{prf:ref}`rigid<def-gen-rigid>` graph that is not 3-connected with vertices $u,v$ separting $G$ into $G_1,G_2$.
Then $G_1+uv$ and $G_2+uv$ are 2-rigid and $G_1$ or $G_2$ is 2-rigid and
\begin{equation*}
    c_2(G):=
    \begin{cases}
        2 c_2(G_1)c_2(G_2+uv), & \text{ if $uv\not\in E$ and $G_1$ is 2-rigid, $G_2$ is not 2-rigid,}\\
        2 c_2(G_1+uv)c_2(G_2+uv), & \text{if $uv\in E$ or both $G_1$ and $G_2$ are 2-rigid.}
    \end{cases}
\end{equation*}


{{references}} {cite:p}`JacksonOwen2019{Thm 6.6}`
:::

:::{prf:theorem}
:label: thm-realization-rigid-3-connected

Let $G=(V,E)$ be a 2-{prf:ref}`rigid<def-gen-rigid>` graph that is 3-{prf:ref}`connected<def-k-connected>` but not {prf:ref}`redundantly<def-redundantly-rigid-graph>` 2-rigid.
Let $e\in E$ such that $G-e$ is not 2-rigid and let $G_1,\ldots,G_m$ be the maximal rigid subgraphs of $G-e$.
Let further be $H_i$ be a minimally rigid subgraph of $G_i$ for each $i\in\{1,\ldots,m\}$
and let $H=H_1\cup\cdots\cup H_m\cup\{e\}$. Then

\begin{equation*}
    c_2(G):= c_2(H) \prod_{i=1}^m \frac{c_2(G_i)}{c_2(H_i)}.
\end{equation*}


{{references}} {cite:p}`DewarGraseggerEtAl2025{Thm 1}`
:::
