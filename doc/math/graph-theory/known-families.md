# Known Families of Graphs

Here we introduce several known graphs that can be accessed in the PyRigi graph database.

:::{prf:definition} $n$-Frustum
:label: def-n-frustum

Assume that $n\geq 3$. The graph $G=(V,E)$ is called the _$n$-Frustum_ if it is the Cartesian product $G=C_n\,\square \, K_2$ of a cycle graph $C_n$ on $n$ vertices and the complete graph $K_2$ on two vertices.

As a {prf:ref}`framework <def-framework>`, the $n$-Frustum is typically realized as two regular $n$-sided polygons on circles centered in the origin with radii $r_1<r_2$. It has a {prf:ref}`nontrivial infinitessimal flex <def-trivial-inf-flex>` given by the rotation of the outer polygon while the inner polygon ramains fixed. This motion does not extend to a {prf:ref}`continuous flex <def-flex>`.

{{pyrigi_crossref}} {func}`.graphDB.Frustum`
{func}`.frameworkDB.Frustum`
:::


:::{prf:definition} Counterexample for the symmetry-adjusted Laman count with a free group action
:label: def-Cn-symmetric

Let $n\geq8$ be even, and let $C_n$ denote the anti-clockwise rotation around the origin by $2\pi/n$. 
Define a {prf:ref}`framework <def-framework>` $(G,p)$ on $n$ joints $\{p(v_1),\dots,p(v_n)\}$ such that $p(v_1)=C_np(v_n)$ and $p(v_i)=C_np(v_{i-1})$ for all $2\leq i\leq n$. In addition, each joint $p(v)$ of $(G,p)$, $p(v)$ is adjacent exactly to the following joints: $C_np(v)$, $C_n^{-1}p(v)$, $C_n^3p(v)$ and $C_n^{-3}p(v)$. For some vector $t$ on the line from the origin to a joint in $\{v_1,\dots,v_n\}$, such a framework has an {prf:ref}`infinitesimal motion <def-inf-flex>` $m$ which satisfies the system of equations
\begin{equation*}
    m(v_i)=\begin{cases}
        t & \text{if } i \text{ is even}\\
        -t & \text{if } i \text{ is odd},
    \end{cases}
\end{equation*}
where $1\leq i\leq n$.

{{pyrigi_crossref}} {func}`~.graphDB.CnSymmetricFourRegular`
{func}`~.frameworkDB.CnSymmetricFourRegular`

{{references}} {cite:p}`LaPorta2024`
:::

:::{prf:definition} Counterexample for the symmetry-adjusted Laman count which contains a joint at the origin
:label: def-Cn-symmetric-joint-at-origin

The previous example can be extended: add a joint $p(u)$ at the origin, and add other $n$ joints $\{p(u_1),\dots,p(u_n)\}$ such that $p(u_1)=C_np(u_n)$ and $p(u_i)=C_np(u_{i-1})$ for all $2\leq i\leq n$. Add the following edges to $G$:
1. For all $1\leq i\leq n$, add the edge $\{u,u_i\}$.
2. For all $1\leq i\leq n$, add the edge $\{u_i,v_i\}$.
3. For all $3\leq i\leq n$, add the edge $\{u_i,u_{i-2}\}$. Also add the edges $\{u_1,u_{n-1}\}$ and $\{u_2,u_n\}$.

The new edges given in (iii) form two disjoint regular $n/2$-gons. The {prf:ref}`infinitesimal motion <def-inf-flex>` from the previous example extends to an infinitesimal motion of the new framework which rotates the two $n/2$-gons clockwise and anti-clockwise, respectively.

{{pyrigi_crossref}} {func}`~.graphDB.CnSymmetricFourRegularWithFixedVertex`
{func}`~.frameworkDB.CnSymmetricFourRegularWithFixedVertex`

{{references}} {cite:p}`LaPorta2024`
:::