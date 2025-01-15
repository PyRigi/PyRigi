# Known Families of Graphs

Here we introduce several known graphs that can be accessed in the PyRigi graph database.

:::{prf:definition} $n$-Frustum
:label: def-n-frustum

Assume that $n\geq 3$. The graph $G=(V,E)$ is called the 
_$n$-Frustum_ if it is the Cartesian product
 $G=C_n\,\square \, K_2$ of a cycle graph $C_n$ on $n$ vertices 
 and the complete graph $K_2$ on two vertices.

As a {prf:ref}`framework <def-framework>`, the $n$-Frustum is 
typically realized as two regular $n$-sided polygons on circles 
centered in the origin of the Euclidean plane with radii $r_1<r_2$. 
It has a {prf:ref}`nontrivial infinitesimal flex <def-trivial-inf-flex>` 
given by the rotation of the outer polygon while the inner polygon remains 
fixed. This infinitesimal flex does not extend to a 
{prf:ref}`continuous flex <def-motion>`.

{{pyrigi_crossref}} {func}`.graphDB.Frustum`
{func}`.frameworkDB.Frustum`
:::


:::{prf:definition} Counterexample for the symmetry-adjusted Laman count with a free group action
:label: def-Cn-symmetric

Let $n\geq8$ be even, and let $C_n$ denote the anti-clockwise 
rotation around the origin by $2\pi/n$. Define a 4-regular 
graph $H_n=(V_n,E_n)$ on the vertex set $V_n=\{v_1,\dots,v_n\}$ 
such that each vertex $v_i$ of $H_n$ is exactly adjacent to 
$v_{i+1}$, $v_{i-1}$, $v_{i+3}$ and $v_{i-3}$ with $v_{n+k}=v_k$ 
and $v_{1-k}=v_{n+1-k}$ for $k\in \{1,2,3\}$.

Define a {prf:ref}`framework <def-framework>` $(H_n,p)$ on 
$H_n$ realized on the unit circle in $\mathbb{R}^2$ such that 
$p(v_1)=C_np(v_n)$ and $p(v_i)=C_np(v_{i-1})$ for all 
$2\leq i\leq n$. For some vector $t$ on the line from the 
origin to a vertex from $\{v_1,\dots,v_n\}$, such a framework 
has a 
{prf:ref}`nontrivial infinitesimal motion <def-trivial-inf-flex>` 
$m$ which satisfies the system of equations
\begin{equation*}
    m(v_i)=\begin{cases}
        t & \text{if } i \text{ is even}\\
        -t & \text{if } i \text{ is odd},
    \end{cases}
\end{equation*}
where $1\leq i\leq n$.

{{pyrigi_crossref}} {func}`~.graphDB.CnSymmetricFourRegular`
{func}`~.frameworkDB.CnSymmetricFourRegular`

{{references}} {cite:p}`LaPortaSchulze2024`
:::

:::{prf:definition} Counterexample for the symmetry-adjusted Laman count which contains a joint at the origin
:label: def-Cn-symmetric-joint-at-origin

The previous example can be extended in the following way: 
add a vertex $u$ and the vertices $U_n=\{u_1,\dots,u_n\}$ 
to $H_n$ from the {prf:ref}`previous definition <def-Cn-symmetric>`. 
We also add the following edges to $H_n$:
1. For all $1\leq i\leq n$, add the edge $\{u,u_i\}$.
2. For all $1\leq i\leq n$, add the edge $\{u_i,v_i\}$.
3. For all $3\leq i\leq n$, add the edge $\{u_i,u_{i-2}\}$. 
Also add the edges $\{u_1,u_{n-1}\}$ and $\{u_2,u_n\}$.

Denote the edge set created in this way by $F_n$. This creates 
the graph $G_n=(V_n\cup\{u\}\cup U_n,~E_n \cup F_n)$.

As a framework, this graph can be realized extending the 
realization $p:V_n\rightarrow \mathbb{R}^2$ from before to 
$G_n$: $p(u)$ is placed at the origin, and the other $n$ vertices 
$\{p(u_1),\dots,p(u_n)\}$ are placed on a circle with radius $r>0$ 
so that that $p(u_1)=C_np(u_n)$ and $p(u_i)=C_np(u_{i-1})$ for 
all $2\leq i\leq n$. In this way, the new edges given in (iii) 
form two disjoint regular $n/2$-gons. The 
{prf:ref}`nontrivial infinitesimal motion <def-trivial-inf-flex>` 
from the previous example extends to a nontrivial infinitesimal 
motion of the new framework $(G_n,p)$ which rotates the two 
$n/2$-gons clockwise and anti-clockwise, respectively.

{{pyrigi_crossref}} {func}`~.graphDB.CnSymmetricFourRegularWithFixedVertex`
{func}`~.frameworkDB.CnSymmetricFourRegularWithFixedVertex`

{{references}} {cite:p}`LaPortaSchulze2024`
:::
