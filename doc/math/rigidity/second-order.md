# Second-Order Theory

:::{prf:definition} Prestress Stability
:label: def-prestress-stability

Let $G=(V,E)$ be a graph and let $F=(G,\, p:V\rightarrow \mathbb{R}^d)$ be a 
$d$-dimensional {prf:ref}`framework <def-framework>` on $G$. $F$ is called 
_prestress stable_ if there exists an {prf:ref}`equilibrium stress <def-equilibrium-stress>` 
$\omega$ such that for every {prf:ref}`nontrivial infinitesimal flex <def-trivial-inf-flex>` $q$ 
it holds that
\begin{equation*}
\sum_{ij\in E} \omega_{ij}\cdot ||q_i-q_j||^2\,>\,0.
\end{equation*} 

{{pyrigi_crossref}} {meth}`~.Framework.is_prestress_stable`
:::


:::{prf:definition} Second-order rigidity
:label: def-second-order-rigid

Let $G=(V,E)$ be a graph and let $F=(G,\, p:V\rightarrow \mathbb{R}^d)$ be a $d$-dimensional 
{prf:ref}`framework <def-framework>` on $G$. Let $q$ denote an 
{prf:ref}`infinitesimal flex <def-inf-flex>` of $F$. A _second-order flex_ $p''$ 
of $F$ corresponding to $q$ is defined by the equation
\begin{equation*}
R(G, p)\cdot p'' + R(G, q)\cdot q = 0
\end{equation*} 
for the {prf:ref}`rigidity matrix <def-rigidity-matrix>` $R(G, p)$ of $F$. 
If there is no second-order flex $p''$ for a 
{prf:ref}`nontrivial infinitesimal flex <def-trivial-inf-flex>` $q$, we 
call $F$ _second-order rigid_.

{{pyrigi_crossref}} {meth}`~.Framework.is_second_order_rigid`
:::


:::{prf:theorem} Equivalent criterion for second-order rigidity 
:label: thm-second-order-rigid

A framework $F=(G,p)$ is second-order rigid in $\RR^d$ if and only if for every 
{prf:ref}`nontrivial infinitesimal flex <def-trivial-inf-flex>` $q$ of $F$ there
 is an {prf:ref}`equilibrium stress <def-equilibrium-stress>` $\omega$ such that 
\begin{equation*}
\sum_{ij\in E} \omega_{ij}\cdot ||q_i-q_j||^2\,>\,0.
\end{equation*} 

{{references}} {cite:p}`Connelly2017{Thm 2.5}`
:::


:::{prf:theorem} Implications of the different types of rigidity
:label: thm-second-order-implies-infinitesimal

{prf:ref}`Infinitesimal rigidity<def-inf-rigid-framework>` in $\RR^d$ implies 
{prf:ref}`prestress stability<def-prestress-stability>` in $\RR^d$ which implies 
{prf:ref}`second-order rigidity<def-second-order-rigid>` in $\RR^d$ which implies 
{prf:ref}`continuous rigidity<def-cont-rigid-framework>` in $\RR^d$. 
None of these implications are reversible.

{{references}} {cite:p}`Connelly2017{Thm 2.6}`
:::


:::{prf:theorem} Second-order rigidity and prestress stability in the case of one-dimensional stresses or flexes
:label: thm-second-order-implies-prestress-stability

Let $F=(G,p)$ denote a $d$-dimensional {prf:ref}`second-order rigid <def-second-order-rigid>` 
{prf:ref}`framework <def-framework>` with a one-dimensional space of 
{prf:ref}`nontrivial infinitesimal flexes <def-trivial-inf-flex>` or a one-dimensional space 
of {prf:ref}`equilibrium stresses <def-equilibrium-stress>`. Then $F$ is 
{prf:ref}`prestress stable <def-prestress-stability>`.

{{references}} {cite:p}`Connelly1996`
:::