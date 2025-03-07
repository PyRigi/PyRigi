# Second-Order Theory

:::{prf:definition} Prestress Stability
:label: def-prestress-stability

Let $G=(V,E)$ be a graph and let $F$ be a 
$d$-dimensional {prf:ref}`framework <def-framework>` on $G$. The framework $F$ is called 
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

Let $G=(V,E)$ be a graph and let $F$ be a $d$-dimensional 
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

:::{prf:theorem} Implications of the different types of rigidity
:label: thm-second-order-implies-infinitesimal

{prf:ref}`Infinitesimal rigidity<def-inf-rigid-framework>` in $\RR^d$ implies 
{prf:ref}`prestress stability<def-prestress-stability>` in $\RR^d$ which implies 
{prf:ref}`second-order rigidity<def-second-order-rigid>` in $\RR^d$ which implies 
{prf:ref}`continuous rigidity<def-cont-rigid-framework>` in $\RR^d$. 
None of these implications are reversible.

{{references}} {cite:p}`ConnellyGortler2017{Thm 2.6}`
:::

:::{prf:theorem} Equivalent criterion for second-order rigidity 
:label: thm-second-order-rigid

A framework $F=(G,p)$ is second-order rigid in $\RR^d$ if and only if for every 
{prf:ref}`nontrivial infinitesimal flex <def-trivial-inf-flex>` $q$ of $F$ there
 is an {prf:ref}`equilibrium stress <def-equilibrium-stress>` $\omega$ such that 
\begin{equation*}
\sum_{ij \in E} \omega_{ij}\cdot ||q_i-q_j||^2\,>\,0.
\end{equation*} 

{{references}} {cite:p}`ConnellyGortler2017{Thm 2.5}`
:::

While this theorem shows that second-order rigidity and prestress stability are distinguishable only
by a swap of quantifiers, they are indeed separate concepts, as the following example highlights.

:::{prf:example} Second-order rigid framework that is not prestress stable
:label: exmp-second-order-not-prestress-stable

Let $G = (V, E)$ be the graph where $V = \{1, \dots, 8\}$ and
\begin{equation*}
E=\{\{1, 2\},\, \{1, 3\},\, \{1,4\},\, \{1, 6\},\, \{1, 7\},\,
\{2,3\},\, \{2,5\},\, \{2, 7\},\, \{3, 4\},\, \{3, 5\},\,
\{4,5\},\, \{4,6\},\,
\{5,6\},\, \{5,7\},\,
\{6, 7\}\}.
\end{equation*}
We describe a {prf:ref}`framework <def-framework>` on $G$ by the {prf:ref}`realization <def-framework>` $p:V\rightarrow \RR^3$ defined by
\begin{equation*}
p(1)=(0,0,0),~p(2)=p(6)=(0,1,0),~p(3)=(0,0,1),p(4)=p(7)=(1,0,0),~p(5)=(1,1,0).
\end{equation*}
This framework has a two-dimensional set of {prf:ref}`nontrivial infinitesimal flexes <def-trivial-inf-flex>`
and a two-dimensional set of {prf:ref}`equilibrium stresses <def-equilibrium-stress>`.
It is {prf:ref}`second-order rigid <def-second-order-rigid>`,
but not {prf:ref}`prestress stable <def-prestress-stability>`.
{{references}} {cite:p}`ConnellyWhiteley1996{Exm 5.3.1}`

{{pyrigi_crossref}} {func}`~.frameworkDB.ConnellyExampleSecondOrderRigidity`
:::

Checking prestress stability and second-order rigidity is generally computationally hard. In the case where
there is a single stress or infinitesimal motion, the problem becomes easier:

If there is only one {prf:ref}`nontrivial infinitesimal flex <def-trivial-inf-flex>` $q$, we check for a basis
$(\omega^{(k)})_{k=1}^r$ of the stress space such that the stress energy
$\sum_{ij \in E} \omega^{(k)}_{ij} \cdot ||q(i)-q(j)||^2$
is always non-zero by verifying that not all of these energies
become simultaneously 0 for all $k=1,\dots,m$.

If there is only one {prf:ref}`equilibrium stress <def-equilibrium-stress>`, denote a basis of the infinitesimal flex
space by $(q^{(k)})_{k=1}^s$. Next, we consider the coefficients
of the monomials $({a_i}\cdot{a_j} \,:\, i,j=1,\dots,s)$ in the quadratic polynomial
\begin{equation*}
E_{q,\omega}(a)=\sum_{ij \in E} \omega_{ij} \cdot ||\sum_{k=1}^s a_k \cdot (q^{(k)}(i)-q^{(k)}(j))||^2.
\end{equation*}
A simple result about sums of nonnegative circuits (cf. {cite:p}`IlimandeWolff2016{Thm 3.8}`)
then lets us characterize the positivity of the stress energy. 
The SONC Criterion says that all faces on the boundary of the
Newton polytope need to satisfy the SONC property. It simplifies to
$|c_{ij}| < \sqrt{4 \cdot c_{ii}\cdot  c_{jj}}$ for all $i,j$ for coefficients $c_{ij}$
of the monomials ${a_i}\cdot{a_j}$. In addition, $c_{ii}$ and $c_{jj}$ need
to have the same sign or be zero.

The following theorem then shows that in these two cases, {prf:ref}`prestress stability <def-prestress-stability>`
and {prf:ref}`second-order-rigidity <def-second-order-rigid>` are equivalent.

:::{prf:theorem} Second-order rigidity and prestress stability in the case of one-dimensional stresses or flexes
:label: thm-second-order-implies-prestress-stability

Let $F$ denote a $d$-dimensional {prf:ref}`second-order rigid <def-second-order-rigid>` 
{prf:ref}`framework <def-framework>` with a one-dimensional space of 
{prf:ref}`nontrivial infinitesimal flexes <def-trivial-inf-flex>` or a one-dimensional space 
of {prf:ref}`equilibrium stresses <def-equilibrium-stress>`. Then, $F$ is 
{prf:ref}`prestress stable <def-prestress-stability>`.

{{references}} {cite:p}`ConnellyWhiteley1996`
:::

In the general case, we use the {prf:ref}`stress matrix <def-stress-matrix>` criterion from
{cite:p}`ConnellyWhiteley1996{Prop 3.4.2}`.

:::{prf:theorem} Stress Matrix criterion for prestress stability
:label: thm-stress-matrix-criterion-prestress-stability

A {prf:ref}`framework <def-framework>` is prestress stable for the {prf:ref}`equilibrium stress <def-equilibrium-stress>`
$\omega$ if and only if the associated {prf:ref}`stress matrix <def-stress-matrix>` $\Omega(\omega)$ is positive definite on any subspace of the
{prf:ref}`nontrivial infinitesimal flexes <def-trivial-inf-flex>`.

{{references}} {cite:p}`ConnellyWhiteley1996`
:::


In the method {meth}`~.Framework.is_prestress_stable`,
we examine the contraposition: If this stress matrix is globally negative
definite or at least nonpositive, then the framework cannot be prestress stable.

For showing the second-order rigidity of a framework, we need
to solve a semi-definite program (SDP). 


Given a {prf:ref}`framework <def-framework>` $F$, parametrize the space
of {prf:ref}`infinitesimal flexes <def-inf-flex>` by variables $(a_{i})_{i=1}^s$
and the space of {prf:ref}`stresses <def-equilibrium-stress>` by variables
$(b_{j})_{j=1}^r$. This turns the stress energy into a cubic polynomial that is homogeneous
and quadratic in $a_i$ and homogeneously linear in $b_j$:
\begin{equation*}
E_{q,\omega}(a,b)=\sum_{k=1}^s \sum_{ij \in E} b_k \cdot \omega^{(k)}_{ij} \cdot ||\sum_{\ell=1}^r a_\ell \cdot ( q^{(\ell)}(i)-q^{(\ell)}(j) )||^2
\end{equation*}

:::{prf:corollary} Polynomial criterion for second-order rigidity
:label: thm-polynomial-criterion-second-order

The framework $F$ is second-order rigid if and only if the polynomial system
in the variables $a_i$ arising from the coefficients of $E_{q,\omega}$ in
the linear monomials $b_j$ has only non-real nontrivial solutions.

{{references}} {prf:ref}`equivalent second-order rigidity criterion <thm-second-order-rigid>`
:::

Otherwise, there would be an infinitesimal
flex such that for any equilibrium stress it holds that the stress energy
is zero. 