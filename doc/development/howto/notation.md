(notation)=
# Notation Standards

The following table provides a standard for the notation in the mathematical documentation
and in the code. 

|Object|Math|Code|
|:---:|:---:|:---:|
|Graph|$G,\, H,\, G_1,\, H_1$| `G`, `G1`, `H`, `H1`,  or `graph`, `graph_1`, `subgraph`|
|Framework|$F$|`F`, `F1`, `framework`|
|dimension|$d$|`dim`|
|Realization|$p$|`realization` or ``points`` depending on `dict`/`list`|
|Point|$(x_1,\dots,x_d)\in \RR^d$|``pos`` or ``point``|
|Vertex Set|$V$|`vertices`|
|Edge Set|$E$|`edges`|
|Vertex|$u,\, v, \, w,\, u_1,\, v_1,\, w_1$|`vertex`, `u`, `v`,`w`,`u1`, `v1`,`w1`|
|Edge|$e_1$ and $e_2$|`edge`, `e`, `e1`|
|#V|$n$  or $\|V\|$| `n`|
|#E|$m$ or $\|E\|$|`m`|
|Rigidity Matrix|$R(G,p)$|`rigidity_matrix`|
|Infinitesimal flex|$q$|`inf_flex`|
|Equilibrium stress|$\omega$|`stress`|
|Stress Matrix|$\Omega$|`stress_matrix`|
|Motion|$\alpha:[0,1]\rightarrow (\RR^d)^n$|`motion`|
|$d$-rigidity matroid|$\mathcal{R}_d$| N/A|
|Symbolic `bool`| N/A | `numerical`|
|Tolerance `float`| N/A | `tolerance`|

Iterator Variables are supposed to be denoted in the following order (with descending callback depth):
  1. `i`
  2. `j`
  3. `k` (if `k` is not defined differently in the method)
  4. ``L``
  5. ``ii``...

If single-letter variables are part of the input parameters, that should be reflected in the method name.
Examples of this are {meth}`.Graph.is_kl_sparse` or {meth}`.Graph.is_k_redundantly_rigid`.

Otherwise, single-letter variables should be avoided whenever possible and descriptive variable names should be used instead.
According to PEP8, the letters ``l`` (lower-case ell), ``O`` (upper-case oh) and ``I`` (upper-case eye)
should generally be avoided due to their similarity to other characters. 