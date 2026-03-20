# SQLite Database

We assume that each graph in the database has at least two vertices.

## Stored values

Here we describe how values in the columns of the database are created.

| Column name           | Property                                       |
|-----------------------|------------------------------------------------|
| `graph`               | Graph encoded in `sparse6`???                  |
| `num_vertices`        | The number of vertices                         |
| `num_edges`           | The number of edges                            |
| `min_degree`          | The minimum degree                             |
| `max_degree`          | The maximum degree                             |
| `rigidity`            | [$d$-rigidity](#rigidity)                      |
| `min_rigidity`        | [Minimal $d$-rigidity](#encoding-min-rigidity) |
| `global_rigidity`     | [Global $d$-rigidity](#global-rigidity)        |  



### Rigidity

We store the {prf:ref}`maximum rigid dimension <def-max-rigid-dimension>`.
Hence, a graph is $d$-rigid if and only if $d$ is at most the stored value.

(encoding-min-rigidity)=
### Minimal rigidity

Let $G=(V,E)$ be a connected graph with at least two vertices.
If $G$ is complete, then $G$ is minimally $d$-rigid
for all $|V|-1 \leq d$ and $G$ is not minimally $d$-rigid for all $1\leq d<|V|-1$
(see {prf:ref}`thm-gen-rigidity-small-complete`).
If $G$ is not complete, then there is at most one $d\in\NN$
such that $G$ is minimally $d$-rigid (it follows from {prf:ref}`thm-gen-rigidity-tight`).
Hence, we store the following value for $G=(V,E)$:
\begin{equation*}
    d_\text{min} = 
        \begin{cases}
            -(|V|-1) & \text{if $G$ is complete}\\
            d & \text{if $G$ is non-complete and minimally $d$-rigid} \\
            0 & \text{otherwise}.
        \end{cases}
\end{equation*}
The function is implemented in `graphDB._min_rigidity_dimension` (to be moved later).
Conversely, a graph is minimally $d$-rigid if and only if $d=d_\text{min}$, or
$d_\text{min}<0$ and $|d_\text{min}| \leq d$.



### Global Rigidity

Let $G=(V,E)$ be a graph.
If $G$ is complete, then it is globally $d$-rigid for every $d\in\NN$,
hence $\infty$ is stored.
If $G$ is not globally $1$-rigid, we store $0$.
Otherwise, we store the integer $d\geq 1$ such that
$G$ is globally $d'$-rigid for all $d'\leq d$
and $G$ is not globally $d'$-rigid for all $d'>d$. 
By {prf:ref}`thm-rigidity-dim-monotonicity` and {prf:ref}`thm-globally-necessary`,
$d$ is well-defined.
Therefore, a graph is globally $d$-rigid if and only if $d$ is at most the stored value.
