# SQLite Database

We assume that each graph in the database has at least two vertices.

## Stored values

Here we describe how values in the columns of the database are created.

### Rigidity

We store the {prf:ref}`maximum rigid dimension <def-max-rigid-dimension>`.
Hence, a graph is $d$-rigid if and only if $d$ is at most the stored value.

### Minimal rigidity

Let $G=(V,E)$ be a connected graph with at least two vertices.
If $G$ is complete, then $G$ is minimally $d$-rigid
for all $|V|-1 \leq d$ and $G$ is not minimally $d$-rigid for all $1\leq d<|V|-1$
(see {prf:ref}`thm-gen-rigidity-small-complete`).
If $G$ is not complete, then there is at most one $d\in\NN$
such that $G$ is $d$-rigid (it follows from {prf:ref}`thm-gen-rigidity-tight`).
Hence, we store the following value for $G=(V,E)$:
\begin{equation*}
    d_\text{min} = 
        \begin{cases}
            -(|V|-1) & \text{if $G$ is complete}\\
            d & \text{if $G$ is non-complete and minimally $d$-rigid} \\
            0 & \text{otherwise}.
        \end{cases}
\end{equation*}
Conversely, a graph is minimally $d$-rigid if and only if $d=d_\text{min}$, or
$d_\text{min}<0$ and $|d_\text{min}| \leq d$.