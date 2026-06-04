(graph-database-interface)=
# Using the Graph Database Interface

The `pyrigi.graphDB` subpackage stores graphs together with computed properties in a
single SQLite file and offers a typed, composable query layer over them. It is intended
for offline, exploratory analysis of property distributions across large graph
collections, where recomputing properties on every run would be impractical. Properties
computed in one session are persisted and available in all later sessions.

Callers interact only with the `GraphStoreService` class. The mathematical encoding of
the rigidity columns is described under
[Rigidity column encoding](#rigidity-column-encoding).

The typical workflow is: open a store, ingest graphs, populate computed columns, query.

## Quick start

```python
from pyrigi.graphDB import GraphStoreService, QueryFilter

with GraphStoreService("outputs/graph_store.db") as store:
    store.ingest("outputs/g6")          # load graph6 files
    store.populate_column("rigidity")   # compute the rigidity property

    rows = store.fetch(
        select=["graph", "num_vertices", "rigidity"],
        filters=[QueryFilter("num_vertices", "=", 5)],
        order_by="num_edges",
    )
    store.pretty_print_results(rows, show_index=True)
```

Used as a context manager, the service opens the connection and creates the schema on
entry, and closes the connection on exit.

## Opening a store

```python
store = GraphStoreService(db_path="outputs/graph_store.db", batch_size=500).init()
```

| Parameter    | Default                      | Notes                                                      |
|--------------|------------------------------|------------------------------------------------------------|
| `db_path`    | `"outputs/graph_store.db"`   | Path to the SQLite file. Use `":memory:"` for testing.     |
| `batch_size` | `500`                        | Rows per transaction. Must be `>= 1`, else `ValueError`.   |

`init()` opens the connection and creates the schema. It must be called before any other
method, is idempotent, and returns the service for chaining. `close()` closes the
connection. The context-manager form (`with GraphStoreService(...) as store:`) calls
`init()` and `close()` automatically and is recommended.

## Ingesting graphs

```python
stats = store.ingest("outputs/g6")   # file, .g6.gz, or directory
```

`ingest(source, batch_size=None)` accepts a `.g6` file, a `.g6.gz` file, or a directory
(all `*.g6` and `*.g6.gz` files, in sorted order). Within each file, blank lines and
lines beginning with `>>` are ignored and gzip is handled transparently.

- Undecodable lines are counted as errors and skipped.
- Graphs with fewer than two vertices are skipped.
- Ingestion is **idempotent**: the `graph` column is unique, so graphs already present
  are skipped.
- The four structural columns are computed during ingestion; the rigidity columns are
  left empty for on-demand population.

`ingest` returns an `IngestStats` with fields `inserted`, `skipped`, `errors`,
`files_processed`.

## The default schema

| Column            | Type      | Filled at | Accepted operators                  |
|-------------------|-----------|-----------|-------------------------------------|
| `graph`           | `TEXT`    | ingestion | all (unique identifier)             |
| `num_vertices`    | `INTEGER` | ingestion | all                                 |
| `num_edges`       | `INTEGER` | ingestion | all                                 |
| `min_degree`      | `INTEGER` | ingestion | all                                 |
| `max_degree`      | `INTEGER` | ingestion | all                                 |
| `rigidity`        | `INTEGER` | on demand | `=`, `IN`, `IS NULL`, `IS NOT NULL` |
| `min_rigidity`    | `INTEGER` | on demand | `=`, `IN`, `IS NULL`, `IS NOT NULL` |
| `global_rigidity` | `INTEGER` | on demand | `=`, `IN`, `IS NULL`, `IS NOT NULL` |

The rigidity columns are nullable; a `NULL` value marks a row whose property has not yet
been computed. Their stored encoding is described under
[Rigidity column encoding](#rigidity-column-encoding).

## Populating columns

```python
store.populate_column("rigidity")
store.populate_column("min_rigidity")
store.populate_column("global_rigidity")
```

The rigidity properties are computed on demand because they are far more expensive than
the structural columns.

`populate_column(column, *, populator=None, batch_size=None, all_rows=False)`:

| Argument     | Effect                                                                       |
|--------------|------------------------------------------------------------------------------|
| `populator`  | Override the registered populator for this call only.                        |
| `batch_size` | Override the instance default for this call.                                 |
| `all_rows`   | `True` recomputes every row; `False` (default) computes only `NULL` rows.    |

The populator is resolved in order: the `populator` argument, then an in-memory callable
cached at registration, then the column's importable reference. No populator raises
`RuntimeError`; an unknown column raises `KeyError`.

A failure on one row is logged at `ERROR` level (with the offending graph6 string) and
skipped, so one bad graph does not abort the run. Configure a logging handler to observe
these. Returns a `PopulateStats` with fields `column`, `processed`, `errors`.

## Querying

A single predicate is a `QueryFilter(column, operator, value)`. The operator is
normalised to upper case and validated against eleven operators:

```
=   !=   <   <=   >   >=   IN   BETWEEN   LIKE   IS NULL   IS NOT NULL
```

An unknown operator raises `ValueError`. The form of `value`:

| Operator               | `value`                                       |
|------------------------|-----------------------------------------------|
| `IN`                   | list or tuple, e.g. `[5, 6, 7]`               |
| `BETWEEN`              | two-element tuple `(low, high)`               |
| `IS NULL`, `IS NOT NULL` | ignored (omit it; defaults to `None`)       |
| all others             | the scalar right-hand side                    |

### One-line queries with `fetch`

```python
rows = store.fetch(
    select=["graph", "num_vertices"],
    filters=[QueryFilter("num_vertices", "=", 7)],
    order_by="num_edges",
    asc=False,
    limit=10,
)
```

`fetch` returns a list of row dictionaries. Parameters:

| Parameter  | Meaning                                                          |
|------------|------------------------------------------------------------------|
| `select`   | Columns to return; `None` (default) returns all.                 |
| `filters`  | List of `QueryFilter`, combined with `AND`.                      |
| `expr`     | Optional grouped boolean expression (see below).                 |
| `order_by` | Column to sort by.                                               |
| `asc`      | Ascending if `True` (default), descending if `False`.            |
| `limit`    | Maximum rows to return.                                          |
| `offset`   | Leading rows to skip; requires `limit`.                          |
| `mapper`   | Function applied to each row dictionary before it is returned.   |

### Grouped boolean expressions

A plain `filters` list joins its predicates with `AND`. Anything more complex (`OR`
groups, negation, nesting) is built as an expression tree using three helpers:

| Helper           | Builds                              |
|------------------|-------------------------------------|
| `all_of(*exprs)` | an `AND` over its arguments         |
| `any_of(*exprs)` | an `OR` over its arguments          |
| `not_(expr)`     | the negation of one expression      |

The tree is passed to the `expr` parameter of `fetch` (or to `where_expr` on the
builder). Helpers nest to any depth and may contain `QueryFilter` leaves or other
helpers:

```python
from pyrigi.graphDB.models import all_of, any_of, not_

expr = all_of(
    QueryFilter("num_vertices", "=", 6),
    any_of(
        QueryFilter("rigidity", "=", 2),
        QueryFilter("global_rigidity", "=", 2),
    ),
    not_(QueryFilter("min_rigidity", "=", 2)),
)
rows = store.fetch(select=["graph"], expr=expr)
```

The helpers are shorthand for the node classes `AndExpr`, `OrExpr`, and `NotExpr`: for
example `all_of(a, b)` is exactly `AndExpr([a, b])`. The classes accept a list, the
helpers accept positional arguments; use whichever reads better. `AndExpr` and `OrExpr`
require at least one child, otherwise they raise `ValueError`.

### The fluent builder

`store.query()` returns a `QueryBuilder` whose methods chain; `fetch` runs the query:

```python
rows = (
    store.query()
    .select(["graph", "num_edges"])
    .where([QueryFilter("num_vertices", "=", 5)])
    .where_any([QueryFilter("num_edges", "=", 4), QueryFilter("num_edges", "=", 5)])
    .order_by("num_edges", asc=False)
    .limit(20)
    .fetch()
)
```

| Method                          | Effect                                                        |
|---------------------------------|---------------------------------------------------------------|
| `select(columns)`               | Choose returned columns (default all).                        |
| `where(filters)`                | Add `AND` predicates; calls accumulate.                       |
| `where_any(filters)`            | Add an `OR` group of filters (shortcut for `where_expr(any_of(...))`). |
| `where_expr(expr)`              | Add any expression tree (the general form).                   |
| `filter(column, operator, value)` | Add one predicate without building a `QueryFilter`.         |
| `order_by(column, asc=True)`, `limit(n)`, `offset(n)` | Ordering and paging.                    |

All three predicate methods may be combined on one builder; their conditions are joined
with `AND`. `where_any` is a convenience for the common case of OR-ing a few filters;
`where_expr` handles everything else, including nesting and negation. The builder's
`where(filters)` and `where_expr(expr)` correspond to the `filters` and `expr` parameters
of `fetch`.

`compile()` returns an immutable `CompiledQuery` (SQL string plus bound parameters) that
can be inspected before execution, which is useful for debugging and tests.

### Mapping and streaming

`mapper` transforms each row, for example decoding graph6 strings back to `networkx`
graphs:

```python
import networkx as nx

graphs = store.fetch(
    select=["graph"],
    mapper=lambda row: nx.from_graph6_bytes(row["graph"].encode("ascii")),
)
```

`fetch` builds the full list in memory. For very large results, `iter_fetch` takes the
same arguments but yields rows one at a time:

```python
for row in store.iter_fetch(filters=[QueryFilter("num_vertices", "=", 8)]):
    ...
```

## Rigidity-aware querying

Complete graphs are stored with a sentinel value (`-1` for `rigidity` and
`global_rigidity`; a negative value for `min_rigidity`), because they are rigid in every
dimension. To keep queries correct without exposing this encoding, the three rigidity
columns accept only `=`, `IN`, `IS NULL`, `IS NOT NULL` (any other operator raises
`ValueError`), and the query layer rewrites `=` and `IN` to include complete graphs
automatically:

```python
store.fetch(filters=[QueryFilter("rigidity", "=", 2)])       # dim-2 rigid + complete
store.fetch(filters=[QueryFilter("rigidity", "IN", [1, 2])]) # likewise
```

Complete graphs alone are selected with `QueryFilter("rigidity", "=", -1)`. See
[Rigidity column encoding](#rigidity-column-encoding) for the stored encoding.

(rigidity-column-encoding)=
## Rigidity column encoding

The three rigidity columns store integer encodings of rigidity-theoretic properties.
Every graph in the database is assumed to have at least two vertices.

### Rigidity

The stored value is the {prf:ref}`maximum rigid dimension <def-max-rigid-dimension>`, so a
graph is $d$-rigid if and only if $d$ is at most the stored value. Complete graphs, which
are rigid in every dimension, are stored as $-1$. Since $-1$ is not a valid rigidity
dimension, the sentinel is unambiguous.

(encoding-min-rigidity)=
### Minimal rigidity

Let $G=(V,E)$ be a connected graph with at least two vertices. If $G$ is complete, then $G$
is minimally $d$-rigid for all $|V|-1 \leq d$ and is not minimally $d$-rigid for any
$1\leq d<|V|-1$ (see {prf:ref}`thm-gen-rigidity-small-complete`). If $G$ is not complete,
there is at most one $d\in\NN$ such that $G$ is minimally $d$-rigid (it follows from
{prf:ref}`thm-gen-rigidity-tight`). The stored value is therefore:

\begin{equation*}
    d_\text{min} =
        \begin{cases}
            -(|V|-1) & \text{if $G$ is complete}\\
            d & \text{if $G$ is non-complete and minimally $d$-rigid} \\
            0 & \text{otherwise}.
        \end{cases}
\end{equation*}

Conversely, a graph is minimally $d$-rigid if and only if $d=d_\text{min}$, or
$d_\text{min}<0$ and $|d_\text{min}| \leq d$. The encoding is computed by
`pyrigi.graphDB.small_graphs._min_rigidity_dimension_encoding`.

### Global rigidity

The stored value is the
{prf:ref}`maximum globally rigid dimension <def-max-globally-rigid-dimension>`, so a graph
is globally $d$-rigid if and only if $d$ is at most the stored value. Complete graphs are
stored as $-1$, using the same sentinel as the `rigidity` column.

## Custom columns

```python
store.add_column(
    "density",
    "REAL",
    description="Edge density of the graph",
    populator=lambda row: (
        2 * row["num_edges"] / (row["num_vertices"] * (row["num_vertices"] - 1))
        if row["num_vertices"] > 1 else 0.0
    ),
)
store.populate_column("density")
```

`add_column(name, data_type="INTEGER", description="", *, ...)` keyword-only arguments:

| Argument          | Purpose                                                                  |
|-------------------|--------------------------------------------------------------------------|
| `populator`       | Runtime callable `(row: dict) -> value`. `row` holds all stored columns. |
| `populator_ref`   | Importable reference `"package.module:function"` instead of a callable.  |
| `fetch_strategy`  | Custom query rewriter for the column (see below).                        |
| `fetch_ref`       | Importable reference for the fetch strategy.                             |
| `valid_operators` | `frozenset` of accepted operators; others raise `ValueError`.            |

A name colliding with a built-in default raises `ValueError`.

**Persistence.** A runtime callable (such as the lambda above) lives only for the current
session and must be supplied again later. A function registered through `populator_ref`
is re-imported automatically in any later session. The built-in columns use importable
references and are always available.

## Managing columns and data

| Method                                               | Effect                                                                 |
|------------------------------------------------------|------------------------------------------------------------------------|
| `update_column_populator(name, ...)`                 | Replace a column's populator (e.g. attach a custom rigidity solver).   |
| `update_column_fetch_strategy(name, ...)`            | Replace a column's fetch strategy; re-register runtime ones each session. |
| `drop_column(name)`                                  | Remove a custom column (defaults cannot be dropped; missing raises `KeyError`). |
| `delete_graph(g6)`                                   | Delete one row by graph6 string; returns `True` if found.              |
| `delete_where(filters=None, expr=None)`              | Delete all matching rows; returns the count. No arguments deletes all. |

## Inspecting the store

| Method                       | Returns                                                              |
|------------------------------|----------------------------------------------------------------------|
| `count()`                    | Total number of graphs.                                              |
| `count_unpopulated(column)`  | Number of rows where `column` is `NULL`.                             |
| `info()`                     | Dict with `total_graphs` and `columns` (`name`, `type`, `default`, `description`, `has_populator`). |
| `list_columns()`             | `ColumnDef` objects for all columns.                                 |
| `get_column(name)`           | `ColumnDef` for one column, or `None`.                               |

## Displaying results

```python
store.pretty_print_results(rows, show_index=True)
```

`format_results` returns the ASCII table as a string; `pretty_print_results` prints it
and returns it. Both take the same keyword-only arguments:

| Argument        | Default | Notes                                                          |
|-----------------|---------|----------------------------------------------------------------|
| `columns`       | `None`  | Explicit column order; otherwise taken from the first row.     |
| `max_rows`      | `20`    | `None` renders all; negative raises `ValueError`; a notice is added when truncated. |
| `max_col_width` | `48`    | Longer cells are truncated with an ellipsis.                   |
| `show_index`    | `False` | Add a leading `#` index column.                                |

`pretty_print_results` also takes `file` (default standard output). An empty result
renders as `(no rows)`. The fluent builder exposes the same two helpers as
`format_results` and `pretty_print`.

## Advanced: custom fetch strategies

A fetch strategy controls how a `QueryFilter` becomes SQL. Its signature is
`(column, operator, value) -> (sql_fragment, params)`. Columns without one use a default
pass-through that handles all operators. A custom strategy is needed when a column's
storage encoding differs from how it is queried; the rigidity columns are the built-in
example. Supply it through the `fetch_strategy` argument of `add_column`, or persistently
through `fetch_ref`.

## Reference

`IngestStats` (from `ingest`): `inserted`, `skipped`, `errors`, `files_processed`.
`PopulateStats` (from `populate_column`): `column`, `processed`, `errors`.

**See also**

- {doc}`/userguide/api/graphDB` for the auto-generated API reference.
