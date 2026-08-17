(graph-database-interface)=
# Using the Graph Database Interface

The `pyrigi.graphDB` subpackage stores graphs together with computed properties in a
single SQLite file and offers a typed, composable query layer over them. It is intended
for offline, exploratory analysis of property distributions across large graph
collections, where recomputing properties on every run would be impractical. Properties
computed in one session are persisted and available in all later sessions.

Callers interact only with the {class}`~pyrigi.graphDB.service.GraphStoreService` class.
The mathematical encoding of the rigidity columns is described under
[Rigidity column encoding](#rigidity-column-encoding).

The typical workflow is: open a store, ingest graphs, populate computed columns, query.

## Quick start

The following example runs the whole workflow end to end: open a store, ingest graphs,
compute a property, then query it.

```python
from pyrigi.graphDB import GraphStoreService, QueryFilter

with GraphStoreService("outputs/graph_store.db") as store:
    store.ingest("outputs/g6")          # a directory of .g6 files (a single .g6/.g6.gz file also works)
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

Create a store by giving it a database path and, optionally, a batch size, then activate it
with {meth}`~pyrigi.graphDB.service.GraphStoreService.init`:

```python
store = GraphStoreService(db_path="outputs/graph_store.db", batch_size=500).init()
```

| Parameter    | Default                      | Notes                                                      |
|--------------|------------------------------|------------------------------------------------------------|
| `db_path`    | `"outputs/graph_store.db"`   | Path to the SQLite file. Use `":memory:"` for testing.     |
| `batch_size` | `500`                        | Rows per transaction. Must be `>= 1`, else `ValueError`.   |

The `init()` method opens the connection and creates the schema. It must be called before
any other method, is idempotent, and returns the service for chaining. The {meth}`~pyrigi.graphDB.service.GraphStoreService.close` method
closes the connection. The context-manager form (`with GraphStoreService(...) as store:`)
calls `init()` and `close()` automatically and is recommended.

## Ingesting graphs

Load graphs into the store from graph6 data, either a single file or a whole directory:

```python
stats = store.ingest("outputs/g6")   # file, .g6.gz, or directory
```

The {meth}`ingest(source, batch_size=None) <pyrigi.graphDB.service.GraphStoreService.ingest>` method accepts a `.g6` file, a `.g6.gz` file, or a
directory (all `*.g6` and `*.g6.gz` files, in sorted order). Within each file, blank lines and
lines beginning with `>>` are ignored and gzip is handled transparently.

- Undecodable lines are counted as errors and skipped.
- Graphs with fewer than two vertices are skipped.
- Ingestion is **idempotent**: the `graph` column is unique, so graphs already present
  are skipped.
- The four structural columns are computed during ingestion; the rigidity columns are
  left empty for on-demand population.

`ingest` returns an {class}`~pyrigi.graphDB.models.stats.IngestStats` with fields
`inserted`, `skipped`, `errors`, `files_processed`.

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

Compute the rigidity properties and store them, one column at a time:

```python
store.populate_column("rigidity")
store.populate_column("min_rigidity")
store.populate_column("global_rigidity")
```

The rigidity properties are computed on demand because they are far more expensive than
the structural columns.

{meth}`populate_column(column, *, populator=None, batch_size=None, recompute_all=False) <pyrigi.graphDB.service.GraphStoreService.populate_column>`:

| Argument     | Effect                                                                       |
|--------------|------------------------------------------------------------------------------|
| `populator`  | Override the registered populator for this call only.                        |
| `batch_size` | Override the instance default for this call.                                 |
| `recompute_all` | `True` recomputes every row; `False` (default) computes only `NULL` rows.  |

The populator is resolved in order: the `populator` argument, then an in-memory callable
cached at registration, then the column's importable reference. No populator raises
`RuntimeError`; an unknown column raises `KeyError`.

A failure on one row is logged at `ERROR` level (with the offending graph6 string) and
skipped, so one bad graph does not abort the run. These `ERROR` messages already print to
stderr by default (Python's last-resort handler shows `WARNING` and above, unformatted).
Configure logging to format them, or to also see the `INFO`-level operational messages the
default threshold hides. Do this in your own driver script, before `populate_column`
(logging is the application's responsibility, not the library's; for example at the top of
`main()` in `pyrigi/graphDB/scripts/demo.py`):

```python
import logging
logging.basicConfig(level=logging.INFO)                 # INFO+ from every library
# or scope to this subpackage only:
logging.getLogger("pyrigi.graphDB").setLevel(logging.INFO)
logging.getLogger("pyrigi.graphDB").addHandler(logging.StreamHandler())
```

The `pyrigi.graphDB` logger emits per-row populate failures at `ERROR` (with the graph6
string), ingestion parse/read errors, and `INFO` operational messages.

The method returns a {class}`~pyrigi.graphDB.models.stats.PopulateStats` with fields
`column`, `processed`, and `errors`.

## Querying

A single predicate is a {class}`~pyrigi.graphDB.models.filters.QueryFilter`, written
`QueryFilter(column, operator, value)`. The operator is
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

The {meth}`~pyrigi.graphDB.service.GraphStoreService.fetch` method runs a whole query in a single call, taking the columns, filters,
ordering, and paging as arguments:

```python
rows = store.fetch(
    select=["graph", "num_vertices"],
    filters=[QueryFilter("num_vertices", "=", 7)],
    order_by="num_edges",
    ascending=False,
    limit=10,
)
```

This returns up to ten graphs on exactly seven vertices, ordered from most to fewest edges,
giving each graph's graph6 string and vertex count.

The `fetch` method returns a list of row dictionaries. Parameters:

| Parameter  | Meaning                                                          |
|------------|------------------------------------------------------------------|
| `select`   | Columns to return; `None` (default) returns all.                 |
| `filters`  | List of `QueryFilter`, combined with `AND`.                      |
| `expr`     | Optional grouped boolean expression (see below).                 |
| `order_by` | Column to sort by.                                               |
| `ascending` | Ascending if `True` (default), descending if `False`.           |
| `limit`    | Maximum rows to return.                                          |
| `offset`   | Leading rows to skip; requires `limit`.                          |
| `mapper`   | Function applied to each row dictionary before it is returned.   |

### Grouped boolean expressions

A plain `filters` list joins its predicates with `AND`. Anything more complex (`OR`
groups, negation, nesting) is built as an expression tree using three helpers:

| Helper           | Builds                              |
|------------------|-------------------------------------|
| {func}`all_of(*exprs) <pyrigi.graphDB.models.expressions.all_of>` | an `AND` over its arguments |
| {func}`any_of(*exprs) <pyrigi.graphDB.models.expressions.any_of>` | an `OR` over its arguments |
| {func}`not_(expr) <pyrigi.graphDB.models.expressions.not_>`     | the negation of one expression |

The tree is passed to the `expr` parameter of {meth}`~pyrigi.graphDB.service.GraphStoreService.fetch` (or to {meth}`~pyrigi.graphDB.query.QueryBuilder.where_expr` on the
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

This returns the graphs on six vertices that are 2-rigid or globally 2-rigid but not
minimally 2-rigid.

The helpers are shorthand for the node classes
{class}`~pyrigi.graphDB.models.expressions.AndExpr`,
{class}`~pyrigi.graphDB.models.expressions.OrExpr`, and
{class}`~pyrigi.graphDB.models.expressions.NotExpr`: for
example `all_of(a, b)` is exactly `AndExpr([a, b])`. The classes accept a list, the
helpers accept positional arguments; use whichever reads better. `AndExpr` and `OrExpr`
require at least one child, otherwise they raise `ValueError`.

### The fluent builder

{meth}`store.query() <pyrigi.graphDB.service.GraphStoreService.query>` returns a
{class}`~pyrigi.graphDB.query.QueryBuilder` whose methods chain;
{meth}`~pyrigi.graphDB.query.QueryBuilder.fetch` runs the query:

```python
rows = (
    store.query()
    .select(["graph", "num_edges"])
    .where([QueryFilter("num_vertices", "=", 5)])
    .where_any([QueryFilter("num_edges", "=", 4), QueryFilter("num_edges", "=", 5)])
    .order_by("num_edges", ascending=False)
    .limit(20)
    .fetch()
)
```

| Method                          | Effect                                                        |
|---------------------------------|---------------------------------------------------------------|
| {meth}`select(columns) <pyrigi.graphDB.query.QueryBuilder.select>` | Choose returned columns (default all). |
| {meth}`where(filters) <pyrigi.graphDB.query.QueryBuilder.where>` | Add `AND` predicates; calls accumulate. |
| {meth}`where_any(filters) <pyrigi.graphDB.query.QueryBuilder.where_any>` | Add an `OR` group of filters (shortcut for `where_expr(any_of(...))`). |
| {meth}`where_expr(expr) <pyrigi.graphDB.query.QueryBuilder.where_expr>` | Add any expression tree (the general form). |
| {meth}`filter(column, operator, value) <pyrigi.graphDB.query.QueryBuilder.filter>` | Add one predicate without building a `QueryFilter`. |
| {meth}`order_by(column, ascending=True) <pyrigi.graphDB.query.QueryBuilder.order_by>`, {meth}`limit(n) <pyrigi.graphDB.query.QueryBuilder.limit>`, {meth}`offset(n) <pyrigi.graphDB.query.QueryBuilder.offset>` | Ordering and paging. |

All three predicate methods may be combined on one builder; their conditions are joined
with `AND`. The `where_any` method is a convenience for the common case of OR-ing a few
filters; `where_expr` handles everything else, including nesting and negation. The builder's
`where(filters)` and `where_expr(expr)` correspond to the `filters` and `expr` parameters
of `fetch`.

The {meth}`~pyrigi.graphDB.query.QueryBuilder.compile` method returns an immutable {class}`~pyrigi.graphDB.query.CompiledQuery`
(SQL string plus bound parameters) that can be inspected before execution, which is useful
for debugging and tests.

### Mapping and streaming

`mapper` transforms each row. The most common need, turning the stored graph6 strings back
into graph objects, is covered by two ready-made mappers,
{func}`~pyrigi.graphDB.utils.mappers.to_networkx` and
{func}`~pyrigi.graphDB.utils.mappers.to_pyrigi`:

```python
from pyrigi.graphDB import to_networkx, to_pyrigi

# pyrigi.Graph objects (a networkx.Graph subclass, with PyRigi's rigidity methods)
graphs = store.fetch(select=["graph"], mapper=to_pyrigi)

# plain networkx.Graph objects
nx_graphs = store.fetch(select=["graph"], mapper=to_networkx)
```

Both read the `graph` column, so the query must select it (include `"graph"` in `select`,
or use `select=None`), and both work the same way with
{meth}`~pyrigi.graphDB.service.GraphStoreService.iter_fetch`. Any other callable is
still accepted for custom transforms.

{meth}`~pyrigi.graphDB.service.GraphStoreService.fetch` builds the full list in memory. For very large results, `iter_fetch` takes the
same arguments but yields rows one at a time:

```python
for row in store.iter_fetch(filters=[QueryFilter("num_vertices", "=", 8)]):
    ...
```

(rigidity-column-encoding)=
## Rigidity columns: encoding and queries

The three rigidity columns store integer encodings of rigidity-theoretic properties, and
the query layer exposes them through a small operator set so callers need not handle the
encoding directly. All three accept only `=`, `IN`, `IS NULL`, and `IS NOT NULL`; any other
operator raises `ValueError`. Every graph in the database is assumed to have at least two
vertices.

### Rigidity and global rigidity

**Encoding.** The stored value is the {prf:ref}`maximum rigid dimension
<def-max-rigid-dimension>` (respectively the {prf:ref}`maximum globally rigid dimension
<def-max-globally-rigid-dimension>`), so a graph is $d$-rigid if and only if $d$ is at most
the stored value. Complete graphs, rigid in every dimension, are stored as the sentinel
$-1$; since $-1$ is not a valid dimension, the sentinel is unambiguous.

**Querying.** Because a graph is rigid in every dimension up to its stored maximum, `=`
means "is d-rigid": the query layer rewrites `rigidity = d` to match every graph whose
stored value is at least `d`, plus every complete graph. `IN` is the disjunction of such
tests.

```python
# graphs that are 2-rigid (stored maximum >= 2), plus complete graphs
store.fetch(filters=[QueryFilter("rigidity", "=", 2)])
# is 1-rigid or 2-rigid (a disjunction of the above)
store.fetch(filters=[QueryFilter("rigidity", "IN", [1, 2])])
```

Because the property is monotone, `rigidity = 1` returns every connected graph (all are
1-rigid), while `rigidity = 2` returns the strictly smaller set that is also 2-rigid. The
`global_rigidity` column behaves identically with its own maximum.

(encoding-min-rigidity)=
### Minimal rigidity

**Encoding.** Let $G=(V,E)$ be a connected graph with at least two vertices. If $G$ is
complete, then $G$ is minimally $d$-rigid for all $|V|-1 \leq d$ and is not minimally
$d$-rigid for any $1\leq d<|V|-1$ (see {prf:ref}`thm-gen-rigidity-small-complete`). If $G$
is not complete, there is at most one $d\in\NN$ such that $G$ is minimally $d$-rigid (it
follows from {prf:ref}`thm-gen-rigidity-tight`). The stored value is therefore:

\begin{equation*}
    d_\text{min} =
        \begin{cases}
            -(|V|-1) & \text{if $G$ is complete}\\
            d & \text{if $G$ is non-complete and minimally $d$-rigid} \\
            0 & \text{otherwise}.
        \end{cases}
\end{equation*}

A graph is minimally $d$-rigid if and only if $d=d_\text{min}$, or $d_\text{min}<0$ and
$|d_\text{min}| \leq d$. The encoding is computed by
`pyrigi.graphDB.small_graphs._min_rigidity_dimension_encoding`.

**Querying.** Minimal $d$-rigidity is not monotone, so, unlike the other two columns,
`min_rigidity = d` matches graphs that are *exactly* minimally $d$-rigid. The `=` expansion
covers both branches of the encoding (the non-complete value and the complete-graph range).

## Custom columns

Register a column of your own to store an additional computed property. The example below
adds a `density` column and fills it from the edge and vertex counts:

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

{meth}`add_column(name, data_type="INTEGER", description="", *, ...) <pyrigi.graphDB.service.GraphStoreService.add_column>` keyword-only arguments:

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
| {meth}`update_column_populator(name, ...) <pyrigi.graphDB.service.GraphStoreService.update_column_populator>` | Replace a column's populator (e.g. attach a custom rigidity solver). |
| {meth}`update_column_fetch_strategy(name, ...) <pyrigi.graphDB.service.GraphStoreService.update_column_fetch_strategy>` | Replace a column's fetch strategy; re-register runtime ones each session. |
| {meth}`drop_column(name) <pyrigi.graphDB.service.GraphStoreService.drop_column>` | Remove a custom column (defaults cannot be dropped; missing raises `KeyError`). |
| {meth}`delete_graph(g6) <pyrigi.graphDB.service.GraphStoreService.delete_graph>` | Delete one row by graph6 string; returns `True` if found. |
| {meth}`delete_where(filters=None, expr=None) <pyrigi.graphDB.service.GraphStoreService.delete_where>` | Delete all matching rows; returns the count. No arguments deletes all. |

## Inspecting the store

| Method                       | Returns                                                              |
|------------------------------|----------------------------------------------------------------------|
| {meth}`~pyrigi.graphDB.service.GraphStoreService.count` | Total number of graphs. |
| {meth}`count_unpopulated(column) <pyrigi.graphDB.service.GraphStoreService.count_unpopulated>` | Number of rows where `column` is `NULL`. |
| {meth}`~pyrigi.graphDB.service.GraphStoreService.info` | Dict with `total_graphs` and `columns` (`name`, `type`, `default`, `description`, `has_populator`). |
| {meth}`~pyrigi.graphDB.service.GraphStoreService.list_columns` | {class}`~pyrigi.graphDB.models.column_def.ColumnDef` objects for all columns. |
| {meth}`get_column(name) <pyrigi.graphDB.service.GraphStoreService.get_column>` | `ColumnDef` for one column, or `None`. |

## Displaying results

Print a result set as a formatted table:

```python
store.pretty_print_results(rows, show_index=True)
```

{meth}`~pyrigi.graphDB.service.GraphStoreService.format_results` returns the ASCII table as a string; {meth}`~pyrigi.graphDB.service.GraphStoreService.pretty_print_results` prints it
and returns it. Both take the same keyword-only arguments:

| Argument        | Default | Notes                                                          |
|-----------------|---------|----------------------------------------------------------------|
| `columns`       | `None`  | Explicit column order; otherwise taken from the first row.     |
| `max_rows`      | `20`    | `None` renders all; negative raises `ValueError`; a notice is added when truncated. |
| `max_col_width` | `48`    | Longer cells are truncated with an ellipsis.                   |
| `show_index`    | `False` | Add a leading `#` index column.                                |

`pretty_print_results` also takes `file` (default standard output). An empty result
renders as `(no rows)`. The fluent builder exposes the same two helpers as
`format_results` and {meth}`~pyrigi.graphDB.query.QueryBuilder.pretty_print`.

## Advanced: custom fetch strategies

A fetch strategy controls how a `QueryFilter` becomes SQL. Its signature is
`(column, operator, value) -> (sql_fragment, params)`. Columns without one use a default
pass-through that handles all operators. A custom strategy is needed when a column's
storage encoding differs from how it is queried; the rigidity columns are the built-in
example. Supply it through the `fetch_strategy` argument of {meth}`~pyrigi.graphDB.service.GraphStoreService.add_column`, or persistently
through `fetch_ref`.

## Reference

`IngestStats` (from {meth}`~pyrigi.graphDB.service.GraphStoreService.ingest`): `inserted`, `skipped`, `errors`, `files_processed`.
`PopulateStats` (from {meth}`~pyrigi.graphDB.service.GraphStoreService.populate_column`): `column`, `processed`, `errors`.

**See also**

- {doc}`/userguide/api/graph_store` for the auto-generated API reference.
