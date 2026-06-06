# Graph store

The graph store persists graphs together with computed properties in a single SQLite file
and provides a typed, composable query layer over them. The single entry point is
{class}`~pyrigi.graphDB.service.GraphStoreService`; the other classes on this page
({class}`~pyrigi.graphDB.models.filters.QueryFilter`, the expression helpers,
{class}`~pyrigi.graphDB.query.QueryBuilder`,
{class}`~pyrigi.graphDB.models.column_def.ColumnDef`, and the statistics containers)
support its query and reporting interface.

For a step-by-step walkthrough with explanations, see the how-to guide
{ref}`graph-database-interface`.

## Examples

Open a store, ingest graph6 files, compute a property, and query the result:

```python
from pyrigi.graphDB import GraphStoreService, QueryFilter

with GraphStoreService("outputs/graph_store.db") as store:
    store.ingest("outputs/g6")          # load .g6 / .g6.gz files or a directory
    store.populate_column("rigidity")   # compute the rigidity property on demand

    rows = store.fetch(
        select=["graph", "num_vertices", "rigidity"],
        filters=[QueryFilter("num_vertices", "=", 5)],
        order_by="num_edges",
    )
    store.pretty_print_results(rows, show_index=True)
```

Combine predicates with grouped boolean expressions:

```python
from pyrigi.graphDB import GraphStoreService, QueryFilter
from pyrigi.graphDB.models import all_of, any_of

with GraphStoreService("outputs/graph_store.db") as store:
    expr = all_of(
        QueryFilter("num_vertices", "=", 6),
        any_of(
            QueryFilter("rigidity", "=", 2),
            QueryFilter("global_rigidity", "=", 2),
        ),
    )
    rows = store.fetch(select=["graph", "rigidity", "global_rigidity"], expr=expr)
```

## Service

```{eval-rst}
.. automodule:: pyrigi.graphDB.service
   :members:
```

## Query building

```{eval-rst}
.. automodule:: pyrigi.graphDB.query
   :members:

.. automodule:: pyrigi.graphDB.models.filters
   :members:

.. automodule:: pyrigi.graphDB.models.expressions
   :members:
```

## Column definitions and statistics

```{eval-rst}
.. automodule:: pyrigi.graphDB.models.column_def
   :members:

.. automodule:: pyrigi.graphDB.models.stats
   :members:
```
