# Graph store internals

The modules below implement the graph store and are not part of the stable public
interface. Callers should use {class}`~pyrigi.graphDB.service.GraphStoreService`; the
classes and functions here may change in future releases.

## Database connection

```{eval-rst}
.. automodule:: pyrigi.graphDB.db
   :members:
```

## Repositories

```{eval-rst}
.. automodule:: pyrigi.graphDB.repositories.graph_repo
   :members:

.. automodule:: pyrigi.graphDB.repositories.column_registry
   :members:
```

## Ingestion

```{eval-rst}
.. automodule:: pyrigi.graphDB.ingestion.reader
   :members:

.. automodule:: pyrigi.graphDB.ingestion.parser
   :members:

.. automodule:: pyrigi.graphDB.ingestion.default_computer
   :members:
```

## Default columns, populators, and fetch strategies

```{eval-rst}
.. automodule:: pyrigi.graphDB.defaults.columns
   :members:

.. automodule:: pyrigi.graphDB.defaults.populators
   :members:
   :private-members:

.. automodule:: pyrigi.graphDB.defaults.fetch_strategies
   :members:
   :private-members:
```

## Resolvers and helpers

```{eval-rst}
.. automodule:: pyrigi.graphDB.models.resolvers
   :members:
   :private-members:

.. automodule:: pyrigi.graphDB.utils.pretty
   :members:
```

## Constants

```{eval-rst}
.. automodule:: pyrigi.graphDB.constants.operators
   :members:

.. automodule:: pyrigi.graphDB.constants.schema
   :members:
   :private-members:

.. automodule:: pyrigi.graphDB.constants.identifiers
   :members:
```
