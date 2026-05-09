"""
pyrigi.graphDB.repositories
~~~~~~~~~~~~~~~~~~~~~
Data-access layer — one class per repository.

Re-exports both repository classes for convenient imports::

    from pyrigi.graphDB.repositories import GraphRepository, ColumnRegistryRepo
"""
from pyrigi.graphDB.repositories.graph_repo import GraphRepository
from pyrigi.graphDB.repositories.column_registry import ColumnRegistryRepo

__all__ = ["GraphRepository", "ColumnRegistryRepo"]
