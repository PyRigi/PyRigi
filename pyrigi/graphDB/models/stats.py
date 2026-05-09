"""
pyrigi.graphDB.models.stats
~~~~~~~~~~~~~~~~~~~~~~
Lightweight result containers returned by service methods.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IngestStats:
    """Returned by :meth:`~pyrigi.graphDB.service.GraphStoreService.ingest`."""
    inserted: int = 0
    skipped: int = 0
    errors: int = 0
    files_processed: int = 0


@dataclass
class PopulateStats:
    """Returned by :meth:`~pyrigi.graphDB.service.GraphStoreService.populate_column`."""
    column: str = ""
    processed: int = 0
    errors: int = 0
