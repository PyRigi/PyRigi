"""
pyrigi.graphDB.models.stats
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Lightweight result containers returned by service methods.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IngestStats:
    """Summary of an ingestion run.

    Returned by :meth:`~pyrigi.graphDB.service.GraphStoreService.ingest`.

    Attributes
    ----------
    inserted:
        Number of new graphs written to the database.
    skipped:
        Number of graphs that were read but not inserted, either because
        they were already present (duplicate ``graph`` value) or because
        they had fewer than two vertices.
    errors:
        Number of lines that could not be decoded as graph6 strings.
    files_processed:
        Number of files read during the run.
    """

    inserted: int = 0
    skipped: int = 0
    errors: int = 0
    files_processed: int = 0


@dataclass
class PopulateStats:
    """Summary of a column population run.

    Returned by
    :meth:`~pyrigi.graphDB.service.GraphStoreService.populate_column`.

    Attributes
    ----------
    column:
        Name of the column that was populated.
    processed:
        Number of rows for which a value was computed and written.
    errors:
        Number of rows whose populator raised an exception and were
        skipped (the failure is logged, the run continues).
    """

    column: str = ""
    processed: int = 0
    errors: int = 0
