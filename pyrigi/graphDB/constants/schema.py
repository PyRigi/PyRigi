"""
pyrigi.graphDB.constants.schema
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DDL strings for the core ``graphs`` and ``column_registry`` tables.
Centralised here so that ``db.py`` stays free of inline SQL literals.
"""

_GRAPHS_DDL = """
CREATE TABLE IF NOT EXISTS graphs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    graph           TEXT    NOT NULL UNIQUE,
    num_vertices    INTEGER NOT NULL,
    num_edges       INTEGER NOT NULL,
    min_degree      INTEGER NOT NULL,
    max_degree      INTEGER NOT NULL,
    rigidity        INTEGER,
    min_rigidity    INTEGER,
    global_rigidity INTEGER
);
"""

_REGISTRY_DDL = """
CREATE TABLE IF NOT EXISTS column_registry (
    name            TEXT PRIMARY KEY,
    data_type       TEXT    NOT NULL,
    description     TEXT    NOT NULL DEFAULT '',
    populator_ref   TEXT,
    fetch_ref       TEXT,
    valid_operators TEXT,
    is_default      INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""
