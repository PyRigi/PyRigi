"""
pyrigi.graphDB.ingestion
~~~~~~~~~~~~~~~~~~
Pipeline for reading ``.g6`` / ``.g6.gz`` files and inserting graphs into
the database.

Three decoupled responsibilities
---------------------------------
G6Reader
    Yields raw graph6 strings from one file or an entire directory.
GraphParser
    Converts a graph6 string → networkx Graph.
DefaultColumnComputer
    Computes the values for the five always-populated default columns
    (``graph``, ``num_vertices``, ``num_edges``, ``min_degree``,
    ``max_degree``).

The :class:`~pyrigi.graphDB.service.GraphStoreService` orchestrates these three
objects; they are independent and individually testable.
"""
from pyrigi.graphDB.ingestion.reader import G6Reader
from pyrigi.graphDB.ingestion.parser import GraphParser
from pyrigi.graphDB.ingestion.default_computer import DefaultColumnComputer

__all__ = ["G6Reader", "GraphParser", "DefaultColumnComputer"]
