"""
pyrigi.graphDB.ingestion.reader
~~~~~~~~~~~~~~~~~~~~~~~~~
:class:`G6Reader` — iterates over graph6 strings from files or directories.
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)


class G6Reader:
    """Iterates over graph6 strings from a file or directory.

    Parameters
    ----------
    source:
        Path to a ``.g6`` file, a ``.g6.gz`` compressed file, or a
        directory.  Directory mode recurses one level deep and picks up
        all ``*.g6`` and ``*.g6.gz`` files.
    """

    _SUFFIXES = {".g6", ".g6.gz", ".gz"}

    def __init__(self, source: str | Path) -> None:
        self._source = Path(source)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def files(self) -> list[Path]:
        """Return a sorted list of graph files to process."""
        p = self._source
        if p.is_file():
            return [p]
        if p.is_dir():
            result = sorted(f for f in p.iterdir() if f.is_file() and self._is_g6(f))
            if not result:
                log.warning("No .g6 or .g6.gz files found in %s", p)
            return result
        raise FileNotFoundError(f"Source not found: {p}")

    def iter_strings(self) -> Iterator[str]:
        """Yield every graph6 string from all discovered files."""
        for path in self.files():
            yield from self._read_file(path)

    def iter_strings_with_file(self) -> Iterator[tuple[Path, str]]:
        """Yield ``(file_path, g6_string)`` pairs."""
        for path in self.files():
            for g6 in self._read_file(path):
                yield path, g6

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_g6(path: Path) -> bool:
        name = path.name
        return name.endswith(".g6") or name.endswith(".g6.gz")

    @staticmethod
    def _read_file(path: Path) -> Iterator[str]:
        """Yield non-empty, non-comment lines."""
        try:
            if path.suffix == ".gz" or path.name.endswith(".g6.gz"):
                opener = gzip.open(path, "rt", encoding="ascii")
            else:
                opener = open(path, "r", encoding="ascii")

            with opener as fh:
                for line in fh:
                    line = line.strip()
                    if line and not line.startswith(">>"):
                        yield line
        except Exception as exc:
            log.error("Failed to read %s: %s", path, exc)
