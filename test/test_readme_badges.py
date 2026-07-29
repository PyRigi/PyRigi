"""
Consistency checks for the package version badges in ``README.md``.

The package badges use the shields.io ``dynamic/toml`` endpoint, which accepts only a
restricted JSONPath subset: a dependency can be selected by its position in
``[project] dependencies``. Reordering the ``[project] dependencies`` array
makes a badge advertise the wrong package.
These tests check that the index of the package in the badge and
in the ``[project] dependencies`` array are consistent.
"""

import re
import tomllib
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

_ROOT = Path(__file__).parents[1]

#: Packages that are expected to have a version badge in ``README.md``.
_EXPECTED = {"networkx", "numpy", "sympy"}

#: Matches ``[![name](https://img.shields.io/badge/dynamic/toml?...)](link)``.
_BADGE = re.compile(
    r"\[!\[(?P<name>[^\]]+)\]"
    r"\((?P<url>https://img\.shields\.io/badge/dynamic/toml\?[^)]+)\)\]"
)

#: Matches the JSONPath a badge uses to index into the dependency array.
_INDEX_QUERY = re.compile(r"\$\.project\.dependencies\[(?P<index>\d+)\]")


def _project() -> dict:
    """Return the ``[project]`` table of ``pyproject.toml``."""
    with open(_ROOT / "pyproject.toml", "rb") as file:
        return tomllib.load(file)["project"]


def _badges() -> list[tuple[str, str]]:
    """Return the ``(name, url)`` pairs of all dynamic TOML badges."""
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    return [(m.group("name"), m.group("url")) for m in _BADGE.finditer(readme)]


def _query(url: str) -> str:
    """Return the JSONPath that ``url`` queries ``pyproject.toml`` with."""
    return parse_qs(urlparse(url).query)["query"][0]


def _dependency_badges() -> list[tuple[str, str, int]]:
    """Return the ``(name, url, index)`` triples of the badges reading a
    position in ``[project] dependencies``."""
    badges = []
    for name, url in _badges():
        match = _INDEX_QUERY.fullmatch(_query(url))
        if match is not None:
            badges.append((name, url, int(match.group("index"))))
    return badges


def test_readme_badges_exist():
    names = {name for name, _, _ in _dependency_badges()}
    assert _EXPECTED <= names, f"missing version badges for {_EXPECTED - names}"


@pytest.mark.parametrize(
    "name, index",
    [(name, index) for name, _, index in _dependency_badges()],
    ids=[name for name, _, _ in _dependency_badges()],
)
def test_readme_badge_index(name, index):
    dependencies = _project()["dependencies"]
    assert index < len(dependencies), (
        f"the {name} badge points at dependency {index}, but only "
        f"{len(dependencies)} are declared"
    )
    assert dependencies[index].startswith(name), (
        f"the {name} badge points at dependency {index}, which is now "
        f"{dependencies[index]!r}; update the index in README.md"
    )
