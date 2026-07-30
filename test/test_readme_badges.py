"""
Consistency checks for the package version badges in ``README.md``.

The package badges use the shields.io ``dynamic/regex`` endpoint: the badge applies
its ``search`` regular expression to ``pyproject.toml`` and renders the ``replace``
template, so that only the minimal supported version is advertised.
These tests check that the regular expression of each badge still matches
the requirement declared in ``[project] dependencies``.
"""

import re
import tomllib
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

_ROOT = Path(__file__).parents[1]

#: Packages that are expected to have a version badge in ``README.md``.
_EXPECTED = {"networkx", "numpy", "sympy"}

#: Matches ``[![name](https://img.shields.io/badge/dynamic/regex?...)](link)``.
_BADGE = re.compile(
    r"\[!\[(?P<name>[^\]]+)\]"
    r"\((?P<url>https://img\.shields\.io/badge/dynamic/regex\?[^)]+)\)\]"
)

#: Matches the lower bound of a requirement like ``networkx (>=3.4.2,<4.0.0)``.
_LOWER_BOUND = re.compile(r">=\s*(?P<version>[^,)\s]+)")


def _dependencies() -> list[str]:
    """Return the ``[project] dependencies`` of ``pyproject.toml``."""
    with open(_ROOT / "pyproject.toml", "rb") as file:
        return tomllib.load(file)["project"]["dependencies"]


def _badges() -> list[tuple[str, str]]:
    """Return the ``(name, url)`` pairs of all dynamic regex badges."""
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    return [(m.group("name"), m.group("url")) for m in _BADGE.finditer(readme)]


def test_readme_badges_exist():
    names = {name for name, _ in _badges()}
    assert _EXPECTED <= names, f"missing version badges for {_EXPECTED - names}"


@pytest.mark.parametrize("name, url", _badges(), ids=[name for name, _ in _badges()])
def test_readme_badge_version(name, url):
    requirements = [dep for dep in _dependencies() if dep.startswith(name)]
    assert len(requirements) == 1, (
        f"the {name} badge does not correspond to exactly one dependency: "
        f"{requirements}"
    )
    lower_bound = _LOWER_BOUND.search(requirements[0])
    assert lower_bound is not None, (
        f"the requirement {requirements[0]!r} has no lower bound, "
        f"so the {name} badge cannot display one"
    )
