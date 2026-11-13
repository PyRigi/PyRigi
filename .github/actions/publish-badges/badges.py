"""
Write shields.io endpoint JSON files with the results of the jobs of a workflow run.

The jobs, as returned by the GitHub API, are read from stdin.
The job names are expected in the form
"<caller job> on <os>, Python <version>[...] / <called job>",
as given by the regular-test-extras-doc workflows.
The files are written to the folder given as the first argument,
one for each job, OS and Python version, e.g. ``lnumber-ubuntu-latest-3.12.json``.
"""

import json
import pathlib
import re
import sys

JOBS = {
    "Test extra lnumber": "lnumber",
    "Test extra meshing": "meshing",
    "Test doctests and documentation": "doc",
}
RESULTS = {
    "success": ("passing", "brightgreen"),
    "failure": ("failing", "red"),
    "timed_out": ("timed out", "red"),
    "skipped": ("skipped", "lightgrey"),
    "cancelled": ("cancelled", "lightgrey"),
}
PATTERN = re.compile(
    r" on (?P<os>[^,]+), Python (?P<python>[0-9.]+)[^/]* / (?P<job>.+)$"
)

folder = pathlib.Path(sys.argv[1])
folder.mkdir(parents=True, exist_ok=True)
written = 0
for job in json.load(sys.stdin)["jobs"]:
    match = PATTERN.search(job["name"])
    if match is None or match["job"] not in JOBS:
        continue
    message, color = RESULTS.get(
        job["conclusion"], (str(job["conclusion"]), "lightgrey")
    )
    path = folder / f"{JOBS[match['job']]}-{match['os']}-{match['python']}.json"
    badge = {"schemaVersion": 1, "label": "", "message": message, "color": color}
    path.write_text(json.dumps(badge) + "\n")
    print(f"{path}: {message}")
    written += 1
if written == 0:
    sys.exit("No job matching the expected names was found.")
