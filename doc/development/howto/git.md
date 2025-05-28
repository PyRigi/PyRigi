(git)=
# Git(flow)

We use [Git](https://git-scm.com/) for version control and the project is hosted at [Github](https://github.com/PyRigi/Pyrigi).
We use [Gitflow](https://nvie.com/posts/a-successful-git-branching-model/) (see also [this description](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)) for PyRigi development.
In a nutshell, this means that there are two prominent branches in PyRigi's Git repository:

- `main`, which contains the stable version of the package
- `dev`, which is used for the development.

Collaborators are not allowed to push their Git commits directly to these two branches.
Rather, they should employ _pull requests_ (PR).
Say Alice and Bob want to implement feature X in PyRigi.
These are the tasks to be performed:

1. they branch from `dev`, creating a branch called `feature-X`, and there they develop the intended functionality;
2. once they are done, they push `feature-X` to GitHub and solicit a pull request of `feature-X` into `dev`;
3. After creating a pull request, but before the branch is merged into `dev`, the code is checked
by the maintainers, who may ask some other collaborator to serve as reviewer to ensure that the coding
standards and other requirements are satisfied. In a review, Alice and Bob will get comments about specific
pieces of code or other further suggestions. Once they think that they have adequately
addressed a comment, Alice and Bob can use a tick in the GitHub GUI (`:white_check_mark:`✅ or
`:heavy_check_mark:`✔️ ) to indicate that. If the reviewer agrees, they will resolve the comment.
4. Once the pull request is approved, a maintainer merges `feature-X` into `dev` and during the next version release
cycle, it will be merged into `main`, making the code available through the `pip` installation of PyRigi.

The branch prefixes should be named according the following convention:

| Branch prefix | Usage                                                    | Usual PR prefix                |
|---------------|----------------------------------------------------------|--------------------------------|
| `feature-`    | adding new features, parameters or algorithms\*          | `Feature:`                     |
| `bugfix-`     | fixing bugs on `dev` introduced since the latest release | `Feature:`                     |
| `doc-`        | changing the documentation (including fixes)             | `Doc:`, `Guide:` or `Minor`    |
| `hotfix-`     | fixing bugs on `main`                                    | `Fix:`                         |
| `test-`       | adding and refactoring tests                             | `Test:` or `Minor:`            |
| `refactor-`   | refactoring the package source code\*                    | `Code:`, `Minor:` or `Update:` |
| `release-`    | creating new release                                     | `Setup:`                       |
| `setup-`      | changing technical setting (e.g. GitHub workflows)       | `Setup:`                       |
| `major-`      | introducing backward incompatible changes                | `Update:`                      |

\* Only in backward compatible manner.

:::{warning}
Changes that are not backward compatible can only be introduced on a `major-` branch.
When such a branch is merged to `dev`, only a major release can follow.
:::

## Pull request titles

The pull request titles are used for [Release Notes](https://github.com/PyRigi/PyRigi/releases).
Hence, each PR title has to start with one of the prefixes from the table above
followed by a capitalized verb as a past participle.
The maintainer who approves and merges a PR is responsible for checking (and possible adjusting)
the prefix to match the section in which it should be listed in the Release Notes according to the following tables.
The first part of Release Notes is aimed at users:

| PR prefix  | Release notes section | Information about                           |
|------------|-----------------------|---------------------------------------------|
| `Feature:` | New features          | new functionality                           |
| `Update:`  | Updates               | improvements, interface changes             |
| `Fix:`     | Bug fixes             | fixed bugs since the last release           |
| `Doc:`     | Documentation         | changes to User guide or Math documentation |

The second part is meant for developers:

| PR prefix | Release notes section | Information about                      |
|-----------|-----------------------|----------------------------------------|
| `Test:`   | Testing               | new or changed tests                   |
| `Setup:`  | Technical setup       | changes to technical setting           |
| `Code:`   | Code changes          | refactoring and restructuring the code |
| `Guide:`  | Development guide     | changes to Development guide           |
| `Minor:`  | Minor changes         | fixed typos, minor refactoring etc.    |

The release notes generated on GitHub stored in `doc/release_notes.txt` can be automatically
grouped according to the prefixes by `python sort_release_notes.py release_notes.txt` in the `doc` folder.
After that, manual adjustments should be made.
For instance:

* If a feature was developed (and bug fixed) on several PRs,
  the latter should be listed under the same item under New features.
* If a bug is discovered on `main`, only hot fixed on `main` by disabling the functionality
  and properly fixed on `dev` via a `feature-` branch, all corresponding PRs should be in the same item under Bug fixes.

## Version Release

Once in a while, the maintainers merge the branch `dev` into `main` and create a new release.
The release numbers follow this scheme:

* MAJOR version: significant functionality extensions yielding possibly incompatible API changes (x+1.y.z)
* MINOR version: new functionality in a backward compatible manner (x.y+1.z)
* PATCH version: backward compatible bug fixes (x.y.z+1).

To create a new MAJOR/MINOR version, the following steps should be taken by the **maintainers**:

1. Create a release branch `release-x.y.z` on `dev`.
2. Update the `version` and `release` in `doc/conf.py` and the `version` in `pyproject.toml` and `.zenodo.json`.
3. Update the `contributors.md`.
4. Merge the branch into `dev`.
5. Continue on the release branch and remove the files that are not supposed to be in the release (e.g. `poetry.lock`).
6. Merge the branch into `main`.
7. Check that the online documentation has been deployed correctly.
8. Add a new release tag in GitHub and generate the corresponding release notes according to the instructions above.
9. Review the upload to PyPi.
10. Run `poetry update` and commit `poetry.lock` to update the dependencies on `dev` .

To release a new PATCH version, the following should be taken using some steps from above:
* Create a release branch `release-x.y.z` on `main`.
* Step 2. (and 3. if the patch involves a new contributor).
* Steps 6.-9.
* Pull `main`.
* Create branch `release-x.y.z-main-to-dev` on `dev`.
* Merge `main` into `release-x.y.z-main-to-dev` while keeping the `poetry.lock` file from `release-x.y.z-main-to-dev`.
* Merge the branch `release-x.y.z-main-to-dev` via a PR to `dev`.
