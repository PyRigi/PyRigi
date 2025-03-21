(git)=
# Git(flow)

We use [Git](https://git-scm.com/) for version control and the project is hosted at [Github](https://github.com/PyRigi/Pyrigi).
We use [Gitflow](https://nvie.com/posts/a-successful-git-branching-model/) (see also [this description](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)) for PyRigi development.
In a nutshell, this means that there are two prominent branches in PyRigi's Git repository:

- `main`, which contains the stable version of the package
- `dev`, which is used for the development.

Collaborators are not allowed to push their Git commits directly to these two branches.
Rather, they should employ _pull requests_.
Say Alice and Bob want to implement feature X in PyRigi.
These are the tasks to be performed:

1. they branch from `dev`, creating a branch called `feature-X`, and there they develop the intended functionality;
2. once they are done, they push `feature-X` to GitHub and solicit a pull request of `feature-X` into `dev`;
3. After creating a pull request, but before your code is merged into the `dev` branch, the code is checked
by the maintainers, who may ask some other collaborator to serve as reviewer to ensure that the coding
standards and other requirements are satisfied. In a review, Alice and Bob will get comments about specific
pieces of code or other further suggestions. Once they think that they have adequately
addressed a comment, Alica and Bob can use a tick in the GitHub GUI (`:white_check_mark:`✅ or
`:heavy_check_mark:`✔️ ) to indicate that. If the reviewer agrees, they will resolve the comment. 
4. Once the pull request is approved, a maintainer merges `feature-X` into `dev` and during the next version release
cycle, it will be merged into `main`, making the code available through the `pip` installation of PyRigi.


We propose a few categories for contributing branches:
* _features_: branches to implement new features/improvements to the current status; their name should start by `feature-`
* _documentation_: branches to modify the documentation; their name should start by `doc-`, or `docfix-` when solving an error
* _bugs_: branches to solve known bugs; their name should start by `bugfix-`
* _hotfix_: branches to solve an urgent error; their name should start by `hotfix-`
* _testing_: branches to add tests; their name should start by `test-`
* _refactoring_: branches to refactor the code; their name should start by `refactor-`

When a bug is discovered on `main`, the corresponding `bugfix-`/`hotfix-` branch should be created on `main`.


## Version Release

Once in a while, the maintainers merge the branch `dev` into `main` and create a new release.
The release numbers follow this scheme:

* MAJOR version: significant functionality extensions yielding possibly incompatible API changes (x+1.y.z)
* MINOR version: new functionality in a backward compatible manner (x.y+1.z)
* PATCH version: backward compatible bug fixes (x.y.z+1).

To create a new MAJOR/MINOR version, the following steps should be taken by the **maintainers**:

1. Create a release branch `release-x.y.z` on `dev`.
2. Update the `version` and `release` in `doc/conf.py` and the `version` in `pyproject.toml`.
3. Update the `contributors.md`.
4. Merge the branch into `dev`.
5. Continue on the release branch and remove the files that are not supposed to be in the release (e.g. `poetry.lock`).
6. Merge the branch into `main`.
7. Check that the online documentation has been deployed correctly.
8. Add a new release tag in GitHub and generate the corresponding release notes.
9. Review the upload to PyPi.
10. Run `poetry update` and commit `poetry.lock` to update the dependencies on `dev` .

To release a new PATCH version, the following should be taken using some steps from above:
* Create a release branch `release-x.y.z` on `main`.
* Step 2. (and 3. if the patch involves a new contributor).
* Steps 6.-10.
* Create branch `release-x.y.z-main-to-dev` on `dev`.
* Merge `main` into `release-x.y.z-main-to-dev` while keeping the `poetry.lock` file from `release-x.y.z-main-to-dev`.
* Merge the branch `release-x.y.z-main-to-dev` via a PR to `dev`. 