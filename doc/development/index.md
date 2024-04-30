## Tools we use

https://github.com/sphinx-doc/sphinx-autobuild

pip install sphinx-autobuild

in the doc folder:

sphinx-autobuild . _build/html --open-browser

Documentation can be generated from docstrings using `sphinx`
by executing `make html` in the folder `doc`.
The extensions `sphinx-math-dollar`(dollar signs to typeset inline and displayed math expressions), `sphinx_design` (tabs), `sphinx_proof`, `sphinxcontrib.bibtex`, `furo`, `sphinx_copybutton`, `myst-parser`,  have to be installed, for example via `pip install extension_name`.

New `latex` commands can be created by modifying `mathjax3_config` in `doc/conf.py`.
