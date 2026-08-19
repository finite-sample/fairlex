"""Sphinx configuration — fleet standard via py-canon."""

from py_canon.sphinx import configure

# `configure` injects its settings into this module's namespace, so the two it
# amends below are read back through `globals()` rather than by name: to a
# static reader they are undefined until the call has run.
_ns = globals()
configure(_ns)

# myst_nb renders docs/examples/*.ipynb. It supersedes myst_parser — it loads
# and extends it — so registering both raises "extension already registered".
# Preferred to nbsphinx, which shells out to a pandoc binary that the fleet's
# reusable docs workflow does not install.
_ns["extensions"] = [e for e in _ns["extensions"] if e != "myst_parser"] + ["myst_nb"]

# Without this, napoleon turns `CalibrationResult`'s Attributes section into
# py:attribute descriptions that collide with the ones autodoc generates from
# the dataclass fields, and `sphinx-build -W` fails on the duplicates.
napoleon_use_ivar = True

# The notebook is committed without outputs, so it has to run at build time.
nb_execution_mode = "cache"
nb_execution_timeout = 600
nb_execution_raise_on_error = True

_ns["intersphinx_mapping"] |= {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

html_theme_options = {
    "source_repository": "https://github.com/finite-sample/fairlex",
    "source_branch": "main",
    "source_directory": "docs/",
}
