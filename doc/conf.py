"""Sphinx configuration file for an LSST stack package.

This configuration only affects single-package Sphinx documentation builds.
"""
# ruff: noqa: F403, F405

from documenteer.conf.guide import *

exclude_patterns.append("changes/*")

# Citations are resolved against the local refs.bib (documenteer's guide
# configuration does not enable sphinxcontrib-bibtex by default).
# TODO[deferred]: this only covers the single-package build. Revisit when the
# package is mature enough that building its docs together with the rest of
# the pipelines docs matters, in case the integrated build handles citations
# differently.
extensions.append("sphinxcontrib.bibtex")
bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "alpha"
