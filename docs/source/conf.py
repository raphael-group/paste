# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

from pathlib import Path


# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import sys
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent)) 

import src

# -- Project information -----------------------------------------------------

project = 'paste'
copyright = '2022, Raphael Lab'
author = 'Ron Zeira, Max Land, Alexander Strzalkowski, Benjamin J. Raphael'

# The full version, including alpha/beta/rc tags
release = '1.2.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    'sphinx.ext.autosummary', 
    "sphinx.ext.autodoc",
    'sphinx.ext.napoleon',
    "sphinx_autodoc_typehints",
    "nbsphinx",
    "sphinx_gallery.load_style",
    "sphinx.ext.viewcode"
]

# Moves Type hints from function header into description
autodoc_typehints = "description"


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False



# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_show_sourcelink = False
html_show_sphinx = False



nbsphinx_thumbnails = {
    "notebooks/getting-started": "_static/images/breast_stack_2d.png",
}