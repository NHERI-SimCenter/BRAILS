# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.abspath('./sphinx_ext/'))
from datetime import datetime
#sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'BRAILS'
copyright = f"{str(datetime.today().year)}, The Regents of the University of California"
author = 'Barbaros Cetiner, Chaofeng Wang, Frank McKenna, Sascha Hornauer, Yunhui Guo'
# The short X.Y version
#version = '1.0'
# The full version, including alpha/beta/rc tags
release = '3.0.0'

rst_prolog = """
.. |app| replace:: BRAILS
.. |appName| replace:: BRAILS
.. |s3harkName| replace:: SURF
.. |surfName| replace:: SURF
.. |brailsName| replace:: BRAILS
.. |full tool name| replace:: Building Recognition using AI at Large-Scale
.. _MessageBoard: https://simcenter-messageboard.designsafe-ci.org/smf/index.php?board=10.0
.. |messageBoard| replace:: `MessageBoard`_
.. |short tool name| replace:: BRAILS
.. |short tool id| replace:: BRAILS
.. |tool github link| replace:: `BRAILS Github page`_
.. _brails Github page: https://github.com/NHERI-SimCenter/BRAILS
.. |tool version| replace:: 3.0.0
.. |SimCenter| replace:: `SimCenter`_
.. _SimCenter: https://simcenter.designsafe-ci.org/

.. |EE-UQ short name| replace:: EE-UQ app
.. |EE-UQ app link| replace:: `EE-UQ app`_
.. _EE-UQ app: https://simcenter.designsafe-ci.org/research-tools/ee-uq-application/
.. |user survey link| replace:: `user survey`_
.. _user survey: https://docs.google.com/forms/d/e/1FAIpQLSfh20kBxDmvmHgz9uFwhkospGLCeazZzL770A2GuYZ2KgBZBA/viewform?usp=sf_link
"""

rst_prolog += f"""
.. |developers| replace:: {", ".join(f"**{auth}** " for auth in author.split(", "))}
"""

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions =  [
    "sphinx-jsonschema",
    "sphinxcontrib.bibtex",
    "toctree_filter",
    "sphinxcontrib.images",
    "sphinx.ext.extlinks",
    "sphinxcontrib.images",
    "rendre.sphinx",
    "sphinx.ext.autodoc",
    "crate.sphinx.csv",
    "sphinx_panels",
    #"sphinxcontrib.spelling",
    'sphinx_toolbox.collapse',
    'sphinx.ext.todo',
    'sphinx.ext.githubpages',
    'sphinxcontrib.bibtex',
    'toctree_filter',
    'rst2pdf.pdfbuilder',
    'sphinx.ext.mathjax'
]

pdf_documents = [('index', u'rst2pdf', u'BRAILS', u'NHERI SimCenter'),]
bibtex_bibfiles = ['common/technical_manual/references.bib']

mathjax_path="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}
#---sphinx-themes-----
html_theme = 'sphinx_rtd_theme' # press

html_theme_options = {
	'logo_only': True,
	'style_nav_header_background': '#F2F2F2', #64B5F6 #607D8B
}

html_logo = 'images/logo/SimCenter_BRAILS_logo_solo.png'

numfig = True
numfig_secnum_depth = 2

html_css_files = [
	'custom.css'
]

# you need to modify custom.js for different git channels
html_js_files = [
    #'custom.js',
	#'https://sidecar.gitter.im/dist/sidecar.v1.js'
]

