# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------

project = 'Comet-emu'
author = 'A. Eggemeier'
release = '1.4.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',    # Include documentation from docstrings
    'sphinx.ext.napoleon',   # Support for NumPy and Google style docstrings
    'sphinx.ext.viewcode',   # Add links to highlighted source code
]
source_suffix = '.rst'
master_doc = 'source/index'
#templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

#html_theme = 'alabaster'
#html_static_path = ['_static']