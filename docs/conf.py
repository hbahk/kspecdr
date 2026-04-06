# Configuration file for the Sphinx documentation builder.

import os
import sys
import importlib.util
from docutils import nodes
from docutils.parsers.rst import Directive
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------

project = 'kspecdr'
copyright = '2025-2026, KSPEC Team'
author = 'Hyeonguk Bahk'

version = '0.1.0'
release = '0.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'myst_nb',
]

# Keep autodoc robust on Read the Docs even when optional runtime deps are absent.
autodoc_mock_imports = [
    "astroscrappy",
    "pywt",
    "tqdm",
]

autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

HAS_MERMAID = importlib.util.find_spec("sphinxcontrib.mermaid") is not None
if HAS_MERMAID:
    extensions.append("sphinxcontrib.mermaid")

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

HAS_PYDATA_THEME = importlib.util.find_spec("pydata_sphinx_theme") is not None
html_theme = "pydata_sphinx_theme" if HAS_PYDATA_THEME else "alabaster"
html_theme_options = {
    "show_nav_level": 2,
}
html_static_path = ['_static']
html_logo = '_static/kspec_logo.png'

# -- Options for MyST NB -----------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
]
myst_heading_anchors = 3

nb_execution_mode = 'off'


class MermaidFallbackDirective(Directive):
    """
    Fallback mermaid directive used when sphinxcontrib-mermaid is unavailable.
    Renders the mermaid source as a literal code block so docs still build.
    """

    has_content = True

    def run(self):
        code = "\n".join(self.content)
        literal = nodes.literal_block(code, code)
        literal["classes"].append("mermaid")
        return [literal]


def setup(app):
    if not HAS_MERMAID:
        app.add_directive("mermaid", MermaidFallbackDirective)
