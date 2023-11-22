# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sphinx_bootstrap_theme

project = 'mvcmi'
copyright = '2023, Mainak Jas'
author = 'Mainak Jas'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'numpydoc',
    'sphinx_copybutton',
    'sphinxcontrib.jquery',
]

autosummary_generate = True
autodoc_default_options = {'inherited-members': None}
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = True
default_role = 'autolink'

# Sphinx-Copybutton configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store',
                    "auto_examples/*.ipynb",
                    "auto_examples/*.py"]

source_suffix = '.rst'
master_doc = 'index'

pygments_style = 'sphinx'

sphinx_gallery_conf = {
    "doc_module": "mvcmi",
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    'backreferences_dir': 'generated',
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/devdocs', None),
    'scipy': ('https://scipy.github.io/devdocs', None),
    'matplotlib': ('https://matplotlib.org', None)
}
intersphinx_timeout = 5

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'bootstrap'
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

html_theme_options = {
    'navbar_sidebarrel': False,
    'navbar_links': [
        ("Examples", "auto_examples/index"),
        ("API", "api"),
        ("GitHub", "https://github.com/mvcmi/mvcmi/", True)
    ],
    'bootswatch_theme': "yeti",
    'globaltoc_depth': -1,
}
