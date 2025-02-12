import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../toddlerbot"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "toddlerbot"
copyright = "2025, ToddlerBot"
author = "Haochen Shi, Weizhuo(Ken) Wang, Shuran Song, C. Karen Liu"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_tabs.tabs",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "display_item",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_theme_options = {
    "sidebar_hide_name": True,
    "light_logo": "logo_light.png",
    "dark_logo": "logo_dark.png",
}
html_static_path = ["_static"]
# Add the custom CSS file
html_css_files = [
    "custom.css",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "special-members": False,
    "exclude-members": "staticMetaObject",
}
