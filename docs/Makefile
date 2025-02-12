# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
SPHINXAPIDOC  = sphinx-apidoc
PACKAGE       = ../toddlerbot
APIDIR        = api

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile apidoc

# Generate API documentation before building HTML
apidoc:
	@mkdir -p "$(SOURCEDIR)/$(APIDIR)"
	@$(SPHINXAPIDOC) -o "$(SOURCEDIR)/$(APIDIR)" "$(PACKAGE)"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option. $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile apidoc
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
