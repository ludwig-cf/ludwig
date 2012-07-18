############################################################################
#
#  Makefile
#
#  make docs
#
############################################################################

TEX_SOURCE = ludwig.tex

LATEX = latex
RM = /bin/rm

docs:	ludwig.tex Makefile
	$(LATEX) $(TEX_SOURCE)

clean:
	$(RM) -f *aux *dvi *log *toc 
