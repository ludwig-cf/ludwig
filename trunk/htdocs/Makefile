############################################################################
#
#  Makefile
#
#  make docs
#
############################################################################

TEX_SOURCE = ludwig.tex

LATEX = pdflatex
RM = /bin/rm

docs:	ludwig.tex Makefile
	$(LATEX) $(TEX_SOURCE)

clean:
	$(RM) -f *aux *dvi *log *toc 
