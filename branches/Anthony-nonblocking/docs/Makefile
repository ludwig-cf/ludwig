############################################################################
#
#  Makefile
#
#  Makefile for documentation
#
#  make dvi
#  make pdf
#  make clean (probably the most useful)
#
#  should both work.
#
#  Edinburgh Soft Matter and Statistical Physics Group and
#  Edinburgh Parallel Computing Centre
#
#  (c) 2014-2015 The University of Edinburgh
#
############################################################################

TEX_SOURCE = ludwig.tex

dvi:	$(TEX_SOURCE) Makefile
	latex $(TEX_SOURCE)

pdf:	$(TEX_SOURCE)
	pdflatex $(TEX_SOURCE)

clean:
	rm -f *pdf *aux *dvi *log *toc *out 
