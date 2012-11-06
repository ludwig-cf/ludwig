BOOK=BookGPU


all:
	pdflatex ${BOOK}
	pdflatex ${BOOK}
	bibtex bu1
	bibtex bu2

	makeindex  ${BOOK}.idx
	pdflatex ${BOOK}
	pdflatex ${BOOK}


clean:
	rm -rf *~ *.aux *.bbl *.blg *.dvi *.lon *.not *.lof *.log *.lot *.toc *.ind *.idx *.ilg 

