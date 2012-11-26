# Name of manuscript
manuscript = snamc-tallyserver

# PdfLaTeX compilation options
latexopt = -halt-on-error -file-line-error

# List of images to include
images = events.pdf time.pdf model.pdf

#=================================================================
# Generate PDF of manuscript using PdfLaTeX
#=================================================================

all: $(manuscript).pdf

$(manuscript).pdf: $(manuscript).tex $(images) references.bib
	pdflatex $(latexopt) $(manuscript)
	bibtex -terse $(manuscript)
	pdflatex $(latexopt) $(manuscript)
	pdflatex $(latexopt) $(manuscript)

#=================================================================
# Generate Images
#=================================================================

%.pdf: make_plots.py
	python $<

#=================================================================
# Other
#=================================================================

clean:
	@rm -f *.aux *.bbl *.blg *.log *.out *.spl $(manuscript).pdf \
    $(images)

.PHONY: all clean
