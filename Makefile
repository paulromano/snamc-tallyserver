# Name of manuscript
manuscript = snamc-tallyserver

# PdfLaTeX compilation options
latexopt = -halt-on-error -file-line-error

# List of images to include
images = baseline.pdf events.pdf mira_cs.pdf mira_r1.pdf mira_r3.pdf \
    mira_r7.pdf mira_r15.pdf mira_cs.pdf model_negative.pdf model.pdf \
    time.pdf

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
