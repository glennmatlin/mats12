LATEXMK := latexmk
MAIN    := main.tex
OUTDIR  := build
BASENAME := $(basename $(notdir $(MAIN)))

.PHONY: all watch clean help

help:
	@echo "Targets:"
	@echo "  make           build $(MAIN) → $(OUTDIR)/$(BASENAME).pdf"
	@echo "  make watch     continuous build (latexmk -pvc)"
	@echo "  make clean     remove $(OUTDIR)/ and stray aux files"

all:
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=$(OUTDIR) $(MAIN)

watch:
	$(LATEXMK) -pdf -pvc -interaction=nonstopmode -file-line-error -outdir=$(OUTDIR) $(MAIN)

clean:
	$(LATEXMK) -C -outdir=$(OUTDIR)
	rm -rf $(OUTDIR)
	rm -f $(BASENAME).aux $(BASENAME).bbl $(BASENAME).blg $(BASENAME).fdb_latexmk \
		$(BASENAME).fls $(BASENAME).log $(BASENAME).out $(BASENAME).pdf \
		$(BASENAME).synctex.gz $(BASENAME).toc $(BASENAME).run.xml
