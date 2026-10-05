paper.pdf: main.tex sections/*.tex
	latexmk -pdf -interaction=nonstopmode -outdir=build main.tex
	cp build/main.pdf paper.pdf

clean:
	rm -rf build paper.pdf

.PHONY: clean
