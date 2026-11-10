NAME  = practical_significance
BUILD = build
OUT   = ..

all:
	mkdir -p $(BUILD)
	pdflatex -interaction=nonstopmode -output-directory=$(BUILD) $(NAME).tex
	pdflatex -interaction=nonstopmode -output-directory=$(BUILD) $(NAME).tex
	cp $(BUILD)/$(NAME).pdf $(OUT)/$(NAME).pdf

clean:
	rm -rf $(BUILD)