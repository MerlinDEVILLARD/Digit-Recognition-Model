SHELL = powershell -Command
PYTHON = python
ARGS =

help:
	$(SHELL) echo "Makefile for US President Prediction Model project."
	$(SHELL) echo "\thelp:tPrint this."
	$(SHELL) echo "\tclean:\tClean the repository from unecessary files."
	$(SHELL) echo "\ttrain:\tTrain the model. Use args to change the training parameter."
	$(SHELL) echo "\tinf:\tDo the inference of the model. Use args to set the input."

all: help

.PHONY: help all clean train inf