SUBDIR := $(shell find -mindepth 2 -maxdepth 2 -type f -iname Makefile -exec dirname {} \;)

all:
	$(MAKE) -C "$(SUBDIR)" clean
	$(MAKE) -C "$(SUBDIR)"

.PHONY: all
