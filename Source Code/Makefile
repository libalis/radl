PROGRAM := tf
CC := gcc
CFLAGS := $(shell pkg-config --cflags --libs glib-2.0) -pthread -lm
BUILD_DIR := ./build
C_DIR := ./c
H_DIR := ./h

IO := $(C_DIR)/io.c $(H_DIR)/io.h
MAIN := $(C_DIR)/main.c $(H_DIR)/main.h
MATRIX := $(C_DIR)/matrix.c $(H_DIR)/matrix.h
MT := $(C_DIR)/mt.c $(H_DIR)/mt.h
TF := $(C_DIR)/tf.c $(H_DIR)/tf.h
TIMER := $(C_DIR)/timer.c $(H_DIR)/timer.h
UTILS := $(C_DIR)/utils.c $(H_DIR)/utils.h

all: $(BUILD_DIR)/$(PROGRAM)

clean:
	rm -rf $(BUILD_DIR)

debug:
	$(MAKE) CFLAGS="$(CFLAGS) -g -DDEBUG"

$(BUILD_DIR)/$(PROGRAM): $(IO) $(MAIN) $(MATRIX) $(MT) $(TF) $(TIMER) $(UTILS)
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $^

.PHONY: all clean debug
