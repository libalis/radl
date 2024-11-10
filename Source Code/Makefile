PROGRAM := tf
CC := gcc
BUILD_DIR := ./build
C_DIR := ./c
H_DIR := ./h

IO := $(C_DIR)/io.c $(H_DIR)/io.h
MAIN := $(C_DIR)/main.c
MATRIX := $(C_DIR)/matrix.c $(H_DIR)/matrix.h
MT := $(C_DIR)/mt.c $(H_DIR)/mt.h
TF := $(C_DIR)/tf.c $(H_DIR)/tf.h
TIMER := $(C_DIR)/timer.c $(H_DIR)/timer.h
UTILS := $(C_DIR)/utils.c $(H_DIR)/utils.h

all: $(BUILD_DIR)/$(PROGRAM)

clean:
	rm -rf $(BUILD_DIR)

debug:
	$(MAKE) CFLAGS="-g -DDEBUG"

$(BUILD_DIR)/$(PROGRAM): $(IO) $(MAIN) $(MATRIX) $(MT) $(TF) $(TIMER) $(UTILS)
	mkdir -p $(BUILD_DIR)
ifdef CFLAGS
	$(CC) $(CFLAGS) -o $@ $^ -pthread -lm
else
	$(CC) -o $@ $^ -pthread -lm
endif

.PHONY: all clean debug
