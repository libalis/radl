PROGRAM = main
CC = gcc
CFLAGS = -g
BUILD_DIR = ./build
C_DIR = ./c
H_DIR = ./h

all: $(BUILD_DIR)/$(PROGRAM)

clean:
	rm -rf $(BUILD_DIR)

$(BUILD_DIR)/$(PROGRAM): $(C_DIR)/io.c $(C_DIR)/main.c $(C_DIR)/matrix.c $(C_DIR)/tf.c $(C_DIR)/timer.c $(H_DIR)/io.h $(H_DIR)/matrix.h $(H_DIR)/tf.h $(H_DIR)/timer.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $^ -lm

.PHONY: all clean
