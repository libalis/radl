PROGRAM := tf
CC := gcc
BUILD_DIR := ./build
C_DIR := ./c
H_DIR := ./h

all: $(BUILD_DIR)/$(PROGRAM)

clean:
	rm -rf $(BUILD_DIR)

debug:
	$(MAKE) CFLAGS="-g -DDEBUG"

$(BUILD_DIR)/$(PROGRAM): $(C_DIR)/io.c $(C_DIR)/main.c $(C_DIR)/matrix.c $(C_DIR)/tf.c $(C_DIR)/timer.c $(H_DIR)/io.h $(H_DIR)/matrix.h $(H_DIR)/tf.h $(H_DIR)/timer.h
	mkdir -p $(BUILD_DIR)
ifdef CFLAGS
	$(CC) $(CFLAGS) -o $@ $^ -lm
else
	$(CC) -o $@ $^ -lm
endif

.PHONY: all clean debug
