PROGRAM := tf
CC := gcc
CFLAGS :=
BUILD_DIR := ./build
C_DIR := ./c
H_DIR := ./h

all: $(BUILD_DIR)/$(PROGRAM)

clean:
	rm -rf $(BUILD_DIR)

debug:
	$(MAKE) CFLAGS="$(CFLAGS) -g -DDEBUG"

benchmark:
	$(MAKE) clean
	$(MAKE)
	$(BUILD_DIR)/$(PROGRAM)

$(BUILD_DIR)/$(PROGRAM): $(C_DIR)/io.c $(C_DIR)/main.c $(C_DIR)/matrix.c $(C_DIR)/tf.c $(C_DIR)/timer.c $(H_DIR)/io.h $(H_DIR)/matrix.h $(H_DIR)/tf.h $(H_DIR)/timer.h
	mkdir -p $(BUILD_DIR)
	if [ -n "$(CFLAGS)" ]; then \
        $(CC) $(CFLAGS) -o $@ $^ -lm; \
    else \
        $(CC) -o $@ $^ -lm; \
    fi

.PHONY: all clean debug benchmark
