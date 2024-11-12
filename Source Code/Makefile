PROGRAM := tf
CC := gcc
ICX := /opt/intel/oneapi/compiler/latest/bin/icx
CFLAGS := $(shell pkg-config --cflags --libs glib-2.0) -pthread -lm
BUILD_DIR := ./build
C_DIR := ./c
H_DIR := ./h

all: $(BUILD_DIR)/$(PROGRAM)

clean:
	rm -rf $(BUILD_DIR)

debug:
	$(MAKE) CFLAGS="$(CFLAGS) -g -DDEBUG"

intel:
	$(MAKE) CC="$(ICX)" CFLAGS="$(CFLAGS) -DBENCHMARK=\"\\\"./csv/benchmark_intel.csv\\\"\""

debug_intel:
	$(MAKE) CC="$(ICX)" CFLAGS="$(CFLAGS) -g -DDEBUG -DBENCHMARK=\"\\\"./csv/benchmark_intel.csv\\\"\""

$(BUILD_DIR)/$(PROGRAM): $(patsubst $(C_DIR)/%.c, $(BUILD_DIR)/%.o, $(wildcard $(C_DIR)/*.c))
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: $(C_DIR)/%.c
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ -c $<

.PHONY: all clean debug intel debug_intel
