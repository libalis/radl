PROGRAM := radl
CC := gcc
ICX := /opt/intel/oneapi/compiler/latest/bin/icx
CFLAGS := $(shell pkg-config --cflags --libs glib-2.0) -pthread
LDFLAGS := -lm
BUILD_DIR := ./build
C_DIR := ./c
H_DIR := ./h

all: $(BUILD_DIR)/$(PROGRAM)

clean:
	rm -rf $(BUILD_DIR)

debug:
	$(MAKE) CFLAGS="$(CFLAGS) \
	-g -DDEBUG"

xl:
	$(MAKE) CFLAGS="$(CFLAGS) \
	-DBENCHMARK=\"\\\"./csv/benchmark_xl.csv\\\"\" \
	-DEXPORT=\"\\\"bash -c \"./py/export_xl.py\"\\\"\" \
	-DCONV_BIAS=\"\\\"./tmp/conv_bias.txt\\\"\" \
	-DFC_BIAS=\"\\\"./tmp/fc_bias.txt\\\"\" \
	-DFC_WEIGHTS=\"\\\"./tmp/fc_weights.txt\\\"\" \
	-DMASKS_LEN=\"\\\"./tmp/masks_len.txt\\\"\" \
	-DMASKS=\"\\\"./tmp/masks_%d.txt\\\"\""

intel:
	$(MAKE) CC="$(ICX)" CFLAGS="$(CFLAGS) \
	-DBENCHMARK=\"\\\"./csv/benchmark_intel.csv\\\"\""

debug_intel:
	$(MAKE) CC="$(ICX)" CFLAGS="$(CFLAGS) \
	-g -DDEBUG \
	-DBENCHMARK=\"\\\"./csv/benchmark_intel.csv\\\"\""

xl_intel:
	$(MAKE) CC="$(ICX)" CFLAGS="$(CFLAGS) \
	-DBENCHMARK=\"\\\"./csv/benchmark_xl_intel.csv\\\"\" \
	-DEXPORT=\"\\\"bash -c \"./py/export_xl.py\"\\\"\" \
	-DCONV_BIAS=\"\\\"./tmp/conv_bias.txt\\\"\" \
	-DFC_BIAS=\"\\\"./tmp/fc_bias.txt\\\"\" \
	-DFC_WEIGHTS=\"\\\"./tmp/fc_weights.txt\\\"\" \
	-DMASKS_LEN=\"\\\"./tmp/masks_len.txt\\\"\" \
	-DMASKS=\"\\\"./tmp/masks_%d.txt\\\"\""

benchmark:
	$(MAKE) clean
	$(MAKE)
	$(BUILD_DIR)/$(PROGRAM)
	$(MAKE) clean
	$(MAKE) xl
	$(BUILD_DIR)/$(PROGRAM)

benchmark_intel:
	$(MAKE) clean
	$(MAKE) intel
	$(BUILD_DIR)/$(PROGRAM)
	$(MAKE) clean
	$(MAKE) xl_intel
	$(BUILD_DIR)/$(PROGRAM)

$(BUILD_DIR)/$(PROGRAM): $(patsubst $(C_DIR)/%.c, $(BUILD_DIR)/%.o, $(wildcard $(C_DIR)/*.c))
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(C_DIR)/%.c $(H_DIR)/%.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ -c $< $(LDFLAGS)

.PHONY: all clean debug xl intel debug_intel xl_intel benchmark benchmark_intel
