PROGRAM := tf
CC := g++
ICX := /opt/intel/oneapi/compiler/latest/bin/icpx
CFLAGS := $(shell pkg-config --cflags --libs glib-2.0) -pthread
LDFLAGS := -lm
BUILD_DIR := ./build
CPP_DIR := ./cpp
HPP_DIR := ./hpp

all: $(BUILD_DIR)/$(PROGRAM)

clean:
	rm -rf $(BUILD_DIR)

debug:
	$(MAKE) CFLAGS="$(CFLAGS) \
	-g -pg -DDEBUG"

omp:
	$(MAKE) CFLAGS="$(shell pkg-config --cflags --libs glib-2.0) -fopenmp \
	-DOMP \
	-DBENCHMARK=\"\\\"./csv/benchmark_omp.csv\\\"\""

omp_xl:
	$(MAKE) CFLAGS="$(shell pkg-config --cflags --libs glib-2.0) -fopenmp \
	-DOMP \
	-DBENCHMARK=\"\\\"./csv/benchmark_omp_xl.csv\\\"\" \
	-DEXPORT=\"\\\"bash -c \"./py/export_xl.py\"\\\"\" \
	-DCONV_BIAS=\"\\\"./tmp/conv_bias.txt\\\"\" \
	-DFC_BIAS=\"\\\"./tmp/fc_bias.txt\\\"\" \
	-DFC_WEIGHTS=\"\\\"./tmp/fc_weights.txt\\\"\" \
	-DMASKS_LEN=\"\\\"./tmp/masks_len.txt\\\"\" \
	-DMASKS=\"\\\"./tmp/masks_%d.txt\\\"\""

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

omp_intel:
	$(MAKE) CC="$(ICX)" CFLAGS="$(shell pkg-config --cflags --libs glib-2.0) -qopenmp \
	-DOMP \
	-DBENCHMARK=\"\\\"./csv/benchmark_omp_intel.csv\\\"\""

omp_xl_intel:
	$(MAKE) CC="$(ICX)" CFLAGS="$(shell pkg-config --cflags --libs glib-2.0) -qopenmp \
	-DOMP \
	-DBENCHMARK=\"\\\"./csv/benchmark_omp_xl_intel.csv\\\"\" \
	-DEXPORT=\"\\\"bash -c \"./py/export_xl.py\"\\\"\" \
	-DCONV_BIAS=\"\\\"./tmp/conv_bias.txt\\\"\" \
	-DFC_BIAS=\"\\\"./tmp/fc_bias.txt\\\"\" \
	-DFC_WEIGHTS=\"\\\"./tmp/fc_weights.txt\\\"\" \
	-DMASKS_LEN=\"\\\"./tmp/masks_len.txt\\\"\" \
	-DMASKS=\"\\\"./tmp/masks_%d.txt\\\"\""

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
	./$(BUILD_DIR)/$(PROGRAM)
	$(MAKE) clean
	$(MAKE) omp
	./$(BUILD_DIR)/$(PROGRAM)
	$(MAKE) clean
	$(MAKE) omp_xl
	./$(BUILD_DIR)/$(PROGRAM)
	$(MAKE) clean
	$(MAKE) xl
	./$(BUILD_DIR)/$(PROGRAM)

benchmark_intel:
	$(MAKE) benchmark
	$(MAKE) clean
	$(MAKE) intel
	./$(BUILD_DIR)/$(PROGRAM)
	$(MAKE) clean
	$(MAKE) omp_intel
	./$(BUILD_DIR)/$(PROGRAM)
	$(MAKE) clean
	$(MAKE) omp_xl_intel
	./$(BUILD_DIR)/$(PROGRAM)
	$(MAKE) clean
	$(MAKE) xl_intel
	./$(BUILD_DIR)/$(PROGRAM)

$(BUILD_DIR)/$(PROGRAM): $(patsubst $(CPP_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(wildcard $(CPP_DIR)/*.cpp))
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(CPP_DIR)/%.cpp $(HPP_DIR)/%.hpp
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ -c $< $(LDFLAGS)

.PHONY: all clean debug omp omp_xl xl intel omp_intel omp_xl_intel xl_intel benchmark benchmark_intel
