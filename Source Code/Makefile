PROGRAM := tf
CC := clang
ICX := /opt/intel/oneapi/compiler/latest/bin/icpx
CFLAGS := -Wno-unused-command-line-argument
LDFLAGS := $(shell pkg-config --cflags --libs glib-2.0) -lm -lstdc++
BUILD_DIR := ./build
CPP_DIR := ./cpp
CU_DIR := ./cu
HPP_DIR := ./hpp

all: CFLAGS += -pthread
all: $(BUILD_DIR)/$(PROGRAM)

clean:
	rm -rf $(BUILD_DIR)

debug:
	$(MAKE) CFLAGS="$(CFLAGS) \
	-pthread \
	-g -pg -DDEBUG"

omp:
	$(MAKE) CFLAGS="$(CFLAGS) \
	-fopenmp \
	-DOMP \
	-DBENCHMARK=\"\\\"./csv/benchmark_omp.csv\\\"\""

omp_xl:
	$(MAKE) CFLAGS="$(CFLAGS) \
	-fopenmp \
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
	-pthread \
	-DBENCHMARK=\"\\\"./csv/benchmark_xl.csv\\\"\" \
	-DEXPORT=\"\\\"bash -c \"./py/export_xl.py\"\\\"\" \
	-DCONV_BIAS=\"\\\"./tmp/conv_bias.txt\\\"\" \
	-DFC_BIAS=\"\\\"./tmp/fc_bias.txt\\\"\" \
	-DFC_WEIGHTS=\"\\\"./tmp/fc_weights.txt\\\"\" \
	-DMASKS_LEN=\"\\\"./tmp/masks_len.txt\\\"\" \
	-DMASKS=\"\\\"./tmp/masks_%d.txt\\\"\""

intel:
	$(MAKE) CC="$(ICX)" CFLAGS="$(CFLAGS) \
	-pthread \
	-DBENCHMARK=\"\\\"./csv/benchmark_intel.csv\\\"\""

omp_intel:
	$(MAKE) CC="$(ICX)" CFLAGS="$(CFLAGS) \
	-qopenmp \
	-DOMP \
	-DBENCHMARK=\"\\\"./csv/benchmark_omp_intel.csv\\\"\""

omp_xl_intel:
	$(MAKE) CC="$(ICX)" CFLAGS="$(CFLAGS) \
	-qopenmp \
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
	-pthread \
	-DBENCHMARK=\"\\\"./csv/benchmark_xl_intel.csv\\\"\" \
	-DEXPORT=\"\\\"bash -c \"./py/export_xl.py\"\\\"\" \
	-DCONV_BIAS=\"\\\"./tmp/conv_bias.txt\\\"\" \
	-DFC_BIAS=\"\\\"./tmp/fc_bias.txt\\\"\" \
	-DFC_WEIGHTS=\"\\\"./tmp/fc_weights.txt\\\"\" \
	-DMASKS_LEN=\"\\\"./tmp/masks_len.txt\\\"\" \
	-DMASKS=\"\\\"./tmp/masks_%d.txt\\\"\""

nvidia:
	$(MAKE) CC="nvcc" CFLAGS="$(shell echo $(CFLAGS) | sed 's/-Wno-unused-command-line-argument//g') \
	-DNVIDIA \
	-DBENCHMARK=\"\\\"./csv/benchmark_nvidia.csv\\\"\"" \
	LDFLAGS="$(shell echo $(LDFLAGS) | sed 's/-pthread//g')"

xl_nvidia:
	$(MAKE) CC="nvcc" CFLAGS="$(shell echo $(CFLAGS) | sed 's/-Wno-unused-command-line-argument//g') \
	-DNVIDIA \
	-DBENCHMARK=\"\\\"./csv/benchmark_xl_nvidia.csv\\\"\" \
	-DEXPORT=\"\\\"bash -c \"./py/export_xl.py\"\\\"\" \
	-DCONV_BIAS=\"\\\"./tmp/conv_bias.txt\\\"\" \
	-DFC_BIAS=\"\\\"./tmp/fc_bias.txt\\\"\" \
	-DFC_WEIGHTS=\"\\\"./tmp/fc_weights.txt\\\"\" \
	-DMASKS_LEN=\"\\\"./tmp/masks_len.txt\\\"\" \
	-DMASKS=\"\\\"./tmp/masks_%d.txt\\\"\"" \
	LDFLAGS="$(shell echo $(LDFLAGS) | sed 's/-pthread//g')"

benchmark:
	$(MAKE) clean
	$(MAKE)
	$(BUILD_DIR)/$(PROGRAM)
	$(MAKE) clean
	$(MAKE) omp
	$(BUILD_DIR)/$(PROGRAM)
	$(MAKE) clean
	$(MAKE) omp_xl
	$(BUILD_DIR)/$(PROGRAM)
	$(MAKE) clean
	$(MAKE) xl
	$(BUILD_DIR)/$(PROGRAM)

benchmark_intel:
	$(MAKE) benchmark
	$(MAKE) clean
	$(MAKE) intel
	$(BUILD_DIR)/$(PROGRAM)
	$(MAKE) clean
	$(MAKE) omp_intel
	$(BUILD_DIR)/$(PROGRAM)
	$(MAKE) clean
	$(MAKE) omp_xl_intel
	$(BUILD_DIR)/$(PROGRAM)
	$(MAKE) clean
	$(MAKE) xl_intel
	$(BUILD_DIR)/$(PROGRAM)

benchmark_nvidia:
	$(MAKE) benchmark_intel
	$(MAKE) clean
	$(MAKE) nvidia
	$(BUILD_DIR)/$(PROGRAM)
	$(MAKE) clean
	$(MAKE) xl_nvidia
	$(BUILD_DIR)/$(PROGRAM)

$(BUILD_DIR)/$(PROGRAM): $(patsubst $(CPP_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(wildcard $(CPP_DIR)/*.cpp))
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(CPP_DIR)/%.cpp $(HPP_DIR)/%.hpp
	mkdir -p $(BUILD_DIR)
	if [ $(CC) = "nvcc" ] && [ -f $(CU_DIR)/$*.cu ]; then \
	   $(CC) $(CFLAGS) -o $@ -c $(CU_DIR)/$*.cu $(LDFLAGS); \
	else \
	   $(CC) $(CFLAGS) -o $@ -c $< $(LDFLAGS); \
	fi

.PHONY: all clean debug omp omp_xl xl intel omp_intel omp_xl_intel xl_intel nvidia xl_nvidia benchmark benchmark_intel benchmark_nvidia
