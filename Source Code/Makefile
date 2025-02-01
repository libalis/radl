PROGRAM := tf

CC := clang++
ICPX := /opt/intel/oneapi/compiler/latest/bin/icpx
NVCC := nvcc

CFLAGS := -O3 $(shell pkg-config --cflags glib-2.0 | sed 's/ -pthread//g') -Xcompiler -pthread -Xcompiler -Wno-unused-command-line-argument -Xcompiler -Wno-unused-result -std=c++11
LDFLAGS := $(shell pkg-config --libs glib-2.0) -lm

BUILD_DIR := ./build
CPP_DIR := ./cpp
CU_DIR := ./cu
HPP_DIR := ./hpp

UNAME_S := $(shell uname -s)
AVX512 := $(shell grep -m1 -o "avx512" /proc/cpuinfo)

ifneq ($(UNAME_S), Darwin)
	ifneq ($(AVX512),)
	   CFLAGS += -Xcompiler -mavx512f
	else
	   CFLAGS += -Xcompiler -mavx2
	endif
else
	CFLAGS += -I/opt/homebrew/opt/libomp/include
	LDFLAGS += -L/opt/homebrew/opt/libomp/lib -lomp
endif

AMX := -DAMX
DEBUG := -g -pg -fsanitize=address -DDEBUG
INT := -DINT
NVIDIA := -DNVIDIA
OMP := -Xcompiler -fopenmp -DOMP
OMP_DARWIN := -Xclang -fopenmp -DOMP
OMP_INTEL := -Xcompiler -qopenmp -DOMP
XL := -DEXPORT=\"\\\"bash -c \"./py/export_xl.py\"\\\"\" \
	  -DCONV_BIAS=\"\\\"./tmp/conv_bias.txt\\\"\" \
	  -DFC_BIAS=\"\\\"./tmp/fc_bias.txt\\\"\" \
	  -DFC_WEIGHTS=\"\\\"./tmp/fc_weights.txt\\\"\" \
	  -DMASKS_LEN=\"\\\"./tmp/masks_len.txt\\\"\" \
	  -DMASKS=\"\\\"./tmp/masks_%d.txt\\\"\"

all: $(BUILD_DIR)/$(PROGRAM)

clean:
	rm -rf $(BUILD_DIR)

config:
	sh/config.sh

debug:
	$(MAKE) CFLAGS="$(CFLAGS) $(DEBUG)"

ifneq ($(UNAME_S), Darwin)
omp:
	$(MAKE) CFLAGS="$(CFLAGS) $(OMP) \
	-DBENCHMARK=\"\\\"./csv/benchmark_omp.csv\\\"\""
else
omp:
	$(MAKE) CFLAGS="$(CFLAGS) $(OMP_DARWIN) \
	-DBENCHMARK=\"\\\"./csv/benchmark_omp.csv\\\"\""
endif

ifneq ($(UNAME_S), Darwin)
omp_xl:
	$(MAKE) CFLAGS="$(CFLAGS) $(OMP) $(XL) \
	-DBENCHMARK=\"\\\"./csv/benchmark_omp_xl.csv\\\"\""
else
omp_xl:
	$(MAKE) CFLAGS="$(CFLAGS) $(OMP_DARWIN) $(XL) \
	-DBENCHMARK=\"\\\"./csv/benchmark_omp_xl.csv\\\"\""
endif

xl:
	$(MAKE) CFLAGS="$(CFLAGS) $(XL) \
	-DBENCHMARK=\"\\\"./csv/benchmark_xl.csv\\\"\""

intel:
	$(MAKE) CC="$(ICPX)" CFLAGS="$(CFLAGS) \
	-DBENCHMARK=\"\\\"./csv/benchmark_intel.csv\\\"\""

omp_intel:
	$(MAKE) CC="$(ICPX)" CFLAGS="$(CFLAGS) $(OMP_INTEL) \
	-DBENCHMARK=\"\\\"./csv/benchmark_omp_intel.csv\\\"\""

omp_xl_intel:
	$(MAKE) CC="$(ICPX)" CFLAGS="$(CFLAGS) $(OMP_INTEL) $(XL) \
	-DBENCHMARK=\"\\\"./csv/benchmark_omp_xl_intel.csv\\\"\""

xl_intel:
	$(MAKE) CC="$(ICPX)" CFLAGS="$(CFLAGS) $(XL) \
	-DBENCHMARK=\"\\\"./csv/benchmark_xl_intel.csv\\\"\""

nvidia:
	$(MAKE) CC="$(NVCC)" CFLAGS="$(CFLAGS) $(NVIDIA) \
	-DBENCHMARK=\"\\\"./csv/benchmark_nvidia.csv\\\"\""

xl_nvidia:
	$(MAKE) CC="$(NVCC)" CFLAGS="$(CFLAGS) $(NVIDIA) $(XL) \
	-DBENCHMARK=\"\\\"./csv/benchmark_xl_nvidia.csv\\\"\""

debug_amx:
	$(MAKE) CFLAGS="$(CFLAGS) $(AMX) $(DEBUG) \
	-DBENCHMARK=\"\\\"./csv/benchmark_amx.csv\\\"\""

amx:
	$(MAKE) CFLAGS="$(CFLAGS) $(AMX) \
	-DBENCHMARK=\"\\\"./csv/benchmark_amx.csv\\\"\""

amx_xl:
	$(MAKE) CFLAGS="$(CFLAGS) $(AMX) $(XL) \
	-DBENCHMARK=\"\\\"./csv/benchmark_amx_xl.csv\\\"\""

int:
	$(MAKE) CFLAGS="$(CFLAGS) $(INT) \
	-DBENCHMARK=\"\\\"./csv/benchmark_int.csv\\\"\""

int_xl:
	$(MAKE) CFLAGS="$(CFLAGS) $(INT) $(XL) \
	-DBENCHMARK=\"\\\"./csv/benchmark_int_xl.csv\\\"\""

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
	$(MAKE) clean
	$(MAKE) int
	$(BUILD_DIR)/$(PROGRAM)
	$(MAKE) clean
	$(MAKE) int_xl
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

benchmark_darwin:
	$(MAKE) benchmark
	$(MAKE) clean
	$(MAKE) amx
	$(BUILD_DIR)/$(PROGRAM)
	$(MAKE) clean
	$(MAKE) amx_xl
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

.PHONY: all clean config debug omp omp_xl xl intel omp_intel omp_xl_intel xl_intel nvidia xl_nvidia debug_amx amx amx_xl int int_xl benchmark benchmark_intel benchmark_nvidia benchmark_darwin
