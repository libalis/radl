#!/bin/bash
PROGRAM="radl"
ICPX="/opt/intel/oneapi/compiler/latest/bin/icpx"
CFLAGS="-O3 $(pkg-config --cflags glib-2.0 | sed 's/ -pthread//g') -Xcompiler -Wno-unused-command-line-argument -Xcompiler -Wno-unused-result -std=c++11"
LDFLAGS="$(pkg-config --libs glib-2.0) -lm"
BUILD_DIR="./build"
UNAME="$(uname -m)"

amx() {
    AMX=$(dialog --title "AMX" --defaultno --yesno \
        "\nDo you want to enable AMX, or use Neon by default?" \
        10 60 3>&1 1>&2 2>&3)
    if [[ $? -eq 0 ]]; then
        CFLAGS="$CFLAGS -DAMX"
    fi
}

architecture() {
    ARCHITECTURE=$(dialog --title "Architecture" --msgbox \
        "\nDetected CPU architecture: $UNAME" \
        10 60 3>&1 1>&2 2>&3)
    if [[ $? -ne 0 ]]; then
        exit
    fi
}

avx() {
    if grep -q "avx512" /proc/cpuinfo; then
        AVX=$(dialog --title "AVX-512" --msgbox \
            "\nAVX-512 support detected" \
            10 60 3>&1 1>&2 2>&3)
        if [[ $? -ne 0 ]]; then
            exit
        fi
        CFLAGS="$CFLAGS -Xcompiler -mavx512f -Xcompiler -mavx512bw -Xcompiler -mavx512vl"
    else
        AVX=$(dialog --title "AVX-512" --msgbox \
            "\nNo AVX-512 support detected\
            \nFalling back to AVX-2" \
            10 60 3>&1 1>&2 2>&3)
        if [[ $? -ne 0 ]]; then
            exit
        fi
        CFLAGS="$CFLAGS -Xcompiler -mavx2"
    fi
}

compiler() {
    COMPILER=$(dialog --title "Compiler" --menu \
        "\nPlease select your desired compiler:" 10 60 2 \
        1 "Clang" \
        2 "ICPX" \
        3>&1 1>&2 2>&3)
    if [[ $? -ne 0 ]]; then
        exit
    elif [[ "$COMPILER" =~ "1" ]]; then
        CC="clang++"
    elif [[ "$COMPILER" =~ "2" ]]; then
        CC="$ICPX"
        CFLAGS="$(CFLAGS)"
    fi
    openmp
}

data_type() {
    DATA_TYPE=$(dialog --title "Data type" --menu \
        "\nPlease select your desired data type:" 10 60 2 \
        1 "Float" \
        2 "Integer" \
        3>&1 1>&2 2>&3)
    if [[ $? -ne 0 ]]; then
        exit
    elif [[ "$DATA_TYPE" =~ "2" ]]; then
        CFLAGS="$CFLAGS -DINT"
    fi
    scale_factor
}

main() {
    MAIN=$(dialog --title "Optimizing AI Workloads for Enhanced Performance" --msgbox \
        "\nA Comprehensive Framework for Comparing CPU and GPU Architectures
        \n\nThe following dialogs will guide you through the process of building the perfect executable for your needs" \
        10 60 3>&1 1>&2 2>&3)
    if [[ $? -ne 0 ]]; then
        exit
    fi
    target
}

openmp() {
    OPENMP=$(dialog --title "OpenMP" --defaultno --yesno \
        "\nDo you want to enable OpenMP, or use pthreads by default?" \
        10 60 3>&1 1>&2 2>&3)
    if [[ $? -eq 0 ]]; then
        if [[ "$UNAME" == "aarch64" || "$UNAME" == "armv7l" || "$UNAME" == "armv6l" || "$UNAME" == "arm64" ]]; then
            CFLAGS="$CFLAGS -Xclang -fopenmp -DOMP"
        elif [[ "$CC" == "$ICPX" ]]; then
            CFLAGS="$CFLAGS -Xcompiler -qopenmp -DOMP"
        else
            CFLAGS="$CFLAGS -Xcompiler -fopenmp -DOMP"
        fi
    else
        CFLAGS="$CFLAGS -Xcompiler -pthread"
        simd
    fi
    data_type
}

options() {
    OPTIONS=$(dialog --title "Optional flags" --checklist \
        "\nPlease select your desired options:" 10 60 1 \
        1 "Debugging" on \
        3>&1 1>&2 2>&3)
    if [[ $? -ne 0 ]]; then
        exit
    elif [[ "$OPTIONS" =~ "1" ]]; then
        CFLAGS="$CFLAGS -g -pg"
        if [[ "$CC" != "nvcc" ]]; then
            CFLAGS="$CFLAGS -fsanitize=address"
        fi
        CFLAGS="$CFLAGS -DDEBUG"
    fi
    summary
}

scale_factor() {
    SCALE_FACTOR=$(dialog --title "Scale factor" \
        --inputbox "\nPlease select your desired scale factor (>= 1):" \
        10 60 1 3>&1 1>&2 2>&3)
    if [[ $? -ne 0 || -z "$SCALE_FACTOR" || ! "$SCALE_FACTOR" =~ ^[1-9]+[0-9]*$ ]]; then
        exit
    elif [[ $SCALE_FACTOR != "1" ]]; then
        CFLAGS="$CFLAGS -DEXPORT=\"\\\"bash -c \"./py/export_xl.py\"\\\"\" \
            -DCONV_BIAS=\"\\\"./tmp/conv_bias.txt\\\"\" \
            -DFC_BIAS=\"\\\"./tmp/fc_bias.txt\\\"\" \
            -DFC_WEIGHTS=\"\\\"./tmp/fc_weights.txt\\\"\" \
            -DMASKS_LEN=\"\\\"./tmp/masks_len.txt\\\"\" \
            -DMASKS=\"\\\"./tmp/masks_%d.txt\\\"\""
        sed -i "/^scale_factor =/c\scale_factor = $SCALE_FACTOR" py/export_xl.py
    fi
    options
}

simd() {
    SIMD=$(dialog --title "SIMD" --defaultno --yesno \
        "\nDo you want to disable SIMD, or use it by default?" \
        10 60 3>&1 1>&2 2>&3)
    if [[ $? -ne 0 ]]; then
        if [[ "$UNAME" == "aarch64" || "$UNAME" == "armv7l" || "$UNAME" == "armv6l" || "$UNAME" == "arm64" ]]; then
            amx
        else
            avx
        fi
    else
        CFLAGS="$CFLAGS -DNO_SIMD"
    fi
}

summary() {
    SUMMARY=$(dialog --title "Summary" --colors --msgbox \
        "\nThe following flags have been selected: \
        \n\nCC=\"\Z1$CC\Zn\" \
        \nCFLAGS=\"\Z2$CFLAGS\Zn\" \
        \nLDFLAGS=\"\Z4$LDFLAGS\Zn\"" \
        20 60 3>&1 1>&2 2>&3)
    if [[ $? -ne 0 ]]; then
        exit
    fi
}

target() {
    TARGET=$(dialog --title "Target" --menu \
        "\nPlease select your desired target:" 10 60 2 \
        1 "CPU" \
        2 "GPU" \
        3>&1 1>&2 2>&3)
    if [[ $? -ne 0 ]]; then
        exit
    elif [[ "$TARGET" =~ "1" ]]; then
        uname
    elif [[ "$TARGET" =~ "2" ]]; then
        CC="nvcc"
        CFLAGS="$CFLAGS -DNVIDIA"
        architecture
        data_type
    fi
}

uname() {
    if [[ "$UNAME" == "x86_64" || "$UNAME" == "i386" || "$UNAME" == "i686" ]]; then
        architecture
        compiler
    elif [[ "$UNAME" == "aarch64" || "$UNAME" == "armv7l" || "$UNAME" == "armv6l" || "$UNAME" == "arm64" ]]; then
        CFLAGS="$CFLAGS -I/opt/homebrew/opt/libomp/include"
        LDFLAGS="$LDFLAGS -L/opt/homebrew/opt/libomp/lib -lomp"
        architecture
        CC="clang++"
        openmp
    else
        ARCHITECTURE=$(dialog --title "Architecture" --msgbox \
            "\nUnknown CPU architecture: $UNAME" \
            10 60 3>&1 1>&2 2>&3)
        exit
    fi
}

main
clear
make clean
make CC="$CC" CFLAGS="$CFLAGS" LDFLAGS="$LDFLAGS"
$BUILD_DIR/$PROGRAM
