#!/bin/bash
PROGRAM="tf"
ICPX="/opt/intel/oneapi/compiler/latest/bin/icpx"
CFLAGS="-O3 $(pkg-config --cflags glib-2.0 | sed 's/ -pthread//g') -Xcompiler -Wno-unused-command-line-argument -Xcompiler -Wno-unused-result"
LDFLAGS="$(pkg-config --libs glib-2.0) -lstdc++ -lm"
BUILD_DIR="./build"
UNAME="$(uname -m)"

architecture() {
    ARCHITECTURE=$(dialog --title "Architecture" --msgbox \
        "\nDetected CPU architecture: $UNAME" \
        10 60 3>&1 1>&2 2>&3)
    if [[ $? -ne 0 ]]; then
        exit
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
        CC="clang"
    elif [[ "$COMPILER" =~ "2" ]]; then
        CC="$ICPX"
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
    INTO=$(dialog --title "Optimizing Deep Learning Performance" --msgbox \
        "\nA Hybrid CPU-GPU Framework with Multithreading, SIMD,\
        \nand Evaluation of Efficiency Metrics\
        \n\nThe following dialogs will guide you through the process of building the perfect executable for your needs" \
        10 60 3>&1 1>&2 2>&3)
    if [[ $? -ne 0 ]]; then
        exit
    fi
    target
}

openmp() {
    OPENMP=$(dialog --yesno --title "OpenMP" --defaultno \
        "\nEnable OpenMP for parallelization, or use pthreads by default?" \
        10 60 3>&1 1>&2 2>&3)
    if $OPENMP; then
        if [[ "$UNAME" == "aarch64" || "$UNAME" == "armv7l" || "$UNAME" == "armv6l" || "$UNAME" == "arm64" ]]; then
            CFLAGS="$CFLAGS -Xclang -fopenmp -DOMP"
        elif [[ "$CC" == "$ICPX" ]]; then
            CFLAGS="$CFLAGS -Xcompiler -qopenmp -DOMP"
        else
            CFLAGS="$CFLAGS -Xcompiler -fopenmp -DOMP"
        fi
    else
        CFLAGS="$CFLAGS -Xcompiler -pthread"
    fi
    data_type
}

options() {
    OPTIONS=$(dialog --title "Optional flags" --checklist \
        "\nPlease select your desired options:" 10 60 1 \
        1 "Debugging" off \
        3>&1 1>&2 2>&3)
    if [[ $? -ne 0 ]]; then
        exit
    elif [[ "$OPTIONS" =~ "1" ]]; then
        CFLAGS="$CFLAGS -g -pg -DDEBUG"
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

summary() {
    SUMMARY=$(dialog --colors --title "Summary" --msgbox \
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
        CFLAGS="$CFLAGS -DNVIDIA -Xcompiler -mavx512f"
        data_type
    fi
}

uname() {
    if [[ "$UNAME" == "x86_64" || "$UNAME" == "i386" || "$UNAME" == "i686" ]]; then
        CFLAGS="$CFLAGS -Xcompiler -mavx512f"
        architecture
        compiler
    elif [[ "$UNAME" == "aarch64" || "$UNAME" == "armv7l" || "$UNAME" == "armv6l" || "$UNAME" == "arm64" ]]; then
        CFLAGS="$CFLAGS -I/opt/homebrew/opt/libomp/include"
        LDFLAGS="$LDFLAGS -L/opt/homebrew/opt/libomp/lib -lomp"
        architecture
        CC="clang"
        openmp
    else
        ARCHITECTURE=$(dialog --title "Architecture" --msgbox \
            "\nUnknown CPU architecture: $UNAME" \
            10 60 3>&1 1>&2 2>&3)
        exit
    fi
}

main
#clear
make clean
make CC="$CC" CFLAGS="$CFLAGS" LDFLAGS="$LDFLAGS"
$BUILD_DIR/$PROGRAM
