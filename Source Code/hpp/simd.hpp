#ifndef SIMD_H
    #define SIMD_H

    #if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
        #define x86
        #include <immintrin.h>
        #define zero (_mm_setzero_ps())
    #elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
        #define ARM
        #include <arm_neon.h>
    #endif

    #define VECTOR_SIZE (128)
    #define CHUNK_SIZE (VECTOR_SIZE / 8 / sizeof(float))

    #include "../hpp/mt.hpp"

    void add_simd(mt_arg *mt);
    void biasing_simd(mt_arg *mt);
    void conv2d_simd(mt_arg *mt);
    void matmul_simd(mt_arg *mt);
#endif
