#include "../hpp/simd.hpp"
#include "../hpp/utils.hpp"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
    #define x86
    #include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
    #define ARM
    #include <arm_neon.h>
#endif

#define VECTOR_SIZE (128)
#define CHUNK_SIZE (VECTOR_SIZE / 8 / sizeof(float))

void add_simd(mt_arg *mt) {
    return;
}

void biasing_simd(mt_arg *mt) {
    #ifdef x86
        __m128 result;
        __m128 accu;
    #else
        float32x4_t result;
        float32x4_t accu;
    #endif
    for(int j = 0; j < mt->a_ptr[mt->m]->y - CHUNK_SIZE; j += CHUNK_SIZE) {
        #ifdef x86
            result = _mm_load_ps(&mt->a_ptr[mt->m]->m[get_idx(mt->i, j, mt->a_ptr[mt->m]->y)]);
            accu = _mm_load_ps1(&mt->b->m[get_idx(mt->m, 0, mt->b->y)]);
            result = _mm_add_ps(result, accu);
            _mm_store_ps(&mt->c_ptr[mt->m]->m[get_idx(mt->i, j, mt->c_ptr[mt->m]->y)], result);
        #else
            result = vld1q_f32(&mt->a_ptr[mt->m]->m[get_idx(mt->i, j, mt->a_ptr[mt->m]->y)]);
            accu = vdupq_n_f32(mt->b->m[get_idx(mt->m, 0, mt->b->y)]);
            result = vaddq_f32(result, accu);
            vst1q_f32(&mt->c_ptr[mt->m]->m[get_idx(mt->i, j, mt->c_ptr[mt->m]->y)], result);
        #endif
    }
    for(int j = mt->a_ptr[mt->m]->y - CHUNK_SIZE; j < mt->a_ptr[mt->m]->y; j++) {
        mt->c_ptr[mt->m]->m[get_idx(mt->i, j, mt->c_ptr[mt->m]->y)] = mt->a_ptr[mt->m]->m[get_idx(mt->i, j, mt->a_ptr[mt->m]->y)] + mt->b->m[get_idx(mt->m, 0, mt->b->y)];
    }
}

void conv2d_simd(mt_arg *mt) {
    return;
}

void flatten_simd(mt_arg *mt) {
    return;
}

void flip_kernels_simd(mt_arg *mt) {
    return;
}

void hyperbolic_tangent_simd(mt_arg *mt) {
    return;
}

void matmul_simd(mt_arg *mt) {
    return;
}

void maxpool_simd(mt_arg *arg) {
    return;
}

void relu_simd(mt_arg *arg) {
    return;
}

void transpose_simd(mt_arg *arg) {
    return;
}
