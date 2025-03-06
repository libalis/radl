#ifndef SIMD_HPP
    #define SIMD_HPP

    #ifdef INT
        #include "matrix_int8.hpp"
    #endif
    #include "mt_arg.hpp"
    #include "utils.hpp"

    #if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
        #define x86
        #include <immintrin.h>
    #elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
        #include <arm_neon.h>
        #ifdef AMX
            #include "../amx/aarch64.h"
        #endif
    #endif

    __attribute__((always_inline)) inline void add_simd(mt_arg *mt) {
        #ifdef x86
            #ifdef __AVX512F__
                #ifdef INT
                    int CHUNK_SIZE = sizeof(__m512i) / sizeof(DATA_TYPE);
                    __m512i a, b;
                    __mmask16 m;
                    for(int j = 0; j < (*mt->c)->y; j += CHUNK_SIZE) {
                        m = (__mmask16)((1 << (((j + CHUNK_SIZE) <= (*mt->c)->y) ? CHUNK_SIZE : (*mt->c)->y - j)) - 1);
                        a = _mm512_maskz_loadu_epi32(m, &((matrix*)*mt->a)->m[get_idx(mt->i, j, ((matrix*)*mt->a)->y)]);
                        b = _mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(m, &((matrix_int8*)*mt->b)->m[get_idx(mt->i, j, ((matrix*)*mt->b)->y)]));
                        _mm512_mask_storeu_epi32(&(*mt->c)->m[get_idx(mt->i, j, (*mt->c)->y)], m, _mm512_add_epi32(a, b));
                    }
                #else
                    int CHUNK_SIZE = sizeof(__m512) / sizeof(DATA_TYPE);
                    __m512 a, b;
                    __mmask16 m;
                    for(int j = 0; j < (*mt->c)->y; j += CHUNK_SIZE) {
                        m = (__mmask16)((1 << (((j + CHUNK_SIZE) <= (*mt->c)->y) ? CHUNK_SIZE : (*mt->c)->y - j)) - 1);
                        a = _mm512_maskz_loadu_ps(m, &((matrix*)*mt->a)->m[get_idx(mt->i, j, ((matrix*)*mt->a)->y)]);
                        b = _mm512_maskz_loadu_ps(m, &((matrix*)*mt->b)->m[get_idx(mt->i, j, ((matrix*)*mt->b)->y)]);
                        _mm512_mask_storeu_ps(&(*mt->c)->m[get_idx(mt->i, j, (*mt->c)->y)], m, _mm512_add_ps(a, b));
                    }
                #endif
            #else
                #ifdef INT
                    int CHUNK_SIZE = sizeof(__m256i) / sizeof(DATA_TYPE);
                    __m256i a, b;
                    for(int j = 0; j + CHUNK_SIZE - 1 < (*mt->c)->y; j += CHUNK_SIZE) {
                        a = _mm256_loadu_si256((__m256i*)&((matrix*)*mt->a)->m[get_idx(mt->i, j, ((matrix*)*mt->a)->y)]);
                        b = _mm256_cvtepi8_epi32(_mm_loadu_si64(&((matrix_int8*)*mt->b)->m[get_idx(mt->i, j, ((matrix*)*mt->b)->y)]));
                        _mm256_storeu_si256((__m256i*)&(*mt->c)->m[get_idx(mt->i, j, (*mt->c)->y)], _mm256_add_epi32(a, b));
                    }
                    for(int j = (*mt->c)->y - ((*mt->c)->y % CHUNK_SIZE); j < (*mt->c)->y; j++) {
                        (*mt->c)->m[get_idx(mt->i, j, (*mt->c)->y)] = ((matrix*)*mt->a)->m[get_idx(mt->i, j, ((matrix*)*mt->a)->y)] + ((matrix_int8*)*mt->b)->m[get_idx(mt->i, j, ((matrix*)*mt->b)->y)];
                    }
                #else
                    int CHUNK_SIZE = sizeof(__m256) / sizeof(DATA_TYPE);
                    __m256 a, b;
                    for(int j = 0; j + CHUNK_SIZE - 1 < (*mt->c)->y; j += CHUNK_SIZE) {
                        a = _mm256_loadu_ps(&((matrix*)*mt->a)->m[get_idx(mt->i, j, ((matrix*)*mt->a)->y)]);
                        b = _mm256_loadu_ps(&((matrix*)*mt->b)->m[get_idx(mt->i, j, ((matrix*)*mt->b)->y)]);
                        _mm256_storeu_ps(&(*mt->c)->m[get_idx(mt->i, j, (*mt->c)->y)], _mm256_add_ps(a, b));
                    }
                    for(int j = (*mt->c)->y - ((*mt->c)->y % CHUNK_SIZE); j < (*mt->c)->y; j++) {
                        (*mt->c)->m[get_idx(mt->i, j, (*mt->c)->y)] = ((matrix*)*mt->a)->m[get_idx(mt->i, j, ((matrix*)*mt->a)->y)] + ((matrix*)*mt->b)->m[get_idx(mt->i, j, ((matrix*)*mt->b)->y)];
                    }
                #endif
            #endif
        #else
            #ifdef INT
                int CHUNK_SIZE = 8;
                int32x4_t a1, a2, b1, b2, c1, c2;
                for(int j = 0; j + CHUNK_SIZE - 1 < (*mt->c)->y; j += CHUNK_SIZE) {
                    a1 = vld1q_s32(&((matrix*)*mt->a)->m[get_idx(mt->i, j, ((matrix*)*mt->a)->y)]);
                    a2 = vld1q_s32(&((matrix*)*mt->a)->m[get_idx(mt->i, j + 4, ((matrix*)*mt->a)->y)]);
                    b1 = vmovl_s16(vget_low_s16(vmovl_s8(vld1_s8(&((matrix_int8*)*mt->b)->m[get_idx(mt->i, j, ((matrix*)*mt->b)->y)]))));
                    b2 = vmovl_s16(vget_high_s16(vmovl_s8(vld1_s8(&((matrix_int8*)*mt->b)->m[get_idx(mt->i, j, ((matrix*)*mt->b)->y)]))));
                    c1 = vaddq_s32(a1, b1);
                    c2 = vaddq_s32(a2, b2);
                    vst1q_s32(&(*mt->c)->m[get_idx(mt->i, j, (*mt->c)->y)], c1);
                    vst1q_s32(&(*mt->c)->m[get_idx(mt->i, j + 4, (*mt->c)->y)], c2);
                }
                for(int j = (*mt->c)->y - ((*mt->c)->y % CHUNK_SIZE); j < (*mt->c)->y; j++) {
                    (*mt->c)->m[get_idx(mt->i, j, (*mt->c)->y)] = ((matrix*)*mt->a)->m[get_idx(mt->i, j, ((matrix*)*mt->a)->y)] + ((matrix_int8*)*mt->b)->m[get_idx(mt->i, j, ((matrix*)*mt->b)->y)];
                }
            #else
                int CHUNK_SIZE = sizeof(float32x4_t) / sizeof(DATA_TYPE);
                float32x4_t a, b;
                for(int j = 0; j + CHUNK_SIZE - 1 < (*mt->c)->y; j += CHUNK_SIZE) {
                    a = vld1q_f32(&((matrix*)*mt->a)->m[get_idx(mt->i, j, ((matrix*)*mt->a)->y)]);
                    b = vld1q_f32(&((matrix*)*mt->b)->m[get_idx(mt->i, j, ((matrix*)*mt->b)->y)]);
                    vst1q_f32(&(*mt->c)->m[get_idx(mt->i, j, (*mt->c)->y)], vaddq_f32(a, b));
                }
                for(int j = (*mt->c)->y - ((*mt->c)->y % CHUNK_SIZE); j < (*mt->c)->y; j++) {
                    (*mt->c)->m[get_idx(mt->i, j, (*mt->c)->y)] = ((matrix*)*mt->a)->m[get_idx(mt->i, j, ((matrix*)*mt->a)->y)] + ((matrix*)*mt->b)->m[get_idx(mt->i, j, ((matrix*)*mt->b)->y)];
                }
            #endif
        #endif
    }

    __attribute__((always_inline)) inline void biasing_simd(mt_arg *mt) {
        #ifdef x86
            #ifdef __AVX512F__
                #ifdef INT
                    int CHUNK_SIZE = sizeof(__m512i) / sizeof(DATA_TYPE);
                    __m512i a, b;
                    __mmask16 m;
                    for(int j = 0; j < ((matrix*)mt->a[mt->m])->y; j += CHUNK_SIZE) {
                        m = (__mmask16)((1 << (((j + CHUNK_SIZE) <= ((matrix*)mt->a[mt->m])->y) ? CHUNK_SIZE : ((matrix*)mt->a[mt->m])->y - j)) - 1);
                        a = _mm512_maskz_loadu_epi32(m, &((matrix*)mt->a[mt->m])->m[get_idx(mt->i, j, ((matrix*)mt->a[mt->m])->y)]);
                        b = _mm512_set1_epi32((DATA_TYPE)((matrix_int8*)*mt->b)->m[get_idx(mt->m, 0, ((matrix*)*mt->b)->y)]);
                        _mm512_mask_storeu_epi32(&mt->c[mt->m]->m[get_idx(mt->i, j, mt->c[mt->m]->y)], m, _mm512_add_epi32(a, b));
                    }
                #else
                    int CHUNK_SIZE = sizeof(__m512) / sizeof(DATA_TYPE);
                    __m512 a, b;
                    __mmask16 m;
                    for(int j = 0; j < ((matrix*)mt->a[mt->m])->y; j += CHUNK_SIZE) {
                        m = (__mmask16)((1 << (((j + CHUNK_SIZE) <= ((matrix*)mt->a[mt->m])->y) ? CHUNK_SIZE : ((matrix*)mt->a[mt->m])->y - j)) - 1);
                        a = _mm512_maskz_loadu_ps(m, &((matrix*)mt->a[mt->m])->m[get_idx(mt->i, j, ((matrix*)mt->a[mt->m])->y)]);
                        b = _mm512_set1_ps(((matrix*)*mt->b)->m[get_idx(mt->m, 0, ((matrix*)*mt->b)->y)]);
                        _mm512_mask_storeu_ps(&mt->c[mt->m]->m[get_idx(mt->i, j, mt->c[mt->m]->y)], m, _mm512_add_ps(a, b));
                    }
                #endif
            #else
                #ifdef INT
                    int CHUNK_SIZE = sizeof(__m256i) / sizeof(DATA_TYPE);
                    __m256i a, b;
                    for(int j = 0; j + CHUNK_SIZE - 1 < ((matrix*)mt->a[mt->m])->y; j += CHUNK_SIZE) {
                        a = _mm256_loadu_si256((__m256i*)&((matrix*)mt->a[mt->m])->m[get_idx(mt->i, j, ((matrix*)mt->a[mt->m])->y)]);
                        b = _mm256_set1_epi32((DATA_TYPE)((matrix_int8*)*mt->b)->m[get_idx(mt->m, 0, ((matrix*)*mt->b)->y)]);
                        _mm256_storeu_si256((__m256i*)&mt->c[mt->m]->m[get_idx(mt->i, j, mt->c[mt->m]->y)], _mm256_add_epi32(a, b));
                    }
                    for(int j = ((matrix*)mt->a[mt->m])->y - (((matrix*)mt->a[mt->m])->y % CHUNK_SIZE); j < ((matrix*)mt->a[mt->m])->y; j++) {
                        mt->c[mt->m]->m[get_idx(mt->i, j, mt->c[mt->m]->y)] = ((matrix*)mt->a[mt->m])->m[get_idx(mt->i, j, ((matrix*)mt->a[mt->m])->y)] + ((matrix_int8*)*mt->b)->m[get_idx(mt->m, 0, ((matrix*)*mt->b)->y)];
                    }
                #else
                    int CHUNK_SIZE = sizeof(__m256) / sizeof(DATA_TYPE);
                    __m256 a, b;
                    for(int j = 0; j + CHUNK_SIZE - 1 < ((matrix*)mt->a[mt->m])->y; j += CHUNK_SIZE) {
                        a = _mm256_loadu_ps(&((matrix*)mt->a[mt->m])->m[get_idx(mt->i, j, ((matrix*)mt->a[mt->m])->y)]);
                        b = _mm256_set1_ps(((matrix*)*mt->b)->m[get_idx(mt->m, 0, ((matrix*)*mt->b)->y)]);
                        _mm256_storeu_ps(&mt->c[mt->m]->m[get_idx(mt->i, j, mt->c[mt->m]->y)], _mm256_add_ps(a, b));
                    }
                    for(int j = ((matrix*)mt->a[mt->m])->y - (((matrix*)mt->a[mt->m])->y % CHUNK_SIZE); j < ((matrix*)mt->a[mt->m])->y; j++) {
                        mt->c[mt->m]->m[get_idx(mt->i, j, mt->c[mt->m]->y)] = ((matrix*)mt->a[mt->m])->m[get_idx(mt->i, j, ((matrix*)mt->a[mt->m])->y)] + ((matrix*)*mt->b)->m[get_idx(mt->m, 0, ((matrix*)*mt->b)->y)];
                    }
                #endif
            #endif
        #else
            #ifdef INT
                int CHUNK_SIZE = sizeof(int32x4_t) / sizeof(DATA_TYPE);
                int32x4_t a, b;
                for(int j = 0; j + CHUNK_SIZE - 1 < ((matrix*)mt->a[mt->m])->y; j += CHUNK_SIZE) {
                    a = vld1q_s32(&((matrix*)mt->a[mt->m])->m[get_idx(mt->i, j, ((matrix*)mt->a[mt->m])->y)]);
                    b = vdupq_n_s32((int32_t)((matrix_int8*)*mt->b)->m[get_idx(mt->m, 0, ((matrix*)*mt->b)->y)]);
                    vst1q_s32(&mt->c[mt->m]->m[get_idx(mt->i, j, mt->c[mt->m]->y)], vaddq_s32(a, b));
                }
                for(int j = ((matrix*)mt->a[mt->m])->y - (((matrix*)mt->a[mt->m])->y % CHUNK_SIZE); j < ((matrix*)mt->a[mt->m])->y; j++) {
                    mt->c[mt->m]->m[get_idx(mt->i, j, mt->c[mt->m]->y)] = ((matrix*)mt->a[mt->m])->m[get_idx(mt->i, j, ((matrix*)mt->a[mt->m])->y)] + ((matrix_int8*)*mt->b)->m[get_idx(mt->m, 0, ((matrix*)*mt->b)->y)];
                }
            #else
                int CHUNK_SIZE = sizeof(float32x4_t) / sizeof(DATA_TYPE);
                float32x4_t a, b;
                for(int j = 0; j + CHUNK_SIZE - 1 < ((matrix*)mt->a[mt->m])->y; j += CHUNK_SIZE) {
                    a = vld1q_f32(&((matrix*)mt->a[mt->m])->m[get_idx(mt->i, j, ((matrix*)mt->a[mt->m])->y)]);
                    b = vdupq_n_f32(((matrix*)*mt->b)->m[get_idx(mt->m, 0, ((matrix*)*mt->b)->y)]);
                    vst1q_f32(&mt->c[mt->m]->m[get_idx(mt->i, j, mt->c[mt->m]->y)], vaddq_f32(a, b));
                }
                for(int j = ((matrix*)mt->a[mt->m])->y - (((matrix*)mt->a[mt->m])->y % CHUNK_SIZE); j < ((matrix*)mt->a[mt->m])->y; j++) {
                    mt->c[mt->m]->m[get_idx(mt->i, j, mt->c[mt->m]->y)] = ((matrix*)mt->a[mt->m])->m[get_idx(mt->i, j, ((matrix*)mt->a[mt->m])->y)] + ((matrix*)*mt->b)->m[get_idx(mt->m, 0, ((matrix*)*mt->b)->y)];
                }
            #endif
        #endif
    }

    __attribute__((always_inline)) inline void conv2d_simd(mt_arg *mt) {
        #ifdef x86
            #ifdef __AVX512F__
                #ifdef INT
                    int CHUNK_SIZE = sizeof(__m512i) / sizeof(DATA_TYPE);
                    __m512i a, b, c = _mm512_setzero_si512();
                    __mmask16 m;
                    for(int k = 0; k < ((matrix*)mt->b[mt->m])->x; k++) {
                        for(int l = 0; l < ((matrix*)mt->b[mt->m])->y; l += CHUNK_SIZE) {
                            m = (__mmask16)((1 << (((l + CHUNK_SIZE) <= ((matrix*)mt->b[mt->m])->y) ? CHUNK_SIZE : ((matrix*)mt->b[mt->m])->y - l)) - 1);
                            a = _mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(m, &((matrix_int8*)*mt->a)->m[get_idx(mt->i + k, mt->j + l, ((matrix*)*mt->a)->y)]));
                            b = _mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(m, &((matrix_int8*)mt->b[mt->m])->m[get_idx(k, l, ((matrix*)mt->b[mt->m])->y)]));
                            c = _mm512_add_epi32(_mm512_mullo_epi32(a, b), c);
                        }
                    }
                    mt->c[mt->m]->m[get_idx(mt->i, mt->j, mt->c[mt->m]->y)] = _mm512_reduce_add_epi32(c);
                #else
                    int CHUNK_SIZE = sizeof(__m512) / sizeof(DATA_TYPE);
                    __m512 a, b, c = _mm512_setzero_ps();
                    __mmask16 m;
                    for(int k = 0; k < ((matrix*)mt->b[mt->m])->x; k++) {
                        for(int l = 0; l < ((matrix*)mt->b[mt->m])->y; l += CHUNK_SIZE) {
                            m = (__mmask16)((1 << (((l + CHUNK_SIZE) <= ((matrix*)mt->b[mt->m])->y) ? CHUNK_SIZE : ((matrix*)mt->b[mt->m])->y - l)) - 1);
                            a = _mm512_maskz_loadu_ps(m, &((matrix*)*mt->a)->m[get_idx(mt->i + k, mt->j + l, ((matrix*)*mt->a)->y)]);
                            b = _mm512_maskz_loadu_ps(m, &((matrix*)mt->b[mt->m])->m[get_idx(k, l, ((matrix*)mt->b[mt->m])->y)]);
                            c = _mm512_add_ps(_mm512_mul_ps(a, b), c);
                        }
                    }
                    mt->c[mt->m]->m[get_idx(mt->i, mt->j, mt->c[mt->m]->y)] = _mm512_reduce_add_ps(c);
                #endif
            #else
                #ifdef INT
                    int CHUNK_SIZE = sizeof(__m256i) / sizeof(DATA_TYPE);
                    __m256i a, b, c;
                    DATA_TYPE sum = 0;
                    for(int k = 0; k < ((matrix*)mt->b[mt->m])->x; k++) {
                        for(int l = 0; l + CHUNK_SIZE - 1 < ((matrix*)mt->b[mt->m])->y; l += CHUNK_SIZE) {
                            a = _mm256_cvtepi8_epi32(_mm_loadu_si64(&((matrix_int8*)*mt->a)->m[get_idx(mt->i + k, mt->j + l, ((matrix*)*mt->a)->y)]));
                            b = _mm256_cvtepi8_epi32(_mm_loadu_si64(&((matrix_int8*)mt->b[mt->m])->m[get_idx(k, l, ((matrix*)mt->b[mt->m])->y)]));
                            c = _mm256_mullo_epi32(a, b);
                            sum += _mm_extract_epi32(_mm_hadd_epi32(_mm_hadd_epi32(_mm_add_epi32(_mm256_castsi256_si128(c), _mm256_extracti128_si256(c, 1)), _mm_setzero_si128()), _mm_setzero_si128()), 0);
                        }
                        for(int l = ((matrix*)mt->b[mt->m])->y - (((matrix*)mt->b[mt->m])->y % CHUNK_SIZE); l < ((matrix*)mt->b[mt->m])->y; l++) {
                            sum += ((matrix_int8*)*mt->a)->m[get_idx(mt->i + k, mt->j + l, ((matrix*)*mt->a)->y)] * ((matrix_int8*)mt->b[mt->m])->m[get_idx(k, l, ((matrix*)mt->b[mt->m])->y)];
                        }
                    }
                    mt->c[mt->m]->m[get_idx(mt->i, mt->j, mt->c[mt->m]->y)] = sum;
                #else
                    int CHUNK_SIZE = sizeof(__m256) / sizeof(DATA_TYPE);
                    __m256 a, b, c;
                    DATA_TYPE sum = 0;
                    for(int k = 0; k < ((matrix*)mt->b[mt->m])->x; k++) {
                        for(int l = 0; l + CHUNK_SIZE - 1 < ((matrix*)mt->b[mt->m])->y; l += CHUNK_SIZE) {
                            a = _mm256_loadu_ps(&((matrix*)*mt->a)->m[get_idx(mt->i + k, mt->j + l, ((matrix*)*mt->a)->y)]);
                            b = _mm256_loadu_ps(&((matrix*)mt->b[mt->m])->m[get_idx(k, l, ((matrix*)mt->b[mt->m])->y)]);
                            c = _mm256_mul_ps(a, b);
                            sum += _mm_extract_ps(_mm_hadd_ps(_mm_hadd_ps(_mm_add_ps(_mm256_castps256_ps128(c), _mm256_extractf128_ps(c, 1)), _mm_setzero_ps()), _mm_setzero_ps()), 0);
                        }
                        for(int l = ((matrix*)mt->b[mt->m])->y - (((matrix*)mt->b[mt->m])->y % CHUNK_SIZE); l < ((matrix*)mt->b[mt->m])->y; l++) {
                            sum += ((matrix*)*mt->a)->m[get_idx(mt->i + k, mt->j + l, ((matrix*)*mt->a)->y)] * ((matrix*)mt->b[mt->m])->m[get_idx(k, l, ((matrix*)mt->b[mt->m])->y)];
                        }
                    }
                    mt->c[mt->m]->m[get_idx(mt->i, mt->j, mt->c[mt->m]->y)] = sum;
                #endif
            #endif
        #else
            #if !defined(AMX)
                #ifdef INT
                    int CHUNK_SIZE = sizeof(int16x8_t) / sizeof(int16_t);
                    int8x16_t a, b;
                    int16x8_t c_low, c_high;
                    DATA_TYPE sum = 0;
                    for(int k = 0; k < ((matrix*)mt->b[mt->m])->x; k++) {
                        for(int l = 0; l + CHUNK_SIZE - 1 < ((matrix*)mt->b[mt->m])->y; l += CHUNK_SIZE) {
                            a = vld1q_s8(&((matrix_int8*)*mt->a)->m[get_idx(mt->i + k, mt->j + l, ((matrix*)*mt->a)->y)]);
                            b = vld1q_s8(&((matrix_int8*)mt->b[mt->m])->m[get_idx(k, l, ((matrix*)mt->b[mt->m])->y)]);
                            c_low = vmulq_s16(vmovl_s8(vget_low_s8(a)), vmovl_s8(vget_low_s8(b)));
                            c_high = vmulq_s16(vmovl_s8(vget_high_s8(a)), vmovl_s8(vget_high_s8(b)));
                            sum += vaddvq_s16(vaddq_s16(c_low, c_high));
                        }
                        for(int l = ((matrix*)mt->b[mt->m])->y - (((matrix*)mt->b[mt->m])->y % CHUNK_SIZE); l < ((matrix*)mt->b[mt->m])->x; l++) {
                            sum += ((matrix_int8*)*mt->a)->m[get_idx(mt->i + k, mt->j + l, ((matrix*)*mt->a)->y)] * ((matrix_int8*)mt->b[mt->m])->m[get_idx(k, l, ((matrix*)mt->b[mt->m])->y)];
                        }
                    }
                    mt->c[mt->m]->m[get_idx(mt->i, mt->j, mt->c[mt->m]->y)] = sum;
                #else
                    int CHUNK_SIZE = sizeof(float32x4_t) / sizeof(DATA_TYPE);
                    float32x4_t a, b;
                    DATA_TYPE sum = 0;
                    for(int k = 0; k < ((matrix*)mt->b[mt->m])->x; k++) {
                        for(int l = 0; l + CHUNK_SIZE - 1 < ((matrix*)mt->b[mt->m])->y; l += CHUNK_SIZE) {
                            a = vld1q_f32(&((matrix*)*mt->a)->m[get_idx(mt->i + k, mt->j + l, ((matrix*)*mt->a)->y)]);
                            b = vld1q_f32(&((matrix*)mt->b[mt->m])->m[get_idx(k, l, ((matrix*)mt->b[mt->m])->y)]);
                            sum += vaddvq_f32(vmulq_f32(a, b));
                        }
                        for(int l = ((matrix*)mt->b[mt->m])->y - (((matrix*)mt->b[mt->m])->y % CHUNK_SIZE); l < ((matrix*)mt->b[mt->m])->y; l++) {
                            sum += ((matrix*)*mt->a)->m[get_idx(mt->i + k, mt->j + l, ((matrix*)*mt->a)->y)] * ((matrix*)mt->b[mt->m])->m[get_idx(k, l, ((matrix*)mt->b[mt->m])->y)];
                        }
                    }
                    mt->c[mt->m]->m[get_idx(mt->i, mt->j, mt->c[mt->m]->y)] = sum;
                #endif
            #else
                #ifdef INT
                    // int16_t
                    const long LANE_SIZE = 32; // 32 int16_t values can be performed in one instruction
                    const int CHUNK_SIZE = 4 * LANE_SIZE;
                    DATA_TYPE z_reg[LANE_SIZE];
                    DATA_TYPE sum = 0;
                    uint64_t stz = (uint64_t)&z_reg;
                    for(int k = 0; k < mt->b[mt->m]->x; k++) {
                        for(int l = 0; l + CHUNK_SIZE - 1 < mt->b[mt->m]->y; l += CHUNK_SIZE) {
                            uint64_t ldx = (uint64_t)&(*mt->a)->m[get_idx(mt->i + k, mt->j + l, (*mt->a)->y)];
                            ldx = ldx | 1ull << 60; // four registers
                            ldx = ldx | 1ull << 62; // multiple registers
                            uint64_t ldy = (uint64_t)&mt->b[mt->m]->m[get_idx(k, l, mt->b[mt->m]->y)];
                            ldy = ldy | 1ull << 60; // four registers
                            ldy = ldy | 1ull << 62; // multiple registers
                            AMX_SET();
                            AMX_LDX(ldx);
                            AMX_LDY(ldy);
                            uint64_t mac16 = 1ull << 63; // vector mode
                            // mac16 = mac16 | 1ull << 62; // Z is i32
                            for(int i = 0; i < CHUNK_SIZE / LANE_SIZE; i++) {
                                size_t i_offset = LANE_SIZE * i * sizeof(DATA_TYPE);
                                AMX_MAC16(mac16 + i_offset + (i_offset << 10)); // x and y offset
                            }
                            AMX_STZ(stz);
                            AMX_CLR();
                            for(int i = 0; i < LANE_SIZE; i++) {
                                sum += z_reg[i];
                            }
                            mt->c[mt->m]->m[get_idx(mt->i, mt->j, mt->c[mt->m]->y)] = sum;
                        }
                        for(int l = mt->b[mt->m]->y - (mt->b[mt->m]->y % CHUNK_SIZE); l < mt->b[mt->m]->y; l++) {
                            sum += (*mt->a)->m[get_idx(mt->i + k, mt->j + l, (*mt->a)->y)] * mt->b[mt->m]->m[get_idx(k, l, mt->b[mt->m]->y)];
                        }
                    }
                    mt->c[mt->m]->m[get_idx(mt->i, mt->j, mt->c[mt->m]->y)] = sum;
                #else
                    const long LANE_SIZE = 16; // only 16 values can be performed at once
                    const int CHUNK_SIZE = 4 * LANE_SIZE;
                    DATA_TYPE z_reg[LANE_SIZE];
                    DATA_TYPE sum = 0;
                    uint64_t stz = (uint64_t)&z_reg;
                    for(int k = 0; k < ((matrix*)mt->b[mt->m])->x; k++) {
                        for(int l = 0; l + CHUNK_SIZE - 1 < ((matrix*)mt->b[mt->m])->y; l += CHUNK_SIZE) {
                            uint64_t ldx = (uint64_t)&((matrix*)*mt->a)->m[get_idx(mt->i + k, mt->j + l, ((matrix*)*mt->a)->y)];
                            ldx = ldx | 1ull << 60; // four registers
                            ldx = ldx | 1ull << 62; // multiple registers
                            uint64_t ldy = (uint64_t)&((matrix*)mt->b[mt->m])->m[get_idx(k, l, ((matrix*)mt->b[mt->m])->y)];
                            ldy = ldy | 1ull << 60; // four registers
                            ldy = ldy | 1ull << 62; // multiple registers
                            AMX_SET();
                            AMX_LDX(ldx);
                            AMX_LDY(ldy);
                            uint64_t fma32 = 1ull << 63; // vector mode
                            for(int i = 0; i < CHUNK_SIZE / LANE_SIZE; i++) {
                                size_t i_offset = LANE_SIZE * i * sizeof(DATA_TYPE);
                                AMX_FMA32(fma32 + i_offset + (i_offset << 10)); // x and y offset
                            }
                            AMX_STZ(stz);
                            AMX_CLR();
                            for(int i = 0; i < LANE_SIZE; i++) {
                                sum += z_reg[i];
                            }
                            mt->c[mt->m]->m[get_idx(mt->i, mt->j, mt->c[mt->m]->y)] = sum;
                        }
                        for(int l = ((matrix*)mt->b[mt->m])->y - (((matrix*)mt->b[mt->m])->y % CHUNK_SIZE); l < ((matrix*)mt->b[mt->m])->y; l++) {
                            sum += ((matrix*)*mt->a)->m[get_idx(mt->i + k, mt->j + l, ((matrix*)*mt->a)->y)] * ((matrix*)mt->b[mt->m])->m[get_idx(k, l, ((matrix*)mt->b[mt->m])->y)];
                        }
                    }
                    mt->c[mt->m]->m[get_idx(mt->i, mt->j, mt->c[mt->m]->y)] = sum;
                #endif
            #endif
        #endif
    }

    __attribute__((always_inline)) inline void matmul_simd(mt_arg *mt) {
        #ifdef x86
            #ifdef __AVX512F__
                #ifdef INT
                    int CHUNK_SIZE = sizeof(__m512i) / sizeof(DATA_TYPE);
                    __m512i a, b, c = _mm512_setzero_si512();
                    __mmask16 m;
                    for(int k = 0; k < ((matrix*)*mt->a)->y; k += CHUNK_SIZE) {
                        m = (__mmask16)((1 << (((k + CHUNK_SIZE) <= ((matrix*)*mt->a)->y) ? CHUNK_SIZE : ((matrix*)*mt->a)->y - k)) - 1);
                        a = _mm512_maskz_loadu_epi32(m, &((matrix*)*mt->a)->m[get_idx(mt->i, k, ((matrix*)*mt->a)->y)]);
                        b = _mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(m, &((matrix_int8*)*mt->b)->m[get_idx(mt->j, k, ((matrix*)*mt->b)->y)]));
                        c = _mm512_add_epi32(_mm512_mullo_epi32(a, b), c);
                    }
                    (*mt->c)->m[get_idx(mt->i, mt->j, (*mt->c)->y)] = _mm512_reduce_add_epi32(c);
                #else
                    int CHUNK_SIZE = sizeof(__m512) / sizeof(DATA_TYPE);
                    __m512 a, b, c = _mm512_setzero_ps();
                    __mmask16 m;
                    for(int k = 0; k < ((matrix*)*mt->a)->y; k += CHUNK_SIZE) {
                        m = (__mmask16)((1 << (((k + CHUNK_SIZE) <= ((matrix*)*mt->a)->y) ? CHUNK_SIZE : ((matrix*)*mt->a)->y - k)) - 1);
                        a = _mm512_maskz_loadu_ps(m, &((matrix*)*mt->a)->m[get_idx(mt->i, k, ((matrix*)*mt->a)->y)]);
                        b = _mm512_maskz_loadu_ps(m, &((matrix*)*mt->b)->m[get_idx(mt->j, k, ((matrix*)*mt->b)->y)]);
                        c = _mm512_add_ps(_mm512_mul_ps(a, b), c);
                    }
                    (*mt->c)->m[get_idx(mt->i, mt->j, (*mt->c)->y)] = _mm512_reduce_add_ps(c);
                #endif
            #else
                #ifdef INT
                    int CHUNK_SIZE = sizeof(__m256i) / sizeof(DATA_TYPE);
                    __m256i a, b, c = _mm256_setzero_si256();
                    DATA_TYPE sum = 0;
                    for(int k = 0; k + CHUNK_SIZE - 1 < ((matrix*)*mt->a)->y; k += CHUNK_SIZE) {
                        a = _mm256_loadu_si256((__m256i*)&((matrix*)*mt->a)->m[get_idx(mt->i, k, ((matrix*)*mt->a)->y)]);
                        b = _mm256_cvtepi8_epi32(_mm_loadu_si64(&((matrix_int8*)*mt->b)->m[get_idx(mt->j, k, ((matrix*)*mt->b)->y)]));
                        c = _mm256_add_epi32(_mm256_mullo_epi32(a, b), c);
                    }
                    sum += _mm_extract_epi32(_mm_hadd_epi32(_mm_hadd_epi32(_mm_add_epi32(_mm256_castsi256_si128(c), _mm256_extracti128_si256(c, 1)), _mm_setzero_si128()), _mm_setzero_si128()), 0);
                    for(int k = ((matrix*)*mt->a)->y - (((matrix*)*mt->a)->y % CHUNK_SIZE); k < ((matrix*)*mt->a)->y; k++) {
                        sum += ((matrix*)*mt->a)->m[get_idx(mt->i, k, ((matrix*)*mt->a)->y)] * ((matrix_int8*)*mt->b)->m[get_idx(mt->j, k, ((matrix*)*mt->b)->y)];
                    }
                    (*mt->c)->m[get_idx(mt->i, mt->j, (*mt->c)->y)] = sum;
                #else
                    int CHUNK_SIZE = sizeof(__m256) / sizeof(DATA_TYPE);
                    __m256 a, b, c = _mm256_setzero_ps();
                    DATA_TYPE sum = 0;
                    for(int k = 0; k + CHUNK_SIZE - 1 < ((matrix*)*mt->a)->y; k += CHUNK_SIZE) {
                        a = _mm256_loadu_ps(&((matrix*)*mt->a)->m[get_idx(mt->i, k, ((matrix*)*mt->a)->y)]);
                        b = _mm256_loadu_ps(&((matrix*)*mt->b)->m[get_idx(mt->j, k, ((matrix*)*mt->b)->y)]);
                        c = _mm256_add_ps(_mm256_mul_ps(a, b), c);
                    }
                    sum += _mm_extract_ps(_mm_hadd_ps(_mm_hadd_ps(_mm_add_ps(_mm256_castps256_ps128(c), _mm256_extractf128_ps(c, 1)), _mm_setzero_ps()), _mm_setzero_ps()), 0);
                    for(int k = ((matrix*)*mt->a)->y - (((matrix*)*mt->a)->y % CHUNK_SIZE); k < ((matrix*)*mt->a)->y; k++) {
                        sum += ((matrix*)*mt->a)->m[get_idx(mt->i, k, ((matrix*)*mt->a)->y)] * ((matrix*)*mt->b)->m[get_idx(mt->j, k, ((matrix*)*mt->b)->y)];
                    }
                    (*mt->c)->m[get_idx(mt->i, mt->j, (*mt->c)->y)] = sum;
                #endif
            #endif
        #else
            #if !defined(AMX)
                #ifdef INT
                    int CHUNK_SIZE = 16;
                    int8x16_t b;
                    int32x2_t total_half;
                    int32x4_t a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4, total;
                    int32_t sum = 0;
                    for(int k = 0; k < ((matrix*)*mt->a)->y; k += CHUNK_SIZE) {
                        a1 = vld1q_s32(&((matrix*)*mt->a)->m[get_idx(mt->i, k, ((matrix*)*mt->a)->y)]);
                        a2 = vld1q_s32(&((matrix*)*mt->a)->m[get_idx(mt->i, k + 4, ((matrix*)*mt->a)->y)]);
                        a3 = vld1q_s32(&((matrix*)*mt->a)->m[get_idx(mt->i, k + 8, ((matrix*)*mt->a)->y)]);
                        a4 = vld1q_s32(&((matrix*)*mt->a)->m[get_idx(mt->i, k + 12, ((matrix*)*mt->a)->y)]);
                        b = vld1q_s8(&((matrix_int8*)*mt->b)->m[get_idx(mt->j, k, ((matrix*)*mt->b)->y)]);
                        b1 = vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(b))));
                        b2 = vmovl_s16(vget_high_s16(vmovl_s8(vget_low_s8(b))));
                        b3 = vmovl_s16(vget_low_s16(vmovl_s8(vget_high_s8(b))));
                        b4 = vmovl_s16(vget_high_s16(vmovl_s8(vget_high_s8(b))));
                        c1 = vmulq_s32(a1, b1);
                        c2 = vmulq_s32(a2, b2);
                        c3 = vmulq_s32(a3, b3);
                        c4 = vmulq_s32(a4, b4);
                        total = vaddq_s32(vaddq_s32(c1, c2), vaddq_s32(c3, c4));
                        total_half = vadd_s32(vget_low_s32(total), vget_high_s32(total));
                        sum += vget_lane_s32(vpadd_s32(total_half, vdup_n_s32(0)), 0);
                    }
                    (*mt->c)->m[get_idx(mt->i, mt->j, (*mt->c)->y)] += sum;
                #else
                    int CHUNK_SIZE = sizeof(float32x4_t) / sizeof(DATA_TYPE);
                    float32x4_t a, b;
                    for(int k = 0; k + CHUNK_SIZE - 1 < ((matrix*)*mt->a)->y; k += CHUNK_SIZE) {
                        a = vld1q_f32(&((matrix*)*mt->a)->m[get_idx(mt->i, k, ((matrix*)*mt->a)->y)]);
                        b = vld1q_f32(&((matrix*)*mt->b)->m[get_idx(mt->j, k, ((matrix*)*mt->b)->y)]);
                        (*mt->c)->m[get_idx(mt->i, mt->j, (*mt->c)->y)] += vaddvq_f32(vmulq_f32(a, b));
                    }
                    for(int k = ((matrix*)*mt->a)->y - (((matrix*)*mt->a)->y % CHUNK_SIZE); k < ((matrix*)*mt->a)->y; k++) {
                        (*mt->c)->m[get_idx(mt->i, mt->j, (*mt->c)->y)] += ((matrix*)*mt->a)->m[get_idx(mt->i, k, ((matrix*)*mt->a)->y)] * ((matrix*)*mt->b)->m[get_idx(mt->j, k, ((matrix*)*mt->b)->y)];
                    }
                #endif
            #else
                #ifdef INT
                    // int16_t
                    const long LANE_SIZE = 32; // only 32 values can be performed at once, because int16_t is 2B
                    const int CHUNK_SIZE = 4 * LANE_SIZE;
                    DATA_TYPE z_reg[LANE_SIZE];
                    DATA_TYPE sum = 0;
                    uint64_t ldx = (uint64_t)&(*mt->a)->m[get_idx(mt->i, 0, (*mt->a)->y)];
                    ldx = ldx | 1ull << 60; // four registers
                    ldx = ldx | 1ull << 62; // multiple registers
                    uint64_t ldy = (uint64_t)&(*mt->b)->m[get_idx(mt->j, 0, (*mt->b)->y)];
                    ldy = ldy | 1ull << 60; // four registers
                    ldy = ldy | 1ull << 62; // multiple registers
                    uint64_t mac16 = 1ull << 63; // vector mode
                    // mac16 = mac16 | 1ull << 62; // z is i32
                    uint64_t stz = (uint64_t)&z_reg;
                    AMX_SET();
                    for(int k = 0; k + CHUNK_SIZE - 1 < (*mt->a)->y; k += CHUNK_SIZE) {
                        size_t k_offset = k * sizeof(DATA_TYPE);
                        AMX_LDX(ldx + k_offset);
                        AMX_LDY(ldy + k_offset);
                        for(int i = 0; i < CHUNK_SIZE / LANE_SIZE; i++) {
                            size_t i_offset = LANE_SIZE * i * sizeof(DATA_TYPE);
                            AMX_MAC16(mac16 + i_offset + (i_offset << 10)); // x and y offset
                        }
                    }
                    AMX_STZ(stz);
                    AMX_CLR();
                    for(int i = 0; i < LANE_SIZE; i++) {
                        sum += z_reg[i];
                    }
                    (*mt->c)->m[get_idx(mt->i, mt->j, (*mt->c)->y)] = sum;
                    for(int k = (*mt->a)->y - ((*mt->a)->y % CHUNK_SIZE); k < (*mt->a)->y; k++) {
                        (*mt->c)->m[get_idx(mt->i, mt->j, (*mt->c)->y)] += (*mt->a)->m[get_idx(mt->i, k, (*mt->a)->y)] * (*mt->b)->m[get_idx(mt->j, k, (*mt->b)->y)];
                    }
                #else
                    const long LANE_SIZE = 16; // only 16 values can be performed at once
                    const int CHUNK_SIZE = 4 * LANE_SIZE;
                    DATA_TYPE z_reg[LANE_SIZE];
                    DATA_TYPE sum = 0;
                    uint64_t ldx = (uint64_t)&((matrix*)*mt->a)->m[get_idx(mt->i, 0, ((matrix*)*mt->a)->y)];
                    ldx = ldx | 1ull << 60; // four registers
                    ldx = ldx | 1ull << 62; // multiple registers
                    uint64_t ldy = (uint64_t)&((matrix*)*mt->b)->m[get_idx(mt->j, 0, ((matrix*)*mt->b)->y)];
                    ldy = ldy | 1ull << 60; // four registers
                    ldy = ldy | 1ull << 62; // multiple registers
                    uint64_t fma32 = 1ull << 63; // vector mode
                    uint64_t stz = (uint64_t)&z_reg;
                    AMX_SET();
                    for(int k = 0; k + CHUNK_SIZE - 1 < ((matrix*)*mt->a)->y; k += CHUNK_SIZE) {
                        size_t k_offset = k * sizeof(DATA_TYPE);
                        AMX_LDX(ldx + k_offset);
                        AMX_LDY(ldy + k_offset);
                        for(int i = 0; i < CHUNK_SIZE / LANE_SIZE; i++) {
                            size_t i_offset = LANE_SIZE * i * sizeof(DATA_TYPE);
                            AMX_FMA32(fma32 + i_offset + (i_offset << 10)); // x and y offset
                        }
                    }
                    AMX_STZ(stz);
                    AMX_CLR();
                    for(int i = 0; i < LANE_SIZE; i++) {
                        sum += z_reg[i];
                    }
                    (*mt->c)->m[get_idx(mt->i, mt->j, (*mt->c)->y)] = sum;
                    for(int k = ((matrix*)*mt->a)->y - (((matrix*)*mt->a)->y % CHUNK_SIZE); k < ((matrix*)*mt->a)->y; k++) {
                        (*mt->c)->m[get_idx(mt->i, mt->j, (*mt->c)->y)] += ((matrix*)*mt->a)->m[get_idx(mt->i, k, ((matrix*)*mt->a)->y)] * ((matrix*)*mt->b)->m[get_idx(mt->j, k, ((matrix*)*mt->b)->y)];
                    }
                #endif
            #endif
        #endif
    }
#endif
