#ifndef SIMD_HPP
    #define SIMD_HPP

    #include "mt_arg.hpp"
    //#include "utils.hpp"

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
        /*
        #ifdef x86
            #ifdef __AVX512F__
                #ifdef INT
                    long CHUNK_SIZE = sizeof(__m512i) / sizeof(DATA_TYPE);
                    __m512i a, b;
                    __mmask16 m;
                    for(int j = 0; j < (*mt->c)->y; j += CHUNK_SIZE) {
                        m = (__mmask16)((1 << (((j + CHUNK_SIZE) <= (*mt->c)->y) ? CHUNK_SIZE : (*mt->c)->y - j)) - 1);
                        a = _mm512_maskz_loadu_epi32(m, &(*mt->a)->m[get_idx(mt->i, j, (*mt->a)->y)]);
                        b = _mm512_maskz_loadu_epi32(m, &(*mt->b)->m[get_idx(mt->i, j, (*mt->b)->y)]);
                        _mm512_mask_storeu_epi32(&(*mt->c)->m[get_idx(mt->i, j, (*mt->c)->y)], m, _mm512_add_epi32(a, b));
                    }
                #else
                    long CHUNK_SIZE = sizeof(__m512) / sizeof(DATA_TYPE);
                    __m512 a, b;
                    __mmask16 m;
                    for(int j = 0; j < (*mt->c)->y; j += CHUNK_SIZE) {
                        m = (__mmask16)((1 << (((j + CHUNK_SIZE) <= (*mt->c)->y) ? CHUNK_SIZE : (*mt->c)->y - j)) - 1);
                        a = _mm512_maskz_loadu_ps(m, &(*mt->a)->m[get_idx(mt->i, j, (*mt->a)->y)]);
                        b = _mm512_maskz_loadu_ps(m, &(*mt->b)->m[get_idx(mt->i, j, (*mt->b)->y)]);
                        _mm512_mask_storeu_ps(&(*mt->c)->m[get_idx(mt->i, j, (*mt->c)->y)], m, _mm512_add_ps(a, b));
                    }
                #endif
            #else
                #ifdef INT
                    long CHUNK_SIZE = sizeof(__m256i) / sizeof(DATA_TYPE);
                    __m256i a, b;
                    for(int j = 0; j + CHUNK_SIZE - 1 < (*mt->c)->y; j += CHUNK_SIZE) {
                        a = _mm256_loadu_si256((__m256i*)&(*mt->a)->m[get_idx(mt->i, j, (*mt->a)->y)]);
                        b = _mm256_loadu_si256((__m256i*)&(*mt->b)->m[get_idx(mt->i, j, (*mt->b)->y)]);
                        _mm256_storeu_si256((__m256i*)&(*mt->c)->m[get_idx(mt->i, j, (*mt->c)->y)], _mm256_add_epi32(a, b));
                    }
                    for(int j = (*mt->c)->y - ((*mt->c)->y % CHUNK_SIZE); j < (*mt->c)->y; j++) {
                        (*mt->c)->m[get_idx(mt->i, j, (*mt->c)->y)] = (*mt->a)->m[get_idx(mt->i, j, (*mt->a)->y)] + (*mt->b)->m[get_idx(mt->i, j, (*mt->b)->y)];
                    }
                #else
                    long CHUNK_SIZE = sizeof(__m256) / sizeof(DATA_TYPE);
                    __m256 a, b;
                    for(int j = 0; j + CHUNK_SIZE - 1 < (*mt->c)->y; j += CHUNK_SIZE) {
                        a = _mm256_loadu_ps(&(*mt->a)->m[get_idx(mt->i, j, (*mt->a)->y)]);
                        b = _mm256_loadu_ps(&(*mt->b)->m[get_idx(mt->i, j, (*mt->b)->y)]);
                        _mm256_storeu_ps(&(*mt->c)->m[get_idx(mt->i, j, (*mt->c)->y)], _mm256_add_ps(a, b));
                    }
                    for(int j = (*mt->c)->y - ((*mt->c)->y % CHUNK_SIZE); j < (*mt->c)->y; j++) {
                        (*mt->c)->m[get_idx(mt->i, j, (*mt->c)->y)] = (*mt->a)->m[get_idx(mt->i, j, (*mt->a)->y)] + (*mt->b)->m[get_idx(mt->i, j, (*mt->b)->y)];
                    }
                #endif
            #endif
        #else
            #ifdef INT
                long CHUNK_SIZE = sizeof(int32x4_t) / sizeof(DATA_TYPE);
                int32x4_t a, b;
                for (int j = 0; j + CHUNK_SIZE - 1 < (*mt->c)->y; j += CHUNK_SIZE) {
                    a = vld1q_s32(&(*mt->a)->m[get_idx(mt->i, j, (*mt->a)->y)]);
                    b = vld1q_s32(&(*mt->b)->m[get_idx(mt->i, j, (*mt->b)->y)]);
                    vst1q_s32(&(*mt->c)->m[get_idx(mt->i, j, (*mt->c)->y)], vaddq_s32(a, b));
                }
                for(int j = (*mt->c)->y - ((*mt->c)->y % CHUNK_SIZE); j < (*mt->c)->y; j++) {
                    (*mt->c)->m[get_idx(mt->i, j, (*mt->c)->y)] = (*mt->a)->m[get_idx(mt->i, j, (*mt->a)->y)] + (*mt->b)->m[get_idx(mt->i, j, (*mt->b)->y)];
                }
            #else
                long CHUNK_SIZE = sizeof(float32x4_t) / sizeof(DATA_TYPE);
                float32x4_t a, b;
                for(int j = 0; j + CHUNK_SIZE - 1 < (*mt->c)->y; j += CHUNK_SIZE) {
                    a = vld1q_f32(&(*mt->a)->m[get_idx(mt->i, j, (*mt->a)->y)]);
                    b = vld1q_f32(&(*mt->b)->m[get_idx(mt->i, j, (*mt->b)->y)]);
                    vst1q_f32(&(*mt->c)->m[get_idx(mt->i, j, (*mt->c)->y)], vaddq_f32(a, b));
                }
                for(int j = (*mt->c)->y - ((*mt->c)->y % CHUNK_SIZE); j < (*mt->c)->y; j++) {
                    (*mt->c)->m[get_idx(mt->i, j, (*mt->c)->y)] = (*mt->a)->m[get_idx(mt->i, j, (*mt->a)->y)] + (*mt->b)->m[get_idx(mt->i, j, (*mt->b)->y)];
                }
            #endif
        #endif
        */
    }

    __attribute__((always_inline)) inline void biasing_simd(mt_arg *mt) {
        /*
        #ifdef x86
            #ifdef __AVX512F__
                #ifdef INT
                    long CHUNK_SIZE = sizeof(__m512i) / sizeof(DATA_TYPE);
                    __m512i a, b;
                    __mmask16 m;
                    for(int j = 0; j < mt->a[mt->m]->y; j += CHUNK_SIZE) {
                        m = (__mmask16)((1 << (((j + CHUNK_SIZE) <= mt->a[mt->m]->y) ? CHUNK_SIZE : mt->a[mt->m]->y - j)) - 1);
                        a = _mm512_maskz_loadu_epi32(m, &mt->a[mt->m]->m[get_idx(mt->i, j, mt->a[mt->m]->y)]);
                        b = _mm512_set1_epi32((*mt->b)->m[get_idx(mt->m, 0, (*mt->b)->y)]);
                        _mm512_mask_storeu_epi32(&mt->c[mt->m]->m[get_idx(mt->i, j, mt->c[mt->m]->y)], m, _mm512_add_epi32(a, b));
                    }
                #else
                    long CHUNK_SIZE = sizeof(__m512) / sizeof(DATA_TYPE);
                    __m512 a, b;
                    __mmask16 m;
                    for(int j = 0; j < mt->a[mt->m]->y; j += CHUNK_SIZE) {
                        m = (__mmask16)((1 << (((j + CHUNK_SIZE) <= mt->a[mt->m]->y) ? CHUNK_SIZE : mt->a[mt->m]->y - j)) - 1);
                        a = _mm512_maskz_loadu_ps(m, &mt->a[mt->m]->m[get_idx(mt->i, j, mt->a[mt->m]->y)]);
                        b = _mm512_set1_ps((*mt->b)->m[get_idx(mt->m, 0, (*mt->b)->y)]);
                        _mm512_mask_storeu_ps(&mt->c[mt->m]->m[get_idx(mt->i, j, mt->c[mt->m]->y)], m, _mm512_add_ps(a, b));
                    }
                #endif
            #else
                #ifdef INT
                    long CHUNK_SIZE = sizeof(__m256i) / sizeof(DATA_TYPE);
                    __m256i a, b;
                    for(int j = 0; j + CHUNK_SIZE - 1 < mt->a[mt->m]->y; j += CHUNK_SIZE) {
                        a = _mm256_loadu_si256((__m256i*)&mt->a[mt->m]->m[get_idx(mt->i, j, mt->a[mt->m]->y)]);
                        b = _mm256_set1_epi32((*mt->b)->m[get_idx(mt->m, 0, (*mt->b)->y)]);
                        _mm256_storeu_si256((__m256i*)&mt->c[mt->m]->m[get_idx(mt->i, j, mt->c[mt->m]->y)], _mm256_add_epi32(a, b));
                    }
                    for(int j = mt->a[mt->m]->y - (mt->a[mt->m]->y % CHUNK_SIZE); j < mt->a[mt->m]->y; j++) {
                        mt->c[mt->m]->m[get_idx(mt->i, j, mt->c[mt->m]->y)] = mt->a[mt->m]->m[get_idx(mt->i, j, mt->a[mt->m]->y)] + (*mt->b)->m[get_idx(mt->m, 0, (*mt->b)->y)];
                    }
                #else
                    long CHUNK_SIZE = sizeof(__m256) / sizeof(DATA_TYPE);
                    __m256 a, b;
                    for(int j = 0; j + CHUNK_SIZE - 1 < mt->a[mt->m]->y; j += CHUNK_SIZE) {
                        a = _mm256_loadu_ps(&mt->a[mt->m]->m[get_idx(mt->i, j, mt->a[mt->m]->y)]);
                        b = _mm256_set1_ps((*mt->b)->m[get_idx(mt->m, 0, (*mt->b)->y)]);
                        _mm256_storeu_ps(&mt->c[mt->m]->m[get_idx(mt->i, j, mt->c[mt->m]->y)], _mm256_add_ps(a, b));
                    }
                    for(int j = mt->a[mt->m]->y - (mt->a[mt->m]->y % CHUNK_SIZE); j < mt->a[mt->m]->y; j++) {
                        mt->c[mt->m]->m[get_idx(mt->i, j, mt->c[mt->m]->y)] = mt->a[mt->m]->m[get_idx(mt->i, j, mt->a[mt->m]->y)] + (*mt->b)->m[get_idx(mt->m, 0, (*mt->b)->y)];
                    }
                #endif
            #endif
        #else
            #ifdef INT
                long CHUNK_SIZE = sizeof(int32x4_t) / sizeof(DATA_TYPE);
                int32x4_t a, b;
                for(int j = 0; j + CHUNK_SIZE - 1 < mt->a[mt->m]->y; j += CHUNK_SIZE) {
                    a = vld1q_s32(&mt->a[mt->m]->m[get_idx(mt->i, j, mt->a[mt->m]->y)]);
                    b = vdupq_n_s32((*mt->b)->m[get_idx(mt->m, 0, (*mt->b)->y)]);
                    vst1q_s32(&mt->c[mt->m]->m[get_idx(mt->i, j, mt->c[mt->m]->y)], vaddq_s32(a, b));
                }
                for(int j = mt->a[mt->m]->y - (mt->a[mt->m]->y % CHUNK_SIZE); j < mt->a[mt->m]->y; j++) {
                    mt->c[mt->m]->m[get_idx(mt->i, j, mt->c[mt->m]->y)] = mt->a[mt->m]->m[get_idx(mt->i, j, mt->a[mt->m]->y)] + (*mt->b)->m[get_idx(mt->m, 0, (*mt->b)->y)];
                }
            #else
                long CHUNK_SIZE = sizeof(float32x4_t) / sizeof(DATA_TYPE);
                float32x4_t a, b;
                for(int j = 0; j + CHUNK_SIZE - 1 < mt->a[mt->m]->y; j += CHUNK_SIZE) {
                    a = vld1q_f32(&mt->a[mt->m]->m[get_idx(mt->i, j, mt->a[mt->m]->y)]);
                    b = vdupq_n_f32((*mt->b)->m[get_idx(mt->m, 0, (*mt->b)->y)]);
                    vst1q_f32(&mt->c[mt->m]->m[get_idx(mt->i, j, mt->c[mt->m]->y)], vaddq_f32(a, b));
                }
                for(int j = mt->a[mt->m]->y - (mt->a[mt->m]->y % CHUNK_SIZE); j < mt->a[mt->m]->y; j++) {
                    mt->c[mt->m]->m[get_idx(mt->i, j, mt->c[mt->m]->y)] = mt->a[mt->m]->m[get_idx(mt->i, j, mt->a[mt->m]->y)] + (*mt->b)->m[get_idx(mt->m, 0, (*mt->b)->y)];
                }
            #endif
        #endif
        */
    }

    __attribute__((always_inline)) inline void conv2d_simd(mt_arg *mt) {
        /*
        #ifdef x86
            #ifdef __AVX512F__
                #ifdef INT
                    long CHUNK_SIZE = sizeof(__m512i) / sizeof(DATA_TYPE);
                    __m512i a, b;
                    __mmask16 m;
                    DATA_TYPE sum = 0;
                    for(int k = 0; k < mt->b[mt->m]->x; k++) {
                        for(int l = 0; l < mt->b[mt->m]->y; l += CHUNK_SIZE) {
                            m = (__mmask16)((1 << (((l + CHUNK_SIZE) <= mt->b[mt->m]->y) ? CHUNK_SIZE : mt->b[mt->m]->y - l)) - 1);
                            a = _mm512_maskz_loadu_epi32(m, &(*mt->a)->m[get_idx(mt->i + k, mt->j + l, (*mt->a)->y)]);
                            b = _mm512_maskz_loadu_epi32(m, &mt->b[mt->m]->m[get_idx(k, l, mt->b[mt->m]->y)]);
                            sum += _mm512_reduce_add_epi32(_mm512_mullo_epi32(a, b));
                        }
                    }
                    mt->c[mt->m]->m[get_idx(mt->i, mt->j, mt->c[mt->m]->y)] = sum;
                #else
                    long CHUNK_SIZE = sizeof(__m512) / sizeof(DATA_TYPE);
                    __m512 a, b;
                    __mmask16 m;
                    DATA_TYPE sum = 0;
                    for(int k = 0; k < mt->b[mt->m]->x; k++) {
                        for(int l = 0; l < mt->b[mt->m]->y; l += CHUNK_SIZE) {
                            m = (__mmask16)((1 << (((l + CHUNK_SIZE) <= mt->b[mt->m]->y) ? CHUNK_SIZE : mt->b[mt->m]->y - l)) - 1);
                            a = _mm512_maskz_loadu_ps(m, &(*mt->a)->m[get_idx(mt->i + k, mt->j + l, (*mt->a)->y)]);
                            b = _mm512_maskz_loadu_ps(m, &mt->b[mt->m]->m[get_idx(k, l, mt->b[mt->m]->y)]);
                            sum += _mm512_reduce_add_ps(_mm512_mul_ps(a, b));
                        }
                    }
                    mt->c[mt->m]->m[get_idx(mt->i, mt->j, mt->c[mt->m]->y)] = sum;
                #endif
            #else
                #ifdef INT
                    long CHUNK_SIZE = sizeof(__m256i) / sizeof(DATA_TYPE);
                    __m256i a, b, result;
                    DATA_TYPE sum = 0;
                    for(int k = 0; k < mt->b[mt->m]->x; k++) {
                        for(int l = 0; l + CHUNK_SIZE - 1 < mt->b[mt->m]->y; l += CHUNK_SIZE) {
                            a = _mm256_loadu_si256((__m256i*)&(*mt->a)->m[get_idx(mt->i + k, mt->j + l, (*mt->a)->y)]);
                            b = _mm256_loadu_si256((__m256i*)&mt->b[mt->m]->m[get_idx(k, l, mt->b[mt->m]->y)]);
                            result = _mm256_mullo_epi32(a, b);
                            sum += _mm_extract_epi32(_mm_hadd_epi32(_mm_hadd_epi32(_mm_add_epi32(_mm256_castsi256_si128(result), _mm256_extracti128_si256(result, 1)), _mm_setzero_si128()), _mm_setzero_si128()), 0);
                        }
                        for(int l = mt->b[mt->m]->y - (mt->b[mt->m]->y % CHUNK_SIZE); l < mt->b[mt->m]->y; l++) {
                            sum += (*mt->a)->m[get_idx(mt->i + k, mt->j + l, (*mt->a)->y)] * mt->b[mt->m]->m[get_idx(k, l, mt->b[mt->m]->y)];
                        }
                    }
                    mt->c[mt->m]->m[get_idx(mt->i, mt->j, mt->c[mt->m]->y)] = sum;
                #else
                    long CHUNK_SIZE = sizeof(__m256) / sizeof(DATA_TYPE);
                    __m256 a, b, result;
                    DATA_TYPE sum = 0;
                    for(int k = 0; k < mt->b[mt->m]->x; k++) {
                        for(int l = 0; l + CHUNK_SIZE - 1 < mt->b[mt->m]->y; l += CHUNK_SIZE) {
                            a = _mm256_loadu_ps(&(*mt->a)->m[get_idx(mt->i + k, mt->j + l, (*mt->a)->y)]);
                            b = _mm256_loadu_ps(&mt->b[mt->m]->m[get_idx(k, l, mt->b[mt->m]->y)]);
                            result = _mm256_mul_ps(a, b);
                            sum += _mm_cvtss_f32(_mm_hadd_ps(_mm_hadd_ps(_mm_add_ps(_mm256_castps256_ps128(result), _mm256_extractf128_ps(result, 1)), _mm_setzero_ps()), _mm_setzero_ps()));
                        }
                        for(int l = mt->b[mt->m]->y - (mt->b[mt->m]->y % CHUNK_SIZE); l < mt->b[mt->m]->y; l++) {
                            sum += (*mt->a)->m[get_idx(mt->i + k, mt->j + l, (*mt->a)->y)] * mt->b[mt->m]->m[get_idx(k, l, mt->b[mt->m]->y)];
                        }
                    }
                    mt->c[mt->m]->m[get_idx(mt->i, mt->j, mt->c[mt->m]->y)] = sum;
                #endif
            #endif
        #else
            #if !defined(AMX)
                #ifdef INT
                    long CHUNK_SIZE = sizeof(int32x4_t) / sizeof(DATA_TYPE);
                    int32x4_t a, b;
                    DATA_TYPE sum = 0;
                    for(int k = 0; k < mt->b[mt->m]->x; k++) {
                        for(int l = 0; l + CHUNK_SIZE - 1 < mt->b[mt->m]->y; l += CHUNK_SIZE) {
                            a = vld1q_s32(&(*mt->a)->m[get_idx(mt->i + k, mt->j + l, (*mt->a)->y)]);
                            b = vld1q_s32(&mt->b[mt->m]->m[get_idx(k, l, mt->b[mt->m]->y)]);
                            sum += vaddvq_s32(vmulq_s32(a, b));
                        }
                        for(int l = mt->b[mt->m]->y - (mt->b[mt->m]->y % CHUNK_SIZE); l < mt->b[mt->m]->y; l++) {
                            sum += (*mt->a)->m[get_idx(mt->i + k, mt->j + l, (*mt->a)->y)] * mt->b[mt->m]->m[get_idx(k, l, mt->b[mt->m]->y)];
                        }
                    }
                    mt->c[mt->m]->m[get_idx(mt->i, mt->j, mt->c[mt->m]->y)] = sum;
                #else
                    long CHUNK_SIZE = sizeof(float32x4_t) / sizeof(DATA_TYPE);
                    float32x4_t a, b;
                    DATA_TYPE sum = 0;
                    for(int k = 0; k < mt->b[mt->m]->x; k++) {
                        for(int l = 0; l + CHUNK_SIZE - 1 < mt->b[mt->m]->y; l += CHUNK_SIZE) {
                            a = vld1q_f32(&(*mt->a)->m[get_idx(mt->i + k, mt->j + l, (*mt->a)->y)]);
                            b = vld1q_f32(&mt->b[mt->m]->m[get_idx(k, l, mt->b[mt->m]->y)]);
                            sum += vaddvq_f32(vmulq_f32(a, b));
                        }
                        for(int l = mt->b[mt->m]->y - (mt->b[mt->m]->y % CHUNK_SIZE); l < mt->b[mt->m]->y; l++) {
                            sum += (*mt->a)->m[get_idx(mt->i + k, mt->j + l, (*mt->a)->y)] * mt->b[mt->m]->m[get_idx(k, l, mt->b[mt->m]->y)];
                        }
                    }
                    mt->c[mt->m]->m[get_idx(mt->i, mt->j, mt->c[mt->m]->y)] = sum;
                #endif
            #else
                #ifdef INT
                    // int16_t
                    const long LANE_SIZE = 32; // 32 int16_t values can be performed in one instruction
                    const long CHUNK_SIZE = 4 * LANE_SIZE;
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
                    const long CHUNK_SIZE = 4 * LANE_SIZE;
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
                        for(int l = mt->b[mt->m]->y - (mt->b[mt->m]->y % CHUNK_SIZE); l < mt->b[mt->m]->y; l++) {
                            sum += (*mt->a)->m[get_idx(mt->i + k, mt->j + l, (*mt->a)->y)] * mt->b[mt->m]->m[get_idx(k, l, mt->b[mt->m]->y)];
                        }
                    }
                    mt->c[mt->m]->m[get_idx(mt->i, mt->j, mt->c[mt->m]->y)] = sum;
                #endif
            #endif
        #endif
        */
    }

    __attribute__((always_inline)) inline void matmul_simd(mt_arg *mt) {
        /*
        #ifdef x86
            #ifdef __AVX512F__
                #ifdef INT
                    long CHUNK_SIZE = sizeof(__m512i) / sizeof(DATA_TYPE);
                    __m512i a, b;
                    __mmask16 m;
                    for(int k = 0; k < (*mt->a)->y; k += CHUNK_SIZE) {
                        m = (__mmask16)((1 << (((k + CHUNK_SIZE) <= (*mt->a)->y) ? CHUNK_SIZE : (*mt->a)->y - k)) - 1);
                        a = _mm512_maskz_loadu_epi32(m, &(*mt->a)->m[get_idx(mt->i, k, (*mt->a)->y)]);
                        b = _mm512_maskz_loadu_epi32(m, &(*mt->b)->m[get_idx(mt->j, k, (*mt->b)->y)]);
                        (*mt->c)->m[get_idx(mt->i, mt->j, (*mt->c)->y)] += _mm512_reduce_add_epi32(_mm512_mullo_epi32(a, b));
                    }
                #else
                    long CHUNK_SIZE = sizeof(__m512) / sizeof(DATA_TYPE);
                    __m512 a, b;
                    __mmask16 m;
                    for(int k = 0; k < (*mt->a)->y; k += CHUNK_SIZE) {
                        m = (__mmask16)((1 << (((k + CHUNK_SIZE) <= (*mt->a)->y) ? CHUNK_SIZE : (*mt->a)->y - k)) - 1);
                        a = _mm512_maskz_loadu_ps(m, &(*mt->a)->m[get_idx(mt->i, k, (*mt->a)->y)]);
                        b = _mm512_maskz_loadu_ps(m, &(*mt->b)->m[get_idx(mt->j, k, (*mt->b)->y)]);
                        (*mt->c)->m[get_idx(mt->i, mt->j, (*mt->c)->y)] += _mm512_reduce_add_ps(_mm512_mul_ps(a, b));
                    }
                #endif
            #else
                #ifdef INT
                    long CHUNK_SIZE = sizeof(__m256i) / sizeof(DATA_TYPE);
                    __m256i a, b, result;
                    for(int k = 0; k + CHUNK_SIZE - 1 < (*mt->a)->y; k += CHUNK_SIZE) {
                        a = _mm256_loadu_si256((__m256i*)&(*mt->a)->m[get_idx(mt->i, k, (*mt->a)->y)]);
                        b = _mm256_loadu_si256((__m256i*)&(*mt->b)->m[get_idx(mt->j, k, (*mt->b)->y)]);
                        result = _mm256_mullo_epi32(a, b);
                        (*mt->c)->m[get_idx(mt->i, mt->j, (*mt->c)->y)] += _mm_extract_epi32(_mm_hadd_epi32(_mm_hadd_epi32(_mm_add_epi32(_mm256_castsi256_si128(result), _mm256_extracti128_si256(result, 1)), _mm_setzero_si128()), _mm_setzero_si128()), 0);
                    }
                    for(int k = (*mt->a)->y - ((*mt->a)->y % CHUNK_SIZE); k < (*mt->a)->y; k++) {
                        (*mt->c)->m[get_idx(mt->i, mt->j, (*mt->c)->y)] += (*mt->a)->m[get_idx(mt->i, k, (*mt->a)->y)] * (*mt->b)->m[get_idx(mt->j, k, (*mt->b)->y)];
                    }
                #else
                    long CHUNK_SIZE = sizeof(__m256) / sizeof(DATA_TYPE);
                    __m256 a, b, result;
                    for(int k = 0; k + CHUNK_SIZE - 1 < (*mt->a)->y; k += CHUNK_SIZE) {
                        a = _mm256_loadu_ps(&(*mt->a)->m[get_idx(mt->i, k, (*mt->a)->y)]);
                        b = _mm256_loadu_ps(&(*mt->b)->m[get_idx(mt->j, k, (*mt->b)->y)]);
                        result = _mm256_mul_ps(a, b);
                        (*mt->c)->m[get_idx(mt->i, mt->j, (*mt->c)->y)] += _mm_cvtss_f32(_mm_hadd_ps(_mm_hadd_ps(_mm_add_ps(_mm256_castps256_ps128(result), _mm256_extractf128_ps(result, 1)), _mm_setzero_ps()), _mm_setzero_ps()));
                    }
                    for(int k = (*mt->a)->y - ((*mt->a)->y % CHUNK_SIZE); k < (*mt->a)->y; k++) {
                        (*mt->c)->m[get_idx(mt->i, mt->j, (*mt->c)->y)] += (*mt->a)->m[get_idx(mt->i, k, (*mt->a)->y)] * (*mt->b)->m[get_idx(mt->j, k, (*mt->b)->y)];
                    }
                #endif
            #endif
        #else
            #if !defined(AMX)
                #ifdef INT
                    long CHUNK_SIZE = sizeof(int32x4_t) / sizeof(DATA_TYPE);
                    int32x4_t a, b;
                    for(int k = 0; k + CHUNK_SIZE - 1 < (*mt->a)->y; k += CHUNK_SIZE) {
                        a = vld1q_s32(&(*mt->a)->m[get_idx(mt->i, k, (*mt->a)->y)]);
                        b = vld1q_s32(&(*mt->b)->m[get_idx(mt->j, k, (*mt->b)->y)]);
                        (*mt->c)->m[get_idx(mt->i, mt->j, (*mt->c)->y)] += vaddvq_s32(vmulq_s32(a, b));
                    }
                    for(int k = (*mt->a)->y - ((*mt->a)->y % CHUNK_SIZE); k < (*mt->a)->y; k++) {
                        (*mt->c)->m[get_idx(mt->i, mt->j, (*mt->c)->y)] += (*mt->a)->m[get_idx(mt->i, k, (*mt->a)->y)] * (*mt->b)->m[get_idx(mt->j, k, (*mt->b)->y)];
                    }
                #else
                    long CHUNK_SIZE = sizeof(float32x4_t) / sizeof(DATA_TYPE);
                    float32x4_t a, b;
                    for(int k = 0; k + CHUNK_SIZE - 1 < (*mt->a)->y; k += CHUNK_SIZE) {
                        a = vld1q_f32(&(*mt->a)->m[get_idx(mt->i, k, (*mt->a)->y)]);
                        b = vld1q_f32(&(*mt->b)->m[get_idx(mt->j, k, (*mt->b)->y)]);
                        (*mt->c)->m[get_idx(mt->i, mt->j, (*mt->c)->y)] += vaddvq_f32(vmulq_f32(a, b));
                    }
                    for(int k = (*mt->a)->y - ((*mt->a)->y % CHUNK_SIZE); k < (*mt->a)->y; k++) {
                        (*mt->c)->m[get_idx(mt->i, mt->j, (*mt->c)->y)] += (*mt->a)->m[get_idx(mt->i, k, (*mt->a)->y)] * (*mt->b)->m[get_idx(mt->j, k, (*mt->b)->y)];
                    }
                #endif
            #else
                #ifdef INT
                    // int16_t
                    const long LANE_SIZE = 32; // only 32 values can be performed at once, because int16_t is 2B
                    const long CHUNK_SIZE = 4 * LANE_SIZE;
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
                    const long CHUNK_SIZE = 4 * LANE_SIZE;
                    DATA_TYPE z_reg[LANE_SIZE];
                    DATA_TYPE sum = 0;
                    uint64_t ldx = (uint64_t)&(*mt->a)->m[get_idx(mt->i, 0, (*mt->a)->y)];
                    ldx = ldx | 1ull << 60; // four registers
                    ldx = ldx | 1ull << 62; // multiple registers
                    uint64_t ldy = (uint64_t)&(*mt->b)->m[get_idx(mt->j, 0, (*mt->b)->y)];
                    ldy = ldy | 1ull << 60; // four registers
                    ldy = ldy | 1ull << 62; // multiple registers
                    uint64_t fma32 = 1ull << 63; // vector mode
                    uint64_t stz = (uint64_t)&z_reg;
                    AMX_SET();
                    for(int k = 0; k + CHUNK_SIZE - 1 < (*mt->a)->y; k += CHUNK_SIZE) {
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
                    for(int k = (*mt->a)->y - ((*mt->a)->y % CHUNK_SIZE); k < (*mt->a)->y; k++) {
                        (*mt->c)->m[get_idx(mt->i, mt->j, (*mt->c)->y)] += (*mt->a)->m[get_idx(mt->i, k, (*mt->a)->y)] * (*mt->b)->m[get_idx(mt->j, k, (*mt->b)->y)];
                    }
                #endif
            #endif
        #endif
        */
    }
#endif
