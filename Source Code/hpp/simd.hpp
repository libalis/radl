#ifndef SIMD_H
    #define SIMD_H

    #include "../hpp/mt.hpp"
    #include "../hpp/utils.hpp"

    __attribute__((always_inline)) inline void add_simd(mt_arg *mt) {
        #ifdef x86
            if(is_avx512_supported()) {
                long CHUNK_SIZE = sizeof(__m512) / sizeof(DATA_TYPE);
                __m512 a, b;
                __mmask16 m;
                for(int j = 0; j < mt->c->y; j += CHUNK_SIZE) {
                        m = (__mmask16)((1 << (((j + CHUNK_SIZE) <= mt->c->y) ? CHUNK_SIZE : mt->c->y - j)) - 1);
                        a = _mm512_maskz_loadu_ps(m, &mt->a->m[get_idx(mt->i, j, mt->a->y)]);
                        b = _mm512_maskz_loadu_ps(m, &mt->b->m[get_idx(mt->i, j, mt->b->y)]);
                        _mm512_mask_storeu_ps(&mt->c->m[get_idx(mt->i, j, mt->c->y)], m, _mm512_add_ps(a, b));
                }
            } else {
                long CHUNK_SIZE = sizeof(__m256) / sizeof(DATA_TYPE);
                __m256 a, b;
                for(int j = 0; j + CHUNK_SIZE - 1 < mt->c->y; j += CHUNK_SIZE) {
                    a = _mm256_loadu_ps(&mt->a->m[get_idx(mt->i, j, mt->a->y)]);
                    b = _mm256_loadu_ps(&mt->b->m[get_idx(mt->i, j, mt->b->y)]);
                    _mm256_storeu_ps(&mt->c->m[get_idx(mt->i, j, mt->c->y)], _mm256_add_ps(a, b));
                }
                for(int j = mt->c->y - (mt->c->y % CHUNK_SIZE); j < mt->c->y; j++) {
                    mt->c->m[get_idx(mt->i, j, mt->c->y)] = mt->a->m[get_idx(mt->i, j, mt->a->y)] + mt->b->m[get_idx(mt->i, j, mt->b->y)];
                }
            }
        #else
            long CHUNK_SIZE = sizeof(float32x4_t) / sizeof(DATA_TYPE);
            float32x4_t a, b;
            for(int j = 0; j + CHUNK_SIZE - 1 < mt->c->y; j += CHUNK_SIZE) {
                a = vld1q_f32(&mt->a->m[get_idx(mt->i, j, mt->a->y)]);
                b = vld1q_f32(&mt->b->m[get_idx(mt->i, j, mt->b->y)]);
                vst1q_f32(&mt->c->m[get_idx(mt->i, j, mt->c->y)], vaddq_f32(a, b));
            }
            for(int j = mt->c->y - (mt->c->y % CHUNK_SIZE); j < mt->c->y; j++) {
                mt->c->m[get_idx(mt->i, j, mt->c->y)] = mt->a->m[get_idx(mt->i, j, mt->a->y)] + mt->b->m[get_idx(mt->i, j, mt->b->y)];
            }
        #endif
    }

    __attribute__((always_inline)) inline void biasing_simd(mt_arg *mt) {
        #ifdef x86
            if(is_avx512_supported()) {
                long CHUNK_SIZE = sizeof(__m512) / sizeof(DATA_TYPE);
                __m512 a, b;
                __mmask16 m;
                for(int j = 0; j < mt->a_ptr[mt->m]->y; j += CHUNK_SIZE) {
                    m = (__mmask16)((1 << (((j + CHUNK_SIZE) <= mt->a_ptr[mt->m]->y) ? CHUNK_SIZE : mt->a_ptr[mt->m]->y - j)) - 1);
                    a = _mm512_maskz_loadu_ps(m, &mt->a_ptr[mt->m]->m[get_idx(mt->i, j, mt->a_ptr[mt->m]->y)]);
                    b = _mm512_set1_ps(mt->b->m[get_idx(mt->m, 0, mt->b->y)]);
                    _mm512_mask_storeu_ps(&mt->c_ptr[mt->m]->m[get_idx(mt->i, j, mt->c_ptr[mt->m]->y)], m, _mm512_add_ps(a, b));
                }
            } else {
                long CHUNK_SIZE = sizeof(__m256) / sizeof(DATA_TYPE);
                __m256 a, b;
                for(int j = 0; j + CHUNK_SIZE - 1 < mt->a_ptr[mt->m]->y; j += CHUNK_SIZE) {
                    a = _mm256_loadu_ps(&mt->a_ptr[mt->m]->m[get_idx(mt->i, j, mt->a_ptr[mt->m]->y)]);
                    b = _mm256_set1_ps(mt->b->m[get_idx(mt->m, 0, mt->b->y)]);
                    _mm256_storeu_ps(&mt->c_ptr[mt->m]->m[get_idx(mt->i, j, mt->c_ptr[mt->m]->y)], _mm256_add_ps(a, b));
                }
                for(int j = mt->a_ptr[mt->m]->y - (mt->a_ptr[mt->m]->y % CHUNK_SIZE); j < mt->a_ptr[mt->m]->y; j++) {
                    mt->c_ptr[mt->m]->m[get_idx(mt->i, j, mt->c_ptr[mt->m]->y)] = mt->a_ptr[mt->m]->m[get_idx(mt->i, j, mt->a_ptr[mt->m]->y)] + mt->b->m[get_idx(mt->m, 0, mt->b->y)];
                }
            }
        #else
            long CHUNK_SIZE = sizeof(float32x4_t) / sizeof(DATA_TYPE);
            float32x4_t a, b;
            for(int j = 0; j + CHUNK_SIZE - 1 < mt->a_ptr[mt->m]->y; j += CHUNK_SIZE) {
                a = vld1q_f32(&mt->a_ptr[mt->m]->m[get_idx(mt->i, j, mt->a_ptr[mt->m]->y)]);
                b = vdupq_n_f32(mt->b->m[get_idx(mt->m, 0, mt->b->y)]);
                vst1q_f32(&mt->c_ptr[mt->m]->m[get_idx(mt->i, j, mt->c_ptr[mt->m]->y)], vaddq_f32(a, b));
            }
            for(int j = mt->a_ptr[mt->m]->y - (mt->a_ptr[mt->m]->y % CHUNK_SIZE); j < mt->a_ptr[mt->m]->y; j++) {
                mt->c_ptr[mt->m]->m[get_idx(mt->i, j, mt->c_ptr[mt->m]->y)] = mt->a_ptr[mt->m]->m[get_idx(mt->i, j, mt->a_ptr[mt->m]->y)] + mt->b->m[get_idx(mt->m, 0, mt->b->y)];
            }
        #endif
    }

    __attribute__((always_inline)) inline void conv2d_simd(mt_arg *mt) {
        #ifdef x86
            if(is_avx512_supported()) {
                long CHUNK_SIZE = sizeof(__m512) / sizeof(DATA_TYPE);
                __m512 a, b;
                __mmask16 m;
                DATA_TYPE sum = 0;
                for(int k = 0; k < mt->b_ptr[mt->m]->x; k++) {
                    for(int l = 0; l < mt->b_ptr[mt->m]->y; l += CHUNK_SIZE) {
                        m = (__mmask16)((1 << (((l + CHUNK_SIZE) <= mt->b_ptr[mt->m]->y) ? CHUNK_SIZE : mt->b_ptr[mt->m]->y - l)) - 1);
                        a = _mm512_maskz_loadu_ps(m, &mt->a->m[get_idx(mt->i + k, mt->j + l, mt->a->y)]);
                        b = _mm512_maskz_loadu_ps(m, &mt->b_ptr[mt->m]->m[get_idx(k, l, mt->b_ptr[mt->m]->y)]);
                        sum += _mm512_reduce_add_ps(_mm512_mul_ps(a, b));
                    }
                }
                mt->c_ptr[mt->m]->m[get_idx(mt->i, mt->j, mt->c_ptr[mt->m]->y)] = sum;
            } else {
                long CHUNK_SIZE = sizeof(__m256) / sizeof(DATA_TYPE);
                __m256 a, b, result;
                DATA_TYPE sum = 0;
                for(int k = 0; k < mt->b_ptr[mt->m]->x; k++) {
                    for(int l = 0; l + CHUNK_SIZE - 1 < mt->b_ptr[mt->m]->y; l += CHUNK_SIZE) {
                        a = _mm256_loadu_ps(&mt->a->m[get_idx(mt->i + k, mt->j + l, mt->a->y)]);
                        b = _mm256_loadu_ps(&mt->b_ptr[mt->m]->m[get_idx(k, l, mt->b_ptr[mt->m]->y)]);
                        result = _mm256_mul_ps(a, b);
                        sum += _mm_cvtss_f32(_mm_hadd_ps(_mm_hadd_ps(_mm_add_ps(_mm256_castps256_ps128(result), _mm256_extractf128_ps(result, 1)), _mm_setzero_ps()), _mm_setzero_ps()));
                    }
                    for(int l = mt->b_ptr[mt->m]->y - (mt->b_ptr[mt->m]->y % CHUNK_SIZE); l < mt->b_ptr[mt->m]->y; l++) {
                        sum += mt->a->m[get_idx(mt->i + k, mt->j + l, mt->a->y)] * mt->b_ptr[mt->m]->m[get_idx(k, l, mt->b_ptr[mt->m]->y)];
                    }
                }
                mt->c_ptr[mt->m]->m[get_idx(mt->i, mt->j, mt->c_ptr[mt->m]->y)] = sum;
            }
        #else
            #ifndef AMX
                long CHUNK_SIZE = sizeof(float32x4_t) / sizeof(DATA_TYPE);
                float32x4_t a, b;
                DATA_TYPE sum = 0;
                for(int k = 0; k < mt->b_ptr[mt->m]->x; k++) {
                    for(int l = 0; l + CHUNK_SIZE - 1 < mt->b_ptr[mt->m]->y; l += CHUNK_SIZE) {
                        a = vld1q_f32(&mt->a->m[get_idx(mt->i + k, mt->j + l, mt->a->y)]);
                        b = vld1q_f32(&mt->b_ptr[mt->m]->m[get_idx(k, l, mt->b_ptr[mt->m]->y)]);
                        sum += vaddvq_f32(vmulq_f32(a, b));
                    }
                    for(int l = mt->b_ptr[mt->m]->y - (mt->b_ptr[mt->m]->y % CHUNK_SIZE); l < mt->b_ptr[mt->m]->y; l++) {
                        sum += mt->a->m[get_idx(mt->i + k, mt->j + l, mt->a->y)] * mt->b_ptr[mt->m]->m[get_idx(k, l, mt->b_ptr[mt->m]->y)];
                    }
                }
                mt->c_ptr[mt->m]->m[get_idx(mt->i, mt->j, mt->c_ptr[mt->m]->y)] = sum;
            #else
                long CHUNK_SIZE = 32;
                // length of reg_z = CHUNK_SIZE * CHUNK_SIZE
                float sum_arr[1024] = {0.0};
                DATA_TYPE sum = 0.0;
                for(int k = 0; k < mt->b_ptr[mt->m]->x; k++) {
                    for(int l = 0; l + CHUNK_SIZE - 1 < mt->b_ptr[mt->m]->y; l += CHUNK_SIZE) {
                        AMX_SET();
                        for(int o = 0; o < CHUNK_SIZE; o++) {
                            AMX_LDX(&mt->a->m[get_idx(mt->i + k, mt->j + l + o, mt->a->y)]);
                            AMX_LDY(&mt->b_ptr[mt->m]->m[get_idx(k, l + o, mt->b_ptr[mt->m]->y)]);
                            AMX_FMA32(1ull << 62);
                            AMX_STZ(&sum_arr + o * 128);
                        }
                        AMX_CLR();
                        for(int i = 0; i < 1024; i++) {
                            sum += sum_arr[i];
                            sum_arr[i] = 0.0;
                        }
                    }
                    for(int l = mt->b_ptr[mt->m]->y - (mt->b_ptr[mt->m]->y % CHUNK_SIZE); l < mt->b_ptr[mt->m]->y; l++) {
                        sum += mt->a->m[get_idx(mt->i + k, mt->j + l, mt->a->y)] * mt->b_ptr[mt->m]->m[get_idx(k, l, mt->b_ptr[mt->m]->y)];
                    }
                }
                mt->c_ptr[mt->m]->m[get_idx(mt->i, mt->j, mt->c_ptr[mt->m]->y)] = sum;
            #endif
        #endif
    }

    __attribute__((always_inline)) inline void matmul_simd(mt_arg *mt) {
        #ifdef x86
            if(is_avx512_supported()) {
                long CHUNK_SIZE = sizeof(__m512) / sizeof(DATA_TYPE);
                __m512 a, b;
                __mmask16 m;
                for(int k = 0; k < mt->a->y; k += CHUNK_SIZE) {
                    m = (__mmask16)((1 << (((k + CHUNK_SIZE) <= mt->a->y) ? CHUNK_SIZE : mt->a->y - k)) - 1);
                    a = _mm512_maskz_loadu_ps(m, &mt->a->m[get_idx(mt->i, k, mt->a->y)]);
                    b = _mm512_maskz_loadu_ps(m, &mt->b->m[get_idx(mt->j, k, mt->b->y)]);
                    mt->c->m[get_idx(mt->i, mt->j, mt->c->y)] += _mm512_reduce_add_ps(_mm512_mul_ps(a, b));
                }
            } else {
                long CHUNK_SIZE = sizeof(__m256) / sizeof(DATA_TYPE);
                __m256 a, b, result;
                for(int k = 0; k + CHUNK_SIZE - 1 < mt->a->y; k += CHUNK_SIZE) {
                    a = _mm256_loadu_ps(&mt->a->m[get_idx(mt->i, k, mt->a->y)]);
                    b = _mm256_loadu_ps(&mt->b->m[get_idx(mt->j, k, mt->b->y)]);
                    result = _mm256_mul_ps(a, b);
                    mt->c->m[get_idx(mt->i, mt->j, mt->c->y)] += _mm_cvtss_f32(_mm_hadd_ps(_mm_hadd_ps(_mm_add_ps(_mm256_castps256_ps128(result), _mm256_extractf128_ps(result, 1)), _mm_setzero_ps()), _mm_setzero_ps()));
                }
                for(int k = mt->a->y - (mt->a->y % CHUNK_SIZE); k < mt->a->y; k++) {
                    mt->c->m[get_idx(mt->i, mt->j, mt->c->y)] += mt->a->m[get_idx(mt->i, k, mt->a->y)] * mt->b->m[get_idx(mt->j, k, mt->b->y)];
                }
            }
        #else
            #ifndef AMX
                long CHUNK_SIZE = sizeof(float32x4_t) / sizeof(DATA_TYPE);
                float32x4_t a, b;
                for(int k = 0; k + CHUNK_SIZE - 1 < mt->a->y; k += CHUNK_SIZE) {
                    a = vld1q_f32(&mt->a->m[get_idx(mt->i, k, mt->a->y)]);
                    b = vld1q_f32(&mt->b->m[get_idx(mt->j, k, mt->b->y)]);
                    mt->c->m[get_idx(mt->i, mt->j, mt->c->y)] += vaddvq_f32(vmulq_f32(a, b));
                }
                for(int k = mt->a->y - (mt->a->y % CHUNK_SIZE); k < mt->a->y; k++) {
                    mt->c->m[get_idx(mt->i, mt->j, mt->c->y)] += mt->a->m[get_idx(mt->i, k, mt->a->y)] * mt->b->m[get_idx(mt->j, k, mt->b->y)];
                }
            #else
                long CHUNK_SIZE = 32;
                uint64_t reset_z = 1ull << 62;
                AMX_SET();
                for(int k = 0; k < mt->a->y; k += CHUNK_SIZE) {
                    for(int o = 0; o < CHUNK_SIZE; o++) {
                        AMX_LDX(&mt->a->m[get_idx(mt->i, k + o, mt->a->y)]);
                        AMX_LDY(&mt->b->m[get_idx(mt->j, k + o, mt->b->y)]);
                        AMX_FMA32(reset_z);
                        reset_z = 0;
                        AMX_STZ(&mt->c->m[get_idx(mt->i, mt->j, mt->c->y)] + o * 64);
                    }
                    // for(int i = 0; i < CHUNK_SIZE; i++) {
                    //     // safe 1 float = 4B --> 16 float = 64B
                    //     AMX_STZ(&mt->c->m[get_idx(mt->i, mt->j, mt->c->y)] + i * 64);
                    // }
                }
                AMX_CLR();
            #endif
        #endif
    }
#endif
