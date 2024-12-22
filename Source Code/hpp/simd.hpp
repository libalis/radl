#ifndef SIMD_H
    #define SIMD_H

    #if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
        #define x86
        #include <immintrin.h>
    #elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
        #define ARM
        #include <arm_neon.h>
    #endif

    #ifdef x86
        #define VECTOR_SIZE (sizeof(__m512))
    #else
        #define VECTOR_SIZE (sizeof(float32x4_t))
    #endif

    #define CHUNK_SIZE (VECTOR_SIZE / sizeof(DATA_TYPE))

    #include "../hpp/mt.hpp"
    #include "../hpp/utils.hpp"

    __attribute__((always_inline)) inline void add_simd(mt_arg *mt) {
        #ifdef x86
            __m512 a, b;
            __mmask16 m;
        #else
            float32x4_t a, b;
        #endif
        for(int j = 0; j < mt->c->y; j += CHUNK_SIZE) {
            #ifdef x86
                m = (__mmask16)((1 << (((j + CHUNK_SIZE) <= mt->c->y) ? CHUNK_SIZE : mt->c->y - j)) - 1);
                a = _mm512_maskz_loadu_ps(m, &mt->a->m[get_idx(mt->i, j, mt->a->y)]);
                b = _mm512_maskz_loadu_ps(m, &mt->b->m[get_idx(mt->i, j, mt->b->y)]);
                _mm512_mask_storeu_ps(&mt->c->m[get_idx(mt->i, j, mt->c->y)], m, _mm512_add_ps(a, b));
            #else
                a = vld1q_f32(&mt->a->m[get_idx(mt->i, j, mt->a->y)]);
                b = vld1q_f32(&mt->b->m[get_idx(mt->i, j, mt->b->y)]);
                vst1q_f32(&mt->c->m[get_idx(mt->i, j, mt->c->y)], vaddq_f32(a, b));
            #endif
        }
    }

    __attribute__((always_inline)) inline void biasing_simd(mt_arg *mt) {
        #ifdef x86
            __m512 a, b;
            __mmask16 m;
        #else
            float32x4_t a, b;
        #endif
        for(int j = 0; j < mt->a_ptr[mt->m]->y; j += CHUNK_SIZE) {
            #ifdef x86
                m = (__mmask16)((1 << (((j + CHUNK_SIZE) <= mt->a_ptr[mt->m]->y) ? CHUNK_SIZE : mt->a_ptr[mt->m]->y - j)) - 1);
                a = _mm512_maskz_loadu_ps(m, &mt->a_ptr[mt->m]->m[get_idx(mt->i, j, mt->a_ptr[mt->m]->y)]);
                b = _mm512_set1_ps(mt->b->m[get_idx(mt->m, 0, mt->b->y)]);
                _mm512_mask_storeu_ps(&mt->c_ptr[mt->m]->m[get_idx(mt->i, j, mt->c_ptr[mt->m]->y)], m, _mm512_add_ps(a, b));
            #else
                a = vld1q_f32(&mt->a_ptr[mt->m]->m[get_idx(mt->i, j, mt->a_ptr[mt->m]->y)]);
                b = vdupq_n_f32(mt->b->m[get_idx(mt->m, 0, mt->b->y)]);
                vst1q_f32(&mt->c_ptr[mt->m]->m[get_idx(mt->i, j, mt->c_ptr[mt->m]->y)], vaddq_f32(a, b));
            #endif
        }
    }

    __attribute__((always_inline)) inline void conv2d_simd(mt_arg *mt) {
        #ifdef x86
            __m512 a, b;
            __mmask16 m;
        #else
            float32x4_t a, b;
        #endif
        DATA_TYPE sum = 0;
        for(int k = 0; k < mt->b_ptr[mt->m]->x; k++) {
            for(int l = 0; l < mt->b_ptr[mt->m]->y; l += CHUNK_SIZE) {
                #ifdef x86
                    m = (__mmask16)((1 << (((l + CHUNK_SIZE) <= mt->b_ptr[mt->m]->y) ? CHUNK_SIZE : mt->b_ptr[mt->m]->y - l)) - 1);
                    a = _mm512_maskz_loadu_ps(m, &mt->a->m[get_idx(mt->i + k, mt->j + l, mt->a->y)]);
                    b = _mm512_maskz_loadu_ps(m, &mt->b_ptr[mt->m]->m[get_idx(k, l, mt->b_ptr[mt->m]->y)]);
                    sum += _mm512_reduce_add_ps(_mm512_mul_ps(a, b));
                #else
                    a = vld1q_f32(&mt->a->m[get_idx(mt->i + k, mt->j + l, mt->a->y)]);
                    b = vld1q_f32(&mt->b_ptr[mt->m]->m[get_idx(k, l, mt->b_ptr[mt->m]->y)]);
                    sum += vaddvq_f32(vmulq_f32(a, b));
                #endif
            }
        }
        mt->c_ptr[mt->m]->m[get_idx(mt->i, mt->j, mt->c_ptr[mt->m]->y)] = sum;
    }

    __attribute__((always_inline)) inline void matmul_simd(mt_arg *mt) {
        #ifdef x86
            __m512 a, b;
            __mmask16 m;
        #else
            float32x4_t a, b;
        #endif
        for(int k = 0; k < mt->a->y; k += CHUNK_SIZE) {
            #ifdef x86
                m = (__mmask16)((1 << (((k + CHUNK_SIZE) <= mt->a->y) ? CHUNK_SIZE : mt->a->y - k)) - 1);
                a = _mm512_maskz_loadu_ps(m, &mt->a->m[get_idx(mt->i, k, mt->a->y)]);
                b = _mm512_maskz_loadu_ps(m, &mt->b->m[get_idx(mt->j, k, mt->b->y)]);
                mt->c->m[get_idx(mt->i, mt->j, mt->c->y)] += _mm512_reduce_add_ps(_mm512_mul_ps(a, b));
            #else
                a = vld1q_f32(&mt->a->m[get_idx(mt->i, k, mt->a->y)]);
                b = vld1q_f32(&mt->b->m[get_idx(mt->j, k, mt->b->y)]);
                mt->c->m[get_idx(mt->i, mt->j, mt->c->y)] += vaddvq_f32(vmulq_f32(a, b));
            #endif
        }
    }
#endif
