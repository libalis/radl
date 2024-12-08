#include "../hpp/mt.hpp"
#include "../hpp/simd.hpp"
#include "../hpp/utils.hpp"

void add_simd(mt_arg *mt) {
    #ifdef x86
        __m128 a, b;
    #else
        float32x4_t a, b;
    #endif
    for(int j = 0; j + CHUNK_SIZE - 1 < mt->c->y; j += CHUNK_SIZE) {
        #ifdef x86
            a = _mm_loadu_ps(&mt->a->m[get_idx(mt->i, j, mt->a->y)]);
            b = _mm_loadu_ps(&mt->b->m[get_idx(mt->i, j, mt->b->y)]);
            _mm_storeu_ps(&mt->c->m[get_idx(mt->i, j, mt->c->y)], _mm_add_ps(a, b));
        #else
            a = vld1q_f32(&mt->a->m[get_idx(mt->i, j, mt->a->y)]);
            b = vld1q_f32(&mt->b->m[get_idx(mt->i, j, mt->b->y)]);
            vst1q_f32(&mt->c->m[get_idx(mt->i, j, mt->c->y)], vaddq_f32(a, b));
        #endif
    }
    for(int j = mt->c->y - (mt->c->y % CHUNK_SIZE); j < mt->c->y; j++) {
        mt->c->m[get_idx(mt->i, j, mt->c->y)] = mt->a->m[get_idx(mt->i, j, mt->a->y)] + mt->b->m[get_idx(mt->i, j, mt->b->y)];
    }
}

void biasing_simd(mt_arg *mt) {
    #ifdef x86
        __m128 a, b;
    #else
        float32x4_t a, b;
    #endif
    for(int j = 0; j + CHUNK_SIZE - 1 < mt->a_ptr[mt->m]->y; j += CHUNK_SIZE) {
        #ifdef x86
            a = _mm_loadu_ps(&mt->a_ptr[mt->m]->m[get_idx(mt->i, j, mt->a_ptr[mt->m]->y)]);
            b = _mm_load_ps1(&mt->b->m[get_idx(mt->m, 0, mt->b->y)]);
            _mm_storeu_ps(&mt->c_ptr[mt->m]->m[get_idx(mt->i, j, mt->c_ptr[mt->m]->y)], _mm_add_ps(a, b));
        #else
            a = vld1q_f32(&mt->a_ptr[mt->m]->m[get_idx(mt->i, j, mt->a_ptr[mt->m]->y)]);
            b = vdupq_n_f32(mt->b->m[get_idx(mt->m, 0, mt->b->y)]);
            vst1q_f32(&mt->c_ptr[mt->m]->m[get_idx(mt->i, j, mt->c_ptr[mt->m]->y)], vaddq_f32(a, b));
        #endif
    }
    for(int j = mt->a_ptr[mt->m]->y - (mt->a_ptr[mt->m]->y % CHUNK_SIZE); j < mt->a_ptr[mt->m]->y; j++) {
        mt->c_ptr[mt->m]->m[get_idx(mt->i, j, mt->c_ptr[mt->m]->y)] = mt->a_ptr[mt->m]->m[get_idx(mt->i, j, mt->a_ptr[mt->m]->y)] + mt->b->m[get_idx(mt->m, 0, mt->b->y)];
    }
}

void conv2d_simd(mt_arg *mt) {
    #ifdef x86
        __m128 a, b;
    #else
        float32x4_t a, b;
    #endif
    float sum = 0.0;
    for(int k = 0; k < mt->b_ptr[mt->m]->x; k++) {
        for(int l = 0; l + CHUNK_SIZE - 1 < mt->b_ptr[mt->m]->y; l += CHUNK_SIZE) {
            #ifdef x86
                a = _mm_set_ps(mt->a->m[get_idx(mt->i + k + 3, mt->j + l, mt->a->y)],
                               mt->a->m[get_idx(mt->i + k + 2, mt->j + l, mt->a->y)],
                               mt->a->m[get_idx(mt->i + k + 1, mt->j + l, mt->a->y)],
                               mt->a->m[get_idx(mt->i + k, mt->j + l, mt->a->y)]);
                b = _mm_set_ps(mt->b_ptr[mt->m]->m[get_idx(k + 3, l, mt->b_ptr[mt->m]->y)],
                               mt->b_ptr[mt->m]->m[get_idx(k + 2, l, mt->b_ptr[mt->m]->y)],
                               mt->b_ptr[mt->m]->m[get_idx(k + 1, l, mt->b_ptr[mt->m]->y)],
                               mt->b_ptr[mt->m]->m[get_idx(k, l, mt->b_ptr[mt->m]->y)]);
                sum += _mm_cvtss_f32(_mm_hadd_ps(_mm_hadd_ps(_mm_mul_ps(a, b), zero), zero));
            #else
                a = vsetq_lane_f32(mt->a->m[get_idx(mt->i + k, mt->j + l, mt->a->y)], a, 0);
                a = vsetq_lane_f32(mt->a->m[get_idx(mt->i + k + 1, mt->j + l, mt->a->y)], a, 1);
                a = vsetq_lane_f32(mt->a->m[get_idx(mt->i + k + 2, mt->j + l, mt->a->y)], a, 2);
                a = vsetq_lane_f32(mt->a->m[get_idx(mt->i + k + 3, mt->j + l, mt->a->y)], a, 3);
                b = vsetq_lane_f32(mt->b_ptr[mt->m]->m[get_idx(k, l, mt->b_ptr[mt->m]->y)], b, 0);
                b = vsetq_lane_f32(mt->b_ptr[mt->m]->m[get_idx(k + 1, l, mt->b_ptr[mt->m]->y)], b, 1);
                b = vsetq_lane_f32(mt->b_ptr[mt->m]->m[get_idx(k + 2, l, mt->b_ptr[mt->m]->y)], b, 2);
                b = vsetq_lane_f32(mt->b_ptr[mt->m]->m[get_idx(k + 3, l, mt->b_ptr[mt->m]->y)], b, 3);
                sum += vaddvq_f32(vmulq_f32(a, b));
            #endif
        }
        for(int l = mt->b_ptr[mt->m]->y - (mt->b_ptr[mt->m]->y % CHUNK_SIZE); l < mt->b_ptr[mt->m]->y; l++) {
            sum += mt->a->m[get_idx(mt->i + k, mt->j + l, mt->a->y)] * mt->b_ptr[mt->m]->m[get_idx(k, l, mt->b_ptr[mt->m]->y)];
        }
    }
    mt->c_ptr[mt->m]->m[get_idx(mt->i, mt->j, mt->c_ptr[mt->m]->y)] = sum;
}

void matmul_simd(mt_arg *mt) {
    #ifdef x86
        __m128 a, b;
    #else
        float32x4_t a, b;
    #endif
    for(int k = 0; k + CHUNK_SIZE - 1 < mt->a->y; k += CHUNK_SIZE) {
        #ifdef x86
            a = _mm_loadu_ps(&mt->a->m[get_idx(mt->i, k, mt->a->y)]);
            b = _mm_set_ps(mt->b->m[get_idx(k + 3, mt->j, mt->b->y)],
                           mt->b->m[get_idx(k + 2, mt->j, mt->b->y)],
                           mt->b->m[get_idx(k + 1, mt->j, mt->b->y)],
                           mt->b->m[get_idx(k, mt->j, mt->b->y)]);
            mt->c->m[get_idx(mt->i, mt->j, mt->c->y)] += _mm_cvtss_f32(_mm_hadd_ps(_mm_hadd_ps(_mm_mul_ps(a, b), zero), zero));
        #else
            a = vld1q_f32(&mt->a->m[get_idx(mt->i, k, mt->a->y)]);
            b = vsetq_lane_f32(mt->b->m[get_idx(k, mt->j, mt->b->y)], b, 0);
            b = vsetq_lane_f32(mt->b->m[get_idx(k + 1, mt->j, mt->b->y)], b, 1);
            b = vsetq_lane_f32(mt->b->m[get_idx(k + 2, mt->j, mt->b->y)], b, 2);
            b = vsetq_lane_f32(mt->b->m[get_idx(k + 3, mt->j, mt->b->y)], b, 3);
            mt->c->m[get_idx(mt->i, mt->j, mt->c->y)] += vaddvq_f32(vmulq_f32(a, b));
        #endif
    }
    for(int k = mt->a->y - (mt->a->y % CHUNK_SIZE); k < mt->a->y; k++) {
        mt->c->m[get_idx(mt->i, mt->j, mt->c->y)] += mt->a->m[get_idx(mt->i, k, mt->a->y)] * mt->b->m[get_idx(k, mt->j, mt->b->y)];
    }
}
