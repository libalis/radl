#include <math.h>
#include <pthread.h>
#include <stdlib.h>

#include "../h/mt.h"

void* add_routine(void* arg) {
    mt_arg* mt = arg;
    for(int i = mt->idx; i < mt->c->x; i += THREADS) {
        for(int j = 0; j < mt->c->y; j++) {
            mt->c->m[i][j] = mt->a->m[i][j] + mt->b->m[i][j];
        }
    }
    return NULL;
}

void* matmul_routine(void* arg) {
    mt_arg* mt = arg;
    for(int i = mt->idx; i < mt->c->x; i += THREADS) {
        for(int k = 0; k < mt->c->y; k++) {
            for(int j = 0; j < mt->a->y; j++) {
                mt->c->m[i][k] = mt->c->m[i][k] + mt->a->m[i][j] * mt->b->m[j][k];
            }
        }
    }
    return NULL;
}

void* flatten_routine(void* arg) {
    mt_arg* mt = arg;
    for(int i = mt->idx; i < mt->a_ptr[0]->x; i += THREADS) {
        for(int j = 0; j < mt->a_ptr[0]->y; j++) {
            for(int m = 0; m < mt->len; m++) {
                int index = i * mt->a_ptr[0]->y * mt->len + j * mt->len + m;
                mt->c->m[index][0] = mt->a_ptr[m]->m[i][j];
            }
        }
    }
    return NULL;
}

void* maxpool_routine(void* arg) {
    mt_arg* mt = arg;
    int pool_len = 2;
    for(int i = mt->idx * pool_len; i < mt->a_ptr[mt->m]->x; i += THREADS * pool_len) {
        for(int j = 0; j < mt->a_ptr[mt->m]->y; j += pool_len) {
            float max_val = mt->a_ptr[mt->m]->m[i][j];
            for(int k = 0; k < pool_len; k++) {
                for(int l = 0; l < pool_len; l++) {
                    float curr_val = mt->a_ptr[mt->m]->m[i + k][j + l];
                    if(curr_val > max_val) {
                        max_val = curr_val;
                    }
                }
            }
            mt->c_ptr[mt->m]->m[i / pool_len][j / pool_len] = max_val;
        }
    }
    return NULL;
}

void* hyperbolic_tangent_routine(void* arg) {
    mt_arg* mt = arg;
    for(int i = mt->idx; i < mt->a_ptr[mt->m]->x; i += THREADS) {
        for(int j = 0; j < mt->a_ptr[mt->m]->y; j++) {
            mt->c_ptr[mt->m]->m[i][j] = tanh(mt->a_ptr[mt->m]->m[i][j]);
        }
    }
    return NULL;
}

void* relu_routine(void* arg) {
    mt_arg* mt = arg;
    for(int i = mt->idx; i < mt->a_ptr[mt->m]->x; i += THREADS) {
        for(int j = 0; j < mt->a_ptr[mt->m]->y; j++) {
            if(mt->a_ptr[mt->m]->m[i][j] < 0.0) {
                mt->c_ptr[mt->m]->m[i][j] = 0.0;
            } else {
                mt->c_ptr[mt->m]->m[i][j] = mt->a_ptr[mt->m]->m[i][j];
            }
        }
    }
    return NULL;
}

void* biasing_routine(void* arg) {
    mt_arg* mt = arg;
    for(int i = mt->idx; i < mt->a_ptr[mt->m]->x; i += THREADS) {
        for(int j = 0; j < mt->a_ptr[mt->m]->y; j++) {
            mt->c_ptr[mt->m]->m[i][j] = mt->a_ptr[mt->m]->m[i][j] + mt->b->m[mt->m][0];
        }
    }
    return NULL;
}

void* conv2d_routine(void* arg) {
    mt_arg* mt = arg;
    int cx = mt->a->x - mt->b_ptr[mt->m]->x + 1;
    int cy = mt->a->y - mt->b_ptr[mt->m]->y + 1;
    for(int i = mt->idx; i < cx; i += THREADS) {
        for(int j = 0; j < cy; j++) {
            float sum = 0.0;
            for(int k = 0; k < mt->b_ptr[mt->m]->x; k++) {
                for(int l = 0; l < mt->b_ptr[mt->m]->y; l++) {
                    sum += mt->a->m[i + k][j + l] * mt->b_ptr[mt->m]->m[k][l];
                }
            }
            mt->c_ptr[mt->m]->m[i][j] = sum;
        }
    }
    return NULL;
}

void mt(void* (*start_routine)(void*), mt_arg* arg) {
    pthread_t tids[THREADS];
    for(int i = 0; i < THREADS; i++) {
        arg[i].idx = i;
        pthread_create(&tids[i], NULL, start_routine, &arg[i]);
    }
    for(int i = 0; i < THREADS; i++) {
        pthread_join(tids[i], NULL);
    }
}
