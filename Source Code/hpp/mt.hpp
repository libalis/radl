#ifndef MT_HPP
    #define MT_HPP

    #include <glib.h>
    #include <math.h>
    #include <pthread.h>

    #include "mt_arg.hpp"
    #include "simd.hpp"
    #include "tf.hpp"

    extern int COUNTER;
    extern int THREADS;
    extern int MAX_THREADS;

    extern GAsyncQueue *queue;

    extern pthread_cond_t cond;
    extern pthread_mutex_t mutex;
    extern pthread_t tids[];

    __attribute__((always_inline)) inline void wait_mt() {
        pthread_mutex_lock(&mutex);
        COUNTER++;
        if(COUNTER == THREADS + 1) {
            COUNTER = 0;
            pthread_cond_broadcast(&cond);
        } else {
            pthread_cond_wait(&cond, &mutex);
        }
        pthread_mutex_unlock(&mutex);
    }

    __attribute__((always_inline)) inline void add_mt(mt_arg *mt) {
        if(mt->single_core) {
            for(int i = 0; i < (*mt->c)->x; i++) {
                mt->i = i;
                add_simd(mt);
            }
        } else {
            for(int i = mt->idx * ((*mt->c)->x / THREADS); i < ((mt->idx + 1) * ((*mt->c)->x / THREADS)) + (mt->idx == THREADS - 1 ? (*mt->c)->x % THREADS : 0); i++) {
                mt->i = i;
                add_simd(mt);
            }
            wait_mt();
        }
    }

    __attribute__((always_inline)) inline void biasing_mt(mt_arg *mt) {
        if(mt->single_core) {
            for(int i = 0; i < mt->a[mt->m]->x; i++) {
                mt->i = i;
                biasing_simd(mt);
            }
        } else {
            for(int i = mt->idx * (mt->a[mt->m]->x / THREADS); i < ((mt->idx + 1) * (mt->a[mt->m]->x / THREADS)) + (mt->idx == THREADS - 1 ? mt->a[mt->m]->x % THREADS : 0); i++) {
                mt->i = i;
                biasing_simd(mt);
            }
            wait_mt();
        }
    }

    __attribute__((always_inline)) inline void conv2d_mt(mt_arg *mt) {
        if(mt->single_core) {
            for(int i = 0; i < (*mt->a)->x - mt->b[mt->m]->x + 1; i++) {
                for(int j = 0; j < (*mt->a)->y - mt->b[mt->m]->y + 1; j++) {
                    mt->i = i;
                    mt->j = j;
                    conv2d_simd(mt);
                }
            }
        } else {
            for(int i = mt->idx * (((*mt->a)->x - mt->b[mt->m]->x + 1) / THREADS); i < ((mt->idx + 1) * (((*mt->a)->x - mt->b[mt->m]->x + 1) / THREADS)) + (mt->idx == THREADS - 1 ? ((*mt->a)->x - mt->b[mt->m]->x + 1) % THREADS : 0); i++) {
                for(int j = 0; j < (*mt->a)->y - mt->b[mt->m]->y + 1; j++) {
                    mt->i = i;
                    mt->j = j;
                    conv2d_simd(mt);
                }
            }
            wait_mt();
        }
    }

    __attribute__((always_inline)) inline void flatten_mt(mt_arg *mt) {
        if(mt->single_core) {
            for(int i = 0; i < (*mt->a)->x / mt->len; i++) {
                for(int j = 0; j < (*mt->a)->y; j++) {
                    for(int m = 0; m < mt->len; m++) {
                        int idx = i * (*mt->a)->y * mt->len + j * mt->len + m;
                        (*mt->c)->m[get_idx(0, idx, (*mt->c)->y)] = (*mt->a)->m[get_idx(i, j, (*mt->a)->y) + m * (((*mt->a)->x / mt->len) * (*mt->a)->y)];
                    }
                }
            }
        } else {
            for(int i = mt->idx * ((*mt->a)->x / mt->len / THREADS); i < ((mt->idx + 1) * ((*mt->a)->x / mt->len / THREADS)) + (mt->idx == THREADS - 1 ? (*mt->a)->x / mt->len % THREADS : 0); i++) {
                for(int j = 0; j < (*mt->a)->y; j++) {
                    for(int m = 0; m < mt->len; m++) {
                        int idx = i * (*mt->a)->y * mt->len + j * mt->len + m;
                        (*mt->c)->m[get_idx(0, idx, (*mt->c)->y)] = (*mt->a)->m[get_idx(i, j, (*mt->a)->y) + m * (((*mt->a)->x / mt->len) * (*mt->a)->y)];
                    }
                }
            }
            wait_mt();
        }
    }

    __attribute__((always_inline)) inline void flip_kernels_mt(mt_arg *mt) {
        if(mt->single_core) {
            for(int i = 0; i < mt->a[mt->m]->x; i++) {
                for(int j = 0; j < mt->a[mt->m]->y; j++) {
                    mt->c[mt->m]->m[get_idx(i, j, mt->c[mt->m]->y)] = mt->a[mt->m]->m[get_idx(mt->a[mt->m]->x - i - 1, mt->a[mt->m]->y - j - 1, mt->a[mt->m]->y)];
                }
            }
        } else {
            for(int i = mt->idx * (mt->a[mt->m]->x / THREADS); i < ((mt->idx + 1) * (mt->a[mt->m]->x / THREADS)) + (mt->idx == THREADS - 1 ? mt->a[mt->m]->x % THREADS : 0); i++) {
                for(int j = 0; j < mt->a[mt->m]->y; j++) {
                    mt->c[mt->m]->m[get_idx(i, j, mt->c[mt->m]->y)] = mt->a[mt->m]->m[get_idx(mt->a[mt->m]->x - i - 1, mt->a[mt->m]->y - j - 1, mt->a[mt->m]->y)];
                }
            }
            wait_mt();
        }
    }

    __attribute__((always_inline)) inline void hyperbolic_tangent_mt(mt_arg *mt) {
        if(mt->single_core) {
            for(int i = 0; i < mt->a[mt->m]->x; i++) {
                for(int j = 0; j < mt->a[mt->m]->y; j++) {
                    mt->c[mt->m]->m[get_idx(i, j, mt->c[mt->m]->y)] = tanh(mt->a[mt->m]->m[get_idx(i, j, mt->a[mt->m]->y)]);
                }
            }
        } else {
            for(int i = mt->idx * (mt->a[mt->m]->x / THREADS); i < ((mt->idx + 1) * (mt->a[mt->m]->x / THREADS)) + (mt->idx == THREADS - 1 ? mt->a[mt->m]->x % THREADS : 0); i++) {
                for(int j = 0; j < mt->a[mt->m]->y; j++) {
                    mt->c[mt->m]->m[get_idx(i, j, mt->c[mt->m]->y)] = tanh(mt->a[mt->m]->m[get_idx(i, j, mt->a[mt->m]->y)]);
                }
            }
            wait_mt();
        }
    }

    __attribute__((always_inline)) inline void matmul_mt(mt_arg *mt) {
        if(mt->single_core) {
            for(int j = 0; j < (*mt->c)->y; j++) {
                (*mt->c)->m[get_idx(mt->i, j, (*mt->c)->y)] = 0;
                mt->j = j;
                matmul_simd(mt);
            }
        } else {
            for(int j = mt->idx * ((*mt->c)->y / THREADS); j < ((mt->idx + 1) * ((*mt->c)->y / THREADS)) + (mt->idx == THREADS - 1 ? (*mt->c)->y % THREADS : 0); j++) {
                (*mt->c)->m[get_idx(mt->i, j, (*mt->c)->y)] = 0;
                mt->j = j;
                matmul_simd(mt);
            }
            wait_mt();
        }
    }

    __attribute__((always_inline)) inline void maxpool_mt(mt_arg *mt) {
        if(mt->single_core) {
            for(int i = 0; i < mt->a[mt->m]->x; i += POOL_LEN) {
                for(int j = 0; j < mt->a[mt->m]->y; j += POOL_LEN) {
                    DATA_TYPE max_val = mt->a[mt->m]->m[get_idx(i, j, mt->a[mt->m]->y)];
                    for(int k = 0; k < POOL_LEN; k++) {
                        for(int l = 0; l < POOL_LEN; l++) {
                            DATA_TYPE curr_val = mt->a[mt->m]->m[get_idx(i + k, j + l, mt->a[mt->m]->y)];
                            if(curr_val > max_val) {
                                max_val = curr_val;
                            }
                        }
                    }
                    (*mt->c)->m[get_idx(i / POOL_LEN, j / POOL_LEN, (*mt->c)->y) + mt->m * (((*mt->c)->x / mt->len) * (*mt->c)->y)] = max_val;
                }
            }
        } else {
            for(int m = mt->idx * ((mt->a[mt->m]->x / POOL_LEN) / THREADS); m < ((mt->idx + 1) * ((mt->a[mt->m]->x / POOL_LEN) / THREADS)) + (mt->idx == THREADS - 1 ? (mt->a[mt->m]->x / POOL_LEN) % THREADS : 0); m++) {
                int i = m * POOL_LEN;
                for(int j = 0; j < mt->a[mt->m]->y; j += POOL_LEN) {
                    DATA_TYPE max_val = mt->a[mt->m]->m[get_idx(i, j, mt->a[mt->m]->y)];
                    for(int k = 0; k < POOL_LEN; k++) {
                        for(int l = 0; l < POOL_LEN; l++) {
                            DATA_TYPE curr_val = mt->a[mt->m]->m[get_idx(i + k, j + l, mt->a[mt->m]->y)];
                            if(curr_val > max_val) {
                                max_val = curr_val;
                            }
                        }
                    }
                    (*mt->c)->m[get_idx(i / POOL_LEN, j / POOL_LEN, (*mt->c)->y) + mt->m * (((*mt->c)->x / mt->len) * (*mt->c)->y)] = max_val;
                }
            }
            wait_mt();
        }
    }

    __attribute__((always_inline)) inline void relu_mt(mt_arg *mt) {
        if(mt->single_core) {
            for(int i = 0; i < mt->a[mt->m]->x; i++) {
                for(int j = 0; j < mt->a[mt->m]->y; j++) {
                    DATA_TYPE val = 0;
                    if(mt->a[mt->m]->m[get_idx(i, j, mt->a[mt->m]->y)] > 0) {
                        val = mt->a[mt->m]->m[get_idx(i, j, mt->a[mt->m]->y)];
                    }
                    mt->c[mt->m]->m[get_idx(i, j, mt->c[mt->m]->y)] = val;
                }
            }
        } else {
            for(int i = mt->idx * (mt->a[mt->m]->x / THREADS); i < ((mt->idx + 1) * (mt->a[mt->m]->x / THREADS)) + (mt->idx == THREADS - 1 ? mt->a[mt->m]->x % THREADS : 0); i++) {
                for(int j = 0; j < mt->a[mt->m]->y; j++) {
                    DATA_TYPE val = 0;
                    if(mt->a[mt->m]->m[get_idx(i, j, mt->a[mt->m]->y)] > 0) {
                        val = mt->a[mt->m]->m[get_idx(i, j, mt->a[mt->m]->y)];
                    }
                    mt->c[mt->m]->m[get_idx(i, j, mt->c[mt->m]->y)] = val;
                }
            }
            wait_mt();
        }
    }

    __attribute__((always_inline)) inline void transpose_mt(mt_arg *mt) {
        if(mt->single_core) {
            for(int i = 0; i < (*mt->a)->x; i++) {
                for(int j = 0; j < (*mt->a)->y; j++) {
                    (*mt->c)->m[get_idx(j, i, (*mt->c)->y)] = (*mt->a)->m[get_idx(i, j, (*mt->a)->y)];
                }
            }
        } else {
            for(int i = mt->idx * ((*mt->a)->x / THREADS); i < ((mt->idx + 1) * ((*mt->a)->x / THREADS)) + (mt->idx == THREADS - 1 ? (*mt->a)->x % THREADS : 0); i++) {
                for(int j = 0; j < (*mt->a)->y; j++) {
                    (*mt->c)->m[get_idx(j, i, (*mt->c)->y)] = (*mt->a)->m[get_idx(i, j, (*mt->a)->y)];
                }
            }
            wait_mt();
        }
    }

    __attribute__((always_inline)) inline static void *start_mt(void *arg) {
        int idx = *(int*)arg;
        wait_mt();
        while(1) {
            mt_arg *head = (mt_arg*)g_async_queue_pop(queue);
            head->idx = idx;
            head->start_routine(head);
        }
        return NULL;
    }

    __attribute__((always_inline)) inline static void stop_mt(mt_arg *mt) {
        pthread_exit(EXIT_SUCCESS);
    }

    __attribute__((always_inline)) inline void push_mt(mt_arg *mt) {
        g_async_queue_push(queue, mt);
    }

    __attribute__((always_inline)) inline void create_mt(int threads) {
        THREADS = threads;
        queue = g_async_queue_new();
        pthread_mutex_init(&mutex, NULL);
        pthread_cond_init(&cond, NULL);
        int idx[THREADS];
        for(int i = 0; i < THREADS; i++) {
            idx[i] = i;
            pthread_create(&tids[i], NULL, start_mt, &idx[i]);
        }
        wait_mt();
    }

    __attribute__((always_inline)) inline void join_mt() {
        mt_arg arg[THREADS];
        for(long i = 0; i < THREADS; i++) {
            arg[i].start_routine = stop_mt;
            push_mt(&arg[i]);
        }
        for(long i = 0; i < THREADS; i++) {
            pthread_join(tids[i], NULL);
        }
        pthread_cond_destroy(&cond);
        pthread_mutex_destroy(&mutex);
        g_async_queue_unref(queue);
    }
#endif
