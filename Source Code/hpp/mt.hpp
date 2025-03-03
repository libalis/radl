#ifndef MT_HPP
    #define MT_HPP

    #ifndef POOL_LEN
        #define POOL_LEN (2)
    #endif

    #include <glib.h>
    #include <limits.h>
    #include <math.h>
    #include <pthread.h>
    #include <unistd.h>

    #include "mt_arg.hpp"
    #include "simd.hpp"

    typedef struct mt_struct {
        void *instance;
        int idx;
    } mt_struct;

    class mt {
    public:
        #ifdef DEBUG
            int accurate;
        #endif

        int COUNTER;
        int THREADS;

        GAsyncQueue *queue;

        pthread_cond_t cond;
        pthread_mutex_t mutex;
        pthread_t tids[(int)(CHAR_BIT * sizeof(void*))];

        mt() : mt(1) {}

        mt(int threads) {
            #ifdef DEBUG
                accurate = 0;
            #endif
            COUNTER = 0;
            THREADS = threads;
            queue = g_async_queue_new();
            pthread_mutex_init(&mutex, NULL);
            pthread_cond_init(&cond, NULL);
            mt_struct arg[THREADS];
            for(int i = 0; i < THREADS; i++) {
                arg[i].instance = this;
                arg[i].idx = i;
                pthread_create(&tids[i], NULL, start_mt, &arg[i]);
            }
            wait_mt();
        }

        ~mt() {
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

        #ifdef DEBUG
            __attribute__((always_inline)) inline void add_to_accuracy(int number) {
                pthread_mutex_lock(&mutex);
                accurate += number;
                pthread_mutex_unlock(&mutex);
            }
        #endif

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

        __attribute__((always_inline)) inline void add_mt(mt_arg *arg) {
            if(arg->single_core) {
                for(int i = 0; i < (*arg->c)->x; i++) {
                    arg->i = i;
                    add_simd(arg);
                }
            } else {
                for(int i = arg->idx * ((*arg->c)->x / THREADS); i < ((arg->idx + 1) * ((*arg->c)->x / THREADS)) + (arg->idx == THREADS - 1 ? (*arg->c)->x % THREADS : 0); i++) {
                    arg->i = i;
                    add_simd(arg);
                }
                wait_mt();
            }
        }

        __attribute__((always_inline)) inline static void add_mt_wrapper(void *instance, mt_arg *arg) {
            ((mt*)instance)->add_mt(arg);
        }

        __attribute__((always_inline)) inline void biasing_mt(mt_arg *arg) {
            if(arg->single_core) {
                for(int i = 0; i < ((matrix*)arg->a[arg->m])->x; i++) {
                    arg->i = i;
                    biasing_simd(arg);
                }
            } else {
                for(int i = arg->idx * (((matrix*)arg->a[arg->m])->x / THREADS); i < ((arg->idx + 1) * (((matrix*)arg->a[arg->m])->x / THREADS)) + (arg->idx == THREADS - 1 ? ((matrix*)arg->a[arg->m])->x % THREADS : 0); i++) {
                    arg->i = i;
                    biasing_simd(arg);
                }
                wait_mt();
            }
        }

        __attribute__((always_inline)) inline static void biasing_mt_wrapper(void *instance, mt_arg *arg) {
            ((mt*)instance)->biasing_mt(arg);
        }

        __attribute__((always_inline)) inline void conv2d_mt(mt_arg *arg) {
            if(arg->single_core) {
                for(int i = 0; i < ((matrix*)*arg->a)->x - ((matrix*)arg->b[arg->m])->x + 1; i++) {
                    for(int j = 0; j < ((matrix*)*arg->a)->y - ((matrix*)arg->b[arg->m])->y + 1; j++) {
                        arg->i = i;
                        arg->j = j;
                        conv2d_simd(arg);
                    }
                }
            } else {
                for(int i = arg->idx * ((((matrix*)*arg->a)->x - ((matrix*)arg->b[arg->m])->x + 1) / THREADS); i < ((arg->idx + 1) * ((((matrix*)*arg->a)->x - ((matrix*)arg->b[arg->m])->x + 1) / THREADS)) + (arg->idx == THREADS - 1 ? (((matrix*)*arg->a)->x - ((matrix*)arg->b[arg->m])->x + 1) % THREADS : 0); i++) {
                    for(int j = 0; j < ((matrix*)*arg->a)->y - ((matrix*)arg->b[arg->m])->y + 1; j++) {
                        arg->i = i;
                        arg->j = j;
                        conv2d_simd(arg);
                    }
                }
                wait_mt();
            }
        }

        __attribute__((always_inline)) inline static void conv2d_mt_wrapper(void *instance, mt_arg *arg) {
            ((mt*)instance)->conv2d_mt(arg);
        }

        __attribute__((always_inline)) inline void flatten_mt(mt_arg *arg) {
            if(arg->single_core) {
                for(int i = 0; i < ((matrix*)*arg->a)->x / arg->len; i++) {
                    for(int j = 0; j < ((matrix*)*arg->a)->y; j++) {
                        for(int m = 0; m < arg->len; m++) {
                            int idx = i * ((matrix*)*arg->a)->y * arg->len + j * arg->len + m;
                            (*arg->c)->m[get_idx(0, idx, (*arg->c)->y)] = ((matrix*)*arg->a)->m[get_idx(i, j, ((matrix*)*arg->a)->y) + m * ((((matrix*)*arg->a)->x / arg->len) * ((matrix*)*arg->a)->y)];
                        }
                    }
                }
            } else {
                for(int i = arg->idx * (((matrix*)*arg->a)->x / arg->len / THREADS); i < ((arg->idx + 1) * (((matrix*)*arg->a)->x / arg->len / THREADS)) + (arg->idx == THREADS - 1 ? ((matrix*)*arg->a)->x / arg->len % THREADS : 0); i++) {
                    for(int j = 0; j < ((matrix*)*arg->a)->y; j++) {
                        for(int m = 0; m < arg->len; m++) {
                            int idx = i * ((matrix*)*arg->a)->y * arg->len + j * arg->len + m;
                            (*arg->c)->m[get_idx(0, idx, (*arg->c)->y)] = ((matrix*)*arg->a)->m[get_idx(i, j, ((matrix*)*arg->a)->y) + m * ((((matrix*)*arg->a)->x / arg->len) * ((matrix*)*arg->a)->y)];
                        }
                    }
                }
                wait_mt();
            }
        }

        __attribute__((always_inline)) inline static void flatten_mt_wrapper(void *instance, mt_arg *arg) {
            ((mt*)instance)->flatten_mt(arg);
        }

        __attribute__((always_inline)) inline void flip_kernels_mt(mt_arg *arg) {
            if(arg->single_core) {
                for(int i = 0; i < ((matrix*)arg->a[arg->m])->x; i++) {
                    for(int j = 0; j < ((matrix*)arg->a[arg->m])->y; j++) {
                        arg->c[arg->m]->m[get_idx(i, j, arg->c[arg->m]->y)] = ((matrix*)arg->a[arg->m])->m[get_idx(((matrix*)arg->a[arg->m])->x - i - 1, ((matrix*)arg->a[arg->m])->y - j - 1, ((matrix*)arg->a[arg->m])->y)];
                    }
                }
            } else {
                for(int i = arg->idx * (((matrix*)arg->a[arg->m])->x / THREADS); i < ((arg->idx + 1) * (((matrix*)arg->a[arg->m])->x / THREADS)) + (arg->idx == THREADS - 1 ? ((matrix*)arg->a[arg->m])->x % THREADS : 0); i++) {
                    for(int j = 0; j < ((matrix*)arg->a[arg->m])->y; j++) {
                        arg->c[arg->m]->m[get_idx(i, j, arg->c[arg->m]->y)] = ((matrix*)arg->a[arg->m])->m[get_idx(((matrix*)arg->a[arg->m])->x - i - 1, ((matrix*)arg->a[arg->m])->y - j - 1, ((matrix*)arg->a[arg->m])->y)];
                    }
                }
                wait_mt();
            }
        }

        __attribute__((always_inline)) inline static void flip_kernels_mt_wrapper(void *instance, mt_arg *arg) {
            ((mt*)instance)->flip_kernels_mt(arg);
        }

        __attribute__((always_inline)) inline void hyperbolic_tangent_mt(mt_arg *arg) {
            if(arg->single_core) {
                for(int i = 0; i < ((matrix*)arg->a[arg->m])->x; i++) {
                    for(int j = 0; j < ((matrix*)arg->a[arg->m])->y; j++) {
                        arg->c[arg->m]->m[get_idx(i, j, arg->c[arg->m]->y)] = tanh(((matrix*)arg->a[arg->m])->m[get_idx(i, j, ((matrix*)arg->a[arg->m])->y)]);
                    }
                }
            } else {
                for(int i = arg->idx * (((matrix*)arg->a[arg->m])->x / THREADS); i < ((arg->idx + 1) * (((matrix*)arg->a[arg->m])->x / THREADS)) + (arg->idx == THREADS - 1 ? ((matrix*)arg->a[arg->m])->x % THREADS : 0); i++) {
                    for(int j = 0; j < ((matrix*)arg->a[arg->m])->y; j++) {
                        arg->c[arg->m]->m[get_idx(i, j, arg->c[arg->m]->y)] = tanh(((matrix*)arg->a[arg->m])->m[get_idx(i, j, ((matrix*)arg->a[arg->m])->y)]);
                    }
                }
                wait_mt();
            }
        }

        __attribute__((always_inline)) inline static void hyperbolic_tangent_mt_wrapper(void *instance, mt_arg *arg) {
            ((mt*)instance)->hyperbolic_tangent_mt(arg);
        }

        __attribute__((always_inline)) inline void matmul_mt(mt_arg *arg) {
            if(arg->single_core) {
                for(int j = 0; j < (*arg->c)->y; j++) {
                    (*arg->c)->m[get_idx(arg->i, j, (*arg->c)->y)] = 0;
                    arg->j = j;
                    matmul_simd(arg);
                }
            } else {
                for(int j = arg->idx * ((*arg->c)->y / THREADS); j < ((arg->idx + 1) * ((*arg->c)->y / THREADS)) + (arg->idx == THREADS - 1 ? (*arg->c)->y % THREADS : 0); j++) {
                    (*arg->c)->m[get_idx(arg->i, j, (*arg->c)->y)] = 0;
                    arg->j = j;
                    matmul_simd(arg);
                }
                wait_mt();
            }
        }

        __attribute__((always_inline)) inline static void matmul_mt_wrapper(void *instance, mt_arg *arg) {
            ((mt*)instance)->matmul_mt(arg);
        }

        __attribute__((always_inline)) inline void maxpool_mt(mt_arg *arg) {
            if(arg->single_core) {
                for(int i = 0; i < ((matrix*)arg->a[arg->m])->x; i += POOL_LEN) {
                    for(int j = 0; j < ((matrix*)arg->a[arg->m])->y; j += POOL_LEN) {
                        DATA_TYPE max_val = ((matrix*)arg->a[arg->m])->m[get_idx(i, j, ((matrix*)arg->a[arg->m])->y)];
                        for(int k = 0; k < POOL_LEN; k++) {
                            for(int l = 0; l < POOL_LEN; l++) {
                                DATA_TYPE curr_val = ((matrix*)arg->a[arg->m])->m[get_idx(i + k, j + l, ((matrix*)arg->a[arg->m])->y)];
                                if(curr_val > max_val) {
                                    max_val = curr_val;
                                }
                            }
                        }
                        (*arg->c)->m[get_idx(i / POOL_LEN, j / POOL_LEN, (*arg->c)->y) + arg->m * (((*arg->c)->x / arg->len) * (*arg->c)->y)] = max_val;
                    }
                }
            } else {
                for(int m = arg->idx * ((((matrix*)arg->a[arg->m])->x / POOL_LEN) / THREADS); m < ((arg->idx + 1) * ((((matrix*)arg->a[arg->m])->x / POOL_LEN) / THREADS)) + (arg->idx == THREADS - 1 ? (((matrix*)arg->a[arg->m])->x / POOL_LEN) % THREADS : 0); m++) {
                    int i = m * POOL_LEN;
                    for(int j = 0; j < ((matrix*)arg->a[arg->m])->y; j += POOL_LEN) {
                        DATA_TYPE max_val = ((matrix*)arg->a[arg->m])->m[get_idx(i, j, ((matrix*)arg->a[arg->m])->y)];
                        for(int k = 0; k < POOL_LEN; k++) {
                            for(int l = 0; l < POOL_LEN; l++) {
                                DATA_TYPE curr_val = ((matrix*)arg->a[arg->m])->m[get_idx(i + k, j + l, ((matrix*)arg->a[arg->m])->y)];
                                if(curr_val > max_val) {
                                    max_val = curr_val;
                                }
                            }
                        }
                        (*arg->c)->m[get_idx(i / POOL_LEN, j / POOL_LEN, (*arg->c)->y) + arg->m * (((*arg->c)->x / arg->len) * (*arg->c)->y)] = max_val;
                    }
                }
                wait_mt();
            }
        }

        __attribute__((always_inline)) inline static void maxpool_mt_wrapper(void *instance, mt_arg *arg) {
            ((mt*)instance)->maxpool_mt(arg);
        }

        __attribute__((always_inline)) inline void relu_mt(mt_arg *arg) {
            if(arg->single_core) {
                for(int i = 0; i < ((matrix*)arg->a[arg->m])->x; i++) {
                    for(int j = 0; j < ((matrix*)arg->a[arg->m])->y; j++) {
                        DATA_TYPE val = 0;
                        if(((matrix*)arg->a[arg->m])->m[get_idx(i, j, ((matrix*)arg->a[arg->m])->y)] > 0) {
                            val = ((matrix*)arg->a[arg->m])->m[get_idx(i, j, ((matrix*)arg->a[arg->m])->y)];
                        }
                        arg->c[arg->m]->m[get_idx(i, j, arg->c[arg->m]->y)] = val;
                    }
                }
            } else {
                for(int i = arg->idx * (((matrix*)arg->a[arg->m])->x / THREADS); i < ((arg->idx + 1) * (((matrix*)arg->a[arg->m])->x / THREADS)) + (arg->idx == THREADS - 1 ? ((matrix*)arg->a[arg->m])->x % THREADS : 0); i++) {
                    for(int j = 0; j < ((matrix*)arg->a[arg->m])->y; j++) {
                        DATA_TYPE val = 0;
                        if(((matrix*)arg->a[arg->m])->m[get_idx(i, j, ((matrix*)arg->a[arg->m])->y)] > 0) {
                            val = ((matrix*)arg->a[arg->m])->m[get_idx(i, j, ((matrix*)arg->a[arg->m])->y)];
                        }
                        arg->c[arg->m]->m[get_idx(i, j, arg->c[arg->m]->y)] = val;
                    }
                }
                wait_mt();
            }
        }

        __attribute__((always_inline)) inline static void relu_mt_wrapper(void *instance, mt_arg *arg) {
            ((mt*)instance)->relu_mt(arg);
        }

        __attribute__((always_inline)) inline void transpose_mt(mt_arg *arg) {
            if(arg->single_core) {
                for(int i = 0; i < ((matrix*)*arg->a)->x; i++) {
                    for(int j = 0; j < ((matrix*)*arg->a)->y; j++) {
                        (*arg->c)->m[get_idx(j, i, (*arg->c)->y)] = ((matrix*)*arg->a)->m[get_idx(i, j, ((matrix*)*arg->a)->y)];
                    }
                }
            } else {
                for(int i = arg->idx * (((matrix*)*arg->a)->x / THREADS); i < ((arg->idx + 1) * (((matrix*)*arg->a)->x / THREADS)) + (arg->idx == THREADS - 1 ? ((matrix*)*arg->a)->x % THREADS : 0); i++) {
                    for(int j = 0; j < ((matrix*)*arg->a)->y; j++) {
                        (*arg->c)->m[get_idx(j, i, (*arg->c)->y)] = ((matrix*)*arg->a)->m[get_idx(i, j, ((matrix*)*arg->a)->y)];
                    }
                }
                wait_mt();
            }
        }

        __attribute__((always_inline)) inline static void transpose_mt_wrapper(void *instance, mt_arg *arg) {
            ((mt*)instance)->transpose_mt(arg);
        }

        __attribute__((always_inline)) inline static void *start_mt(void *arg) {
            mt *instance = (mt*)(((mt_struct*)arg)->instance);
            int idx = ((mt_struct*)arg)->idx;
            instance->wait_mt();
            while(1) {
                mt_arg *head = (mt_arg*)g_async_queue_pop(instance->queue);
                head->idx = idx;
                head->start_routine(instance, head);
            }
            return NULL;
        }

        __attribute__((always_inline)) inline static void stop_mt(void *instance, mt_arg *arg) {
            pthread_exit(EXIT_SUCCESS);
        }

        __attribute__((always_inline)) inline void push_mt(mt_arg *arg) {
            g_async_queue_push(queue, arg);
        }
    };
#endif
