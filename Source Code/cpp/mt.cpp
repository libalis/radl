#include <limits.h>
#include <math.h>

#include "../hpp/mt.hpp"
#include "../hpp/simd.hpp"
#include "../hpp/tf.hpp"
#include "../hpp/utils.hpp"

GAsyncQueue *queue = NULL;
long counter = 0;
pthread_cond_t cond;
pthread_mutex_t mutex;
pthread_t tids[(int)(CHAR_BIT * sizeof(void *))];

void add_mt(mt_arg *mt) {
    if(mt->single_core) {
        for(int i = 0; i < mt->c->x; i++) {
            mt->i = i;
            add_simd(mt);
        }
    } else {
        for(int i = mt->idx; i < mt->c->x; i += THREADS) {
            mt->i = i;
            add_simd(mt);
        }
        wait_mt();
    }
}

void biasing_mt(mt_arg *mt) {
    if(mt->single_core) {
        for(int i = 0; i < mt->a_ptr[mt->m]->x; i++) {
            mt->i = i;
            biasing_simd(mt);
        }
    } else {
        for(int i = mt->idx; i < mt->a_ptr[mt->m]->x; i += THREADS) {
            mt->i = i;
            biasing_simd(mt);
        }
        wait_mt();
    }
}

void conv2d_mt(mt_arg *mt) {
    if(mt->single_core) {
        for(int i = 0; i < mt->a->x - mt->b_ptr[mt->m]->x + 1; i++) {
            for(int j = 0; j < mt->a->y - mt->b_ptr[mt->m]->y + 1; j++) {
                mt->i = i;
                mt->j = j;
                conv2d_simd(mt);
            }
        }
    } else {
        for(int i = mt->idx; i < mt->a->x - mt->b_ptr[mt->m]->x + 1; i += THREADS) {
            for(int j = 0; j < mt->a->y - mt->b_ptr[mt->m]->y + 1; j++) {
                mt->i = i;
                mt->j = j;
                conv2d_simd(mt);
            }
        }
        wait_mt();
    }
}

void flatten_mt(mt_arg *mt) {
    if(mt->single_core) {
        for(int i = 0; i < mt->a->x / mt->len; i++) {
            for(int j = 0; j < mt->a->y; j++) {
                for(int m = 0; m < mt->len; m++) {
                    int idx = i * mt->a->y * mt->len + j * mt->len + m;
                    mt->c->m[get_idx(idx, 0, mt->c->y)] = mt->a->m[get_idx(i, j, mt->a->y) + m * ((mt->a->x / mt->len) * mt->a->y)];
                }
            }
        }
    } else {
        for(int i = mt->idx; i < mt->a->x / mt->len; i += THREADS) {
            for(int j = 0; j < mt->a->y; j++) {
                for(int m = 0; m < mt->len; m++) {
                    int idx = i * mt->a->y * mt->len + j * mt->len + m;
                    mt->c->m[get_idx(idx, 0, mt->c->y)] = mt->a->m[get_idx(i, j, mt->a->y) + m * ((mt->a->x / mt->len) * mt->a->y)];
                }
            }
        }
        wait_mt();
    }
}

void flip_kernels_mt(mt_arg *mt) {
    if(mt->single_core) {
        for (int i = 0; i < mt->a_ptr[mt->m]->x; i++) {
            for (int j = 0; j < mt->a_ptr[mt->m]->y; j++) {
                mt->c_ptr[mt->m]->m[get_idx(i, j, mt->c_ptr[mt->m]->y)] = mt->a_ptr[mt->m]->m[get_idx(mt->a_ptr[mt->m]->x - i - 1, mt->a_ptr[mt->m]->y - j - 1, mt->a_ptr[mt->m]->y)];
            }
        }
    } else {
        for (int i = mt->idx; i < mt->a_ptr[mt->m]->x; i += THREADS) {
            for (int j = 0; j < mt->a_ptr[mt->m]->y; j++) {
                mt->c_ptr[mt->m]->m[get_idx(i, j, mt->c_ptr[mt->m]->y)] = mt->a_ptr[mt->m]->m[get_idx(mt->a_ptr[mt->m]->x - i - 1, mt->a_ptr[mt->m]->y - j - 1, mt->a_ptr[mt->m]->y)];
            }
        }
        wait_mt();
    }
}

void hyperbolic_tangent_mt(mt_arg *mt) {
    if(mt->single_core) {
        for(int i = 0; i < mt->a_ptr[mt->m]->x; i++) {
            for(int j = 0; j < mt->a_ptr[mt->m]->y; j++) {
                mt->c_ptr[mt->m]->m[get_idx(i, j, mt->c_ptr[mt->m]->y)] = tanh(mt->a_ptr[mt->m]->m[get_idx(i, j, mt->a_ptr[mt->m]->y)]);
            }
        }
    } else {
        for(int i = mt->idx; i < mt->a_ptr[mt->m]->x; i += THREADS) {
            for(int j = 0; j < mt->a_ptr[mt->m]->y; j++) {
                mt->c_ptr[mt->m]->m[get_idx(i, j, mt->c_ptr[mt->m]->y)] = tanh(mt->a_ptr[mt->m]->m[get_idx(i, j, mt->a_ptr[mt->m]->y)]);
            }
        }
        wait_mt();
    }
}

void matmul_mt(mt_arg *mt) {
    if(mt->single_core) {
        for(int j = 0; j < mt->c->y; j++) {
            mt->c->m[get_idx(mt->i, j, mt->c->y)] = 0;
            mt->j = j;
            matmul_simd(mt);
        }
    } else {
        for(int j = mt->idx; j < mt->c->y; j += THREADS) {
            mt->c->m[get_idx(mt->i, j, mt->c->y)] = 0;
            mt->j = j;
            matmul_simd(mt);
        }
        wait_mt();
    }
}

void maxpool_mt(mt_arg *mt) {
    if(mt->single_core) {
        for(int i = 0; i < mt->a_ptr[mt->m]->x; i += POOL_LEN) {
            for(int j = 0; j < mt->a_ptr[mt->m]->y; j += POOL_LEN) {
                DATA_TYPE max_val = mt->a_ptr[mt->m]->m[get_idx(i, j, mt->a_ptr[mt->m]->y)];
                for(int k = 0; k < POOL_LEN; k++) {
                    for(int l = 0; l < POOL_LEN; l++) {
                        DATA_TYPE curr_val = mt->a_ptr[mt->m]->m[get_idx(i + k, j + l, mt->a_ptr[mt->m]->y)];
                        if(curr_val > max_val) {
                            max_val = curr_val;
                        }
                    }
                }
                mt->c->m[get_idx(i / POOL_LEN, j / POOL_LEN, mt->c->y) + mt->m * ((mt->c->x / mt->len) * mt->c->y)] = max_val;
            }
        }
    } else {
        for(int i = mt->idx * POOL_LEN; i < mt->a_ptr[mt->m]->x; i += THREADS * POOL_LEN) {
            for(int j = 0; j < mt->a_ptr[mt->m]->y; j += POOL_LEN) {
                DATA_TYPE max_val = mt->a_ptr[mt->m]->m[get_idx(i, j, mt->a_ptr[mt->m]->y)];
                for(int k = 0; k < POOL_LEN; k++) {
                    for(int l = 0; l < POOL_LEN; l++) {
                        DATA_TYPE curr_val = mt->a_ptr[mt->m]->m[get_idx(i + k, j + l, mt->a_ptr[mt->m]->y)];
                        if(curr_val > max_val) {
                            max_val = curr_val;
                        }
                    }
                }
                mt->c->m[get_idx(i / POOL_LEN, j / POOL_LEN, mt->c->y) + mt->m * ((mt->c->x / mt->len) * mt->c->y)] = max_val;
            }
        }
        wait_mt();
    }
}

void relu_mt(mt_arg *mt) {
    if(mt->single_core) {
        for(int i = 0; i < mt->a_ptr[mt->m]->x; i++) {
            for(int j = 0; j < mt->a_ptr[mt->m]->y; j++) {
                DATA_TYPE val = 0;
                if(mt->a_ptr[mt->m]->m[get_idx(i, j, mt->a_ptr[mt->m]->y)] > 0) {
                    val = mt->a_ptr[mt->m]->m[get_idx(i, j, mt->a_ptr[mt->m]->y)];
                }
                mt->c_ptr[mt->m]->m[get_idx(i, j, mt->c_ptr[mt->m]->y)] = val;
            }
        }
    } else {
        for(int i = mt->idx; i < mt->a_ptr[mt->m]->x; i += THREADS) {
            for(int j = 0; j < mt->a_ptr[mt->m]->y; j++) {
                DATA_TYPE val = 0;
                if(mt->a_ptr[mt->m]->m[get_idx(i, j, mt->a_ptr[mt->m]->y)] > 0) {
                    val = mt->a_ptr[mt->m]->m[get_idx(i, j, mt->a_ptr[mt->m]->y)];
                }
                mt->c_ptr[mt->m]->m[get_idx(i, j, mt->c_ptr[mt->m]->y)] = val;
            }
        }
        wait_mt();
    }
}

void transpose_mt(mt_arg *mt) {
    if(mt->single_core) {
        for(int i = 0; i < mt->a->x; i++) {
            for(int j = 0; j < mt->a->y; j++) {
                mt->c->m[get_idx(j, i, mt->c->y)] = mt->a->m[get_idx(i, j, mt->a->y)];
            }
        }
    } else {
        for(int i = mt->idx; i < mt->a->x; i += THREADS) {
            for(int j = 0; j < mt->a->y; j++) {
                mt->c->m[get_idx(j, i, mt->c->y)] = mt->a->m[get_idx(i, j, mt->a->y)];
            }
        }
        wait_mt();
    }
}

static void *start_mt(void *arg) {
    mt_arg *mt = (mt_arg*)arg;
    while(1) {
        mt_arg *head = (mt_arg*)g_async_queue_pop(queue);
        head->idx = mt->idx;
        head->start_routine(head);
    }
}

static void stop_mt(mt_arg *mt) {
    pthread_exit(EXIT_SUCCESS);
}

void push_mt(mt_arg *mt) {
    g_async_queue_push(queue, mt);
}

void wait_mt() {
    pthread_mutex_lock(&mutex);
    counter++;
    if (counter == THREADS + 1) {
        counter = 0;
        pthread_cond_broadcast(&cond);
    } else {
        pthread_cond_wait(&cond, &mutex);
    }
    pthread_mutex_unlock(&mutex);
}

void create_mt(long threads) {
    THREADS = threads;
    if(queue == NULL) {
        queue = g_async_queue_new();
    }
    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&cond, NULL);
    mt_arg *mt = (mt_arg*)malloc(THREADS * sizeof(mt_arg));
    for(long i = 0; i < THREADS; i++) {
        mt[i].idx = i;
        pthread_create(&tids[i], NULL, start_mt, &mt[i]);
    }
}

void join_mt() {
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
}
