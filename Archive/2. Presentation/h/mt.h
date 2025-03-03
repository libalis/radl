#ifndef MT_H
    #define MT_H

    #include <glib.h>
    #include <pthread.h>

    #include "../h/matrix.h"

    #ifndef POOL_LEN
        #define POOL_LEN (2)
    #endif

    extern GAsyncQueue *queue;
    extern long THREADS;
    extern long counter;
    extern pthread_cond_t cond;
    extern pthread_mutex_t mutex;
    extern pthread_t tids[];

    typedef struct mt_arg {
        long idx;
        matrix **a_ptr;
        matrix *a;
        matrix **b_ptr;
        matrix *b;
        int len;
        matrix **c_ptr;
        matrix *c;
        int m;
        void (*start_routine)(struct mt_arg *mt);
    } mt_arg;

    void add_mt(mt_arg *mt);
    void biasing_mt(mt_arg *mt);
    void conv2d_mt(mt_arg *mt);
    void flatten_mt(mt_arg *mt);
    void flip_kernels_mt(mt_arg *mt);
    void hyperbolic_tangent_mt(mt_arg *mt);
    void matmul_mt(mt_arg *mt);
    void maxpool_mt(mt_arg *mt);
    void relu_mt(mt_arg *mt);
    void transpose_mt(mt_arg *mt);
    static void *start_mt(void *arg);
    static void stop_mt(mt_arg *mt);
    void push_mt(mt_arg *mt);
    void wait_mt();
    void create_mt(long threads);
    void join_mt();
#endif
