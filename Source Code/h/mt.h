#ifndef MT_H
    #define MT_H

    #include "../h/matrix.h"

    #ifndef POOL_LEN
        #define POOL_LEN (2)
    #endif

    extern long THREADS;

    typedef struct mt_arg {
        matrix **a_ptr;
        matrix *a;
        matrix **b_ptr;
        matrix *b;
        int len;
        matrix **c_ptr;
        matrix *c;
        int idx;
        int m;
    } mt_arg;

    void *add_mt(void *arg);
    void *biasing_mt(void *arg);
    void *conv2d_mt(void *arg);
    void *flatten_mt(void *arg);
    void *flip_kernels_mt(void *arg);
    void *hyperbolic_tangent_mt(void *arg);
    void *matmul_mt(void *arg);
    void *maxpool_mt(void *arg);
    void *relu_mt(void *arg);
    void *transpose_mt(void *arg);
    void mt(void *(*mt)(void *), mt_arg *arg);
#endif
