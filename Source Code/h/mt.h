#ifndef MT_H
    #define MT_H

    #ifndef THREADS
        #define THREADS (sysconf(_SC_NPROCESSORS_ONLN))
    #endif

    #include <unistd.h>

    #include "../h/matrix.h"

    typedef struct mt_arg {
        matrix** a_ptr;
        matrix* a;
        matrix** b_ptr;
        matrix* b;
        int len;
        matrix** c_ptr;
        matrix* c;
        int idx;
        int m;
    } mt_arg;

    void* add_routine(void* arg);
    void* matmul_routine(void* arg);
    void* flatten_routine(void* arg);
    void* maxpool_routine(void* arg);
    void* hyperbolic_tangent_routine(void* arg);
    void* relu_routine(void* arg);
    void* biasing_routine(void* arg);
    void* conv2d_routine(void* arg);
    void mt(void* (*start_routine)(void*), mt_arg* arg);
#endif
