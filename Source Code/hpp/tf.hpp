#ifndef TF_H
    #define TF_H

    #include "../hpp/matrix.hpp"

    #ifndef POOL_LEN
        #define POOL_LEN (2)
    #endif

    extern long THREADS;

    matrix *add(matrix *a, matrix *b, matrix *c);
    matrix **biasing(matrix **a, int len, matrix *b, matrix **c);
    matrix **conv2d(matrix *a, matrix **b, int len, matrix **c);
    matrix *flatten(matrix *a, int len, matrix *c);
    matrix **flip_kernels(matrix **a, int len, matrix **c);
    matrix **hyperbolic_tangent(matrix **a, int len, matrix **c);
    matrix *matmul(matrix *a, matrix *b, matrix *c);
    matrix *maxpool(matrix **a, int len, matrix *c);
    matrix **relu(matrix **a, int len, matrix **c);
    matrix *transpose(matrix *a, matrix *c);
#endif
