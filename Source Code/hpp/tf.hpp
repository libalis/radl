#ifndef TF_H
    #define TF_H

    #include "../hpp/matrix.hpp"

    matrix *add(matrix *a, matrix *b);
    matrix **biasing(matrix **a, int len, matrix *b);
    matrix **conv2d(matrix *a, matrix **b, int len);
    matrix *flatten(matrix **a, int len);
    matrix **flip_kernels(matrix **a, int len);
    matrix **hyperbolic_tangent(matrix **a, int len);
    matrix *matmul(matrix *a, matrix *b);
    matrix **maxpool(matrix **a, int len);
    matrix **relu(matrix **a, int len);
    matrix *transpose(matrix *a);
#endif
