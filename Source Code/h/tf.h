#ifndef TF_H
    #define TF_H

    #include "matrix.h"

    int max(matrix* a);
    matrix* add(matrix* a, matrix* b);
    matrix* matmul(matrix* a, matrix* b);
    matrix* flatten(matrix** a, int len);
    matrix** maxpool(matrix** a, int len);
    matrix** hyperbolic_tangent(matrix** a, int len);
    matrix** relu(matrix** a, int len);
    matrix** biasing(matrix** a, int len, matrix* b);
    matrix** conv2d(matrix* a, matrix** b, int len);
#endif
