#ifndef TF_H
#define TF_H

#include "matrix.h"

int max(matrix* a);
matrix* add(matrix* a, matrix* b);
matrix* matmul(matrix* a, matrix* b);
matrix* flatten(matrix* a);
matrix* maxpool(matrix* a);
matrix* relu(matrix* a);
matrix* biasing(matrix* a, matrix* b);
matrix* conv2d(matrix* a, matrix* b);

#endif
