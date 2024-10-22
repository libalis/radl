#ifndef TF_H
#define TF_H

#include "matrix.h"

matrix* matadd(matrix* a, matrix* b);
matrix* matmul(matrix* a, matrix* b);
matrix* conv2d(matrix* a, matrix* b);
matrix* maxpool2d(matrix* a, matrix* b);

#endif
