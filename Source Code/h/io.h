#ifndef IO_H
#define IO_H

#include "matrix.h"

typedef struct io {
    matrix* conv_bias;
    matrix* fc_bias;
    matrix* fc_weights;
    matrix** masks;
    int masks_len;
} io;

matrix* io_to_matrix(char* a);
io* malloc_io();
void free_io(io* a);

#endif
