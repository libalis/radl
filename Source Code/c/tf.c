#include "../h/tf.h"
#include <assert.h>

matrix* matmul(matrix* a, matrix* b) {
    assert(a->y == b->x);
    matrix *c = malloc_matrix(a->x, b->y);
    for(int i = 0; i < c->x; i++) {
        for(int k = 0; k < c->y; k++) {
            for(int j = 0; j < a->y; j++) {
                c->m[i][k] = c->m[i][k] + a->m[i][j] * b->m[j][k];
            }
        }
    }
    return c;
}

matrix* conv2d(matrix* a, matrix* b) {
    // TODO
    return malloc_matrix(0, 0);
}

matrix* maxpool2d(matrix* a, matrix* b) {
    // TODO
    return malloc_matrix(0, 0);
}
