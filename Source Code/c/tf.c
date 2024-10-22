#include "../h/tf.h"
#include <assert.h>

matrix* add(matrix* a, matrix* b) {
    assert(a->x == b->x && a->y == b->y);
    matrix *c = malloc_matrix(a->x, a->y);
    for(int i = 0; i < c->x; i++) {
        for(int j = 0; j < c->y; j++) {
            c->m[i][j] = a->m[i][j] + b->m[i][j];
        }
    }
    return c;
}

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
