#include "../h/mt.h"
#include "../h/tf.h"

matrix *add(matrix *a, matrix *b) {
    matrix *c = malloc_matrix(a->x, a->y);
    mt_arg arg[THREADS];
    for(int i = 0; i < THREADS; i++) {
        arg[i].a = a;
        arg[i].b = b;
        arg[i].c = c;
    }
    mt(add_routine, arg);
    return c;
}

matrix **biasing(matrix **a, int len, matrix *b) {
    matrix **c = malloc_matrix_ptr(len, a[0]->x, a[0]->y);
    mt_arg arg[THREADS];
    for(int m = 0; m < len; m++) {
        for(int i = 0; i < THREADS; i++) {
            arg[i].a_ptr = a;
            arg[i].len = len;
            arg[i].b = b;
            arg[i].c_ptr = c;
            arg[i].m = m;
        }
        mt(biasing_routine, arg);
    }
    return c;
}

matrix **conv2d(matrix *a, matrix **b, int len) {
    matrix **c = malloc_matrix_ptr(len, a->x - b[0]->x + 1, a->y - b[0]->y + 1);
    mt_arg arg[THREADS];
    for(int m = 0; m < len; m++) {
        for(int i = 0; i < THREADS; i++) {
            arg[i].a = a;
            arg[i].b_ptr = b;
            arg[i].len = len;
            arg[i].c_ptr = c;
            arg[i].m = m;
        }
        mt(conv2d_routine, arg);
    }
    return c;
}

matrix *flatten(matrix **a, int len) {
    matrix *c = malloc_matrix(len * a[0]->x * a[0]->y, 1);
    mt_arg arg[THREADS];
    for(int i = 0; i < THREADS; i++) {
        arg[i].a_ptr = a;
        arg[i].len = len;
        arg[i].c = c;
    }
    mt(flatten_routine, arg);
    return c;
}

matrix **flip_kernels(matrix **a, int len) {
    matrix **c = malloc_matrix_ptr(len, a[0]->x, a[0]->y);
    mt_arg arg[THREADS];
    for(int m = 0; m < len; m++) {
        for(int i = 0; i < THREADS; i++) {
            arg[i].a_ptr = a;
            arg[i].len = len;
            arg[i].c_ptr = c;
            arg[i].m = m;
        }
        mt(flip_kernels_routine, arg);
    }
    return c;
}

matrix **hyperbolic_tangent(matrix **a, int len) {
    matrix **c = malloc_matrix_ptr(len, a[0]->x, a[0]->y);
    mt_arg arg[THREADS];
    for(int m = 0; m < len; m++) {
        for(int i = 0; i < THREADS; i++) {
            arg[i].a_ptr = a;
            arg[i].len = len;
            arg[i].c_ptr = c;
            arg[i].m = m;
        }
        mt(hyperbolic_tangent_routine, arg);
    }
    return c;
}

matrix *matmul(matrix *a, matrix *b) {
    matrix *c = malloc_matrix(a->x, b->y);
    mt_arg arg[THREADS];
    for(int i = 0; i < THREADS; i++) {
        arg[i].a = a;
        arg[i].b = b;
        arg[i].c = c;
    }
    mt(matmul_routine, arg);
    return c;
}

matrix **maxpool(matrix **a, int len) {
    matrix **c = malloc_matrix_ptr(len, a[0]->x / POOL_LEN, a[0]->y / POOL_LEN);
    mt_arg arg[THREADS];
    for(int m = 0; m < len; m++) {
        for(int i = 0; i < THREADS; i++) {
            arg[i].a_ptr = a;
            arg[i].len = len;
            arg[i].c_ptr = c;
            arg[i].m = m;
        }
        mt(maxpool_routine, arg);
    }
    return c;
}

matrix **relu(matrix **a, int len) {
    matrix **c = malloc_matrix_ptr(len, a[0]->x, a[0]->y);
    mt_arg arg[THREADS];
    for(int m = 0; m < len; m++) {
        for(int i = 0; i < THREADS; i++) {
            arg[i].a_ptr = a;
            arg[i].len = len;
            arg[i].c_ptr = c;
            arg[i].m = m;
        }
        mt(relu_routine, arg);
    }
    return c;
}

matrix *transpose(matrix *a) {
    matrix *c = malloc_matrix(a->y, a->x);
    mt_arg arg[THREADS];
    for(int i = 0; i < THREADS; i++) {
        arg[i].a = a;
        arg[i].c = c;
    }
    mt(transpose_routine, arg);
    return c;
}
