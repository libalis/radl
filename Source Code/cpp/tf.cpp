#include "../hpp/mt.hpp"
#include "../hpp/tf.hpp"

matrix *add(matrix *a, matrix *b) {
    matrix *c = malloc_matrix(a->x, a->y);
    mt_arg arg[THREADS];
    for(int i = 0; i < THREADS; i++) {
        arg[i].a = a;
        arg[i].b = b;
        arg[i].c = c;
        arg[i].start_routine = add_mt;
        push_mt(&arg[i]);
    }
    wait_mt();
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
            arg[i].start_routine = biasing_mt;
            push_mt(&arg[i]);
        }
        wait_mt();
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
            arg[i].start_routine = conv2d_mt;
            push_mt(&arg[i]);
        }
        wait_mt();
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
        arg[i].start_routine = flatten_mt;
        push_mt(&arg[i]);
    }
    wait_mt();
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
            arg[i].start_routine = flip_kernels_mt;
            push_mt(&arg[i]);
        }
        wait_mt();
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
            arg[i].start_routine = hyperbolic_tangent_mt;
            push_mt(&arg[i]);
        }
        wait_mt();
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
        arg[i].start_routine = matmul_mt;
        push_mt(&arg[i]);
    }
    wait_mt();
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
            arg[i].start_routine = maxpool_mt;
            push_mt(&arg[i]);
        }
        wait_mt();
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
            arg[i].start_routine = relu_mt;
            push_mt(&arg[i]);
        }
        wait_mt();
    }
    return c;
}

matrix *transpose(matrix *a) {
    matrix *c = malloc_matrix(a->y, a->x);
    mt_arg arg[THREADS];
    for(int i = 0; i < THREADS; i++) {
        arg[i].a = a;
        arg[i].c = c;
        arg[i].start_routine = transpose_mt;
        push_mt(&arg[i]);
    }
    wait_mt();
    return c;
}
