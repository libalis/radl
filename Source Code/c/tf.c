#include <stdlib.h>
#include <unistd.h>

#include "../h/mt.h"
#include "../h/tf.h"

int max(matrix* a) {
    float max_val = a->m[0][0];
    int index = 0;
    for(int i = 0; i < a->y; i++) {
        float curr_val = a->m[0][i];
        if(curr_val > max_val) {
            max_val = curr_val;
            index = i;
        }
    }
    return index;
}

matrix* add(matrix* a, matrix* b) {
    matrix* c = malloc_matrix(a->x, a->y);
    mt_arg arg[THREADS];
    for(int i = 0; i < THREADS; i++) {
        arg[i].a = a;
        arg[i].b = b;
        arg[i].c = c;
    }
    mt(add_routine, arg);
    return c;
}

matrix* matmul(matrix* a, matrix* b) {
    matrix* c = malloc_matrix(a->x, b->y);
    mt_arg arg[THREADS];
    for(int i = 0; i < THREADS; i++) {
        arg[i].a = a;
        arg[i].b = b;
        arg[i].c = c;
    }
    mt(matmul_routine, arg);
    return c;
}

matrix* flatten(matrix** a, int len) {
    matrix* c = malloc_matrix(len * a[0]->x * a[0]->y, 1);
    mt_arg arg[THREADS];
    for(int i = 0; i < THREADS; i++) {
        arg[i].a_ptr = a;
        arg[i].len = len;
        arg[i].c = c;
    }
    mt(flatten_routine, arg);
    return c;
}

matrix** maxpool(matrix** a, int len) {
    matrix** c = malloc(len * sizeof(matrix*));
    mt_arg arg[THREADS];
    int pool_len = 2;
    for(int m = 0; m < len; m++) {
        c[m] = malloc_matrix(a[0]->x / pool_len, a[0]->y / pool_len);
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

matrix** hyperbolic_tangent(matrix** a, int len) {
    matrix** c = malloc(len * sizeof(matrix*));
    mt_arg arg[THREADS];
    for(int m = 0; m < len; m++) {
        c[m] = malloc_matrix(a[m]->x, a[m]->y);
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

matrix** relu(matrix** a, int len) {
    matrix** c = malloc(len * sizeof(matrix*));
    mt_arg arg[THREADS];
    for(int m = 0; m < len; m++) {
        c[m] = malloc_matrix(a[m]->x, a[m]->y);
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

matrix** biasing(matrix** a, int len, matrix* b) {
    matrix** c = malloc(len * sizeof(matrix*));
    mt_arg arg[THREADS];
    for(int m = 0; m < len; m++) {
        c[m] = malloc_matrix(a[m]->x, a[m]->y);
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

matrix** conv2d(matrix* a, matrix** b, int len) {
    matrix** c = malloc(len * sizeof(matrix*));
    mt_arg arg[THREADS];
    for(int m = 0; m < len; m++) {
        int cx = a->x - b[m]->x + 1;
        int cy = a->y - b[m]->y + 1;
        c[m] = malloc_matrix(cx, cy);
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
