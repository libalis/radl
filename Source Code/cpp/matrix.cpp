#include <stdio.h>
#include <stdlib.h>

#include "../hpp/matrix.hpp"
#include "../hpp/utils.hpp"

void print_matrix(matrix *a) {
    for(int i = 0; i < a->x; i++) {
        for(int j = 0; j < a->y; j++) {
            printf("%f ", a->m[get_idx(i, j, a->y)]);
        }
        printf("\n");
    }
}

matrix *malloc_matrix(int x, int y) {
    matrix *a = (matrix*)malloc(sizeof(matrix));
    a->x = x;
    a->y = y;
    a->m = (float*)malloc(a->x * a->y * sizeof(float));
    for(int i = 0; i < a->x; i++) {
        for(int j = 0; j < a->y; j++) {
            a->m[get_idx(i, j, a->y)] = 0.0;
        }
    }
    return a;
}

matrix **malloc_matrix_ptr(int len, int x, int y) {
    matrix **c = (matrix**)malloc(len * sizeof(matrix*));
    for(int i = 0; i < len; i++) {
        c[i] = malloc_matrix(x, y);
    }
    return c;
}

void free_matrix(matrix *a) {
    free(a->m);
    free(a);
}

void free_matrix_ptr(matrix **a, int len) {
    for(int i = 0; i < len; i++) {
        free_matrix(a[i]);
    }
    free(a);
}
