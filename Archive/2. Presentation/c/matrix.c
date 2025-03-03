#include "stdio.h"
#include "stdlib.h"

#include "../h/matrix.h"

void print_matrix(matrix *a) {
    for(int i = 0; i < a->x; i++) {
        for(int j = 0; j < a->y; j++) {
            printf("%f ", a->m[i][j]);
        }
        printf("\n");
    }
}

matrix *malloc_matrix(int x, int y) {
    matrix *a = malloc(sizeof(matrix));
    a->x = x;
    a->y = y;
    a->m = malloc(a->x * sizeof(float*));
    for(int i = 0; i < a->x; i++) {
        a->m[i] = malloc(a->y * sizeof(float));
    }
    for(int i = 0; i < a->x; i++) {
        for(int j = 0; j < a->y; j++) {
            a->m[i][j] = 0.0;
        }
    }
    return a;
}

matrix **malloc_matrix_ptr(int len, int x, int y) {
    matrix **c = malloc(len * sizeof(matrix*));
    for(int i = 0; i < len; i++) {
        c[i] = malloc_matrix(x, y);
    }
    return c;
}

void free_matrix(matrix *a) {
    for(int i = 0; i < a->x; i++) {
        free(a->m[i]);
    }
    free(a->m);
    free(a);
}

void free_matrix_ptr(matrix **a, int len) {
    for(int i = 0; i < len; i++) {
        free_matrix(a[i]);
    }
    free(a);
}
