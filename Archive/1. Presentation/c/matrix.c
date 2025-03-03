#include "../h/matrix.h"
#include "stdio.h"
#include "stdlib.h"

void print_matrix(matrix* a) {
    for(int i = 0; i < a->x; i++) {
        for(int j = 0; j < a->y; j++) {
            printf("%f ", a->m[i][j]);
        }
        printf("\n");
    }
}

matrix** flip_kernels(matrix** a, int len) {
    matrix** c = malloc(len * sizeof(matrix*));
    for(int i = 0; i < len; i++) {
        c[i] = malloc_matrix(a[i]->x, a[i]->y);
        for (int j = 0; j < a[i]->x; j++) {
            for (int k = 0; k < a[i]->y; k++) {
                c[i]->m[a[i]->x - j - 1][a[i]->y - k - 1] = a[i]->m[j][k];
            }
        }
    }
    return c;
}

matrix* transpose(matrix* a) {
    matrix* c = malloc_matrix(a->y, a->x);
    for(int i = 0; i < a->x; i++) {
        for(int j = 0; j < a->y; j++) {
            c->m[j][i] = a->m[i][j];
        }
    }
    return c;
}

matrix* malloc_matrix(int x, int y) {
    matrix* a = malloc(sizeof(matrix));
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

void free_matrix(matrix* a) {
    for(int i = 0; i < a->x; i++) {
        free(a->m[i]);
    }
    free(a->m);
    free(a);
}

void free_matrix_ptr(matrix** a, int len) {
    for(int i = 0; i < len; i++) {
        free_matrix(a[i]);
    }
    free(a);
}
