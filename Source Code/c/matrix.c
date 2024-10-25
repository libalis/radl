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

matrix* transpose(matrix* a) {
    matrix* c = malloc_matrix(a->y, a->x);
    for(int i = 0; i < a->x; i++) {
        for(int j = 0; j < a->y; j++) {
            c->m[j][i] = a->m[i][j];
        }
    }
    free_matrix(a);
    a = NULL;
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
    if(a == NULL) {
        return;
    }
    for(int i = 0; i < a->x; i++) {
        if(a->m[i] != NULL) {
            free(a->m[i]);
            a->m[i] = NULL;
        }
    }
    if(a->m != NULL) {
        free(a->m);
        a->m = NULL;
    }
    if(a != NULL) {
        free(a);
        a = NULL;
    }
}
