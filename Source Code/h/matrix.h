#ifndef MATRIX_H
#define MATRIX_H

typedef struct matrix {
    int x;
    int y;
    float** m;
} matrix;

void print_matrix(matrix* a);
matrix* transpose(matrix* a);
matrix* malloc_matrix(int x, int y);
void free_matrix(matrix* a);
void free_matrix_ptr(matrix** a, int len);

#endif
