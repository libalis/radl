#ifndef MATRIX_H
    #define MATRIX_H

    #ifndef DATA_TYPE
        #ifdef INT
            typedef int DATA_TYPE;
        #else
            typedef float DATA_TYPE;
        #endif
    #endif

    typedef struct matrix {
        int x;
        int y;
        DATA_TYPE *m;
    } matrix;

    void print_matrix(matrix *a);
    matrix *malloc_matrix(int x, int y);
    matrix **malloc_matrix_ptr(int len, int x, int y);
    void free_matrix(matrix *a);
    void free_matrix_ptr(matrix **a, int len);
#endif
