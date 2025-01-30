#ifndef MATRIX_HPP
    #define MATRIX_HPP

    #include "utils.hpp"

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

    __attribute__((always_inline)) inline void print_matrix(matrix *a) {
        for(int i = 0; i < a->x; i++) {
            for(int j = 0; j < a->y - 1; j++) {
                #ifdef INT
                    printf("%d ", a->m[get_idx(i, j, a->y)]);
                #else
                    printf("%f ", a->m[get_idx(i, j, a->y)]);
                #endif
            }
            #ifdef INT
                printf("%d\n", a->m[get_idx(i, a->y - 1, a->y)]);
            #else
                printf("%f\n", a->m[get_idx(i, a->y - 1, a->y)]);
            #endif
        }
    }

    __attribute__((always_inline)) inline matrix *malloc_matrix(int x, int y) {
        matrix *a = (matrix*)malloc(sizeof(matrix));
        a->x = x;
        a->y = y;
        a->m = (DATA_TYPE*)malloc(a->x * a->y * sizeof(DATA_TYPE));
        for(int i = 0; i < a->x; i++) {
            for(int j = 0; j < a->y; j++) {
                a->m[get_idx(i, j, a->y)] = 0;
            }
        }
        return a;
    }

    __attribute__((always_inline)) inline matrix **malloc_matrix_ptr(int len, int x, int y) {
        matrix **c = (matrix**)malloc(len * sizeof(matrix*));
        for(int i = 0; i < len; i++) {
            c[i] = malloc_matrix(x, y);
        }
        return c;
    }

    __attribute__((always_inline)) inline void free_matrix(matrix *a) {
        free(a->m);
        free(a);
    }

    __attribute__((always_inline)) inline void free_matrix_ptr(matrix **a, int len) {
        for(int i = 0; i < len; i++) {
            free_matrix(a[i]);
        }
        free(a);
    }
#endif
