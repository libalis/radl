#ifndef matrix_int8_HPP
    #define matrix_int8_HPP

    #include "utils.hpp"

    typedef struct matrix_int8 {
        int x;
        int y;
        int8_t *m;
    } matrix_int8;

    __attribute__((always_inline)) inline void print_matrix_int8(matrix_int8 *a) {
        for(int i = 0; i < a->x; i++) {
            for(int j = 0; j < a->y - 1; j++) {
                printf("%d ", a->m[get_idx(i, j, a->y)]);
            }
            printf("%d\n", a->m[get_idx(i, a->y - 1, a->y)]);
        }
    }

    __attribute__((always_inline)) inline matrix_int8 *malloc_matrix_int8(int x, int y) {
        matrix_int8 *a = (matrix_int8*)malloc(sizeof(matrix_int8));
        a->x = x;
        a->y = y;
        a->m = (int8_t*)malloc(a->x * a->y * sizeof(int8_t));
        for(int i = 0; i < a->x; i++) {
            for(int j = 0; j < a->y; j++) {
                a->m[get_idx(i, j, a->y)] = 0;
            }
        }
        return a;
    }

    __attribute__((always_inline)) inline matrix_int8 **malloc_matrix_int8_ptr(int len, int x, int y) {
        matrix_int8 **c = (matrix_int8**)malloc(len * sizeof(matrix_int8*));
        for(int i = 0; i < len; i++) {
            c[i] = malloc_matrix_int8(x, y);
        }
        return c;
    }

    __attribute__((always_inline)) inline void free_matrix_int8(matrix_int8 *a) {
        free(a->m);
        free(a);
    }

    __attribute__((always_inline)) inline void free_matrix_int8_ptr(matrix_int8 **a, int len) {
        for(int i = 0; i < len; i++) {
            free_matrix_int8(a[i]);
        }
        free(a);
    }
#endif
