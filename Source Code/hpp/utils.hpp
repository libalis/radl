#ifndef UTILS_H
    #define UTILS_H

    #include "../hpp/matrix.hpp"

    __attribute__((always_inline)) inline int get_decimals(int a) {
        int c = 1;
        for(int i = a; i > 0; i /= 10) {
            c++;
        }
        return c;
    }

    __attribute__((always_inline)) inline int get_idx(int i, int j, int y) {
        return i * y + j;
    }

    int get_value(const char *a);
    int index_of_max_element(matrix *a);
#endif
