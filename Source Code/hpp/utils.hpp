#ifndef UTILS_H
    #define UTILS_H

    #include <stdio.h>
    #include <stdlib.h>

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

    __attribute__((always_inline)) inline int get_value(const char *a) {
        FILE *f = fopen(a, "r");
        char *line = NULL;
        size_t len = 0;
        getline(&line, &len, f);
        int c = (int)strtof(line, NULL);
        fclose(f);
        free(line);
        line = NULL;
        len = 0;
        return c;
    }

    __attribute__((always_inline)) inline int index_of_max_element(matrix *a) {
        DATA_TYPE max_val = a->m[get_idx(0, 0, a->y)];
        int idx = 0;
        for(int i = 0; i < a->y; i++) {
            DATA_TYPE curr_val = a->m[get_idx(0, i, a->y)];
            if(curr_val > max_val) {
                max_val = curr_val;
                idx = i;
            }
        }
        return idx;
    }

    __attribute__((always_inline)) inline bool is_avx512_supported() {
        int cpu_info[4];
        __asm__ __volatile__("cpuid": "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3]): "a"(7), "c"(0));
        return (cpu_info[1] & (1 << 16)) != 0;
    }
#endif
