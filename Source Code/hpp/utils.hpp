#ifndef UTILS_HPP
    #define UTILS_HPP

    #include <stdio.h>
    #include <stdlib.h>

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
#endif
