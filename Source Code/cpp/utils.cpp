#include <stdio.h>
#include <stdlib.h>

#include "../hpp/utils.hpp"

int get_decimals(int a) {
    int c = 1;
    for(int i = a; i > 0; i /= 10) {
        c++;
    }
    return c;
}

int get_idx(int i, int j, int y) {
    return i * y + j;
}

int get_value(const char *a) {
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

int index_of_max_element(matrix *a) {
    float max_val = a->m[get_idx(0, 0, a->y)];
    int idx = 0;
    for(int i = 0; i < a->y; i++) {
        float curr_val = a->m[get_idx(0, i, a->y)];
        if(curr_val > max_val) {
            max_val = curr_val;
            idx = i;
        }
    }
    return idx;
}
