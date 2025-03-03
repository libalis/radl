#include <stdio.h>
#include <stdlib.h>

#include "../h/utils.h"

int get_decimals(int a) {
    int c = 1;
    for(int i = a; i > 0; i /= 10) {
        c++;
    }
    return c;
}

int get_value(char *a) {
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
    float max_val = a->m[0][0];
    int idx = 0;
    for(int i = 0; i < a->y; i++) {
        float curr_val = a->m[0][i];
        if(curr_val > max_val) {
            max_val = curr_val;
            idx = i;
        }
    }
    return idx;
}
