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

int get_value(char* a) {
    FILE* f = fopen(a, "r");
    char* line = NULL;
    size_t len = 0;
    getline(&line, &len, f);
    int c = (int)strtof(line, NULL);
    fclose(f);
    free(line);
    line = NULL;
    len = 0;
    return c;
}
