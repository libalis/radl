#include "../h/io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

matrix* io_to_matrix(char* a) {
    // first two rows show the dimensions
    // first row: y (except 'conv_bias' and 'fc_bias' because there is only one row, which is x)
    // second row: x
    FILE* f = fopen(a, "r");
    char* line = NULL;
    size_t len = 0;
    getline(&line, &len, f);
    int y = (int)strtof(line, NULL);
    free(line);
    line = NULL;
    getline(&line, &len, f);
    int x;
    if(strcmp(line, "\n")) {
        x = (int)strtof(line, NULL);
        free(line);
        line = NULL;
        getline(&line, &len, f);
    } else {
        x = y;
        y = 1;
    }
    free(line);
    line = NULL;
    matrix* m = malloc_matrix(x, y);
    for(int i = 0; i < m->x; i++) {
        getline(&line, &len, f);
        m->m[i][0] = strtof(strtok(line, " \n"), NULL);
        for(int j = 1; j < m->y; j++) {
            m->m[i][j] = strtof(strtok(NULL, " \n"), NULL);
        }
        free(line);
        line = NULL;
    }
    fclose(f);
    return m;
}

io* malloc_io() {
    io* a = malloc(sizeof(io));
    a->conv_bias = io_to_matrix("./weights/conv_bias.txt");
    a->fc_bias = io_to_matrix("./weights/fc_bias.txt");
    a->fc_weights = io_to_matrix("./weights/fc_weights.txt");
    FILE* f = fopen("./weights/masks.txt", "r");
    char* line = NULL;
    size_t len = 0;
    getline(&line, &len, f);
    a->masks_len = (int)strtof(line, NULL);
    free(line);
    line = NULL;
    fclose(f);
    a->masks = malloc(a->masks_len * sizeof(matrix*));
    for(int i = 0; i < a->masks_len; i++) {
        char* c = malloc(strlen("./weights/masks_.txt") * sizeof(char) + sizeof(int) + 1);
        sprintf(c, "./weights/masks_%d.txt", i);
        a->masks[i] = io_to_matrix(c);
        free(c);
    }
    return a;
}

void free_io(io* a) {
    free_matrix(a->conv_bias);
    free_matrix(a->fc_bias);
    free_matrix(a->fc_weights);
    for(int i = 0; i < a->masks_len; i++) {
        free_matrix(a->masks[i]);
    }
    free(a->masks);
    free(a);
}
