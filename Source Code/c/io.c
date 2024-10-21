#include "../h/io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

matrix* io_to_matrix(char* a) {
    // TODO
    return malloc_matrix(0, 0);
}

io* malloc_io() {
    io* a = malloc(sizeof(io));
    a->conv_bias = io_to_matrix("../weights/conv_bias.txt");
    a->fc_bias = io_to_matrix("../weights/fc_bias.txt");
    a->fc_weights = io_to_matrix("../weights/fc_weights.txt");
    FILE* f = fopen("./weights/masks.txt", "r");
    char *line = NULL;
    size_t len = 0;
    getline(&line, &len, f);
    a->masks_len = strtol(line, NULL, 0);
    free(line);
    fclose(f);
    a->masks = malloc(a->masks_len * sizeof(matrix*));
    for(int i = 0; i < a->masks_len; i++) {
        char* str = malloc(strlen("../weights/masks/masks_0.txt") * sizeof(char) + 1);
        snprintf(str, strlen("../weights/masks/masks_0.txt") * sizeof(char) + 1, "../weights/masks/masks_%d.txt", i);
        a->masks[i] = io_to_matrix(str);
        free(str);
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
