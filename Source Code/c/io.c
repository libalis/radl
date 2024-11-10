#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../h/io.h"
#include "../h/utils.h"

#ifndef CONV_BIAS
    #define CONV_BIAS ("./data/conv_bias.txt")
#endif
#ifndef FC_BIAS
    #define FC_BIAS ("./data/fc_bias.txt")
#endif
#ifndef FC_WEIGHTS
    #define FC_WEIGHTS ("./data/fc_weights.txt")
#endif
#ifndef IMAGE_LEN
    #define IMAGE_LEN ("./tmp/image_len.txt")
#endif
#ifndef IMAGE
    #define IMAGE ("./tmp/image_.txt")
#endif
#ifndef LABEL
    #define LABEL ("./tmp/label_.txt")
#endif
#ifndef MASKS_LEN
    #define MASKS_LEN ("./data/masks_len.txt")
#endif
#ifndef MASKS
    #define MASKS ("./data/masks_.txt")
#endif

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
    assert(x > 0);
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
    a->conv_bias = io_to_matrix(CONV_BIAS);
    a->fc_bias = io_to_matrix(FC_BIAS);
    a->fc_weights = io_to_matrix(FC_WEIGHTS);
    a->image_len = get_value(IMAGE_LEN);

    a->image = malloc(a->image_len * sizeof(matrix*));
    for(int i = 0; i < a->image_len; i++) {
        char* c = malloc(strlen(IMAGE) * sizeof(char) + get_decimals(i) * sizeof(char) + 2);
        snprintf(c, strlen(IMAGE) * sizeof(char) + get_decimals(i) * sizeof(char) + 2, "./tmp/image_%d.txt", i);
        a->image[i] = io_to_matrix(c);
        free(c);
        c = NULL;
    }

    a->label = malloc(a->image_len * sizeof(int));
    for(int i = 0; i < a->image_len; i++) {
        char* c = malloc(strlen(LABEL) * sizeof(char) + get_decimals(i) * sizeof(char) + 2);
        snprintf(c, strlen(LABEL) * sizeof(char) + get_decimals(i) * sizeof(char) + 2, "./tmp/label_%d.txt", i);
        a->label[i] = get_value(c);
        free(c);
        c = NULL;
    }

    a->masks_len = get_value(MASKS_LEN);

    a->masks = malloc(a->masks_len * sizeof(matrix*));
    for(int i = 0; i < a->masks_len; i++) {
        char* c = malloc(strlen(MASKS) * sizeof(char) + get_decimals(i) * sizeof(char) + 2);
        snprintf(c, strlen(MASKS) * sizeof(char) + get_decimals(i) * sizeof(char) + 2, "./data/masks_%d.txt", i);
        a->masks[i] = io_to_matrix(c);
        free(c);
        c = NULL;
    }
    return a;
}

void free_io(io* a) {
    free_matrix(a->conv_bias);
    free_matrix(a->fc_bias);
    free_matrix(a->fc_weights);
    for(int i = 0; i < a->image_len; i++) {
        free_matrix(a->image[i]);
    }
    free(a->image);
    free(a->label);
    for(int i = 0; i < a->masks_len; i++) {
        free_matrix(a->masks[i]);
    }
    free(a->masks);
    free(a);
}
