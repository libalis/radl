#include <glib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../h/io.h"
#include "../h/utils.h"

matrix *io_to_matrix(char *a) {
    FILE *f = fopen(a, "r");
    char *line = NULL;
    size_t len = 0;
    getline(&line, &len, f);
    int x = (int)strtof(line, NULL);
    free(line);
    line = NULL;
    getline(&line, &len, f);
    int y;
    if(strcmp(line, "\n")) {
        y = (int)strtof(line, NULL);
        free(line);
        line = NULL;
        getline(&line, &len, f);
    } else {
        y = 1;
    }
    free(line);
    line = NULL;
    matrix *m = malloc_matrix(x, y);
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

io *malloc_io() {
    io *a = malloc(sizeof(io));
    a->conv_bias = io_to_matrix(CONV_BIAS);
    a->fc_bias = io_to_matrix(FC_BIAS);
    a->fc_weights = io_to_matrix(FC_WEIGHTS);
    a->image_len = get_value(IMAGE_LEN);

    a->image = malloc(a->image_len * sizeof(matrix*));
    for(int i = 0; i < a->image_len; i++) {
        char *c = g_strdup_printf(IMAGE, i);
        a->image[i] = io_to_matrix(c);
        free(c);
    }

    a->label = malloc(a->image_len * sizeof(int));
    for(int i = 0; i < a->image_len; i++) {
        char *c = g_strdup_printf(LABEL, i);
        a->label[i] = get_value(c);
        free(c);
    }

    a->masks_len = get_value(MASKS_LEN);

    a->masks = malloc(a->masks_len * sizeof(matrix*));
    for(int i = 0; i < a->masks_len; i++) {
        char *c = g_strdup_printf(MASKS, i);
        a->masks[i] = io_to_matrix(c);
        free(c);
    }
    return a;
}

void free_io(io *a) {
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
