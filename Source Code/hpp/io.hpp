#ifndef IO_HPP
    #define IO_HPP

    #include <glib.h>

    #include "matrix.hpp"

    #ifndef CONV_BIAS
        #define CONV_BIAS ("./data/conv_bias.txt")
    #endif
    #ifndef FC_BIAS
        #define FC_BIAS ("./data/fc_bias_transposed.txt")
    #endif
    #ifndef FC_WEIGHTS
        #define FC_WEIGHTS ("./data/fc_weights_transposed.txt")
    #endif
    #ifndef IMAGE_LEN
        #define IMAGE_LEN ("./tmp/image_len.txt")
    #endif
    #ifndef IMAGE
        #define IMAGE ("./tmp/image_%d.txt")
    #endif
    #ifndef LABEL
        #define LABEL ("./tmp/label_%d.txt")
    #endif
    #ifndef MASKS_LEN
        #define MASKS_LEN ("./data/masks_len.txt")
    #endif
    #ifndef MASKS
        #define MASKS ("./data/masks_%d.txt")
    #endif

    typedef struct io {
        matrix *conv_bias;
        matrix *fc_bias;
        matrix *fc_weights;
        int image_len;
        matrix **image;
        int *label;
        int masks_len;
        matrix **masks;
    } io;

    __attribute__((always_inline)) inline matrix *io_to_matrix(const char *a) {
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
            m->m[get_idx(i, 0, m->y)] = (DATA_TYPE)strtof(strtok(line, " \n"), NULL);
            for(int j = 1; j < m->y; j++) {
                m->m[get_idx(i, j, m->y)] = (DATA_TYPE)strtof(strtok(NULL, " \n"), NULL);
            }
            free(line);
            line = NULL;
        }
        fclose(f);
        return m;
    }

    __attribute__((always_inline)) inline io *malloc_io() {
        io *a = (io*)malloc(sizeof(io));
        a->conv_bias = io_to_matrix(CONV_BIAS);
        a->fc_bias = io_to_matrix(FC_BIAS);
        a->fc_weights = io_to_matrix(FC_WEIGHTS);
        a->image_len = get_value(IMAGE_LEN);

        a->image = (matrix**)malloc(a->image_len * sizeof(matrix*));
        for(int i = 0; i < a->image_len; i++) {
            char *c = g_strdup_printf(IMAGE, i);
            a->image[i] = io_to_matrix(c);
            free(c);
        }

        a->label = (int*)malloc(a->image_len * sizeof(int));
        for(int i = 0; i < a->image_len; i++) {
            char *c = g_strdup_printf(LABEL, i);
            a->label[i] = get_value(c);
            free(c);
        }

        a->masks_len = get_value(MASKS_LEN);

        a->masks = (matrix**)malloc(a->masks_len * sizeof(matrix*));
        for(int i = 0; i < a->masks_len; i++) {
            char *c = g_strdup_printf(MASKS, i);
            a->masks[i] = io_to_matrix(c);
            free(c);
        }
        return a;
    }

    __attribute__((always_inline)) inline void free_io(io *a) {
        free_matrix(a->conv_bias);
        free_matrix(a->fc_bias);
        free_matrix(a->fc_weights);
        free_matrix_ptr(a->image, a->image_len);
        free(a->label);
        free_matrix_ptr(a->masks, a->masks_len);
        free(a);
    }
#endif
