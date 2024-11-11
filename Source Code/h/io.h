#ifndef IO_H
    #define IO_H

    #include "matrix.h"

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

    matrix *io_to_matrix(char *a);
    io *malloc_io();
    void free_io(io *a);
#endif
