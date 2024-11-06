#include "../h/io.h"
#include "../h/tf.h"
#include "../h/timer.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    system("bash -c \"./py/export_image.py &>/dev/null\"");

    timeval start_time = start_timer();

    io* i = malloc_io();

    printf("malloc time: %ld us\n", delta_time_us(start_time, stop_timer()));

    timeval processing_time = start_timer();

    int accurate = 0;

    for(int n = 0; n < i->image_len; n++) {
        // matrix** flipped_masks = flip_kernels(i->masks, i->masks_len);
        // matrix** c = conv2d(i->image[n], flipped_masks, i->masks_len);
        matrix** c = conv2d(i->image[n], i->masks, i->masks_len);
        matrix** b = biasing(c, i->masks_len, i->conv_bias);
        matrix** r = relu(b, i->masks_len);
        matrix** m = maxpool(r, i->masks_len);
        matrix* f = flatten(m, i->masks_len);
        matrix* transposed_f = transpose(f);
        matrix* mm = matmul(transposed_f, i->fc_weights);
        matrix* transposed_fc_bias = transpose(i->fc_bias);
        matrix* a = add(mm, transposed_fc_bias);

        int max_val = max(a);
        // printf("Epoch %d done: accuracy %f%%\n", n, (float)(max_val == i->label[n]) / 1 * 100);
        if(max_val == i->label[n]) {
            accurate++;
        }

        free_matrix(a);
        free_matrix(transposed_fc_bias);
        free_matrix(mm);
        free_matrix(transposed_f);
        free_matrix(f);
        free_matrix_ptr(m, i->masks_len);
        free_matrix_ptr(r, i->masks_len);
        free_matrix_ptr(b, i->masks_len);
        free_matrix_ptr(c, i->masks_len);
        // free_matrix_ptr(flipped_masks, i->masks_len);
    }

    // printf("accuracy: %f%%\n", (float)accurate / i->image_len * 100);

    printf("processing time: %ld us\n", delta_time_us(processing_time, stop_timer()));

    free_io(i);

    printf("total time: %ld us\n", delta_time_us(start_time, stop_timer()));

    system("bash -c \"rm -rf ./tmp\"");

    exit(EXIT_SUCCESS);
}
