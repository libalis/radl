#include "../h/io.h"
#include "../h/tf.h"
#include "../h/timer.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    start_timer();

    io* i = malloc_io();

    matrix** c = conv2d(i->image, i->masks, i->masks_len);
    matrix** b = biasing(c, i->masks_len, i->conv_bias);
    matrix** r = relu(b, i->masks_len);
    matrix** m = maxpool(r, i->masks_len);
    matrix* f = flatten(m, i->masks_len);
    matrix* transposed_f = transpose(f);
    matrix* transposed_fc_weights = transpose(i->fc_weights);
    matrix* mm = matmul(transposed_f, transposed_fc_weights);
    matrix* transposed_fc_bias = transpose(i->fc_bias);
    matrix* a = add(mm, transposed_fc_bias);

    int max_val = max(a);
    printf("%d\n", max_val);
    printf("accurate: %d\n", max_val == i->label);

    free_matrix(a);
    free_matrix(transposed_fc_bias);
    free_matrix(mm);
    free_matrix(transposed_fc_weights);
    free_matrix(transposed_f);
    free_matrix(f);
    free_matrix_ptr(m, i->masks_len);
    free_matrix_ptr(r, i->masks_len);
    free_matrix_ptr(b, i->masks_len);
    free_matrix_ptr(c, i->masks_len);

    free_io(i);

    printf("%ld us\n", stop_timer_us());
    exit(EXIT_SUCCESS);
}
