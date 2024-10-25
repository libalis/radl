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
    matrix* m = maxpool(r, i->masks_len);
    matrix* f = flatten(m);
    matrix* mm = matmul(f, i->fc_weights);
    matrix* a = add(mm, i->fc_bias);

    print_matrix(a);
    printf("%d\n", max(a));
    printf("accurate: %d\n", max(a) == i->label);

    free_matrix(a);
    free_matrix(mm);
    free_matrix(f);
    free_matrix(m);
    for (int j = 0; j < i->masks_len; j++) {
        free_matrix(r[j]);
        free_matrix(b[j]);
        free_matrix(c[j]);
    }
    free(r);
    free(b);
    free(c);
    free_io(i);

    printf("%ld us\n", stop_timer_us());
    return EXIT_SUCCESS;
}
