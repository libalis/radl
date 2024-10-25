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
    matrix* mm = matmul(transpose(f), transpose(i->fc_weights));
    matrix* a = add(mm, transpose(i->fc_bias));

    int max_val = max(a);
    printf("%d\n", max_val);
    printf("accurate: %d\n", max_val == i->label);

    free_io(i);

    printf("%ld us\n", stop_timer_us());
    exit(EXIT_SUCCESS);
}
