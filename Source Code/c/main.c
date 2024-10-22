#include "../h/io.h"
#include "../h/tf.h"
#include "../h/timer.h"
#include <stdio.h>

int main(int argc, char* argv[]) {
    start_timer();
    io* i = malloc_io();
    matrix* images;
    matrix* masks;
    matrix* c = conv2d(images, masks);
    matrix* b = biasing(c, i->conv_bias);
    matrix* r = relu(b);
    matrix* m = maxpool(r);
    matrix* f = flatten(m);
    matrix* mm = matmul(f, i->fc_weights);
    matrix* a = add(mm, i->fc_bias);
    printf("%d\n", max(a));
    free_io(i);
    printf("%ld us\n", stop_timer_us());
    return 0;
}
