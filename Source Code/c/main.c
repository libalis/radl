#include "../h/io.h"
#include "../h/tf.h"
#include "../h/timer.h"
#include <stdio.h>

int main(int argc, char* argv[]) {
    start_timer();
    io* i = malloc_io();
    // TODO
    matrix* images = malloc_matrix(0, 0);
    // TODO
    matrix* masks = malloc_matrix(0, 0);
    matrix* c = conv2d(images, masks);
    matrix* b = biasing(c, i->conv_bias);
    matrix* r = relu(b);
    matrix* m = maxpool(r);
    matrix* f = flatten(m);
    matrix* mm = matmul(f, i->fc_weights);
    matrix* a = add(mm, i->fc_bias);
    printf("%d\n", max(a));
    free_matrix(a);
    free_matrix(mm);
    free_matrix(f);
    free_matrix(m);
    free_matrix(r);
    free_matrix(b);
    free_matrix(c);
    free_matrix(masks);
    free_matrix(images);
    free_io(i);
    printf("%ld us\n", stop_timer_us());
    return 0;
}
