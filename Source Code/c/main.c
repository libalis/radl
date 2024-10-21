#include "../h/io.h"
#include "../h/tf.h"
#include "../h/timer.h"
#include <stdio.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
    start_timer();
    io* i = malloc_io();
    matrix* a = malloc_matrix(2, 2);
    a->m[0][0] = 1;
    a->m[0][1] = 2;
    a->m[1][0] = 3;
    a->m[1][1] = 4;
    matrix* b = malloc_matrix(2, 2);
    b->m[0][0] = 1;
    b->m[0][1] = 2;
    b->m[1][0] = 3;
    b->m[1][1] = 4;
    matrix* c = matmul(a, b);
    print_matrix(c);
    free_matrix(a);
    free_matrix(b);
    free_matrix(c);
    free_io(i);
    sleep(3);
    printf("%ld us\n", stop_timer_us());
    printf("%lf s\n", stop_timer());
    return 0;
}
