#include "../h/io.h"
#include "../h/timer.h"
#include <stdio.h>

int main(int argc, char* argv[]) {
    start_timer();
    io* i = malloc_io();
    free_io(i);
    printf("%ld us\n", stop_timer_us());
    return 0;
}
