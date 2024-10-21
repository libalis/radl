#include "../h/timer.h"
#include <stdlib.h>

void start_timer() {
    gettimeofday(&start_time, NULL);
}

long stop_timer_us() {
    gettimeofday(&end_time, NULL);
    return (end_time.tv_sec - start_time.tv_sec) * 1e6 + (end_time.tv_usec - start_time.tv_usec);
}

float stop_timer() {
    return (float)stop_timer_us() / 1e6;
}
