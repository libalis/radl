#include <stdlib.h>

#include "../hpp/timer.hpp"

timeval start_timer() {
    timeval start_time;
    gettimeofday(&start_time, NULL);
    return start_time;
}

timeval stop_timer() {
    timeval end_time;
    gettimeofday(&end_time, NULL);
    return end_time;
}

long delta_time_us(timeval start_time, timeval end_time) {
    return (end_time.tv_sec - start_time.tv_sec) * 1e6 + (end_time.tv_usec - start_time.tv_usec);
}

float delta_time_s(timeval start_time, timeval end_time) {
    return (float)delta_time_us(start_time, end_time) / 1e6;
}
