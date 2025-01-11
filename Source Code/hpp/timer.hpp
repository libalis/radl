#ifndef TIMER_HPP
    #define TIMER_HPP

    #include <stdlib.h>
    #include <sys/time.h>

    __attribute__((always_inline)) inline timeval start_timer() {
        timeval start_time;
        gettimeofday(&start_time, NULL);
        return start_time;
    }

    __attribute__((always_inline)) inline timeval stop_timer() {
        timeval end_time;
        gettimeofday(&end_time, NULL);
        return end_time;
    }

    __attribute__((always_inline)) inline long delta_time_us(timeval start_time, timeval end_time) {
        return (end_time.tv_sec - start_time.tv_sec) * 1e6 + (end_time.tv_usec - start_time.tv_usec);
    }

    __attribute__((always_inline)) inline float delta_time_s(timeval start_time, timeval end_time) {
        return (float)delta_time_us(start_time, end_time) / 1e6;
    }
#endif
