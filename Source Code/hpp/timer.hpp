#ifndef TIMER_H
    #define TIMER_H

    #include <sys/time.h>

    timeval start_timer();
    timeval stop_timer();
    long delta_time_us(timeval start_time, timeval end_time);
    float delta_time_s(timeval start_time, timeval end_time);
#endif
