#ifndef TIMER_H
#define TIMER_H

#include <sys/time.h>

static struct timeval start_time, end_time;

void start_timer();
long stop_timer_us();
float stop_timer();

#endif
