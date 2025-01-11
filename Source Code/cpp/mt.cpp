#include <limits.h>

#include "../hpp/mt.hpp"

long COUNTER = 0;
long THREADS = 1;

GAsyncQueue *queue;

pthread_cond_t cond;
pthread_mutex_t mutex;
pthread_t tids[(int)(CHAR_BIT * sizeof(void *))];
