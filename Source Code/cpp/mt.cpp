#include <limits.h>
#include <unistd.h>

#include "../hpp/mt.hpp"

int COUNTER = 0;
int THREADS = 1;
int MAX_THREADS = sysconf(_SC_NPROCESSORS_ONLN);

GAsyncQueue *queue;

pthread_cond_t cond;
pthread_mutex_t mutex;
pthread_t tids[(int)(CHAR_BIT * sizeof(void *))];
