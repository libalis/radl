#include <stdio.h>
#include <stdlib.h>

#include "../hpp/main.hpp"

int main(int argc, char *argv[]) {
    int ts[] = {1, 4, 5, 8, 11, 16};

    system(EXPORT);

    system("bash -c \"mkdir -p ./csv\"");

    FILE *file = fopen(BENCHMARK, "w");
    fprintf(file, "create_mt_time_us,malloc_time_us,processing_time_us,free_time_us,join_mt_time_us,total_time_us,threads\n");
    fclose(file);

    #if defined(DEBUG) || defined(NVIDIA) || defined(OMP)
        int RUNS = 1;
    #else
        int RUNS = sizeof(ts) / sizeof(ts[0]);
    #endif

    for(int t = 0; t < RUNS; t++) {
        for(int i = -2; i < EPOCHS; i++) {
            run(ts[t], i);
        }
    }

    system("bash -c \"rm -rf ./tmp\"");

    exit(EXIT_SUCCESS);
}
