#include "../h/io.h"
#include "../h/tf.h"
#include "../h/timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifndef RUNS
    #define RUNS (10)
#endif

int main(int argc, char* argv[]) {
    long threads[4] = {1, 4, sysconf(_SC_NPROCESSORS_ONLN), sysconf(_SC_NPROCESSORS_ONLN) * 2};

    system("bash -c \"./py/export_image.py\"");

    system("bash -c \"mkdir -p ./csv\"");

    FILE* f = fopen("./csv/benchmark.csv", "w");
    fprintf(f, "malloc_time_us_avg,processing_time_us_avg,total_time_us_avg,threads\n");
    fclose(f);

    for(int t = 0; t < sizeof(threads) / sizeof(threads[0]); t++) {
        #define THREADS (threads[t])
        long malloc_time = 0;
        long processing_time = 0;
        long total_time = 0;
        for(int i = 0; i < RUNS; i++) {
            #ifdef DEBUG
                printf("Cycle %d start", i);
            #endif

            timeval start_time = start_timer();

            io* io = malloc_io();

            malloc_time += delta_time_us(start_time, stop_timer());

            timeval half_time = start_timer();

            int accurate = 0;

            for(int j = 0; j < io->image_len; j++) {
                matrix** c = conv2d(io->image[j], io->masks, io->masks_len);
                matrix** b = biasing(c, io->masks_len, io->conv_bias);
                matrix** r = relu(b, io->masks_len);
                matrix** m = maxpool(r, io->masks_len);
                matrix* f = flatten(m, io->masks_len);
                matrix* transposed_f = transpose(f);
                matrix* mm = matmul(transposed_f, io->fc_weights);
                matrix* transposed_fc_bias = transpose(io->fc_bias);
                matrix* a = add(mm, transposed_fc_bias);

                int max_val = max(a);
                #ifdef DEBUG
                    printf("Epoch %d done: accuracy %f%%\n", j, (float)(max_val == io->label[j]) / 1 * 100);
                #endif
                if(max_val == io->label[j]) {
                    accurate++;
                }

                free_matrix(a);
                free_matrix(transposed_fc_bias);
                free_matrix(mm);
                free_matrix(transposed_f);
                free_matrix(f);
                free_matrix_ptr(m, io->masks_len);
                free_matrix_ptr(r, io->masks_len);
                free_matrix_ptr(b, io->masks_len);
                free_matrix_ptr(c, io->masks_len);
            }

            #ifdef DEBUG
                printf("accuracy: %f%%\n", (float)accurate / io->image_len * 100);
            #endif

            processing_time += delta_time_us(half_time, stop_timer());

            free_io(io);

            total_time += delta_time_us(start_time, stop_timer());
        }

        FILE* f = fopen("./csv/benchmark.csv", "a");
        fprintf(f, "%ld,%ld,%ld,%ld\n", malloc_time / 10, processing_time / 10, total_time / 10, THREADS);
        fclose(f);
    }

    system("bash -c \"rm -rf ./tmp\"");

    exit(EXIT_SUCCESS);
}
