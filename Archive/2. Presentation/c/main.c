#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "../h/io.h"
#include "../h/main.h"
#include "../h/mt.h"
#include "../h/tf.h"
#include "../h/timer.h"
#include "../h/utils.h"

int main(int argc, char *argv[]) {
    long MAX_THREADS = sysconf(_SC_NPROCESSORS_ONLN);
    long ts[] = {1, 4, 5, 8, 11, 16};

    system(EXPORT);

    system("bash -c \"mkdir -p ./csv\"");

    FILE *f = fopen(BENCHMARK, "w");
    fprintf(f, "create_mt_time_us,malloc_io_time_us,processing_time_us,free_io_time_us,join_mt_time_us,total_time_us,threads\n");
    fclose(f);

    int RUNS;
    #ifndef DEBUG
        RUNS = sizeof(ts) / sizeof(ts[0]);
    #else
        RUNS = 1;
    #endif
    for(int t = 0; t < RUNS; t++) {
        for(int i = 0; i < EPOCHS; i++) {
            #ifdef DEBUG
                printf("Run %d start\n", i);
            #endif

            timeval start_time = start_timer();
            create_mt(ts[t]);
            long create_mt_time_us = delta_time_us(start_time, stop_timer());
            timeval next_time = start_timer();

            io *io = malloc_io();
            long malloc_io_time_us = delta_time_us(next_time, stop_timer());
            next_time = start_timer();

            int accurate = 0;

            for(int j = 0; j < io->image_len; j++) {
                #ifdef DEBUG
                    matrix **flipped_masks = flip_kernels(io->masks, io->masks_len);
                    matrix **flipped_c = conv2d(io->image[j], flipped_masks, io->masks_len);
                    free_matrix_ptr(flipped_c, io->masks_len);
                    free_matrix_ptr(flipped_masks, io->masks_len);
                #endif
                matrix **c = conv2d(io->image[j], io->masks, io->masks_len);
                matrix **b = biasing(c, io->masks_len, io->conv_bias);
                #ifndef DEBUG
                    matrix **hyperbolic_r = hyperbolic_tangent(b, io->masks_len);
                    free_matrix_ptr(hyperbolic_r, io->masks_len);
                #endif
                matrix **r = relu(b, io->masks_len);
                matrix **m = maxpool(r, io->masks_len);
                matrix *f = flatten(m, io->masks_len);
                matrix *transposed_f = transpose(f);
                matrix *mm = matmul(transposed_f, io->fc_weights);
                matrix *transposed_fc_bias = transpose(io->fc_bias);
                matrix *a = add(mm, transposed_fc_bias);

                int max_val = index_of_max_element(a);
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

            long processing_time_us = delta_time_us(next_time, stop_timer());
            next_time = start_timer();

            free_io(io);
            long free_io_time_us = delta_time_us(next_time, stop_timer());
            next_time = start_timer();

            join_mt();
            long join_mt_time_us = delta_time_us(next_time, stop_timer());

            long total_time_us = delta_time_us(start_time, stop_timer());

            FILE *f = fopen(BENCHMARK, "a");
            fprintf(f, "%ld,%ld,%ld,%ld,%ld,%ld,%ld\n", create_mt_time_us, malloc_io_time_us, processing_time_us, free_io_time_us, join_mt_time_us, total_time_us, THREADS);
            fclose(f);
        }
    }

    system("bash -c \"rm -rf ./tmp\"");

    exit(EXIT_SUCCESS);
}
