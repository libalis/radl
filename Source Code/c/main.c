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
    long ts[] = {1, 4, sysconf(_SC_NPROCESSORS_ONLN), sysconf(_SC_NPROCESSORS_ONLN) * 2};

    system("bash -c \"./py/export_image.py\"");

    system("bash -c \"mkdir -p ./csv\"");

    FILE *f = fopen("./csv/benchmark.csv", "w");
    fprintf(f, "malloc_time_us_avg,processing_time_us_avg,total_time_us_avg,threads\n");
    fclose(f);

    for(int t = 0; t < RUNS; t++) {
        THREADS = ts[t];
        long malloc_time = 0;
        long processing_time = 0;
        long total_time = 0;
        for(int i = 0; i < EPOCHS; i++) {
            #ifdef DEBUG
                printf("Run %d start\n", i);
            #endif

            timeval start_time = start_timer();

            io *io = malloc_io();

            malloc_time += delta_time_us(start_time, stop_timer());

            timeval half_time = start_timer();

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

            processing_time += delta_time_us(half_time, stop_timer());

            free_io(io);

            total_time += delta_time_us(start_time, stop_timer());
        }

        FILE *f = fopen("./csv/benchmark.csv", "a");
        fprintf(f, "%ld,%ld,%ld,%ld\n", malloc_time / EPOCHS, processing_time / EPOCHS, total_time / EPOCHS, THREADS);
        fclose(f);
    }

    system("bash -c \"rm -rf ./tmp\"");

    exit(EXIT_SUCCESS);
}
