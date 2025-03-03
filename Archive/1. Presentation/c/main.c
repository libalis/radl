#include "../h/io.h"
#include "../h/tf.h"
#include "../h/timer.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    int ts[] = {1};//, 4, 5, 8, 11, 16};

    system("bash -c \"mkdir -p ./csv\"");

    FILE* file = fopen("./csv/benchmark.csv", "w");
    fprintf(file, "create_mt_time_us,malloc_time_us,processing_time_us,free_time_us,join_mt_time_us,total_time_us,threads\n");
    fclose(file);

    #if defined(DEBUG) || defined(NVIDIA) || defined(OMP)
        int RUNS = 1;
    #else
        int RUNS = sizeof(ts) / sizeof(ts[0]);
    #endif

    for(int t = 0; t < RUNS; t++) {
        for(int i = -2; i < 10; i++) {
            #ifdef DEBUG
                printf("Cycle %d start", i);
            #endif

            system("bash -c \"./py/export_image.py &>/dev/null\"");

            timeval start_time = start_timer();

            io* io = malloc_io();

            long malloc_time_us = delta_time_us(start_time, stop_timer());

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

            long processing_time_us = delta_time_us(half_time, stop_timer());

            free_io(io);

            long total_time_us = delta_time_us(start_time, stop_timer());

            system("bash -c \"rm -rf ./tmp\"");

            if(i >= 0) {
                FILE *file = fopen("./csv/benchmark.csv", "a");
                fprintf(file, "%ld,%ld,%ld,%ld,%ld,%ld,%d\n", 0l, malloc_time_us, processing_time_us, 0l, 0l, total_time_us, ts[t]);
                fclose(file);
            }
        }
    }

    exit(EXIT_SUCCESS);
}
