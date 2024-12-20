#ifdef OMP
    #include <omp.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "../hpp/io.hpp"
#include "../hpp/main.hpp"
#include "../hpp/mt.hpp"
#include "../hpp/tf.hpp"
#include "../hpp/timer.hpp"
#include "../hpp/utils.hpp"

int main(int argc, char *argv[]) {
    long MAX_THREADS = sysconf(_SC_NPROCESSORS_ONLN);
    long ts[] = {1, 4, 5, 8, 11, 16};

    long create_mt_time_us = 0;
    long malloc_time_us = 0;
    long processing_time_us = 0;
    long free_time_us = 0;
    long join_mt_time_us = 0;
    long total_time_us = 0;

    system(EXPORT);

    system("bash -c \"mkdir -p ./csv\"");

    FILE *file = fopen(BENCHMARK, "w");
    fprintf(file, "create_mt_time_us,malloc_time_us,processing_time_us,free_time_us,join_mt_time_us,total_time_us,threads\n");
    fclose(file);

    int RUNS;
    #if !defined(DEBUG) && !defined(OMP) && !defined(NVIDIA)
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
            timeval next_time = start_time;

            #if !defined(OMP) && !defined(NVIDIA)
                create_mt(ts[t]);
                create_mt_time_us = delta_time_us(start_time, stop_timer());
                next_time = start_timer();
            #elif defined(OMP)
                THREADS = omp_get_max_threads();
            #endif

            io *io = malloc_io();
            matrix *transposed_fc_weights = transpose(io->fc_weights, NULL);
            matrix **transposed_image = malloc_matrix_ptr(io->image_len, io->image[0]->x, io->image[0]->y);
            for(int i = 0; i < io->image_len; i++) {
                transposed_image[i] = transpose(io->image[i], transposed_image[i]);
            }
            matrix **transposed_masks = malloc_matrix_ptr(io->masks_len, io->masks[0]->x, io->masks[0]->y);
            for(int i = 0; i < io->masks_len; i++) {
                transposed_masks[i] = transpose(io->masks[i], transposed_masks[i]);
            }
            #ifdef DEBUG
                matrix **flipped_masks = malloc_matrix_ptr(io->masks_len, io->masks[0]->x, io->masks[0]->y);
                matrix **flipped_c = malloc_matrix_ptr(io->masks_len, io->image[0]->x - flipped_masks[0]->x + 1, io->image[0]->y - flipped_masks[0]->y + 1);
            #endif
            matrix **c = malloc_matrix_ptr(io->masks_len, io->image[0]->x - io->masks[0]->x + 1, io->image[0]->y - io->masks[0]->y + 1);
            matrix **b = malloc_matrix_ptr(io->masks_len, c[0]->x, c[0]->y);
            #ifdef DEBUG
                matrix **hyperbolic_r = malloc_matrix_ptr(io->masks_len, b[0]->x, b[0]->y);
            #endif
            matrix **r = malloc_matrix_ptr(io->masks_len, b[0]->x, b[0]->y);
            matrix *m = malloc_matrix(io->masks_len * (r[0]->x / POOL_LEN), (r[0]->y / POOL_LEN));
            matrix *f = malloc_matrix(m->x * m->y, 1);
            matrix *transposed_f = malloc_matrix(f->y, f->x);
            matrix *mm = malloc_matrix(transposed_f->x, io->fc_weights->y);
            matrix *transposed_fc_bias = malloc_matrix(io->fc_bias->y, io->fc_bias->x);
            matrix *a = malloc_matrix(mm->x, mm->y);
            malloc_time_us = delta_time_us(next_time, stop_timer());
            next_time = start_timer();

            #ifdef DEBUG
                int accurate = 0;
            #endif

            for(int j = 0; j < io->image_len; j++) {
                #ifdef DEBUG
                    flipped_masks = flip_kernels(transposed_masks, io->masks_len, flipped_masks);
                    flipped_c = conv2d(transposed_image[j], flipped_masks, io->masks_len, flipped_c);
                #endif
                c = conv2d(transposed_image[j], transposed_masks, io->masks_len, c);
                b = biasing(c, io->masks_len, io->conv_bias, b);
                #ifdef DEBUG
                    hyperbolic_r = hyperbolic_tangent(b, io->masks_len, hyperbolic_r);
                #endif
                r = relu(b, io->masks_len, r);
                m = maxpool(r, io->masks_len, m);
                f = flatten(m, io->masks_len, f);
                transposed_f = transpose(f, transposed_f);
                #ifndef NVIDIA
                    mm = matmul(transposed_f, transposed_fc_weights, mm);
                #else
                    mm = matmul(transposed_f, io->fc_weights, mm);
                #endif
                transposed_fc_bias = transpose(io->fc_bias, transposed_fc_bias);
                a = add(mm, transposed_fc_bias, a);

                int max_val = index_of_max_element(a);
                #ifdef DEBUG
                    printf("Epoch %d done: accuracy %f%%\n", j, (float)(max_val == io->label[j]) / 1 * 100);
                    if(max_val == io->label[j]) {
                        accurate++;
                    }
                #endif
            }
            #ifdef DEBUG
                printf("accuracy: %f%%\n", (float)accurate / io->image_len * 100);
            #endif

            processing_time_us = delta_time_us(next_time, stop_timer());
            next_time = start_timer();

            free_matrix(a);
            free_matrix(transposed_fc_bias);
            free_matrix(mm);
            free_matrix(transposed_f);
            free_matrix(f);
            free_matrix(m);
            free_matrix_ptr(r, io->masks_len);
            #ifdef DEBUG
                free_matrix_ptr(hyperbolic_r, io->masks_len);
            #endif
            free_matrix_ptr(b, io->masks_len);
            free_matrix_ptr(c, io->masks_len);
            #ifdef DEBUG
                free_matrix_ptr(flipped_c, io->masks_len);
                free_matrix_ptr(flipped_masks, io->masks_len);
            #endif
            free_io(io);
            free_time_us = delta_time_us(next_time, stop_timer());
            next_time = start_timer();

            #if !defined(OMP) && !defined(NVIDIA)
                join_mt();
                join_mt_time_us = delta_time_us(next_time, stop_timer());
            #endif

            total_time_us = delta_time_us(start_time, stop_timer());

            FILE *file = fopen(BENCHMARK, "a");
            fprintf(file, "%ld,%ld,%ld,%ld,%ld,%ld,%ld\n", create_mt_time_us, malloc_time_us, processing_time_us, free_time_us, join_mt_time_us, total_time_us, THREADS);
            fclose(file);
        }
    }

    system("bash -c \"rm -rf ./tmp\"");

    exit(EXIT_SUCCESS);
}
