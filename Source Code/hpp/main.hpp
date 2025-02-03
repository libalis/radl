#ifndef MAIN_HPP
    #define MAIN_HPP

    #include "img.hpp"
    #include "io.hpp"
    #include "mt.hpp"
    #ifdef NVIDIA
        #include "tf.hpp"
    #endif
    #include "timer.hpp"

    #ifndef BENCHMARK
        #define BENCHMARK ("./csv/benchmark.csv")
    #endif

    #ifndef EPOCHS
        #ifndef DEBUG
            #define EPOCHS (10)
        #else
            #define EPOCHS (-1)
        #endif
    #endif

    #ifndef EXPORT
        #define EXPORT ("bash -c \"./py/export_image.py\"")
    #endif

    int main(int argc, char *argv[]);

    #ifdef NVIDIA
        __attribute__((always_inline)) inline void run_nvidia(int i) {
            timeval start_time;
            timeval next_time;

            long create_mt_time_us = 0;
            long malloc_time_us = 0;
            long processing_time_us = 0;
            long free_time_us = 0;
            long join_mt_time_us = 0;
            long total_time_us = 0;

            #ifdef DEBUG
                printf("Run %d start\n", i);
            #endif

            start_time = next_time = start_timer();

            int THREADS = 16;

            io *io = malloc_io();

            matrix **flipped_masks = malloc_matrix_ptr(io->masks_len, io->masks[0]->x, io->masks[0]->y);
            matrix *d_img = malloc_cuda_matrix(io->image[0]->x, io->image[0]->y);
            #ifdef DEBUG
                matrix **d_masks = malloc_cuda_matrix_ptr(io->masks_len, io->masks[0]->x, io->masks[0]->y);
                for (int m = 0; m < io->masks_len; m++) {
                    copy_cuda_matrix(io->masks[m], d_masks[m], true);
                }
                matrix **d_flipped_masks = malloc_cuda_matrix_ptr(io->masks_len, io->masks[0]->x, io->masks[0]->y);
                matrix **d_flipped_c = malloc_cuda_matrix_ptr(io->masks_len, io->image[0]->x - io->masks[0]->x + 1, io->image[0]->y - io->masks[0]->y + 1);
            #endif
            init_const_memory(io->masks);
            matrix *d_fc_weights = malloc_cuda_matrix(io->fc_weights->x, io->fc_weights->y);
            copy_cuda_matrix(io->fc_weights, d_fc_weights, true);
            matrix **d_c = malloc_cuda_matrix_ptr(io->masks_len, io->image[0]->x - io->masks[0]->x + 1, io->image[0]->y - io->masks[0]->y + 1);
            matrix **d_calc = malloc_cuda_matrix_ptr(io->masks_len, d_c[0]->x, d_c[0]->y);
            #ifdef DEBUG
                matrix **d_hyperbolic_r = malloc_cuda_matrix_ptr(io->masks_len, d_calc[0]->x, d_calc[0]->y);
            #endif
            matrix *d_m = malloc_cuda_matrix(io->masks_len * (d_calc[0]->x / POOL_LEN), (d_calc[0]->y / POOL_LEN));
            matrix* d_f = malloc_cuda_matrix(1, d_m->x * d_m->y);
            matrix* d_end = malloc_cuda_matrix(d_f->x, d_fc_weights->x);
            matrix* d_fc_bias = malloc_cuda_matrix(io->fc_bias->x, io->fc_bias->y);
            copy_cuda_matrix(io->fc_bias, d_fc_bias, true);
            matrix *a = malloc_matrix(d_end->x, d_end->y);
            int max_val = 0;

            malloc_time_us = delta_time_us(next_time, stop_timer());
            next_time = start_timer();

            #ifdef DEBUG
                int accurate = 0;
            #endif

            for(int j = 0; j < io->image_len; j++) {
                copy_cuda_matrix(io->image[j], d_img, true);
                #ifdef DEBUG
                    flip_kernels(d_masks, io->masks_len, d_flipped_masks);
                    for(int m = 0; m < io->masks_len; m++) {
                        copy_cuda_matrix(flipped_masks[m], d_flipped_masks[m], false);
                    }
                    init_const_memory(flipped_masks);
                    conv2d(d_img, d_flipped_masks, io->masks_len, d_flipped_c);
                    init_const_memory(io->masks);
                #endif
                conv2d(d_img, io->masks, io->masks_len, d_c);
                biasing(d_c, io->masks_len, io->conv_bias, d_calc);
                #ifdef DEBUG
                    hyperbolic_tangent(d_calc, io->masks_len, d_hyperbolic_r);
                #endif
                relu(d_calc, io->masks_len, NULL);
                maxpool(d_calc, io->masks_len, d_m);
                flatten(d_m, io->masks_len, d_f);
                matmul(d_f, d_fc_weights, d_end);
                add(d_end, d_fc_bias, NULL);
                copy_cuda_matrix(a, d_end, false);
                max_val = index_of_max_element(a);
                #ifdef DEBUG
                    if (max_val == io->label[j]) {
                        printf("Epoch %d done: accuracy 100%%\n", j);
                        accurate++;
                    } else {
                        printf("Epoch %d done: accuracy 0%%\n", j);
                    }
                #endif
            }
            #ifdef DEBUG
                printf("accuracy: %f%%\n", (float)accurate / io->image_len * 100);
            #endif

            processing_time_us = delta_time_us(next_time, stop_timer());
            next_time = start_timer();

            free_matrix(a);
            free_cuda_matrix(d_fc_bias);
            free_cuda_matrix(d_end);
            free_cuda_matrix(d_f);
            free_cuda_matrix(d_m);
            #ifdef DEBUG
                free_cuda_matrix_ptr(d_hyperbolic_r, io->masks_len);
            #endif
            free_cuda_matrix_ptr(d_calc, io->masks_len);
            free_cuda_matrix_ptr(d_c, io->masks_len);
            free_cuda_matrix(d_fc_weights);
            #ifdef DEBUG
                free_cuda_matrix_ptr(d_flipped_c, io->masks_len);
                free_cuda_matrix_ptr(d_flipped_masks, io->masks_len);
                free_cuda_matrix_ptr(d_masks, io->masks_len);
            #endif
            free_cuda_matrix(d_img);
            free_matrix_ptr(flipped_masks, io->masks_len);

            free_io(io);

            free_time_us = delta_time_us(next_time, stop_timer());
            next_time = start_timer();

            total_time_us = delta_time_us(start_time, stop_timer());

            #ifdef DEBUG
                FILE *file = fopen(BENCHMARK, "a");
                fprintf(file, "%ld,%ld,%ld,%ld,%ld,%ld,%d\n", create_mt_time_us, malloc_time_us, processing_time_us, free_time_us, join_mt_time_us, total_time_us, THREADS);
                fclose(file);
            #else
                if(i >= 0) {
                    FILE *file = fopen(BENCHMARK, "a");
                    fprintf(file, "%ld,%ld,%ld,%ld,%ld,%ld,%d\n", create_mt_time_us, malloc_time_us, processing_time_us, free_time_us, join_mt_time_us, total_time_us, THREADS);
                    fclose(file);
                }
            #endif
        }
    #elif defined(OMP)
        __attribute__((always_inline)) inline void run_omp(int i) {
            timeval start_time;
            timeval next_time;

            long create_mt_time_us = 0;
            long malloc_time_us = 0;
            long processing_time_us = 0;
            long free_time_us = 0;
            long join_mt_time_us = 0;
            long total_time_us = 0;

            #ifdef DEBUG
                printf("Run %d start\n", i);
            #endif

            start_time = next_time = start_timer();

            int THREADS = sysconf(_SC_NPROCESSORS_ONLN);

            io *io = malloc_io();

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
            matrix *f = malloc_matrix(1, m->x * m->y);
            matrix *mm = malloc_matrix(f->x, io->fc_weights->x);
            matrix *a = malloc_matrix(mm->x, mm->y);
            int max_val = 0;

            malloc_time_us = delta_time_us(next_time, stop_timer());
            next_time = start_timer();

            #ifdef DEBUG
                int accurate = 0;
            #endif

            for(int j = 0; j < io->image_len; j++) {
                #ifdef DEBUG
                    flip_kernels(io->masks, io->masks_len, flipped_masks, NULL);
                    conv2d(io->image[j], flipped_masks, io->masks_len, flipped_c, NULL);
                #endif
                conv2d(io->image[j], io->masks, io->masks_len, c, NULL);
                biasing(c, io->masks_len, io->conv_bias, b, NULL);
                #ifdef DEBUG
                    hyperbolic_tangent(b, io->masks_len, hyperbolic_r, NULL);
                #endif
                relu(b, io->masks_len, r, NULL);
                maxpool(r, io->masks_len, m, NULL);
                flatten(m, io->masks_len, f, NULL);
                matmul(f, io->fc_weights, mm, NULL);
                add(mm, io->fc_bias, a, NULL);
                max_val = index_of_max_element(a);
                #ifdef DEBUG
                    if(max_val == io->label[j]) {
                        printf("Epoch %d done: accuracy 100%%\n", j);
                        accurate++;
                    } else {
                        printf("Epoch %d done: accuracy 0%%\n", j);
                    }
                #endif
            }
            #ifdef DEBUG
                printf("accuracy: %f%%\n", (float)accurate / io->image_len * 100);
            #endif

            processing_time_us = delta_time_us(next_time, stop_timer());
            next_time = start_timer();

            free_matrix(a);
            free_matrix(mm);
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

            total_time_us = delta_time_us(start_time, stop_timer());

            #ifdef DEBUG
                FILE *file = fopen(BENCHMARK, "a");
                fprintf(file, "%ld,%ld,%ld,%ld,%ld,%ld,%d\n", create_mt_time_us, malloc_time_us, processing_time_us, free_time_us, join_mt_time_us, total_time_us, THREADS);
                fclose(file);
            #else
                if(i >= 0) {
                    FILE *file = fopen(BENCHMARK, "a");
                    fprintf(file, "%ld,%ld,%ld,%ld,%ld,%ld,%d\n", create_mt_time_us, malloc_time_us, processing_time_us, free_time_us, join_mt_time_us, total_time_us, THREADS);
                    fclose(file);
                }
            #endif
         }
    #else
        __attribute__((always_inline)) inline void run(int t, int i) {
            timeval start_time;
            timeval next_time;

            long create_mt_time_us = 0;
            long malloc_time_us = 0;
            long processing_time_us = 0;
            long free_time_us = 0;
            long join_mt_time_us = 0;
            long total_time_us = 0;

            #ifdef DEBUG
                printf("Run %d start\n", i);
            #endif

            start_time = next_time = start_timer();

            int THREADS = t;
            mt instance(THREADS);

            create_mt_time_us = delta_time_us(start_time, stop_timer());
            next_time = start_timer();

            io *io = malloc_io();

            malloc_time_us = delta_time_us(next_time, stop_timer());
            next_time = start_timer();

            mt_arg arg[instance.THREADS];
            for(int i = 0; i < instance.THREADS; i++) {
                arg[i].io = io;
                arg[i].start_routine = process_images;
                instance.push_mt(&arg[i]);
            }
            instance.wait_mt();

            #ifdef DEBUG
                printf("accuracy: %f%%\n", (float)instance.accurate / io->image_len * 100);
            #endif

            processing_time_us = delta_time_us(next_time, stop_timer());
            next_time = start_timer();

            free_io(io);

            free_time_us = delta_time_us(next_time, stop_timer());
            next_time = start_timer();

            total_time_us = delta_time_us(start_time, stop_timer());

            #ifdef DEBUG
                FILE *file = fopen(BENCHMARK, "a");
                fprintf(file, "%ld,%ld,%ld,%ld,%ld,%ld,%d\n", create_mt_time_us, malloc_time_us, processing_time_us, free_time_us, join_mt_time_us, total_time_us, THREADS);
                fclose(file);
            #else
                if(i >= 0) {
                    FILE *file = fopen(BENCHMARK, "a");
                    fprintf(file, "%ld,%ld,%ld,%ld,%ld,%ld,%d\n", create_mt_time_us, malloc_time_us, processing_time_us, free_time_us, join_mt_time_us, total_time_us, THREADS);
                    fclose(file);
                }
            #endif
        }
    #endif
#endif
