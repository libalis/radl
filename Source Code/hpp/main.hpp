#ifndef MAIN_HPP
    #define MAIN_HPP

    #include "io.hpp"
    #include "mt.hpp"
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

        #if defined(NVIDIA)
            THREADS = 16;
        #elif defined(OMP)
            THREADS = MAX_THREADS;
        #else
            create_mt(t);

            create_mt_time_us = delta_time_us(start_time, stop_timer());
            next_time = start_timer();
        #endif

        io *io = malloc_io();

        #ifdef NVIDIA
            matrix *d_img = malloc_cuda_matrix(io->image[0]->x, io->image[0]->y);
            #ifdef DEBUG
                matrix **d_flipped_masks = malloc_cuda_matrix_ptr(io->masks_len, io->masks[0]->x, io->masks[0]->y);
            #endif
            matrix **d_masks = malloc_cuda_matrix_ptr(io->masks_len, io->masks[0]->x, io->masks[0]->y);
            for (int m = 0; m < io->masks_len; m++) {
                copy_cuda_matrix(io->masks[m], d_masks[m], true);
            }

            matrix *transposed_fc_bias = transpose(io->fc_bias, NULL);
            matrix *transposed_fc_weights = transpose(io->fc_weights, NULL);

            matrix *d_fc_weights = malloc_cuda_matrix(transposed_fc_weights->x, transposed_fc_weights->y);
            copy_cuda_matrix(transposed_fc_weights, d_fc_weights, true);
        #endif

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
        matrix *mm = malloc_matrix(transposed_f->x, io->fc_weights->x);
        matrix *a = malloc_matrix(mm->x, mm->y);

        int max_val = 0;

        malloc_time_us = delta_time_us(next_time, stop_timer());
        next_time = start_timer();

        #ifdef DEBUG
            int accurate = 0;
        #endif

        for(int j = 0; j < io->image_len; j++) {
            #ifdef NVIDIA
                copy_cuda_matrix(io->image[j], d_img, true);
                #ifdef DEBUG
                    flip_kernels(d_masks, io->masks_len, flipped_masks);
                    for(int m = 0; m < io->masks_len; m++) {
                        copy_cuda_matrix(flipped_masks[m], d_flipped_masks[m], true);
                    }
                    conv2d(d_img, d_flipped_masks, io->masks_len, flipped_c);
                #endif
                conv2d(d_img, d_masks, io->masks_len, c);
                biasing(c, io->masks_len, io->conv_bias, b);
                #ifdef DEBUG
                    hyperbolic_tangent(b, io->masks_len, hyperbolic_r);
                #endif
                relu(b, io->masks_len, r);
                maxpool(r, io->masks_len, m);
                flatten(m, io->masks_len, f);
                transpose(f, transposed_f);
                matmul(transposed_f, d_fc_weights, mm);
                add(mm, transposed_fc_bias, a);
                max_val = index_of_max_element(a);
            #else
                #ifdef DEBUG
                    flip_kernels(io->masks, io->masks_len, flipped_masks);
                    conv2d(io->image[j], flipped_masks, io->masks_len, flipped_c);
                #endif
                conv2d(io->image[j], io->masks, io->masks_len, c);
                biasing(c, io->masks_len, io->conv_bias, b);
                #ifdef DEBUG
                    hyperbolic_tangent(b, io->masks_len, hyperbolic_r);
                #endif
                relu(b, io->masks_len, r);
                maxpool(r, io->masks_len, m);
                flatten(m, io->masks_len, f);
                transpose(f, transposed_f);
                matmul(transposed_f, io->fc_weights, mm);
                add(mm, io->fc_bias, a);
                max_val = index_of_max_element(a);
            #endif

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

        #ifdef NVIDIA
            free_cuda_matrix(d_fc_weights);

            free_matrix(transposed_fc_weights);
            free_matrix(transposed_fc_bias);

            free_cuda_matrix_ptr(d_masks, io->masks_len);
            #ifdef DEBUG
                free_cuda_matrix_ptr(d_flipped_masks, io->masks_len);
            #endif
            free_cuda_matrix(d_img);
        #endif

        free_io(io);

        free_time_us = delta_time_us(next_time, stop_timer());
        next_time = start_timer();

        #if !defined(NVIDIA) && !defined(OMP)
            join_mt();

            join_mt_time_us = delta_time_us(next_time, stop_timer());
            next_time = start_timer();
        #endif

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
