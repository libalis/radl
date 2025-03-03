#ifndef IMG_HPP
    #define IMG_HPP

    #ifdef INT
        #include "io_int8.hpp"
    #else
        #include "io.hpp"
    #endif

    #include "tf.hpp"

    #ifndef NVIDIA
        __attribute__((always_inline)) inline void process_images(void *instance, mt_arg *arg) {
            mt *parent_instance = (mt*)instance;
            mt child_instance(parent_instance->THREADS);

            #ifdef INT
                io_int8 *arg_io = (io_int8*)arg->io;
            #else
                io *arg_io = (io*)arg->io;
            #endif

            matrix **c = malloc_matrix_ptr(arg_io->masks_len, arg_io->image[0]->x - arg_io->masks[0]->x + 1, arg_io->image[0]->y - arg_io->masks[0]->y + 1);
            matrix **b = malloc_matrix_ptr(arg_io->masks_len, c[0]->x, c[0]->y);
            matrix **r = malloc_matrix_ptr(arg_io->masks_len, b[0]->x, b[0]->y);
            matrix *m = malloc_matrix(arg_io->masks_len * (r[0]->x / POOL_LEN), (r[0]->y / POOL_LEN));
            matrix *f = malloc_matrix(1, m->x * m->y);
            matrix *mm = malloc_matrix(f->x, arg_io->fc_weights->x);
            matrix *a = malloc_matrix(mm->x, mm->y);
            int max_val = 0;

            for(int i = arg->idx * (arg_io->image_len / parent_instance->THREADS); i < ((arg->idx + 1) * (arg_io->image_len / parent_instance->THREADS)) + (arg->idx == parent_instance->THREADS - 1 ? arg_io->image_len % parent_instance->THREADS : 0); i++) {
                conv2d((void*)arg_io->image[i], (void**)arg_io->masks, arg_io->masks_len, c, &child_instance);
                biasing((void**)c, arg_io->masks_len, (void*)arg_io->conv_bias, b, &child_instance);
                relu((void**)b, arg_io->masks_len, r, &child_instance);
                maxpool((void**)r, arg_io->masks_len, m, &child_instance);
                flatten((void*)m, arg_io->masks_len, f, &child_instance);
                matmul((void*)f, (void*)arg_io->fc_weights, mm, &child_instance);
                add((void*)mm, (void*)arg_io->fc_bias, a, &child_instance);
                max_val = index_of_max_element(a);
                #ifdef DEBUG
                    if (max_val == arg_io->label[i]) {
                        printf("Epoch %d done: accuracy 100%%\n", i);
                        parent_instance->add_to_accuracy(1);
                    } else {
                        printf("Epoch %d done: accuracy 0%%\n", i);
                    }
                #endif
            }

            free_matrix(a);
            free_matrix(mm);
            free_matrix(f);
            free_matrix(m);
            free_matrix_ptr(r, arg_io->masks_len);
            free_matrix_ptr(b, arg_io->masks_len);
            free_matrix_ptr(c, arg_io->masks_len);

            parent_instance->wait_mt();
        }
    #endif
#endif
