#ifndef IMG_HPP
    #define IMG_HPP

    #include "tf.hpp"

    #ifndef NVIDIA
        __attribute__((always_inline)) inline void process_images(void *instance, mt_arg *arg) {
            mt *parent_instance = (mt*)instance;
            mt child_instance(parent_instance->THREADS);

            #ifdef DEBUG
                matrix **flipped_masks = malloc_matrix_ptr(arg->io->masks_len, arg->io->masks[0]->x, arg->io->masks[0]->y);
                matrix **flipped_c = malloc_matrix_ptr(arg->io->masks_len, arg->io->image[0]->x - flipped_masks[0]->x + 1, arg->io->image[0]->y - flipped_masks[0]->y + 1);
            #endif
            matrix **c = malloc_matrix_ptr(arg->io->masks_len, arg->io->image[0]->x - arg->io->masks[0]->x + 1, arg->io->image[0]->y - arg->io->masks[0]->y + 1);
            matrix **b = malloc_matrix_ptr(arg->io->masks_len, c[0]->x, c[0]->y);
            #ifdef DEBUG
                matrix **hyperbolic_r = malloc_matrix_ptr(arg->io->masks_len, b[0]->x, b[0]->y);
            #endif
            matrix **r = malloc_matrix_ptr(arg->io->masks_len, b[0]->x, b[0]->y);
            matrix *m = malloc_matrix(arg->io->masks_len * (r[0]->x / POOL_LEN), (r[0]->y / POOL_LEN));
            matrix *f = malloc_matrix(1, m->x * m->y);
            matrix *mm = malloc_matrix(f->x, arg->io->fc_weights->x);
            matrix *a = malloc_matrix(mm->x, mm->y);
            int max_val = 0;

            for(int i = arg->idx * (arg->io->image_len / parent_instance->THREADS); i < ((arg->idx + 1) * (arg->io->image_len / parent_instance->THREADS)) + (arg->idx == parent_instance->THREADS - 1 ? arg->io->image_len % parent_instance->THREADS : 0); i++) {
                #ifdef DEBUG
                    flip_kernels(arg->io->masks, arg->io->masks_len, flipped_masks, &child_instance);
                    conv2d(arg->io->image[i], flipped_masks, arg->io->masks_len, flipped_c, &child_instance);
                #endif
                conv2d(arg->io->image[i], arg->io->masks, arg->io->masks_len, c, &child_instance);
                biasing(c, arg->io->masks_len, arg->io->conv_bias, b, &child_instance);
                #ifdef DEBUG
                    hyperbolic_tangent(b, arg->io->masks_len, hyperbolic_r, &child_instance);
                #endif
                relu(b, arg->io->masks_len, r, &child_instance);
                maxpool(r, arg->io->masks_len, m, &child_instance);
                flatten(m, arg->io->masks_len, f, &child_instance);
                matmul(f, arg->io->fc_weights, mm, &child_instance);
                add(mm, arg->io->fc_bias, a, &child_instance);
                max_val = index_of_max_element(a);
                #ifdef DEBUG
                    if (max_val == arg->io->label[i]) {
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
            free_matrix_ptr(r, arg->io->masks_len);
            #ifdef DEBUG
                free_matrix_ptr(hyperbolic_r, arg->io->masks_len);
            #endif
            free_matrix_ptr(b, arg->io->masks_len);
            free_matrix_ptr(c, arg->io->masks_len);
            #ifdef DEBUG
                free_matrix_ptr(flipped_c, arg->io->masks_len);
                free_matrix_ptr(flipped_masks, arg->io->masks_len);
            #endif

            parent_instance->wait_mt();
        }
    #endif
#endif
