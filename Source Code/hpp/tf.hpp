#ifndef TF_HPP
    #define TF_HPP

    #include "matrix.hpp"
    #ifndef NVIDIA
        #include "mt.hpp"
    #endif

    #ifdef NVIDIA
        void init_const_memory(matrix** masks);
        matrix *malloc_cuda_matrix(int x, int y);
        matrix **malloc_cuda_matrix_ptr(int len, int x, int y);
        void free_cuda_matrix(matrix *a);
        void free_cuda_matrix_ptr(matrix **a, int len);
        matrix *copy_cuda_matrix(matrix *h_a, matrix *d_a, bool to_device);
        matrix *add(matrix *a, matrix *b, matrix *c);
        matrix **biasing(matrix **a, int len, matrix *b, matrix **c);
        matrix **conv2d(matrix *a, matrix **b, int len, matrix **c);
        matrix *flatten(matrix *a, int len, matrix *c);
        matrix **flip_kernels(matrix **a, int len, matrix **c);
        matrix **hyperbolic_tangent(matrix **a, int len, matrix **c);
        matrix *matmul(matrix *a, matrix *b, matrix *c);
        matrix *maxpool(matrix **a, int len, matrix *c);
        matrix **relu(matrix **a, int len, matrix **c);
        matrix *transpose(matrix *a, matrix *c);
    #else
        matrix *add(matrix *a, matrix *b, matrix *c, mt *instance);
        matrix **biasing(matrix **a, int len, matrix *b, matrix **c, mt *instance);
        matrix **conv2d(matrix *a, matrix **b, int len, matrix **c, mt *instance);
        matrix *flatten(matrix *a, int len, matrix *c, mt *instance);
        matrix **flip_kernels(matrix **a, int len, matrix **c, mt *instance);
        matrix **hyperbolic_tangent(matrix **a, int len, matrix **c, mt *instance);
        matrix *matmul(matrix *a, matrix *b, matrix *c, mt *instance);
        matrix *maxpool(matrix **a, int len, matrix *c, mt *instance);
        matrix **relu(matrix **a, int len, matrix **c, mt *instance);
        matrix *transpose(matrix *a, matrix *c, mt *instance);
    #endif

    __attribute__((always_inline)) inline int index_of_max_element(matrix *a) {
        DATA_TYPE max_val = a->m[get_idx(0, 0, a->y)];
        int idx = 0;
        for(int i = 0; i < a->y; i++) {
            DATA_TYPE curr_val = a->m[get_idx(0, i, a->y)];
            if(curr_val > max_val) {
                max_val = curr_val;
                idx = i;
            }
        }
        return idx;
    }
#endif
