#ifndef SIMD_H
    #define SIMD_H

    #include "../hpp/mt.hpp"

    void add_simd(mt_arg *mt);
    void biasing_simd(mt_arg *mt);
    void conv2d_simd(mt_arg *mt);
    void flatten_simd(mt_arg *mt);
    void flip_kernels_simd(mt_arg *mt);
    void hyperbolic_tangent_simd(mt_arg *mt);
    void matmul_simd(mt_arg *mt);
    void maxpool_simd(mt_arg *mt);
    void relu_simd(mt_arg *mt);
    void transpose_simd(mt_arg *mt);
#endif
