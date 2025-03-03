#ifndef SIMD_H
    #define SIMD_H

    typedef struct simd_arg {
        int i;
    } simd_arg;

    void add_simd(simd_arg *arg);
    void biasing_simd(simd_arg *arg);
    void conv2d_simd(simd_arg *arg);
    void flatten_simd(simd_arg *arg);
    void flip_kernels_simd(simd_arg *arg);
    void hyperbolic_tangent_simd(simd_arg *arg);
    void matmul_simd(simd_arg *arg);
    void maxpool_simd(simd_arg *arg);
    void relu_simd(simd_arg *arg);
    void transpose_simd(simd_arg *arg);
#endif
