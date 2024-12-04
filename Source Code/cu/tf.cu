#include <math.h>

#include "../hpp/tf.hpp"
#include "../hpp/utils.hpp"

long THREADS = 16;

matrix *add(matrix *a, matrix *b, matrix *c) {
    if(c == NULL) {
        c = malloc_matrix(a->x, a->y);
    }

    for(int i = 0; i < c->x; i++) {
        for(int j = 0; j < c->y; j++) {
            c->m[get_idx(i, j, c->y)] = a->m[get_idx(i, j, a->y)] + b->m[get_idx(i, j, b->y)];
        }
    }

    return c;
}

matrix **biasing(matrix **a, int len, matrix *b, matrix **c) {
    if(c == NULL) {
        c = malloc_matrix_ptr(len, a[0]->x, a[0]->y);
    }

    for(int m = 0; m < len; m++) {
        for(int i = 0; i < a[m]->x; i++) {
            for(int j = 0; j < a[m]->y; j++) {
                c[m]->m[get_idx(i, j, c[m]->y)] = a[m]->m[get_idx(i, j, a[m]->y)] + b->m[get_idx(m, 0, b->y)];
            }
        }
    }

    return c;
}

// 2D Convolution Kernel
__global__ void conv2d_kernel(float* matrix, float* result, int N, float* mask, int M) {
    // Calculate the global thread positions
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Starting index for calculation
    int start_r = row;
    int start_c = col;

    // Temp value for accumulating the result
    float temp = 0.0;

    if(row >= 0 && row < (N) && col >= 0 && col < N) {
        // Iterate over all the rows
        for(int i = 0; i < M; i++) {
            // Go over each column
            for(int j = 0; j < M; j++) {
                    // Range check for rows
                if((start_r + i) >= 0 && (start_r + i) < N && (start_c + j) >= 0 && (start_c + j) < N) {
                    temp += matrix[(start_r + i) * N + (start_c + j)] * mask[i * M + j];
                }

            }
        }

        // Write back the result
        if(row < (N - M + 1) && col < (N - M + 1)) {
            result[row * (N - M + 1) + col] = temp;
        }
    }
}

matrix **conv2d(matrix *a, matrix **b, int len, matrix **c) {
    if(c == NULL) {
        c = malloc_matrix_ptr(len, a->x - b[0]->x + 1, a->y - b[0]->y + 1);
    }

    int N = a->x;
    size_t bytes_n = N * N * sizeof(float);
    float* d_matrix;
    cudaMalloc(&d_matrix, bytes_n);
    cudaMemcpy(d_matrix, a->m, bytes_n, cudaMemcpyHostToDevice);

    int M = b[0]->x;
    float* d_result;
    size_t bytes_k = (N - M + 1) * (N - M + 1) * sizeof(float);
    cudaMalloc(&d_result, bytes_k);

    size_t bytes_m = b[0]->x * b[0]->y * sizeof(float);
    float* d_mask;
    cudaMalloc(&d_mask, bytes_m);

    // Calculate grid dimensions
    int BLOCKS = ((N ) + THREADS - 1) / THREADS;

    // Dimension launch arguments
    dim3 block_dim(THREADS, THREADS);
    dim3 grid_dim(BLOCKS, BLOCKS);

    for(int m = 0; m < len; m++) {
        cudaMemcpy(d_mask, b[m]->m, bytes_m, cudaMemcpyHostToDevice);

        conv2d_kernel<<<grid_dim, block_dim>>>(d_matrix, d_result, N, d_mask, M);

        cudaDeviceSynchronize();

        cudaMemcpy(c[m]->m, d_result, bytes_k, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_matrix);
    cudaFree(d_result);
    cudaFree(d_mask);

    return c;
}

matrix *flatten(matrix **a, int len, matrix *c) {
    if(c == NULL) {
        c = malloc_matrix(len * a[0]->x * a[0]->y, 1);
    }

    for(int i = 0; i < a[0]->x; i++) {
        for(int j = 0; j < a[0]->y; j++) {
            for(int m = 0; m < len; m++) {
                int idx = i * a[0]->y * len + j * len + m;
                c->m[get_idx(idx, 0, c->y)] = a[m]->m[get_idx(i, j, a[m]->y)];
            }
        }
    }

    return c;
}

matrix **flip_kernels(matrix **a, int len, matrix **c) {
    if(c == NULL) {
        c = malloc_matrix_ptr(len, a[0]->x, a[0]->y);
    }

    for(int m = 0; m < len; m++) {
        for(int i = 0; i < a[m]->x; i++) {
            for(int j = 0; j < a[m]->y; j++) {
                c[m]->m[get_idx(i, j, c[m]->y)] = a[m]->m[get_idx(a[m]->x - i - 1, a[m]->y - j - 1, a[m]->y)];
            }
        }
    }

    return c;
}

matrix **hyperbolic_tangent(matrix **a, int len, matrix **c) {
    if(c == NULL) {
        c = malloc_matrix_ptr(len, a[0]->x, a[0]->y);
    }

    for(int m = 0; m < len; m++) {
        for(int i = 0; i < a[m]->x; i++) {
            for(int j = 0; j < a[m]->y; j++) {
                c[m]->m[get_idx(i, j, c[m]->y)] = tanh(a[m]->m[get_idx(i, j, a[m]->y)]);
            }
        }
    }

    return c;
}


__global__ void matmul_kernel(const float* a, const float* b, float* c, int N, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N) {
        float cValue = 0.0;
        for(int k = 0; k < K; k++) {
            cValue += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = cValue;
    }
}

matrix *matmul(matrix *a, matrix *b, matrix *c) {
    if(c == NULL) {
        c = malloc_matrix(a->x, b->y);
    }

    int M = a->x;
    int K = a->y;
    int N = b->y;

    size_t bytes_a = M * K * sizeof(float);
    size_t bytes_b = K * N * sizeof(float);
    size_t bytes_c = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes_a);
    cudaMalloc(&d_b, bytes_b);
    cudaMalloc(&d_c, bytes_c);

    cudaMemcpy(d_a, a->m, bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b->m, bytes_b, cudaMemcpyHostToDevice);

    int BLOCKS_X = (N + THREADS - 1) / THREADS; // N / THREADS;
    int BLOCKS_Y = (M + THREADS - 1) / THREADS; // M / THREADS;

    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS_X, BLOCKS_Y);

    matmul_kernel<<<blocks, threads>>>(d_a, d_b, d_c, N, M, K);

    cudaDeviceSynchronize();

    cudaMemcpy(c->m, d_c, bytes_c, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}

matrix **maxpool(matrix **a, int len, matrix **c) {
    if(c == NULL) {
        c = malloc_matrix_ptr(len, a[0]->x / POOL_LEN, a[0]->y / POOL_LEN);
    }

    for(int m = 0; m < len; m++) {
        for(int i = 0; i < a[m]->x; i += POOL_LEN) {
            for(int j = 0; j < a[m]->y; j += POOL_LEN) {
                float max_val = a[m]->m[get_idx(i, j, a[m]->y)];
                for(int k = 0; k < POOL_LEN; k++) {
                    for(int l = 0; l < POOL_LEN; l++) {
                        float curr_val = a[m]->m[get_idx(i + k, j + l, a[m]->y)];
                        if(curr_val > max_val) {
                            max_val = curr_val;
                        }
                    }
                }
                c[m]->m[get_idx(i / POOL_LEN, j / POOL_LEN, c[m]->y)] = max_val;
            }
        }
    }

    return c;
}

matrix **relu(matrix **a, int len, matrix **c) {
    if(c == NULL) {
        c = malloc_matrix_ptr(len, a[0]->x, a[0]->y);
    }

    for(int m = 0; m < len; m++) {
        for(int i = 0; i < a[m]->x; i++) {
            for(int j = 0; j < a[m]->y; j++) {
                float val = 0.0;
                if(a[m]->m[get_idx(i, j, a[m]->y)] > 0.0) {
                    val = a[m]->m[get_idx(i, j, a[m]->y)];
                }
                c[m]->m[get_idx(i, j, c[m]->y)] = val;
            }
        }
    }

    return c;
}

matrix *transpose(matrix *a, matrix *c) {
    if(c == NULL) {
        c = malloc_matrix(a->y, a->x);
    }

    for(int i = 0; i < a->x; i++) {
        for(int j = 0; j < a->y; j++) {
            c->m[get_idx(j, i, c->y)] = a->m[get_idx(i, j, a->y)];
        }
    }

    return c;
}
