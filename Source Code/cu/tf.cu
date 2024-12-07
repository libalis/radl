#include <math.h>

#include "../hpp/tf.hpp"
#include "../hpp/utils.hpp"

long THREADS = 16;

float* malloc_cuda(int N, int M) {
    size_t bytes = N * M * sizeof(float);

    float* d_matrix;
    cudaMalloc(&d_matrix, bytes);

    return d_matrix;
}

void memcpy_cuda(matrix* a, float* d_matrix, bool to_device) {
    size_t bytes = a->x * a->y * sizeof(float);
    if(to_device == true) {
        cudaMemcpy(d_matrix, a->m, bytes, cudaMemcpyHostToDevice);
    } else {
        cudaMemcpy(a->m, d_matrix, bytes, cudaMemcpyDeviceToHost);
    }
}

void free_cuda(float* d_matrix) {
    cudaFree(d_matrix);
}

__global__ void add_kernel(float* matrix, float* bias, int N, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        matrix[row * N + col] = matrix[row * N + col] + bias[row * K + col]; // Add bias to each element
    }
}

matrix *add(matrix *a, matrix *b, matrix *c) {
    if(c == NULL) {
        c = malloc_matrix(a->x, a->y);
    }

    int N = a->x;
    int M = a->y;
    size_t bytes_n = N * M * sizeof(float);
    float* d_matrix;
    cudaMalloc(&d_matrix, bytes_n);
    cudaMemcpy(d_matrix, a->m, bytes_n, cudaMemcpyHostToDevice);

    int K = b->x;
    int L = b->y;
    size_t bytes_b = K * L * sizeof(float);
    float* d_bias;
    cudaMalloc(&d_bias, bytes_b);
    cudaMemcpy(d_bias, b->m, bytes_b, cudaMemcpyHostToDevice);

    int blocks_x = (N + THREADS - 1) / THREADS;
    int blocks_y = (M + THREADS - 1) / THREADS;
    dim3 block_dim(THREADS, THREADS);
    dim3 grid_dim(blocks_x, blocks_y);

    add_kernel<<<grid_dim, block_dim>>>(d_matrix, d_bias, N, M, K);
    cudaDeviceSynchronize();
    cudaMemcpy(c->m, d_matrix, bytes_n, cudaMemcpyDeviceToHost);

    cudaFree(d_matrix);
    cudaFree(d_bias);

    return c;
}

__global__ void biasing_kernel(float* matrix, float* result, int N, int M, float bias) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        result[row * N + col] = matrix[row * N + col] + bias; // Add bias to each element
    }
}

matrix **biasing(matrix **a, int len, matrix *b, matrix **c) {
    if(c == NULL) {
        c = malloc_matrix_ptr(len, a[0]->x, a[0]->y);
    }

    int N = a[0]->x;
    int M = a[0]->y;
    size_t bytes_n = N * M * sizeof(float);
    float* d_matrix;
    float* d_result;
    cudaMalloc(&d_matrix, bytes_n);
    cudaMalloc(&d_result, bytes_n);

    int blocks_x = (N + THREADS - 1) / THREADS;
    int blocks_y = (M + THREADS - 1) / THREADS;
    dim3 block_dim(THREADS, THREADS);
    dim3 grid_dim(blocks_x, blocks_y);

    for(int m = 0; m < len; m++) {
        cudaMemcpy(d_matrix, a[m]->m, bytes_n, cudaMemcpyHostToDevice);
        //
        biasing_kernel<<<grid_dim, block_dim>>>(d_matrix, d_result, N, M, b->m[get_idx(m, 0, b->y)]);
        cudaDeviceSynchronize();
        //
        cudaMemcpy(c[m]->m, d_result, bytes_n, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_matrix);
    cudaFree(d_result);

    return c;
}

// 2D Convolution Kernel
__global__ void conv2d_kernel(float* matrix, float* result, int N, int M, float* mask, int L, int K) {
    // Calculate the global thread positions
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Starting index for calculation
    int start_r = row;
    int start_c = col;

    // Temp value for accumulating the result
    float temp = 0.0;

    if(row < (N - L + 1) && col < (M - K + 1)) {
        // Iterate over all the rows
        for(int i = 0; i < L; i++) {
            // Go over each column
            for(int j = 0; j < K; j++) {
                    // Range check for rows
                if((start_r + i) >= 0 && (start_r + i) < N && (start_c + j) >= 0 && (start_c + j) < M) {
                    temp += matrix[(start_r + i) * N + (start_c + j)] * mask[i * L + j];
                }

            }
        }

        // Write back the result
        if(row < (N - L + 1) && col < (M - K + 1)) {
            result[row * (N - L + 1) + col] = temp;
        }
    }
}

matrix **conv2d(matrix *a, matrix **b, int len, matrix **c) {
    if(c == NULL) {
        c = malloc_matrix_ptr(len, a->x - b[0]->x + 1, a->y - b[0]->y + 1);
    }

    int N = a->x;
    int M = a->y;
    size_t bytes_n = N * M * sizeof(float);
    float* d_matrix;
    cudaMalloc(&d_matrix, bytes_n);
    cudaMemcpy(d_matrix, a->m, bytes_n, cudaMemcpyHostToDevice);

    int L = b[0]->x;
    int K = b[0]->y;
    float* d_result;
    size_t bytes_k = (N - L + 1) * (M - K + 1) * sizeof(float);
    cudaMalloc(&d_result, bytes_k);

    size_t bytes_m = L * K * sizeof(float);
    float* d_mask;
    cudaMalloc(&d_mask, bytes_m);

    // Calculate grid dimensions
    int blocks_x = ((N-L + 1) + THREADS - 1) / THREADS;
    int blocks_y = ((M-K + 1) + THREADS - 1) / THREADS;

    // Dimension launch arguments
    dim3 block_dim(THREADS, THREADS);
    dim3 grid_dim(blocks_x, blocks_y);

    for(int m = 0; m < len; m++) {
        cudaMemcpy(d_mask, b[m]->m, bytes_m, cudaMemcpyHostToDevice);

        conv2d_kernel<<<grid_dim, block_dim>>>(d_matrix, d_result, N, M, d_mask, L, K);

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

__global__ void flip_kernels_kernel(float* matrix, float* result, int N, int M) {
    // Thread identifiers
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index in the kernel
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index in the kernel

    // Check bounds
    if (row < N && col < M) {
        // Compute flipped indices
        int flipped_row = N - 1 - row; // Flip vertically
        int flipped_col = M - 1 - col; // Flip horizontally

        // Compute 1D indices for input and output arrays
        int input_idx = row * M + col;                     // Original index
        int output_idx = flipped_row * M + flipped_col;    // Flipped index

        // Write flipped value to the result array
        result[output_idx] = matrix[input_idx];
    }
}

matrix **flip_kernels(matrix **a, int len, matrix **c) {
    if(c == NULL) {
        c = malloc_matrix_ptr(len, a[0]->x, a[0]->y);
    }

    int N = a[0]->x;
    int M = a[0]->y;
    size_t bytes_n = N * M * sizeof(float);
    float* d_matrix;
    float* d_result;
    cudaMalloc(&d_matrix, bytes_n);
    cudaMalloc(&d_result, bytes_n);

    // Calculate grid dimensions
    int blocks_x = (N + THREADS - 1) / THREADS;
    int blocks_y = (M + THREADS - 1) / THREADS;

    // Dimension launch arguments
    dim3 block_dim(THREADS, THREADS);
    dim3 grid_dim(blocks_x, blocks_y);

    for(int m = 0; m < len; m++) {
        cudaMemcpy(d_matrix, a[m]->m, bytes_n, cudaMemcpyHostToDevice);
        flip_kernels_kernel<<<grid_dim, block_dim>>>(d_matrix, d_result, N, M);
        cudaDeviceSynchronize();
        cudaMemcpy(c[m]->m, d_result, bytes_n, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_matrix);
    cudaFree(d_result);

    return c;
}

__global__ void hyperbolic_tangent_kernel(float* matrix, int N, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        matrix[row * N + col] = tanh(matrix[row * N + col]);
    }
}

matrix **hyperbolic_tangent(matrix **a, int len, matrix **c) {
    if(c == NULL) {
        c = malloc_matrix_ptr(len, a[0]->x, a[0]->y);
    }
    int N = a[0]->x;
    int M = a[0]->y;
    size_t bytes_n = N * M * sizeof(float);
    float* d_matrix;
    cudaMalloc(&d_matrix, bytes_n);

    int blocks_x = (N + THREADS - 1) / THREADS;
    int blocks_y = (M + THREADS - 1) / THREADS;

    dim3 block_dim(THREADS, THREADS);
    dim3 grid_dim(blocks_x, blocks_y);

    for(int m = 0; m < len; m++) {
        cudaMemcpy(d_matrix, a[m]->m, bytes_n, cudaMemcpyHostToDevice);
        //
        hyperbolic_tangent_kernel<<<grid_dim, block_dim>>>(d_matrix, N, M);
        cudaDeviceSynchronize();
        //
        cudaMemcpy(c[m]->m, d_matrix, bytes_n, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_matrix);

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


__global__ void maxpool_kernel(float* matrix, float* result, int N, int M, int L) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < L) {
        // Iterate over all the rows

        int start_row = row * POOL_LEN;
        int start_col = col * POOL_LEN;

        float max_val = matrix[start_row * N + start_col];

        for(int i = 0; i < POOL_LEN; i++) {
            for(int j = 0; j < POOL_LEN; j++) {
                int cur_row = start_row + i;
                int cur_col = start_col + j;
                float curr_val = matrix[cur_row * N + cur_col];

                if(curr_val > max_val) {
                    max_val = curr_val;
                }
            }
        }

        result[row * M + col] = max_val;
    }
}

matrix **maxpool(matrix **a, int len, matrix **c) {
    if(c == NULL) {
        c = malloc_matrix_ptr(len, a[0]->x / POOL_LEN, a[0]->y / POOL_LEN);
    }

    int N = a[0]->x;
    int K = a[0]->y;
    size_t bytes_n = N * K * sizeof(float);
    float* d_matrix;
    cudaMalloc(&d_matrix, bytes_n);
    //d_matrix = malloc_cuda(N, N);

    int M = c[0]->x;
    int L = c[0]->y;
    size_t bytes_m = M * L * sizeof(float);
    float* d_result;
    cudaMalloc(&d_result, bytes_m);

    int blocks_x = (M + THREADS - 1) / THREADS;
    int blocks_y = (L + THREADS - 1) / THREADS;

    dim3 block_dim(THREADS, THREADS);
    dim3 grid_dim(blocks_x, blocks_y);

    for (int m = 0; m < len; m++) {
        cudaMemcpy(d_matrix, a[m]->m, bytes_n, cudaMemcpyHostToDevice);
        //memcpy_cuda(a[m], d_matrix, true);
        //
        maxpool_kernel<<<grid_dim, block_dim>>>(d_matrix, d_result, N, M, L);
        cudaDeviceSynchronize();
        //
        cudaMemcpy(c[m]->m, d_result, bytes_m, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_matrix);
    cudaFree(d_result);

    return c;
}

__global__ void relu_kernel(float* matrix, int N, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        if (matrix[row * N + col] < 0 ){
            matrix[row * N + col] = 0;
        }
    }
}

matrix **relu(matrix **a, int len, matrix **c) {
    if(c == NULL) {
        c = malloc_matrix_ptr(len, a[0]->x, a[0]->y);
    }

    int N = a[0]->x;
    int M = a[0]->y;
    size_t bytes_n = N * M * sizeof(float);
    float* d_matrix;
    cudaMalloc(&d_matrix, bytes_n);

    int blocks_x = (N + THREADS - 1) / THREADS;
    int blocks_y = (M + THREADS - 1) / THREADS;

    dim3 block_dim(THREADS, THREADS);
    dim3 grid_dim(blocks_x, blocks_y);

    for(int m = 0; m < len; m++) {
        cudaMemcpy(d_matrix, a[m]->m, bytes_n, cudaMemcpyHostToDevice);
        //
        relu_kernel<<<grid_dim, block_dim>>>(d_matrix, N, M);
        cudaDeviceSynchronize();
        //
        cudaMemcpy(c[m]->m, d_matrix, bytes_n, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_matrix);

    return c;
}

__global__ void transpose_kernel(float* matrix, float* result, int N, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        result[col * N + row] = matrix[row * M + col];
    }
}

matrix *transpose(matrix *a, matrix *c) {
    if (c == NULL) {
        c = malloc_matrix(a->y, a->x);  // Allocate transposed matrix
    }

    int N = a->y;
    int M = a->x;
    size_t bytes_n = N * M * sizeof(float);
    float* d_matrix;
    float* d_result;

    cudaMalloc(&d_matrix, bytes_n);
    cudaMalloc(&d_result, bytes_n);

    cudaMemcpy(d_matrix, a->m, bytes_n, cudaMemcpyHostToDevice);

    dim3 block_dim(THREADS, THREADS);

    int blocks_x = (M + THREADS - 1) / THREADS; // M NOT N
    int blocks_y = (N + THREADS - 1) / THREADS;
    dim3 grid_dim(blocks_x, blocks_y);

    transpose_kernel<<<grid_dim, block_dim>>>(d_matrix, d_result, N, M);
    cudaDeviceSynchronize();

    cudaMemcpy(c->m, d_result, bytes_n, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_matrix);
    cudaFree(d_result);

    return c;
}
