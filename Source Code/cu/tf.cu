#include <math.h>

#include "../hpp/tf.hpp"
#include "../hpp/utils.hpp"

long THREADS = 16;

matrix *malloc_cuda_matrix(int x, int y) {
    matrix *a = (matrix*)malloc(sizeof(matrix));
    a->x = x;
    a->y = y;
    a->m = NULL;

    size_t bytes = x * y * sizeof(DATA_TYPE);
    cudaMalloc(&(a->m), bytes);

    return a;
}

matrix **malloc_cuda_matrix_ptr(int len, int x, int y) {
    matrix **c = (matrix**)malloc(len * sizeof(matrix*));

    for(int i = 0; i < len; i++) {
        c[i] = malloc_cuda_matrix(x, y);
    }
    return c;
}

void free_cuda_matrix(matrix *a) {
    cudaFree(a->m);
    free(a);
}

void free_cuda_matrix_ptr(matrix **a, int len) {
    for(int i = 0; i < len; i++) {
        free_cuda_matrix(a[i]);
    }
    free(a);
}

matrix *copy_cuda_matrix(matrix *h_a, matrix *d_a, bool to_device) {
    if(to_device == true) {
        if(d_a == NULL) {
            d_a = malloc_cuda_matrix(h_a->x, h_a->y);
        }
        size_t bytes = h_a->x * h_a->y * sizeof(DATA_TYPE);
        cudaMemcpy(d_a->m, h_a->m, bytes, cudaMemcpyHostToDevice);
        return d_a;
    } else {
        if(h_a == NULL) {
            h_a = malloc_matrix(d_a->x, d_a->y);
        }
        size_t bytes = d_a->x * d_a->y * sizeof(DATA_TYPE);
        cudaMemcpy(h_a->m, d_a->m, bytes, cudaMemcpyDeviceToHost);
        return h_a;
    }
}

DATA_TYPE *malloc_cuda(int N, int M) {
    size_t bytes = N * M * sizeof(DATA_TYPE);

    DATA_TYPE *d_matrix;
    cudaMalloc(&d_matrix, bytes);

    return d_matrix;
}

void memcpy_cuda(matrix *a, DATA_TYPE *d_matrix, bool to_device) {
    size_t bytes = a->x * a->y * sizeof(DATA_TYPE);
    if(to_device == true) {
        cudaMemcpy(d_matrix, a->m, bytes, cudaMemcpyHostToDevice);
    } else {
        cudaMemcpy(a->m, d_matrix, bytes, cudaMemcpyDeviceToHost);
    }
}

void free_cuda(DATA_TYPE *d_matrix) {
    cudaFree(d_matrix);
}

__global__ void add_kernel(DATA_TYPE *matrix, DATA_TYPE *bias, int N, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < N && col < M) {
        // Add bias to each element
        matrix[row * N + col] = matrix[row * N + col] + bias[row * K + col];
    }
}

matrix *add(matrix *a, matrix *b, matrix *c) {
    if(c == NULL) {
        c = malloc_matrix(a->x, a->y);
    }

    int N = a->x;
    int M = a->y;
    size_t bytes_n = N * M * sizeof(DATA_TYPE);
    DATA_TYPE *d_matrix;
    cudaMalloc(&d_matrix, bytes_n);
    cudaMemcpy(d_matrix, a->m, bytes_n, cudaMemcpyHostToDevice);

    int K = b->x;
    int L = b->y;
    size_t bytes_b = K * L * sizeof(DATA_TYPE);
    DATA_TYPE *d_bias;
    cudaMalloc(&d_bias, bytes_b);
    cudaMemcpy(d_bias, b->m, bytes_b, cudaMemcpyHostToDevice);

    #ifndef XL
        long threads = 32;
    #else
        long threads = 1024;
    #endif

    int blocks_x = (N + threads - 1) / threads;
    int blocks_y = (M + threads - 1) / threads;
    dim3 block_dim(threads, threads);
    dim3 grid_dim(blocks_x, blocks_y);

    add_kernel<<<grid_dim, block_dim>>>(d_matrix, d_bias, N, M, K);
    cudaDeviceSynchronize();
    cudaMemcpy(c->m, d_matrix, bytes_n, cudaMemcpyDeviceToHost);

    cudaFree(d_matrix);
    cudaFree(d_bias);

    return c;
}

__global__ void biasing_kernel(DATA_TYPE *matrix, DATA_TYPE *result, int N, int M, DATA_TYPE bias) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < N && col < M) {
        // Add bias to each element
        result[row * N + col] = matrix[row * N + col] + bias;
    }
}

matrix **biasing(matrix **a, int len, matrix *b, matrix **c) {
    if(c == NULL) {
        c = malloc_matrix_ptr(len, a[0]->x, a[0]->y);
    }

    int N = a[0]->x;
    int M = a[0]->y;
    size_t bytes_n = N * M * sizeof(DATA_TYPE);
    DATA_TYPE *d_matrix;
    DATA_TYPE *d_result;
    cudaMalloc(&d_matrix, bytes_n);
    cudaMalloc(&d_result, bytes_n);

    #ifndef XL
        long threads = 32;
    #else
        long threads = 1024;
    #endif

    int blocks_x = (N + threads - 1) / threads;
    int blocks_y = (M + threads - 1) / threads;
    dim3 block_dim(threads, threads);
    dim3 grid_dim(blocks_x, blocks_y);

    for(int m = 0; m < len; m++) {
        cudaMemcpy(d_matrix, a[m]->m, bytes_n, cudaMemcpyHostToDevice);

        biasing_kernel<<<grid_dim, block_dim>>>(d_matrix, d_result, N, M, b->m[get_idx(m, 0, b->y)]);
        cudaDeviceSynchronize();

        cudaMemcpy(c[m]->m, d_result, bytes_n, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_matrix);
    cudaFree(d_result);

    return c;
}

// 2D Convolution Kernel
__global__ void conv2d_kernel(DATA_TYPE *matrix, DATA_TYPE *result, int N, int M, DATA_TYPE *mask, int L, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    DATA_TYPE temp = 0.0;
    if(row < (N - L + 1) && col < (M - K + 1)) {
        // Iterate over all the rows
        for(int l = 0; l < L; l++) {
            // Go over each column
            for(int k = 0; k < K; k++) {
                // Range check for rows
                if((row + l) < N && (col + k) < M) {
                    temp += matrix[(row + l) * N + (col + k)] * mask[l * L + k];
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

    int L = b[0]->x;
    int K = b[0]->y;
    DATA_TYPE *d_result;
    size_t bytes_k = (N - L + 1) * (M - K + 1) * sizeof(DATA_TYPE);
    cudaMalloc(&d_result, bytes_k);

    #ifndef XL
        long threads = 32;
    #else
        long threads = 1024;
    #endif

    // Calculate grid dimensions
    int blocks_x = ((N-L + 1) + threads - 1) / threads;
    int blocks_y = ((M-K + 1) + threads - 1) / threads;

    // Dimension launch arguments
    dim3 block_dim(threads, threads);
    dim3 grid_dim(blocks_x, blocks_y);

    for(int m = 0; m < len; m++) {

        conv2d_kernel<<<grid_dim, block_dim>>>(a->m, d_result, N, M, b[m]->m, L, K);
        cudaDeviceSynchronize();

        cudaMemcpy(c[m]->m, d_result, bytes_k, cudaMemcpyDeviceToHost);
    }
    cudaFree(d_result);

    return c;
}

__global__ void flatten_kernel(DATA_TYPE *matrix, DATA_TYPE *result, int a_x, int a_y, int len) {
    // Determine global row, column, and slice index
    // Slices handled using grid z-dimension
    int z = blockIdx.z;
    // Row index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // Column index
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check for valid indices
    if(row < a_y && col < len) {
        int idx = z * a_y * len + row * len + col;
        int matrix_idx = row + col * (a_x / len) * a_y + z * a_y;

        result[idx] = matrix[matrix_idx];
    }
}

matrix *flatten(matrix *a, int len, matrix *c) {
    if(c == NULL) {
        // Allocate output matrix
        c = malloc_matrix((a->x * a->y), 1);
    }

    size_t bytes_a = a->x * a->y * sizeof(DATA_TYPE);

    DATA_TYPE *d_matrix, *d_result;
    cudaMalloc(&d_matrix, bytes_a);
    cudaMalloc(&d_result, bytes_a);

    cudaMemcpy(d_matrix, a->m, a->x * a->y * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    #ifndef XL
        long threads = 32;
    #else
        long threads = 1024;
    #endif

    // Calculate grid dimensions
    int blocks_x = (len + threads - 1) / threads;
    int blocks_y = (a->y + threads - 1) / threads;
    int blocks_z = a->x / len;

    // Define block and grid dimensions
    dim3 block_dim(threads, threads);
    // Using z-dimension for slices
    dim3 grid_dim(blocks_x, blocks_y, blocks_z);

    flatten_kernel<<<grid_dim, block_dim>>>(d_matrix, d_result, a->x, a->y, len);
    cudaDeviceSynchronize();

    cudaMemcpy(c->m, d_result, bytes_a, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_matrix);
    cudaFree(d_result);

    return c;
}

__global__ void flip_kernels_kernel(DATA_TYPE *matrix, DATA_TYPE *result, int N, int M) {
    // Thread identifiers
    // Row index in the kernel
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // Column index in the kernel
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check bounds
    if(row < N && col < M) {
        // Compute flipped indices
        // Flip vertically
        int flipped_row = N - 1 - row;
        // Flip horizontally
        int flipped_col = M - 1 - col;

        // Compute 1D indices for input and output arrays
        // Original index
        int input_idx = row * M + col;
        // Flipped index
        int output_idx = flipped_row * M + flipped_col;

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
    size_t bytes_n = N * M * sizeof(DATA_TYPE);

    DATA_TYPE *d_result;
    cudaMalloc(&d_result, bytes_n);

    #ifndef XL
        long threads = 32;
    #else
        long threads = 1024;
    #endif

    // Calculate grid dimensions
    int blocks_x = (N + threads - 1) / threads;
    int blocks_y = (M + threads - 1) / threads;

    // Dimension launch arguments
    dim3 block_dim(threads, threads);
    dim3 grid_dim(blocks_x, blocks_y);

    for(int m = 0; m < len; m++) {
        flip_kernels_kernel<<<grid_dim, block_dim>>>(a[m]->m, d_result, N, M);
        cudaDeviceSynchronize();

        cudaMemcpy(c[m]->m, d_result, bytes_n, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_result);

    return c;
}

__global__ void hyperbolic_tangent_kernel(DATA_TYPE *matrix, int N, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < N && col < M) {
        matrix[row * N + col] = tanh(matrix[row * N + col]);
    }
}

matrix **hyperbolic_tangent(matrix **a, int len, matrix **c) {
    if(c == NULL) {
        c = malloc_matrix_ptr(len, a[0]->x, a[0]->y);
    }
    int N = a[0]->x;
    int M = a[0]->y;
    size_t bytes_n = N * M * sizeof(DATA_TYPE);
    DATA_TYPE *d_matrix;
    cudaMalloc(&d_matrix, bytes_n);

    #ifndef XL
        long threads = 32;
    #else
        long threads = 1024;
    #endif

    int blocks_x = (N + threads - 1) / threads;
    int blocks_y = (M + threads - 1) / threads;

    dim3 block_dim(threads, threads);
    dim3 grid_dim(blocks_x, blocks_y);

    for(int m = 0; m < len; m++) {
        cudaMemcpy(d_matrix, a[m]->m, bytes_n, cudaMemcpyHostToDevice);

        hyperbolic_tangent_kernel<<<grid_dim, block_dim>>>(d_matrix, N, M);
        cudaDeviceSynchronize();

        cudaMemcpy(c[m]->m, d_matrix, bytes_n, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_matrix);

    return c;
}


__global__ void matmul_kernel(const DATA_TYPE *a, const DATA_TYPE *b, DATA_TYPE *c, int N, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N) {
        DATA_TYPE cValue = 0.0;
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

    size_t bytes_a = M * K * sizeof(DATA_TYPE);
    size_t bytes_c = M * N * sizeof(DATA_TYPE);

    DATA_TYPE *d_a;
    DATA_TYPE *d_c;
    cudaMalloc(&d_a, bytes_a);
    cudaMalloc(&d_c, bytes_c);

    cudaMemcpy(d_a, a->m, bytes_a, cudaMemcpyHostToDevice);

    #ifndef XL
        long threads = 32;
    #else
        long threads = 1024;
    #endif

    // N / threads;
    int BLOCKS_X = (N + threads - 1) / threads;
    // M / threads;
    int BLOCKS_Y = (M + threads - 1) / threads;

    dim3 block_dim(threads, threads);
    dim3 grid_dim(BLOCKS_X, BLOCKS_Y);

    matmul_kernel<<<grid_dim, block_dim>>>(d_a, b->m, d_c, N, M, K);
    cudaDeviceSynchronize();

    cudaMemcpy(c->m, d_c, bytes_c, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    //cudaFree(d_b);
    cudaFree(d_c);

    return c;
}


__global__ void maxpool_kernel(DATA_TYPE *matrix, DATA_TYPE *result, int N, int M, int L) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Iterate over all the rows
    if(row < M && col < L) {
        int start_row = row * POOL_LEN;
        int start_col = col * POOL_LEN;

        DATA_TYPE max_val = matrix[start_row * N + start_col];

        for(int i = 0; i < POOL_LEN; i++) {
            for(int j = 0; j < POOL_LEN; j++) {
                int cur_row = start_row + i;
                int cur_col = start_col + j;
                DATA_TYPE curr_val = matrix[cur_row * N + cur_col];

                if(curr_val > max_val) {
                    max_val = curr_val;
                }
            }
        }

        result[row * M + col] = max_val;
    }
}

matrix *maxpool(matrix **a, int len, matrix *c) {
    if(c == NULL) {
        c = malloc_matrix(len * (a[0]->x / POOL_LEN), (a[0]->y / POOL_LEN));
    }

    int N = a[0]->x;
    int K = a[0]->y;
    size_t bytes_n = N * K * sizeof(DATA_TYPE);
    DATA_TYPE *d_matrix;
    cudaMalloc(&d_matrix, bytes_n);

    int M = c->x / len;
    int L = c->y;
    size_t bytes_m = M * L * sizeof(DATA_TYPE);
    DATA_TYPE *d_result;
    cudaMalloc(&d_result, bytes_m);

    #ifndef XL
        long threads = 32;
    #else
        long threads = 1024;
    #endif

    int blocks_x = (M + threads - 1) / threads;
    int blocks_y = (L + threads - 1) / threads;

    dim3 block_dim(threads, threads);
    dim3 grid_dim(blocks_x, blocks_y);

    for(int m = 0; m < len; m++) {
        cudaMemcpy(d_matrix, a[m]->m, bytes_n, cudaMemcpyHostToDevice);

        maxpool_kernel<<<grid_dim, block_dim>>>(d_matrix, d_result, N, M, L);
        cudaDeviceSynchronize();

        cudaMemcpy(c->m + m * M * L, d_result, bytes_m, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_matrix);
    cudaFree(d_result);

    return c;
}

__global__ void relu_kernel(DATA_TYPE *matrix, int N, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < N && col < M) {
        if(matrix[row * N + col] < 0 ){
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
    size_t bytes_n = N * M * sizeof(DATA_TYPE);
    DATA_TYPE *d_matrix;
    cudaMalloc(&d_matrix, bytes_n);

    #ifndef XL
        long threads = 32;
    #else
        long threads = 1024;
    #endif

    int blocks_x = (N + threads - 1) / threads;
    int blocks_y = (M + threads - 1) / threads;

    dim3 block_dim(threads, threads);
    dim3 grid_dim(blocks_x, blocks_y);

    for(int m = 0; m < len; m++) {
        cudaMemcpy(d_matrix, a[m]->m, bytes_n, cudaMemcpyHostToDevice);

        relu_kernel<<<grid_dim, block_dim>>>(d_matrix, N, M);
        cudaDeviceSynchronize();

        cudaMemcpy(c[m]->m, d_matrix, bytes_n, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_matrix);

    return c;
}

__global__ void transpose_kernel(DATA_TYPE *matrix, DATA_TYPE *result, int N, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < N && col < M) {
        result[col * N + row] = matrix[row * M + col];
    }
}

matrix *transpose(matrix *a, matrix *c) {
    if(c == NULL) {
        // Allocate transposed matrix
        c = malloc_matrix(a->y, a->x);
    }

    int N = a->y;
    int M = a->x;
    size_t bytes_n = N * M * sizeof(DATA_TYPE);
    DATA_TYPE *d_matrix;
    DATA_TYPE *d_result;

    cudaMalloc(&d_matrix, bytes_n);
    cudaMalloc(&d_result, bytes_n);

    cudaMemcpy(d_matrix, a->m, bytes_n, cudaMemcpyHostToDevice);

    #ifndef XL
        long threads = 32;
    #else
        long threads = 1024;
    #endif

    dim3 block_dim(threads, threads);

    // M NOT N
    int blocks_x = (M + threads - 1) / threads;
    int blocks_y = (N + threads - 1) / threads;
    dim3 grid_dim(blocks_x, blocks_y);

    transpose_kernel<<<grid_dim, block_dim>>>(d_matrix, d_result, N, M);
    cudaDeviceSynchronize();

    cudaMemcpy(c->m, d_result, bytes_n, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_matrix);
    cudaFree(d_result);

    return c;
}
