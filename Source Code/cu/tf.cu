#include <math.h>

#include "../hpp/tf.hpp"
#include "../hpp/utils.hpp"

#define NUM_MASKS (4)
#define MASK_SIZE (3)
__constant__ DATA_TYPE c_masks[NUM_MASKS][MASK_SIZE][MASK_SIZE];

void init_const_memory(matrix **masks) {
    DATA_TYPE h_masks[NUM_MASKS][MASK_SIZE][MASK_SIZE];
    for (int i = 0; i < NUM_MASKS; i++) {
        for (int l = 0; l < MASK_SIZE; l++) {
            for (int k = 0; k < MASK_SIZE; k++) {
                h_masks[i][l][k] = masks[i]->m[l * MASK_SIZE + k];
            }
        }
    }

    cudaMemcpyToSymbol(c_masks, h_masks, sizeof(h_masks));
}

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
        matrix[row * N + col] += bias[row * K + col];
    }
}

matrix *add(matrix *a, matrix *b, matrix *c) {
    int N = a->x;
    int M = a->y;

    int K = b->x;

    long threads_per_block = 32;

    int blocks_x = (N + threads_per_block - 1) / threads_per_block;
    int blocks_y = (M + threads_per_block - 1) / threads_per_block;
    dim3 block_dim(threads_per_block, threads_per_block);
    dim3 grid_dim(blocks_x, blocks_y);

    add_kernel<<<grid_dim, block_dim>>>(a->m, b->m, N, M, K);
    cudaDeviceSynchronize();

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
        c = malloc_cuda_matrix_ptr(len, a[0]->x, a[0]->y);
    }

    int N = a[0]->x;
    int M = a[0]->y;

    long threads_per_block = 32;

    int blocks_x = (N + threads_per_block - 1) / threads_per_block;
    int blocks_y = (M + threads_per_block - 1) / threads_per_block;
    dim3 block_dim(threads_per_block, threads_per_block);
    dim3 grid_dim(blocks_x, blocks_y);

    for(int m = 0; m < len; m++) {
        biasing_kernel<<<grid_dim, block_dim>>>(a[m]->m, c[m]->m, N, M, b->m[get_idx(m, 0, b->y)]);
        cudaDeviceSynchronize();
    }
    return c;
}

// 2D Convolution Kernel
__global__ void conv2d_kernel(DATA_TYPE *matrix, DATA_TYPE *result, int N, int M, int maskIdx, int L, int K) {
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
                    temp += matrix[(row + l) * N + (col + k)] * c_masks[maskIdx][l][k];
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
        c = malloc_cuda_matrix_ptr(len, a->x - b[0]->x + 1, a->y - b[0]->y + 1);
    }
    int N = a->x;
    int M = a->y;

    int L = b[0]->x;
    int K = b[0]->y;

    long threads_per_block = 32;
    // Calculate grid dimensions
    int blocks_x = ((N-L + 1) + threads_per_block - 1) / threads_per_block;
    int blocks_y = ((M-K + 1) + threads_per_block - 1) / threads_per_block;

    // Dimension launch arguments
    dim3 block_dim(threads_per_block, threads_per_block);
    dim3 grid_dim(blocks_x, blocks_y);

    for(int m = 0; m < len; m++) {
        conv2d_kernel<<<grid_dim, block_dim>>>(a->m, c[m]->m, N, M, m, L, K);
        cudaDeviceSynchronize();
    }

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
        c = malloc_cuda_matrix((a->x * a->y), 1);
    }
    long threads_per_block = 32;

    // Calculate grid dimensions
    int blocks_x = (len + threads_per_block - 1) / threads_per_block;
    int blocks_y = (a->y + threads_per_block - 1) / threads_per_block;
    int blocks_z = a->x / len;

    // Define block and grid dimensions
    dim3 block_dim(threads_per_block, threads_per_block);
    // Using z-dimension for slices
    dim3 grid_dim(blocks_x, blocks_y, blocks_z);

    flatten_kernel<<<grid_dim, block_dim>>>(a->m, c->m, a->x, a->y, len);
    cudaDeviceSynchronize();

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
        c = malloc_cuda_matrix_ptr(len, a[0]->x, a[0]->y);
    }

    int N = a[0]->x;
    int M = a[0]->y;

    long threads_per_block = 32;

    // Calculate grid dimensions
    int blocks_x = (N + threads_per_block - 1) / threads_per_block;
    int blocks_y = (M + threads_per_block - 1) / threads_per_block;

    // Dimension launch arguments
    dim3 block_dim(threads_per_block, threads_per_block);
    dim3 grid_dim(blocks_x, blocks_y);
    for(int m = 0; m < len; m++) {
        flip_kernels_kernel<<<grid_dim, block_dim>>>(a[m]->m, c[m]->m, N, M);
        cudaDeviceSynchronize();
    }

    return c;
}

__global__ void hyperbolic_tangent_kernel(DATA_TYPE *matrix, DATA_TYPE *result, int N, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < N && col < M) {
        result[row * N + col] = tanhf(matrix[row * N + col]);
    }
}

matrix **hyperbolic_tangent(matrix **a, int len, matrix **c) {
    if(c == NULL) {
        c = malloc_cuda_matrix_ptr(len, a[0]->x, a[0]->y);
    }
    int N = a[0]->x;
    int M = a[0]->y;

    long threads_per_block = 32;

    int blocks_x = (N + threads_per_block - 1) / threads_per_block;
    int blocks_y = (M + threads_per_block - 1) / threads_per_block;

    dim3 block_dim(threads_per_block, threads_per_block);
    dim3 grid_dim(blocks_x, blocks_y);
    for(int m = 0; m < len; m++) {
        hyperbolic_tangent_kernel<<<grid_dim, block_dim>>>(a[m]->m, c[m]->m, N, M);
        cudaDeviceSynchronize();
    }

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
        c = malloc_cuda_matrix(a->x, b->y);
    }

    int M = a->x;
    int K = a->y;
    int N = b->y;

    long threads_per_block = 32;

    // N / threads_per_block;
    int BLOCKS_X = (N + threads_per_block - 1) / threads_per_block;
    // M / threads_per_block;
    int BLOCKS_Y = (M + threads_per_block - 1) / threads_per_block;

    dim3 block_dim(threads_per_block, threads_per_block);
    dim3 grid_dim(BLOCKS_X, BLOCKS_Y);

    matmul_kernel<<<grid_dim, block_dim>>>(a->m, b->m, c->m, N, M, K);
    cudaDeviceSynchronize();

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
        c = malloc_cuda_matrix(len * (a[0]->x / POOL_LEN), (a[0]->y / POOL_LEN));
    }

    int N = a[0]->x;

    int M = c->x / len;
    int L = c->y;

    long threads_per_block = 32;

    int blocks_x = (M + threads_per_block - 1) / threads_per_block;
    int blocks_y = (L + threads_per_block - 1) / threads_per_block;

    dim3 block_dim(threads_per_block, threads_per_block);
    dim3 grid_dim(blocks_x, blocks_y);
    for(int m = 0; m < len; m++) {
        maxpool_kernel<<<grid_dim, block_dim>>>(a[m]->m, c->m + m * M * L, N, M, L);
        cudaDeviceSynchronize();
    }

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
    int N = a[0]->x;
    int M = a[0]->y;

    long threads_per_block = 32;

    int blocks_x = (N + threads_per_block - 1) / threads_per_block;
    int blocks_y = (M + threads_per_block - 1) / threads_per_block;

    dim3 block_dim(threads_per_block, threads_per_block);
    dim3 grid_dim(blocks_x, blocks_y);
    for(int m = 0; m < len; m++) {
        relu_kernel<<<grid_dim, block_dim>>>(a[m]->m, N, M);
        cudaDeviceSynchronize();
    }

    return a;
}

__global__ void transpose_kernel(DATA_TYPE *matrix, DATA_TYPE *result, int N, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < N && col < M) {
        result[row * M + col] = matrix[col * N + row];
    }
}

matrix *transpose(matrix *a, matrix *c) {
    if(c == NULL) {
        c = malloc_cuda_matrix(a->y, a->x);
    }

    int N = a->y;
    int M = a->x;

    long threads_per_block = 32;

    dim3 block_dim(threads_per_block, threads_per_block);

    // M NOT N
    int blocks_x = (M + threads_per_block - 1) / threads_per_block;
    int blocks_y = (N + threads_per_block - 1) / threads_per_block;
    dim3 grid_dim(blocks_x, blocks_y);

    transpose_kernel<<<grid_dim, block_dim>>>(a->m, c->m, N, M);
    cudaDeviceSynchronize();

    return c;
}
