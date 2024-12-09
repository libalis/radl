#ifdef OMP
    #include <math.h>
    #include <omp.h>
#endif

#include "../hpp/mt.hpp"
#include "../hpp/tf.hpp"

#ifdef OMP
    #include "../hpp/utils.hpp"
#endif

long THREADS = 1;

matrix *add(matrix *a, matrix *b, matrix *c) {
    if(c == NULL) {
        c = malloc_matrix(a->x, a->y);
    }
    #ifdef OMP
        #pragma omp parallel for
        for(int i = 0; i < c->x; i++) {
            #pragma omp parallel for simd
            for(int j = 0; j < c->y; j++) {
                c->m[get_idx(i, j, c->y)] = a->m[get_idx(i, j, a->y)] + b->m[get_idx(i, j, b->y)];
            }
        }
    #else
        mt_arg arg[THREADS];
        for(int i = 0; i < THREADS; i++) {
            arg[i].a = a;
            arg[i].b = b;
            arg[i].c = c;
            arg[i].start_routine = add_mt;
            push_mt(&arg[i]);
        }
        wait_mt();
    #endif
    return c;
}

matrix **biasing(matrix **a, int len, matrix *b, matrix **c) {
    if(c == NULL) {
        c = malloc_matrix_ptr(len, a[0]->x, a[0]->y);
    }
    #ifdef OMP
        #pragma omp parallel for collapse(2)
        for(int m = 0; m < len; m++) {
            for(int i = 0; i < a[0]->x; i++) {
                #pragma omp parallel for simd
                for(int j = 0; j < a[0]->y; j++) {
                    c[m]->m[get_idx(i, j, c[0]->y)] = a[m]->m[get_idx(i, j, a[0]->y)] + b->m[get_idx(m, 0, b->y)];
                }
            }
        }
    #else
        mt_arg arg[THREADS];
        for(int m = 0; m < len; m++) {
            for(int i = 0; i < THREADS; i++) {
                arg[i].a_ptr = a;
                arg[i].len = len;
                arg[i].b = b;
                arg[i].c_ptr = c;
                arg[i].m = m;
                arg[i].start_routine = biasing_mt;
                push_mt(&arg[i]);
            }
            wait_mt();
        }
    #endif
    return c;
}

matrix **conv2d(matrix *a, matrix **b, int len, matrix **c) {
    if(c == NULL) {
        c = malloc_matrix_ptr(len, a->x - b[0]->x + 1, a->y - b[0]->y + 1);
    }
    #ifdef OMP
        #pragma omp parallel for collapse(3)
        for(int m = 0; m < len; m++) {
            for(int i = 0; i < a->x - b[0]->x + 1; i++) {
                for(int j = 0; j < a->y - b[0]->y + 1; j++) {
                    float sum = 0.0;
                    #pragma omp simd
                    for(int k = 0; k < b[0]->x; k++) {
                        for(int l = 0; l < b[0]->y; l++) {
                            sum += a->m[get_idx(i + k, j + l, a->y)] * b[m]->m[get_idx(k, l, b[0]->y)];
                        }
                    }
                    c[m]->m[get_idx(i, j, c[0]->y)] = sum;
                }
            }
        }
    #else
        mt_arg arg[THREADS];
        for(int m = 0; m < len; m++) {
            for(int i = 0; i < THREADS; i++) {
                arg[i].a = a;
                arg[i].b_ptr = b;
                arg[i].len = len;
                arg[i].c_ptr = c;
                arg[i].m = m;
                arg[i].start_routine = conv2d_mt;
                push_mt(&arg[i]);
            }
            wait_mt();
        }
    #endif
    return c;
}

matrix *flatten(matrix **a, int len, matrix *c) {
    if(c == NULL) {
        c = malloc_matrix(len * a[0]->x * a[0]->y, 1);
    }
    #ifdef OMP
        #pragma omp parallel for collapse(3)
        for(int i = 0; i < a[0]->x; i++) {
            for(int j = 0; j < a[0]->y; j++) {
                for(int m = 0; m < len; m++) {
                    int idx = i * a[0]->y * len + j * len + m;
                    c->m[get_idx(idx, 0, c->y)] = a[m]->m[get_idx(i, j, a[0]->y)];
                }
            }
        }
    #else
        mt_arg arg[THREADS];
        for(int i = 0; i < THREADS; i++) {
            arg[i].a_ptr = a;
            arg[i].len = len;
            arg[i].c = c;
            arg[i].start_routine = flatten_mt;
            push_mt(&arg[i]);
        }
        wait_mt();
    #endif
    return c;
}

matrix **flip_kernels(matrix **a, int len, matrix **c) {
    if(c == NULL) {
        c = malloc_matrix_ptr(len, a[0]->x, a[0]->y);
    }
    #ifdef OMP
        #pragma omp parallel for collapse(3)
        for(int m = 0; m < len; m++) {
            for (int i = 0; i < a[0]->x; i++) {
                for (int j = 0; j < a[0]->y; j++) {
                    c[m]->m[get_idx(i, j, c[0]->y)] = a[m]->m[get_idx(a[0]->x - i - 1, a[0]->y - j - 1, a[0]->y)];
                }
            }
        }
    #else
        mt_arg arg[THREADS];
        for(int m = 0; m < len; m++) {
            for(int i = 0; i < THREADS; i++) {
                arg[i].a_ptr = a;
                arg[i].len = len;
                arg[i].c_ptr = c;
                arg[i].m = m;
                arg[i].start_routine = flip_kernels_mt;
                push_mt(&arg[i]);
            }
            wait_mt();
        }
    #endif
    return c;
}

matrix **hyperbolic_tangent(matrix **a, int len, matrix **c) {
    if(c == NULL) {
        c = malloc_matrix_ptr(len, a[0]->x, a[0]->y);
    }
    #ifdef OMP
        #pragma omp parallel for collapse(3)
        for(int m = 0; m < len; m++) {
            for(int i = 0; i < a[0]->x; i++) {
                for(int j = 0; j < a[0]->y; j++) {
                    c[m]->m[get_idx(i, j, c[0]->y)] = tanh(a[m]->m[get_idx(i, j, a[0]->y)]);
                }
            }
        }
    #else
        mt_arg arg[THREADS];
        for(int m = 0; m < len; m++) {
            for(int i = 0; i < THREADS; i++) {
                arg[i].a_ptr = a;
                arg[i].len = len;
                arg[i].c_ptr = c;
                arg[i].m = m;
                arg[i].start_routine = hyperbolic_tangent_mt;
                push_mt(&arg[i]);
            }
            wait_mt();
        }
    #endif
    return c;
}

matrix *matmul(matrix *a, matrix *b, matrix *c) {
    if(c == NULL) {
        c = malloc_matrix(a->x, b->y);
    }
    #ifdef OMP
        #pragma omp parallel for collapse(2)
        for(int i = 0; i < c->x; i++) {
            for(int j = 0; j < c->y; j++) {
                c->m[get_idx(i, j, c->y)] = 0.0;
                #pragma omp simd
                for(int k = 0; k < a->y; k++) {
                    c->m[get_idx(i, j, c->y)] += a->m[get_idx(i, k, a->y)] * b->m[get_idx(k, j, b->y)];
                }
            }
        }
    #else
        mt_arg arg[THREADS];
        for(int i = 0; i < THREADS; i++) {
            arg[i].a = a;
            arg[i].b = b;
            arg[i].c = c;
            arg[i].start_routine = matmul_mt;
            push_mt(&arg[i]);
        }
        wait_mt();
    #endif
    return c;
}

matrix **maxpool(matrix **a, int len, matrix **c) {
    if(c == NULL) {
        c = malloc_matrix_ptr(len, a[0]->x / POOL_LEN, a[0]->y / POOL_LEN);
    }
    #ifdef OMP
        #pragma omp parallel for collapse(3)
        for(int m = 0; m < len; m++) {
            for(int i = 0; i < a[0]->x; i += POOL_LEN) {
                for(int j = 0; j < a[0]->y; j += POOL_LEN) {
                    float max_val = a[m]->m[get_idx(i, j, a[0]->y)];
                    for(int k = 0; k < POOL_LEN; k++) {
                        for(int l = 0; l < POOL_LEN; l++) {
                            float curr_val = a[m]->m[get_idx(i + k, j + l, a[0]->y)];
                            if(curr_val > max_val) {
                                max_val = curr_val;
                            }
                        }
                    }
                    c[m]->m[get_idx(i / POOL_LEN, j / POOL_LEN, c[0]->y)] = max_val;
                }
            }
        }
    #else
        mt_arg arg[THREADS];
        for(int m = 0; m < len; m++) {
            for(int i = 0; i < THREADS; i++) {
                arg[i].a_ptr = a;
                arg[i].len = len;
                arg[i].c_ptr = c;
                arg[i].m = m;
                arg[i].start_routine = maxpool_mt;
                push_mt(&arg[i]);
            }
            wait_mt();
        }
    #endif
    return c;
}

matrix **relu(matrix **a, int len, matrix **c) {
    if(c == NULL) {
        c = malloc_matrix_ptr(len, a[0]->x, a[0]->y);
    }
    #ifdef OMP
        #pragma omp parallel for collapse(3)
        for(int m = 0; m < len; m++) {
            for(int i = 0; i < a[0]->x; i++) {
                for(int j = 0; j < a[0]->y; j++) {
                    float val = 0.0;
                    if(a[m]->m[get_idx(i, j, a[0]->y)] > 0.0) {
                        val = a[m]->m[get_idx(i, j, a[0]->y)];
                    }
                    c[m]->m[get_idx(i, j, c[0]->y)] = val;
                }
            }
        }
    #else
        mt_arg arg[THREADS];
        for(int m = 0; m < len; m++) {
            for(int i = 0; i < THREADS; i++) {
                arg[i].a_ptr = a;
                arg[i].len = len;
                arg[i].c_ptr = c;
                arg[i].m = m;
                arg[i].start_routine = relu_mt;
                push_mt(&arg[i]);
            }
            wait_mt();
        }
    #endif
    return c;
}

matrix *transpose(matrix *a, matrix *c) {
    if(c == NULL) {
        c = malloc_matrix(a->y, a->x);
    }
    #ifdef OMP
        #pragma omp parallel for collapse(2)
        for(int i = 0; i < a->x; i++) {
            for(int j = 0; j < a->y; j++) {
                c->m[get_idx(j, i, c->y)] = a->m[get_idx(i, j, a->y)];
            }
        }
    #else
        mt_arg arg[THREADS];
        for(int i = 0; i < THREADS; i++) {
            arg[i].a = a;
            arg[i].c = c;
            arg[i].start_routine = transpose_mt;
            push_mt(&arg[i]);
        }
        wait_mt();
    #endif
    return c;
}
