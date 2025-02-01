#ifdef OMP
    #include <math.h>
    #include <omp.h>
#endif

#include "../hpp/mt.hpp"
#include "../hpp/tf.hpp"

#ifdef OMP
    #include "../hpp/utils.hpp"
#endif

matrix *add(matrix *a, matrix *b, matrix *c, mt *instance) {
    if(c == NULL) {
        c = malloc_matrix(a->x, a->y);
    }
    #ifdef OMP
        #pragma omp parallel for collapse(2)
        for(int i = 0; i < c->x; i++) {
            for(int j = 0; j < c->y; j++) {
                c->m[get_idx(i, j, c->y)] = a->m[get_idx(i, j, a->y)] + b->m[get_idx(i, j, b->y)];
            }
        }
    #else
        mt_arg arg[instance->THREADS];
        for(int i = 0; i < instance->THREADS; i++) {
            arg[i].a = &a;
            arg[i].b = &b;
            arg[i].c = &c;
            if(instance->THREADS < c->x) {
                arg[i].single_core = 1;
                instance->add_mt(&arg[i]);
                return c;
            }
            arg[i].single_core = 0;
            arg[i].start_routine = instance->add_mt_wrapper;
            instance->push_mt(&arg[i]);
        }
        instance->wait_mt();
    #endif
    return c;
}

matrix **biasing(matrix **a, int len, matrix *b, matrix **c, mt *instance) {
    if(c == NULL) {
        c = malloc_matrix_ptr(len, a[0]->x, a[0]->y);
    }
    #ifdef OMP
        #pragma omp parallel for collapse(3)
        for(int m = 0; m < len; m++) {
            for(int i = 0; i < a[0]->x; i++) {
                for(int j = 0; j < a[0]->y; j++) {
                    c[m]->m[get_idx(i, j, c[0]->y)] = a[m]->m[get_idx(i, j, a[0]->y)] + b->m[get_idx(m, 0, b->y)];
                }
            }
        }
    #else
        mt_arg arg[instance->THREADS];
        for(int m = 0; m < len; m++) {
            for(int i = 0; i < instance->THREADS; i++) {
                arg[i].a = a;
                arg[i].b = &b;
                arg[i].c = c;
                arg[i].len = len;
                arg[i].m = m;
                if(instance->THREADS < a[0]->x) {
                    arg[i].single_core = 1;
                    instance->biasing_mt(&arg[i]);
                    break;
                }
                arg[i].single_core = 0;
                arg[i].start_routine = instance->biasing_mt_wrapper;
                instance->push_mt(&arg[i]);
            }
            if(instance->THREADS >= a[0]->x) {
                instance->wait_mt();
            }
        }
    #endif
    return c;
}

matrix **conv2d(matrix *a, matrix **b, int len, matrix **c, mt *instance) {
    if(c == NULL) {
        c = malloc_matrix_ptr(len, a->x - b[0]->x + 1, a->y - b[0]->y + 1);
    }
    #ifdef OMP
        #pragma omp parallel for collapse(3)
        for(int m = 0; m < len; m++) {
            for(int i = 0; i < a->x - b[0]->x + 1; i++) {
                for(int j = 0; j < a->y - b[0]->y + 1; j++) {
                    DATA_TYPE sum = 0;
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
        mt_arg arg[instance->THREADS];
        for(int m = 0; m < len; m++) {
            for(int i = 0; i < instance->THREADS; i++) {
                arg[i].a = &a;
                arg[i].b = b;
                arg[i].c = c;
                arg[i].len = len;
                arg[i].m = m;
                if(instance->THREADS < a->x - b[0]->x + 1) {
                    arg[i].single_core = 1;
                    instance->conv2d_mt(&arg[i]);
                    break;
                }
                arg[i].single_core = 0;
                arg[i].start_routine = instance->conv2d_mt_wrapper;
                instance->push_mt(&arg[i]);
            }
            if(instance->THREADS >= a->x - b[0]->x + 1) {
                instance->wait_mt();
            }
        }
    #endif
    return c;
}

matrix *flatten(matrix *a, int len, matrix *c, mt *instance) {
    if(c == NULL) {
        c = malloc_matrix(1, a->x * a->y);
    }
    #ifdef OMP
        #pragma omp parallel for collapse(3)
        for(int i = 0; i < a->x / len; i++) {
            for(int j = 0; j < a->y; j++) {
                for(int m = 0; m < len; m++) {
                    int idx = i * a->y * len + j * len + m;
                    c->m[get_idx(0, idx, c->y)] = a->m[get_idx(i, j, a->y) + m * ((a->x / len) * a->y)];
                }
            }
        }
    #else
        mt_arg arg[instance->THREADS];
        for(int i = 0; i < instance->THREADS; i++) {
            arg[i].a = &a;
            arg[i].c = &c;
            arg[i].len = len;
            if(instance->THREADS < a->x / len) {
                arg[i].single_core = 1;
                instance->flatten_mt(&arg[i]);
                return c;
            }
            arg[i].single_core = 0;
            arg[i].start_routine = instance->flatten_mt_wrapper;
            instance->push_mt(&arg[i]);
        }
        instance->wait_mt();
    #endif
    return c;
}

matrix **flip_kernels(matrix **a, int len, matrix **c, mt *instance) {
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
        mt_arg arg[instance->THREADS];
        for(int m = 0; m < len; m++) {
            for(int i = 0; i < instance->THREADS; i++) {
                arg[i].a = a;
                arg[i].c = c;
                arg[i].len = len;
                arg[i].m = m;
                if(instance->THREADS < a[0]->x) {
                    arg[i].single_core = 1;
                    instance->flip_kernels_mt(&arg[i]);
                    break;
                }
                arg[i].single_core = 0;
                arg[i].start_routine = instance->flip_kernels_mt_wrapper;
                instance->push_mt(&arg[i]);
            }
            if(instance->THREADS >= a[0]->x) {
                instance->wait_mt();
            }
        }
    #endif
    return c;
}

matrix **hyperbolic_tangent(matrix **a, int len, matrix **c, mt *instance) {
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
        mt_arg arg[instance->THREADS];
        for(int m = 0; m < len; m++) {
            for(int i = 0; i < instance->THREADS; i++) {
                arg[i].a = a;
                arg[i].c = c;
                arg[i].len = len;
                arg[i].m = m;
                if(instance->THREADS < a[0]->x) {
                    arg[i].single_core = 1;
                    instance->hyperbolic_tangent_mt(&arg[i]);
                    break;
                }
                arg[i].single_core = 0;
                arg[i].start_routine = instance->hyperbolic_tangent_mt_wrapper;
                instance->push_mt(&arg[i]);
            }
            if(instance->THREADS >= a[0]->x) {
                instance->wait_mt();
            }
        }
    #endif
    return c;
}

matrix *matmul(matrix *a, matrix *b, matrix *c, mt *instance) {
    if(c == NULL) {
        c = malloc_matrix(a->x, b->x);
    }
    #ifdef OMP
        #pragma omp parallel for collapse(2)
        for(int i = 0; i < c->x; i++) {
            for(int j = 0; j < c->y; j++) {
                c->m[get_idx(i, j, c->y)] = 0;
                for(int k = 0; k < a->y; k++) {
                    c->m[get_idx(i, j, c->y)] += a->m[get_idx(i, k, a->y)] * b->m[get_idx(j, k, b->y)];
                }
            }
        }
    #else
        mt_arg arg[instance->THREADS];
        for(int i = 0; i < c->x; i++) {
            for(int j = 0; j < instance->THREADS; j++) {
                arg[j].a = &a;
                arg[j].b = &b;
                arg[j].c = &c;
                arg[j].i = i;
                if(instance->THREADS < c->y) {
                    arg[j].single_core = 1;
                    instance->matmul_mt(&arg[i]);
                    break;
                }
                arg[j].single_core = 0;
                arg[j].start_routine = instance->matmul_mt_wrapper;
                instance->push_mt(&arg[j]);
            }
            if(instance->THREADS >= c->y) {
                instance->wait_mt();
            }
        }
    #endif
    return c;
}

matrix *maxpool(matrix **a, int len, matrix *c, mt *instance) {
    if(c == NULL) {
        c = malloc_matrix(len * (a[0]->x / POOL_LEN), (a[0]->y / POOL_LEN));
    }
    #ifdef OMP
        #pragma omp parallel for collapse(3)
        for(int m = 0; m < len; m++) {
            for(int i = 0; i < a[0]->x; i += POOL_LEN) {
                for(int j = 0; j < a[0]->y; j += POOL_LEN) {
                    DATA_TYPE max_val = a[m]->m[get_idx(i, j, a[0]->y)];
                    for(int k = 0; k < POOL_LEN; k++) {
                        for(int l = 0; l < POOL_LEN; l++) {
                            DATA_TYPE curr_val = a[m]->m[get_idx(i + k, j + l, a[0]->y)];
                            if(curr_val > max_val) {
                                max_val = curr_val;
                            }
                        }
                    }
                    c->m[get_idx(i / POOL_LEN, j / POOL_LEN, c->y) + m * ((c->x / len) * c->y)] = max_val;
                }
            }
        }
    #else
        mt_arg arg[instance->THREADS];
        for(int m = 0; m < len; m++) {
            for(int i = 0; i < instance->THREADS; i++) {
                arg[i].a = a;
                arg[i].c = &c;
                arg[i].len = len;
                arg[i].m = m;
                if(instance->THREADS < a[0]->x) {
                    arg[i].single_core = 1;
                    instance->maxpool_mt(&arg[i]);
                    break;
                }
                arg[i].single_core = 0;
                arg[i].start_routine = instance->maxpool_mt_wrapper;
                instance->push_mt(&arg[i]);
            }
            if(instance->THREADS >= a[0]->x) {
                instance->wait_mt();
            }
        }
    #endif
    return c;
}

matrix **relu(matrix **a, int len, matrix **c, mt *instance) {
    if(c == NULL) {
        c = malloc_matrix_ptr(len, a[0]->x, a[0]->y);
    }
    #ifdef OMP
        #pragma omp parallel for collapse(3)
        for(int m = 0; m < len; m++) {
            for(int i = 0; i < a[0]->x; i++) {
                for(int j = 0; j < a[0]->y; j++) {
                    DATA_TYPE val = 0;
                    if(a[m]->m[get_idx(i, j, a[0]->y)] > 0) {
                        val = a[m]->m[get_idx(i, j, a[0]->y)];
                    }
                    c[m]->m[get_idx(i, j, c[0]->y)] = val;
                }
            }
        }
    #else
        mt_arg arg[instance->THREADS];
        for(int m = 0; m < len; m++) {
            for(int i = 0; i < instance->THREADS; i++) {
                arg[i].a = a;
                arg[i].c = c;
                arg[i].len = len;
                arg[i].m = m;
                if(instance->THREADS < a[0]->x) {
                    arg[i].single_core = 1;
                    instance->relu_mt(&arg[i]);
                    break;
                }
                arg[i].single_core = 0;
                arg[i].start_routine = instance->relu_mt_wrapper;
                instance->push_mt(&arg[i]);
            }
            if(instance->THREADS >= a[0]->x) {
                instance->wait_mt();
            }
        }
    #endif
    return c;
}

matrix *transpose(matrix *a, matrix *c, mt *instance) {
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
        mt_arg arg[instance->THREADS];
        for(int i = 0; i < instance->THREADS; i++) {
            arg[i].a = &a;
            arg[i].c = &c;
            if(instance->THREADS < a->x) {
                arg[i].single_core = 1;
                instance->transpose_mt(&arg[i]);
                return c;
            }
            arg[i].single_core = 0;
            arg[i].start_routine = instance->transpose_mt_wrapper;
            instance->push_mt(&arg[i]);
        }
        instance->wait_mt();
    #endif
    return c;
}
