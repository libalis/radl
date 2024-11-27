#include <math.h>

#include "../hpp/mt.hpp"
#include "../hpp/tf.hpp"

matrix *add(matrix *a, matrix *b, matrix *c) {
    if(c == NULL) {
        c = malloc_matrix(a->x, a->y);
    }
    if(THREADS == 1) {
        for(int i = 0; i < c->x; i++) {
            for(int j = 0; j < c->y; j++) {
                c->m[i][j] = a->m[i][j] + b->m[i][j];
            }
        }
    } else {
        mt_arg arg[THREADS];
        for(int i = 0; i < THREADS; i++) {
            arg[i].a = a;
            arg[i].b = b;
            arg[i].c = c;
            arg[i].start_routine = add_mt;
            push_mt(&arg[i]);
        }
        wait_mt();
    }
    return c;
}

matrix **biasing(matrix **a, int len, matrix *b, matrix **c) {
    if(c == NULL) {
        c = malloc_matrix_ptr(len, a[0]->x, a[0]->y);
    }
    if(THREADS == 1) {
        for(int m = 0; m < len; m++) {
            for(int i = 0; i < a[m]->x; i++) {
                for(int j = 0; j < a[m]->y; j++) {
                    c[m]->m[i][j] = a[m]->m[i][j] + b->m[m][0];
                }
            }
        }
    } else {
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
    }
    return c;
}

matrix **conv2d(matrix *a, matrix **b, int len, matrix **c) {
    if(c == NULL) {
        c = malloc_matrix_ptr(len, a->x - b[0]->x + 1, a->y - b[0]->y + 1);
    }
    if(THREADS == 1) {
        for(int m = 0; m < len; m++) {
            for(int i = 0; i < a->x - b[m]->x + 1; i++) {
                for(int j = 0; j < a->y - b[m]->y + 1; j++) {
                    float sum = 0.0;
                    for(int k = 0; k < b[m]->x; k++) {
                        for(int l = 0; l < b[m]->y; l++) {
                            sum += a->m[i + k][j + l] * b[m]->m[k][l];
                        }
                    }
                    c[m]->m[i][j] = sum;
                }
            }
        }
    } else {
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
    }
    return c;
}

matrix *flatten(matrix **a, int len, matrix *c) {
    if(c == NULL) {
        c = malloc_matrix(len * a[0]->x * a[0]->y, 1);
    }
    if(THREADS == 1) {
        for(int i = 0; i < a[0]->x; i++) {
            for(int j = 0; j < a[0]->y; j++) {
                for(int m = 0; m < len; m++) {
                    int idx = i * a[0]->y * len + j * len + m;
                    c->m[idx][0] = a[m]->m[i][j];
                }
            }
        }
    } else {
        mt_arg arg[THREADS];
        for(int i = 0; i < THREADS; i++) {
            arg[i].a_ptr = a;
            arg[i].len = len;
            arg[i].c = c;
            arg[i].start_routine = flatten_mt;
            push_mt(&arg[i]);
        }
        wait_mt();
    }
    return c;
}

matrix **flip_kernels(matrix **a, int len, matrix **c) {
    if(c == NULL) {
        c = malloc_matrix_ptr(len, a[0]->x, a[0]->y);
    }
    if(THREADS == 1) {
        for(int m = 0; m < len; m++) {
            for (int i = 0; i < a[m]->x; i++) {
                for (int j = 0; j < a[m]->y; j++) {
                    c[m]->m[i][j] = a[m]->m[a[m]->x - i - 1][a[m]->y - j - 1];
                }
            }
        }
    } else {
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
    }
    return c;
}

matrix **hyperbolic_tangent(matrix **a, int len, matrix **c) {
    if(c == NULL) {
        c = malloc_matrix_ptr(len, a[0]->x, a[0]->y);
    }
    if(THREADS == 1) {
        for(int m = 0; m < len; m++) {
            for(int i = 0; i < a[m]->x; i++) {
                for(int j = 0; j < a[m]->y; j++) {
                    c[m]->m[i][j] = tanh(a[m]->m[i][j]);
                }
            }
        }
    } else {
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
    }
    return c;
}

matrix *matmul(matrix *a, matrix *b, matrix *c) {
    if(c == NULL) {
        c = malloc_matrix(a->x, b->y);
    }
    if(THREADS == 1) {
        for(int i = 0; i < c->x; i++) {
            for(int j = 0; j < c->y; j++) {
                c->m[i][j] = 0.0;
                for(int k = 0; k < a->y; k++) {
                    c->m[i][j] = c->m[i][j] + a->m[i][k] * b->m[k][j];
                }
            }
        }
    } else {
        mt_arg arg[THREADS];
        for(int i = 0; i < THREADS; i++) {
            arg[i].a = a;
            arg[i].b = b;
            arg[i].c = c;
            arg[i].start_routine = matmul_mt;
            push_mt(&arg[i]);
        }
        wait_mt();
    }
    return c;
}

matrix **maxpool(matrix **a, int len, matrix **c) {
    if(c == NULL) {
        c = malloc_matrix_ptr(len, a[0]->x / POOL_LEN, a[0]->y / POOL_LEN);
    }
    if(THREADS == 1) {
        for(int m = 0; m < len; m++) {
            for(int i = 0; i < a[m]->x; i += POOL_LEN) {
                for(int j = 0; j < a[m]->y; j += POOL_LEN) {
                    float max_val = a[m]->m[i][j];
                    for(int k = 0; k < POOL_LEN; k++) {
                        for(int l = 0; l < POOL_LEN; l++) {
                            float curr_val = a[m]->m[i + k][j + l];
                            if(curr_val > max_val) {
                                max_val = curr_val;
                            }
                        }
                    }
                    c[m]->m[i / POOL_LEN][j / POOL_LEN] = max_val;
                }
            }
        }
    } else {
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
    }
    return c;
}

matrix **relu(matrix **a, int len, matrix **c) {
    if(c == NULL) {
        c = malloc_matrix_ptr(len, a[0]->x, a[0]->y);
    }
    if(THREADS == 1) {
        for(int m = 0; m < len; m++) {
            for(int i = 0; i < a[m]->x; i++) {
                for(int j = 0; j < a[m]->y; j++) {
                    float val = 0.0;
                    if(a[m]->m[i][j] > 0.0) {
                        val = a[m]->m[i][j];
                    }
                    c[m]->m[i][j] = val;
                }
            }
        }
    } else {
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
    }
    return c;
}

matrix *transpose(matrix *a, matrix *c) {
    if(c == NULL) {
        c = malloc_matrix(a->y, a->x);
    }
    if(THREADS == 1) {
        for(int i = 0; i < a->x; i++) {
            for(int j = 0; j < a->y; j++) {
                c->m[j][i] = a->m[i][j];
            }
        }
    } else {
        mt_arg arg[THREADS];
        for(int i = 0; i < THREADS; i++) {
            arg[i].a = a;
            arg[i].c = c;
            arg[i].start_routine = transpose_mt;
            push_mt(&arg[i]);
        }
        wait_mt();
    }
    return c;
}
