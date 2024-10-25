#include "../h/tf.h"
#include <stdlib.h>

int max(matrix* a) {
    float max_val = 0.0;
    int index = 0;
    for(int i = 0; i < a->y; i++) {
        float curr_val = a->m[0][i];
        if(curr_val > max_val) {
            max_val = curr_val;
            index = i;
        }
    }
    free_matrix(a);
    a = NULL;
    return index;
}

matrix* add(matrix* a, matrix* b) {
    matrix* c = malloc_matrix(a->x, a->y);
    for(int i = 0; i < c->x; i++) {
        for(int j = 0; j < c->y; j++) {
            c->m[i][j] = a->m[i][j] + b->m[i][j];
        }
    }
    free_matrix(a);
    a = NULL;
    free_matrix(b);
    b = NULL;
    return c;
}

matrix* matmul(matrix* a, matrix* b) {
    matrix* c = malloc_matrix(a->x, b->y);
    for(int i = 0; i < c->x; i++) {
        for(int k = 0; k < c->y; k++) {
            for(int j = 0; j < a->y; j++) {
                c->m[i][k] = c->m[i][k] + a->m[i][j] * b->m[j][k];
            }
        }
    }
    free_matrix(a);
    a = NULL;
    free_matrix(b);
    b = NULL;
    return c;
}

matrix* flatten(matrix* a) {
    matrix* c = malloc_matrix(a->x * a->y, 1);
    int m = 0;
    for(int i = 0; i < a->x; i++) {
        for(int j = 0; j < a->y; j++) {
            c->m[m++][0] = a->m[i][j];
        }
    }
    free_matrix(a);
    a = NULL;
    return c;
}

matrix* maxpool(matrix** a, int len) {
    int pool_len = 1;
    matrix* c = malloc_matrix(a[0]->x / pool_len, a[0]->y / pool_len);
    for(int i = 0; i < c->x; i += pool_len) {
        for(int j = 0; j < c->y; j += pool_len) {
            float max_val = 0.0;
            for(int m = 0; m < len; m++) {
                for(int k = 0; k < pool_len; k++) {
                    for(int l = 0; l < pool_len; l++) {
                        float curr_val = a[m]->m[i + k][j + l];
                        if(curr_val > max_val) {
                            max_val = curr_val;
                        }
                    }
                }
            }
            c->m[i / pool_len][j / pool_len] = max_val;
        }
    }
    for(int i = 0; i < len; i++) {
        free_matrix(a[i]);
        a[i] = NULL;
    }
    free(a);
    a = NULL;
    return c;
}

matrix** relu(matrix** a, int len) {
    matrix** c = malloc(len * sizeof(matrix*));
    for(int m = 0; m < len; m++) {
        c[m] = malloc_matrix(a[m]->x, a[m]->y);
        for(int i = 0; i < a[m]->x; i++) {
            for(int j = 0; j < a[m]->y; j++) {
                if(a[m]->m[i][j] < 0) {
                    c[m]->m[i][j] = 0.0;
                } else {
                    c[m]->m[i][j] = a[m]->m[i][j];
                }
            }
        }
    }
    for(int i = 0; i < len; i++) {
        free_matrix(a[i]);
        a[i] = NULL;
    }
    free(a);
    a = NULL;
    return c;
}

matrix** biasing(matrix** a, int len, matrix* b) {
    matrix** c = malloc(len * sizeof(matrix*));
    for(int m = 0; m < len; m++) {
        c[m] = malloc_matrix(a[m]->x, a[m]->y);
        for(int i = 0; i < a[m]->x; i++) {
            for(int j = 0; j < a[m]->y; j++) {
                c[m]->m[i][j] = a[m]->m[i][j] + b->m[m][0];
            }
        }
    }
    for(int i = 0; i < len; i++) {
        free_matrix(a[i]);
        a[i] = NULL;
    }
    free(a);
    a = NULL;
    free_matrix(b);
    b = NULL;
    return c;
}

matrix** conv2d(matrix* a, matrix** b, int len) {
    matrix** c = malloc(len * sizeof(matrix*));
    for(int m = 0; m < len; m++) {
        int cx = a->x - b[m]->x + 1;
        int cy = a->y - b[m]->y + 1;
        c[m] = malloc_matrix(cx, cy);
        for (int i = 0; i < cx; i++) {
            for (int j = 0; j < cy; j++) {
                float sum = 0.0;
                for (int k = 0; k < b[m]->x; k++) {
                    for (int l = 0; l < b[m]->y; l++) {
                        sum += a->m[i + k][j + l] * b[m]->m[k][l];
                    }
                }
                c[m]->m[i][j] = sum;
            }
        }
    }
    free_matrix(a);
    a = NULL;
    for(int i = 0; i < len; i++) {
        if(b[i] != NULL){
            free_matrix(b[i]);
            b[i] = NULL;
        }
    }
    free(b);
    b = NULL;
    return c;
}
