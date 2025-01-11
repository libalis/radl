#ifndef MT_ARG_HPP
    #define MT_ARG_HPP

    #include "matrix.hpp"

    typedef struct mt_arg {
        long idx;
        matrix **a_ptr;
        matrix *a;
        matrix **b_ptr;
        matrix *b;
        int len;
        matrix **c_ptr;
        matrix *c;
        int m;
        int i;
        int j;
        bool single_core;
        void (*start_routine)(struct mt_arg *mt);
    } mt_arg;
#endif
