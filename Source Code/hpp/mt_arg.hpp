#ifndef MT_ARG_HPP
    #define MT_ARG_HPP

    #include "matrix.hpp"

    typedef struct mt_arg {
        int idx;
        matrix **a;
        matrix **b;
        matrix **c;
        int len;
        int m;
        int i;
        int j;
        bool single_core;
        void (*start_routine)(struct mt_arg *mt);
    } mt_arg;
#endif
