#ifndef MT_ARG_HPP
    #define MT_ARG_HPP

    #include "io.hpp"
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
        struct io *io;
        void (*start_routine)(void *instance, struct mt_arg *arg);
    } mt_arg;
#endif
