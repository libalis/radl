#ifndef MAIN_H
    #define MAIN_H

    #ifndef RUNS
        #ifndef DEBUG
            #define RUNS (4)
        #else
            #define RUNS (1)
        #endif
    #endif

    #ifndef EPOCHS
        #ifndef DEBUG
            #define EPOCHS (10)
        #else
            #define EPOCHS (1)
        #endif
    #endif

    int main(int argc, char *argv[]);
#endif
