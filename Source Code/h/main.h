#ifndef MAIN_H
    #define MAIN_H

    #ifndef BENCHMARK
        #define BENCHMARK ("./csv/benchmark.csv")
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
