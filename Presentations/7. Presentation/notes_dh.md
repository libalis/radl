# Naive Implementation
## 5
- We implemented the MNIST dataset in python, which is a dataset of "handwritten" numbers
- We used the essential functions and trained in python using TensorFlow and Keras libraries
- We exported the images in a human-readable format. As you can easily see it is a six, but we had to train a model to recognize it 
## 6
- Our naive implementation was made in a modular form in c
- We implemented every function of the graph c 
- Additionally we had to implement the input and output functions for reading and saving the intermediate results 
- We also had to implement benchmark modules like time mesurement and memory allocation
- The main module combines everything and runs the model
## 7
- As we expected the m3 is the fastest.
- It is only single core, therefore we see the same effect on the xl benchmark

# OpenMP
## 15
- API for parallel programming in c or c++ for CPU as well as for GPU
- It uses pragmas (so called compiler directives) to define parallize code blocks
- The compiler handels most parallelization tasks, which reduces the transparency and flexibility of the code
## 15 - image
- You can use environment variables to control the number of threads
- To define the parallel regions you have to use pragmas
- It will be translated by the compiler into parallel code and the execution is handled by the runtime library (e.g. thread creation, synchronization etc.)
- At the end you have parallized code
## 16
- Here are some examples of the pragmas, we used
- As I mentioned before, you can use built-in functions to control the number of threads
- In addition, you can set environment variables (such as OMP_NUM_THREADS) to control the number of threads.
## 16 - code 
- Here is our add function, which uses openMP pragmas to parallize two for loops 
## 17
- We used these compiler flags, they are slightly different
- You can also use offload-targets, to switch the execution of the code to other hardware like GPUs
- But we had no available offload-targets, although we recompiled the compiler multiple times
- OpenMP is a powerful tool, but it is not a swiss army knife. For example we received a warning by using openmp with conv2d, because the compiler could not handle the parallelization

# SIMD
## 28
- Single Instruction Multiple Data
- SIMD is a technique to execute multiple operations in parallel
- It uses vector registers to store multiple values and perform operations simultaneously
- We implemented SIMD in different architectures we used
## 29
- Neon is the SIMD architecture for ARM
- It uses 32 128-bit registers to store values
- it supports varius data types, e.g. int8, int32 or float32
- You don't have to think about alignment, as you can see on the graphic
- Useful functions, e.g. vaddv, which enables horizontal addition -> useful for the conv2d-function

- To use low-level instruction, no assembly is needed
- Because 32-bit and 64-bit ARM processors are compatible, you can use the same code for both architectures
## 30
- Neon is slightly beter, but as Robert said, we use it inside the multi-threaded code, therefore the registers are full
## 31
- Reversed engineering of the AMX (Apple Matrix Extensions) chip for the Apple Silicon chips
- Input registers (x and y) each 4 registers which contain 64 floats
- Output registers (z) holds 64x64 float matrix
- We implemented AMX instruction set, because it is the only functions, where a fused multiply adder is useful. The other functions only add values, which is not useful for the AMX instruction set
- The instructions are macros which take a bitmap as an argument. for setting the relevant bits, we refer to the documentation
- As you can see, the power consumption on the XL benchmark is slightly lower than for the neon benchmark
## 32
- Amx is good for larger test sets, as you can see in the xxl benchmark
## 33
- Keep in mind the limited amount of data types 
- Lack of documentation
- limited instruction set -> no simple add function
- AMX and Arm Neon will be replaced by SME (Scalable Matrix Extensions) under Armv9, which will support both 128-bit (AMX, Neon) and 256-bit (SME) vector lengths

# Quantization
## 38
- int8 for input values and int32 for intermediate results
- Slight decrease in the accuracy, but still 90% overall accuracy

- Implemented for AVX2, AVX-512 and Neon, but AMX does not support int8

- We reduced the model size by 15% but the out multi-threaded object has to much overhead. Basically the size decreased due to the fact, the the input values are int8 and the computing values are still int32. 
- With data types smaller than int32 for intermediate results, the accuracy decreases significantly and there a no prebuild data types like int24
## 39
- There is no significant difference in the benchmarks, but for the xl benchmark you can see there is a slight decrease in the total time for the xl test
