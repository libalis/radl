#!/usr/bin/env python
import os
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

label_amd = "AMD Ryzen 7 3800XT"
label_apple = "Apple M3 Pro 11-Core"
label_intel = "Intel Core i7 1065G7"
label_intel_nvidia = "NVIDIA GeForce MX350"
label_amd_nvidia = "NVIDIA GeForce RTX 2080"

foreground_color = '#242424'
background_color = '#ffffff'

threads = [1, 4, 5, 8, 11, 16]
threads_omp = [8, 11, 16]

"""
READ BENCHMARKS
"""
# amd
df_amd_raw = pd.read_csv("./csv/5. Presentation/amd/benchmark.csv")
df_amd_old_raw = pd.read_csv("./csv/4. Presentation/amd/benchmark.csv")
df_amd_int_raw = pd.read_csv("./csv/5. Presentation/amd/benchmark_int.csv")
df_amd_int_xl_raw = pd.read_csv("./csv/5. Presentation/amd/benchmark_int_xl.csv")
df_amd_intel_raw = pd.read_csv("./csv/5. Presentation/amd/benchmark_intel.csv")
df_amd_intel_old_raw = pd.read_csv("./csv/4. Presentation/amd/benchmark_intel.csv")
df_amd_nvidia_raw = pd.read_csv("./csv/5. Presentation/amd/benchmark_nvidia.csv")
df_amd_nvidia_old_raw = pd.read_csv("./csv/4. Presentation/amd/benchmark_nvidia.csv")
df_amd_omp_raw = pd.read_csv("./csv/5. Presentation/amd/benchmark_omp.csv")
df_amd_omp_intel_raw = pd.read_csv("./csv/5. Presentation/amd/benchmark_omp_intel.csv")
df_amd_omp_xl_raw = pd.read_csv("./csv/5. Presentation/amd/benchmark_omp_xl.csv")
df_amd_omp_xl_intel_raw = pd.read_csv("./csv/5. Presentation/amd/benchmark_omp_xl_intel.csv")
df_amd_xl_raw = pd.read_csv("./csv/5. Presentation/amd/benchmark_xl.csv")
df_amd_xl_old_raw = pd.read_csv("./csv/4. Presentation/amd/benchmark_xl.csv")
df_amd_xl_intel_raw = pd.read_csv("./csv/5. Presentation/amd/benchmark_xl_intel.csv")
df_amd_xl_intel_old_raw = pd.read_csv("./csv/4. Presentation/amd/benchmark_xl_intel.csv")
df_amd_xl_nvidia_raw = pd.read_csv("./csv/5. Presentation/amd/benchmark_xl_nvidia.csv")
df_amd_xl_nvidia_old_raw = pd.read_csv("./csv/4. Presentation/amd/benchmark_xl_nvidia.csv")

tt_amd = df_amd_raw['total_time_us'].values
tt_amd_old = df_amd_old_raw['total_time_us'].values
tt_amd_int = df_amd_int_raw['total_time_us'].values
tt_amd_int_xl = df_amd_int_xl_raw['total_time_us'].values
tt_amd_intel = df_amd_intel_raw['total_time_us'].values
tt_amd_intel_old = df_amd_intel_old_raw['total_time_us'].values
tt_amd_nvidia = df_amd_nvidia_raw['total_time_us'].values
tt_amd_nvidia_old = df_amd_nvidia_old_raw['total_time_us'].values
tt_amd_omp = df_amd_omp_raw['total_time_us'].values
tt_amd_omp_intel = df_amd_omp_intel_raw['total_time_us'].values
tt_amd_omp_xl = df_amd_omp_xl_raw['total_time_us'].values
tt_amd_omp_xl_intel = df_amd_omp_xl_intel_raw['total_time_us'].values
tt_amd_xl = df_amd_xl_raw['total_time_us'].values
tt_amd_xl_old = df_amd_xl_old_raw['total_time_us'].values
tt_amd_xl_intel = df_amd_xl_intel_raw['total_time_us'].values
tt_amd_xl_intel_old = df_amd_xl_intel_old_raw['total_time_us'].values
tt_amd_xl_nvidia = df_amd_xl_nvidia_raw['total_time_us'].values
tt_amd_xl_nvidia_old = df_amd_xl_nvidia_old_raw['total_time_us'].values

# apple
df_apple_raw = pd.read_csv("./csv/5. Presentation/apple/benchmark.csv")
df_apple_old_raw = pd.read_csv("./csv/4. Presentation/apple/benchmark.csv")
df_apple_int_raw = pd.read_csv("./csv/5. Presentation/apple/benchmark_int.csv")
df_apple_int_xl_raw = pd.read_csv("./csv/5. Presentation/apple/benchmark_int_xl.csv")
df_apple_omp_raw = pd.read_csv("./csv/5. Presentation/apple/benchmark_omp.csv")
df_apple_omp_xl_raw = pd.read_csv("./csv/5. Presentation/apple/benchmark_omp_xl.csv")
df_apple_xl_raw = pd.read_csv("./csv/5. Presentation/apple/benchmark_xl.csv")
df_apple_xl_old_raw = pd.read_csv("./csv/4. Presentation/apple/benchmark_xl.csv")
df_apple_amx = pd.read_csv("./csv/5. Presentation/apple/benchmark_amx.csv")
df_apple_amx_xl = pd.read_csv("./csv/5. Presentation/apple/benchmark_amx_xl.csv")
df_apple_neon = pd.read_csv("./csv/5. Presentation/apple/benchmark.csv")
df_apple_neon_xl = pd.read_csv("./csv/5. Presentation/apple/benchmark_xl.csv")

tt_apple = df_apple_raw['total_time_us'].values
tt_apple_old = df_apple_old_raw['total_time_us'].values
tt_apple_int = df_apple_int_raw['total_time_us'].values
tt_apple_int_xl = df_apple_int_xl_raw['total_time_us'].values
tt_apple_omp = df_apple_omp_raw['total_time_us'].values
tt_apple_omp_xl = df_apple_omp_xl_raw['total_time_us'].values
tt_apple_xl = df_apple_xl_raw['total_time_us'].values
tt_apple_xl_old = df_apple_xl_old_raw['total_time_us'].values
tt_apple_amx = df_apple_amx['total_time_us'].values
tt_apple_amx_xl = df_apple_amx_xl['total_time_us'].values
tt_apple_neon = df_apple_neon['total_time_us'].values
tt_apple_neon_xl = df_apple_neon_xl['total_time_us'].values

# intel
df_intel_raw = pd.read_csv("./csv/5. Presentation/intel/benchmark.csv")
df_intel_old_raw = pd.read_csv("./csv/4. Presentation/intel/benchmark.csv")
df_intel_int_raw = pd.read_csv("./csv/5. Presentation/intel/benchmark_int.csv")
df_intel_int_xl_raw = pd.read_csv("./csv/5. Presentation/intel/benchmark_int_xl.csv")
df_intel_intel_raw = pd.read_csv("./csv/5. Presentation/intel/benchmark_intel.csv")
df_intel_intel_old_raw = pd.read_csv("./csv/4. Presentation/intel/benchmark_intel.csv")
df_intel_nvidia_raw = pd.read_csv("./csv/5. Presentation/intel/benchmark_nvidia.csv")
df_intel_nvidia_old_raw = pd.read_csv("./csv/4. Presentation/intel/benchmark_nvidia.csv")
df_intel_omp_raw = pd.read_csv("./csv/5. Presentation/intel/benchmark_omp.csv")
df_intel_omp_intel_raw = pd.read_csv("./csv/5. Presentation/intel/benchmark_omp_intel.csv")
df_intel_omp_xl_raw = pd.read_csv("./csv/5. Presentation/intel/benchmark_omp_xl.csv")
df_intel_omp_xl_intel_raw = pd.read_csv("./csv/5. Presentation/intel/benchmark_omp_xl_intel.csv")
df_intel_xl_raw = pd.read_csv("./csv/5. Presentation/intel/benchmark_xl.csv")
df_intel_xl_old_raw = pd.read_csv("./csv/4. Presentation/intel/benchmark_xl.csv")
df_intel_xl_intel_raw = pd.read_csv("./csv/5. Presentation/intel/benchmark_xl_intel.csv")
df_intel_xl_intel_old_raw = pd.read_csv("./csv/4. Presentation/intel/benchmark_xl_intel.csv")
df_intel_xl_nvidia_raw = pd.read_csv("./csv/5. Presentation/intel/benchmark_xl_nvidia.csv")
df_intel_xl_nvidia_old_raw = pd.read_csv("./csv/4. Presentation/intel/benchmark_xl_nvidia.csv")

tt_intel = df_intel_raw['total_time_us'].values
tt_intel_old = df_intel_old_raw['total_time_us'].values
tt_intel_int = df_intel_int_raw['total_time_us'].values
tt_intel_int_xl = df_intel_int_xl_raw['total_time_us'].values
tt_intel_intel = df_intel_intel_raw['total_time_us'].values
tt_intel_intel_old = df_intel_intel_old_raw['total_time_us'].values
tt_intel_nvidia = df_intel_nvidia_raw['total_time_us'].values
tt_intel_nvidia_old = df_intel_nvidia_old_raw['total_time_us'].values
tt_intel_omp = df_intel_omp_raw['total_time_us'].values
tt_intel_omp_intel = df_intel_omp_intel_raw['total_time_us'].values
tt_intel_omp_xl = df_intel_omp_xl_raw['total_time_us'].values
tt_intel_omp_xl_intel = df_intel_omp_xl_intel_raw['total_time_us'].values
tt_intel_xl = df_intel_xl_raw['total_time_us'].values
tt_intel_xl_old = df_intel_xl_old_raw['total_time_us'].values
tt_intel_xl_intel = df_intel_xl_intel_raw['total_time_us'].values
tt_intel_xl_intel_old = df_intel_xl_intel_old_raw['total_time_us'].values
tt_intel_xl_nvidia = df_intel_xl_nvidia_raw['total_time_us'].values
tt_intel_xl_nvidia_old = df_intel_xl_nvidia_old_raw['total_time_us'].values

"""
CREATE DATA FRAMES
"""
# amd
df_amd = pd.DataFrame({
    'min': [0.0] * len(threads_omp),
    'avg': [0.0] * len(threads_omp),
    'max': [0.0] * len(threads_omp),
    'std': [0.0] * len(threads_omp)
})
df_amd_old = df_amd.copy(deep=True)
df_amd_int = df_amd.copy(deep=True)
df_amd_int_xl = df_amd.copy(deep=True)
df_amd_intel = df_amd.copy(deep=True)
df_amd_intel_old = df_amd.copy(deep=True)
df_amd_nvidia = df_amd.copy(deep=True)
df_amd_nvidia_old = df_amd.copy(deep=True)
df_amd_omp = df_amd.copy(deep=True)
df_amd_omp_intel = df_amd.copy(deep=True)
df_amd_omp_xl = df_amd.copy(deep=True)
df_amd_omp_xl_intel = df_amd.copy(deep=True)
df_amd_xl = df_amd.copy(deep=True)
df_amd_xl_old = df_amd.copy(deep=True)
df_amd_xl_intel = df_amd.copy(deep=True)
df_amd_xl_intel_old = df_amd.copy(deep=True)
df_amd_xl_nvidia = df_amd.copy(deep=True)
df_amd_xl_nvidia_old = df_amd.copy(deep=True)

# apple
df_apple = df_amd.copy(deep=True)
df_apple_old = df_amd.copy(deep=True)
df_apple_int = df_amd.copy(deep=True)
df_apple_int_xl = df_amd.copy(deep=True)
df_apple_omp = df_amd.copy(deep=True)
df_apple_omp_xl = df_apple.copy(deep=True)
df_apple_xl = df_amd.copy(deep=True)
df_apple_xl_old = df_amd.copy(deep=True)
df_apple_amx = df_amd.copy(deep=True)
df_apple_amx_xl = df_amd.copy(deep=True)
df_apple_neon = df_amd.copy(deep=True)
df_apple_neon_xl = df_amd.copy(deep=True)

# intel
df_intel = df_amd.copy(deep=True)
df_intel_old = df_amd.copy(deep=True)
df_intel_int = df_amd.copy(deep=True)
df_intel_int_xl = df_amd.copy(deep=True)
df_intel_intel = df_amd.copy(deep=True)
df_intel_intel_old = df_amd.copy(deep=True)
df_intel_nvidia = df_amd.copy(deep=True)
df_intel_nvidia_old = df_amd.copy(deep=True)
df_intel_omp = df_amd.copy(deep=True)
df_intel_omp_intel = df_amd.copy(deep=True)
df_intel_omp_xl = df_amd.copy(deep=True)
df_intel_omp_xl_intel = df_amd.copy(deep=True)
df_intel_xl = df_amd.copy(deep=True)
df_intel_xl_old = df_amd.copy(deep=True)
df_intel_xl_intel = df_amd.copy(deep=True)
df_intel_xl_intel_old = df_amd.copy(deep=True)
df_intel_xl_nvidia = df_amd.copy(deep=True)
df_intel_xl_nvidia_old = df_amd.copy(deep=True)

# Fill DFs with Minimum, Average, Maximum values
for i in range(len(threads)):
    # amd
    if threads[i] == 16:
        df_amd.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_amd[10*i:10*(i+1)])
        df_amd.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_amd[10*i:10*(i+1)])
        df_amd.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_amd[10*i:10*(i+1)])
        df_amd_old.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_amd_old[10*i:10*(i+1)])
        df_amd_old.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_amd_old[10*i:10*(i+1)])
        df_amd_old.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_amd_old[10*i:10*(i+1)])
        df_amd_int.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_amd_int[10*i:10*(i+1)])
        df_amd_int.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_amd_int[10*i:10*(i+1)])
        df_amd_int.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_amd_int[10*i:10*(i+1)])
        df_amd_int_xl.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_amd_int_xl[10*i:10*(i+1)])
        df_amd_int_xl.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_amd_int_xl[10*i:10*(i+1)])
        df_amd_int_xl.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_amd_int_xl[10*i:10*(i+1)])
        df_amd_intel.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_amd_intel[10*i:10*(i+1)])
        df_amd_intel.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_amd_intel[10*i:10*(i+1)])
        df_amd_intel.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_amd_intel[10*i:10*(i+1)])
        df_amd_intel_old.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_amd_intel_old[10*i:10*(i+1)])
        df_amd_intel_old.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_amd_intel_old[10*i:10*(i+1)])
        df_amd_intel_old.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_amd_intel_old[10*i:10*(i+1)])
        df_amd_nvidia.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_amd_nvidia[0:10])
        df_amd_nvidia.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_amd_nvidia[0:10])
        df_amd_nvidia.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_amd_nvidia[0:10])
        df_amd_nvidia.loc[threads_omp.index(threads[i]), 'std'] = np.round(np.std(tt_amd_nvidia[0:10]), 1)
        df_amd_nvidia_old.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_amd_nvidia_old[0:10])
        df_amd_nvidia_old.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_amd_nvidia_old[0:10])
        df_amd_nvidia_old.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_amd_nvidia_old[0:10])
        df_amd_nvidia_old.loc[threads_omp.index(threads[i]), 'std'] = np.round(np.std(tt_amd_nvidia_old[0:10]), 1)
        df_amd_omp.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_amd_omp[0:10])
        df_amd_omp.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_amd_omp[0:10])
        df_amd_omp.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_amd_omp[0:10])
        df_amd_omp_intel.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_amd_omp_intel[0:10])
        df_amd_omp_intel.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_amd_omp_intel[0:10])
        df_amd_omp_intel.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_amd_omp_intel[0:10])
        df_amd_omp_xl.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_amd_omp_xl[0:10])
        df_amd_omp_xl.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_amd_omp_xl[0:10])
        df_amd_omp_xl.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_amd_omp_xl[0:10])
        df_amd_omp_xl_intel.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_amd_omp_xl_intel[0:10])
        df_amd_omp_xl_intel.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_amd_omp_xl_intel[0:10])
        df_amd_omp_xl_intel.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_amd_omp_xl_intel[0:10])
        df_amd_xl.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_amd_xl[10*i:10*(i+1)])
        df_amd_xl.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_amd_xl[10*i:10*(i+1)])
        df_amd_xl.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_amd_xl[10*i:10*(i+1)])
        df_amd_xl_old.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_amd_xl_old[10*i:10*(i+1)])
        df_amd_xl_old.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_amd_xl_old[10*i:10*(i+1)])
        df_amd_xl_old.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_amd_xl_old[10*i:10*(i+1)])
        df_amd_xl_intel.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_amd_xl_intel[10*i:10*(i+1)])
        df_amd_xl_intel.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_amd_xl_intel[10*i:10*(i+1)])
        df_amd_xl_intel.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_amd_xl_intel[10*i:10*(i+1)])
        df_amd_xl_intel_old.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_amd_xl_intel_old[10*i:10*(i+1)])
        df_amd_xl_intel_old.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_amd_xl_intel_old[10*i:10*(i+1)])
        df_amd_xl_intel_old.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_amd_xl_intel_old[10*i:10*(i+1)])
        df_amd_xl_nvidia.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_amd_xl_nvidia[0:10])
        df_amd_xl_nvidia.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_amd_xl_nvidia[0:10])
        df_amd_xl_nvidia.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_amd_xl_nvidia[0:10])
        df_amd_xl_nvidia.loc[threads_omp.index(threads[i]), 'std'] = np.round(np.std(tt_amd_xl_nvidia[0:10]), 1)
        df_amd_xl_nvidia_old.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_amd_xl_nvidia_old[0:10])
        df_amd_xl_nvidia_old.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_amd_xl_nvidia_old[0:10])
        df_amd_xl_nvidia_old.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_amd_xl_nvidia_old[0:10])
        df_amd_xl_nvidia_old.loc[threads_omp.index(threads[i]), 'std'] = np.round(np.std(tt_amd_xl_nvidia_old[0:10]), 1)

    # apple
    if threads[i] == 1:
        df_apple_amx.loc[threads.index(threads[i]), 'min'] = np.min(tt_apple_amx[10*i:10*(i+1)])
        df_apple_amx.loc[threads.index(threads[i]), 'avg'] = np.average(tt_apple_amx[10*i:10*(i+1)])
        df_apple_amx.loc[threads.index(threads[i]), 'max'] = np.max(tt_apple_amx[10*i:10*(i+1)])
        df_apple_amx_xl.loc[threads.index(threads[i]), 'min'] = np.min(tt_apple_amx_xl[10*i:10*(i+1)])
        df_apple_amx_xl.loc[threads.index(threads[i]), 'avg'] = np.average(tt_apple_amx_xl[10*i:10*(i+1)])
        df_apple_amx_xl.loc[threads.index(threads[i]), 'max'] = np.max(tt_apple_amx_xl[10*i:10*(i+1)])
        df_apple_neon.loc[threads.index(threads[i]), 'min'] = np.min(tt_apple_neon[10*i:10*(i+1)])
        df_apple_neon.loc[threads.index(threads[i]), 'avg'] = np.average(tt_apple_neon[10*i:10*(i+1)])
        df_apple_neon.loc[threads.index(threads[i]), 'max'] = np.max(tt_apple_neon[10*i:10*(i+1)])
        df_apple_neon_xl.loc[threads.index(threads[i]), 'min'] = np.min(tt_apple_neon_xl[10*i:10*(i+1)])
        df_apple_neon_xl.loc[threads.index(threads[i]), 'avg'] = np.average(tt_apple_neon_xl[10*i:10*(i+1)])

    if threads[i] == 11:
        df_apple.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_apple[10*i:10*(i+1)])
        df_apple.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_apple[10*i:10*(i+1)])
        df_apple.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_apple[10*i:10*(i+1)])
        df_apple_old.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_apple_old[10*i:10*(i+1)])
        df_apple_old.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_apple_old[10*i:10*(i+1)])
        df_apple_old.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_apple_old[10*i:10*(i+1)])
        df_apple_int.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_apple_int[0:10])
        df_apple_int.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_apple_int[0:10])
        df_apple_int.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_apple_int[0:10])
        df_apple_int_xl.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_apple_int_xl[0:10])
        df_apple_int_xl.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_apple_int_xl[0:10])
        df_apple_int_xl.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_apple_int_xl[0:10])
        df_apple_omp.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_apple_omp[0:10])
        df_apple_omp.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_apple_omp[0:10])
        df_apple_omp.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_apple_omp[0:10])
        df_apple_omp_xl.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_apple_omp_xl[0:10])
        df_apple_omp_xl.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_apple_omp_xl[0:10])
        df_apple_omp_xl.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_apple_omp_xl[0:10])
        df_apple_xl.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_apple_xl[10*i:10*(i+1)])
        df_apple_xl.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_apple_xl[10*i:10*(i+1)])
        df_apple_xl.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_apple_xl[10*i:10*(i+1)])
        df_apple_xl_old.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_apple_xl_old[10*i:10*(i+1)])
        df_apple_xl_old.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_apple_xl_old[10*i:10*(i+1)])
        df_apple_xl_old.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_apple_xl_old[10*i:10*(i+1)])
        df_apple_amx.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_apple_amx[10*i:10*(i+1)])
        df_apple_amx.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_apple_amx[10*i:10*(i+1)])
        df_apple_amx.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_apple_amx[10*i:10*(i+1)])
        df_apple_amx_xl.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_apple_amx_xl[10*i:10*(i+1)])
        df_apple_amx_xl.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_apple_amx_xl[10*i:10*(i+1)])
        df_apple_amx_xl.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_apple_amx_xl[10*i:10*(i+1)])
        df_apple_neon.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_apple_neon[10*i:10*(i+1)])
        df_apple_neon.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_apple_neon[10*i:10*(i+1)])
        df_apple_neon.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_apple_neon[10*i:10*(i+1)])
        df_apple_neon_xl.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_apple_neon_xl[10*i:10*(i+1)])
        df_apple_neon_xl.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_apple_neon_xl[10*i:10*(i+1)])
        df_apple_neon_xl.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_apple_neon_xl[10*i:10*(i+1)])

    # intel
    if threads[i] == 8:
        df_intel.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_intel[10*i:10*(i+1)])
        df_intel.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_intel[10*i:10*(i+1)])
        df_intel.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_intel[10*i:10*(i+1)])
        df_intel_old.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_intel_old[10*i:10*(i+1)])
        df_intel_old.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_intel_old[10*i:10*(i+1)])
        df_intel_old.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_intel_old[10*i:10*(i+1)])
        df_intel_int.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_intel_int[10*i:10*(i+1)])
        df_intel_int.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_intel_int[10*i:10*(i+1)])
        df_intel_int.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_intel_int[10*i:10*(i+1)])
        df_intel_int_xl.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_intel_int_xl[10*i:10*(i+1)])
        df_intel_int_xl.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_intel_int_xl[10*i:10*(i+1)])
        df_intel_int_xl.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_intel_int_xl[10*i:10*(i+1)])
        df_intel_intel.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_intel_intel[10*i:10*(i+1)])
        df_intel_intel.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_intel_intel[10*i:10*(i+1)])
        df_intel_intel.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_intel_intel[10*i:10*(i+1)])
        df_intel_intel_old.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_intel_intel_old[10*i:10*(i+1)])
        df_intel_intel_old.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_intel_intel_old[10*i:10*(i+1)])
        df_intel_intel_old.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_intel_intel_old[10*i:10*(i+1)])
        df_intel_nvidia.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_intel_nvidia[0:10])
        df_intel_nvidia.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_intel_nvidia[0:10])
        df_intel_nvidia.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_intel_nvidia[0:10])
        df_intel_nvidia.loc[threads_omp.index(threads[i]), 'std'] = np.round(np.std(tt_intel_nvidia[0:10]), 1)
        df_intel_nvidia_old.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_intel_nvidia_old[0:10])
        df_intel_nvidia_old.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_intel_nvidia_old[0:10])
        df_intel_nvidia_old.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_intel_nvidia_old[0:10])
        df_intel_nvidia_old.loc[threads_omp.index(threads[i]), 'std'] = np.round(np.std(tt_intel_nvidia_old[0:10]), 1)
        df_intel_omp.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_intel_omp[0:10])
        df_intel_omp.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_intel_omp[0:10])
        df_intel_omp.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_intel_omp[0:10])
        df_intel_omp_intel.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_intel_omp_intel[0:10])
        df_intel_omp_intel.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_intel_omp_intel[0:10])
        df_intel_omp_intel.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_intel_omp_intel[0:10])
        df_intel_omp_xl.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_intel_omp_xl[0:10])
        df_intel_omp_xl.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_intel_omp_xl[0:10])
        df_intel_omp_xl.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_intel_omp_xl[0:10])
        df_intel_omp_xl_intel.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_intel_omp_xl_intel[0:10])
        df_intel_omp_xl_intel.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_intel_omp_xl_intel[0:10])
        df_intel_omp_xl_intel.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_intel_omp_xl_intel[0:10])
        df_intel_xl.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_intel_xl[10*i:10*(i+1)])
        df_intel_xl.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_intel_xl[10*i:10*(i+1)])
        df_intel_xl.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_intel_xl[10*i:10*(i+1)])
        df_intel_xl_old.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_intel_xl_old[10*i:10*(i+1)])
        df_intel_xl_old.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_intel_xl_old[10*i:10*(i+1)])
        df_intel_xl_old.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_intel_xl_old[10*i:10*(i+1)])
        df_intel_xl_intel.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_intel_xl_intel[10*i:10*(i+1)])
        df_intel_xl_intel.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_intel_xl_intel[10*i:10*(i+1)])
        df_intel_xl_intel.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_intel_xl_intel[10*i:10*(i+1)])
        df_intel_xl_intel_old.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_intel_xl_intel_old[10*i:10*(i+1)])
        df_intel_xl_intel_old.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_intel_xl_intel_old[10*i:10*(i+1)])
        df_intel_xl_intel_old.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_intel_xl_intel_old[10*i:10*(i+1)])
        df_intel_xl_nvidia.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_intel_xl_nvidia[0:10])
        df_intel_xl_nvidia.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_intel_xl_nvidia[0:10])
        df_intel_xl_nvidia.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_intel_xl_nvidia[0:10])
        df_intel_xl_nvidia.loc[threads_omp.index(threads[i]), 'std'] = np.round(np.std(tt_intel_xl_nvidia[0:10]), 1)
        df_intel_xl_nvidia_old.loc[threads_omp.index(threads[i]), 'min'] = np.min(tt_intel_xl_nvidia_old[0:10])
        df_intel_xl_nvidia_old.loc[threads_omp.index(threads[i]), 'avg'] = np.average(tt_intel_xl_nvidia_old[0:10])
        df_intel_xl_nvidia_old.loc[threads_omp.index(threads[i]), 'max'] = np.max(tt_intel_xl_nvidia_old[0:10])
        df_intel_xl_nvidia_old.loc[threads_omp.index(threads[i]), 'std'] = np.round(np.std(tt_intel_xl_nvidia_old[0:10]), 1)

"""
CREATE FIGURES
"""
figures = ["int", "int_xl", "simd", "simd_xl", "omp", "omp_xl", "simd_intel", "simd_xl_intel", "omp_intel", "omp_xl_intel", "cuda", "cuda_xl", "cpu_gpu", "cpu_gpu_xl", "amx", "amx_xl"]
for f in figures:
    bar_width = 0.15

    indices = np.arange(len(threads_omp))
    fig, ax = plt.subplots(figsize=(10, 6))

    match f:
        case "int":
            group_width = bar_width * 2 # Space occupied by one group of bars
            group_gap = 0.15 # Gap between groups
            indices = np.arange(len(threads_omp)) * (group_width + group_gap) # Indices for each group

            ax.bar(indices - 0.5 * bar_width, df_intel['max'], width=bar_width, label=f"{label_intel} Max", color='#99c1f1', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel['avg'], width=bar_width, label=f"{label_intel} Avg", color='#3584e4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel['min'], width=bar_width, label=f"{label_intel} Min", color='#1a5fb4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_int['max'], width=bar_width, label=f"{label_intel} Quantization Max", color='#8ff0a4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_int['avg'], width=bar_width, label=f"{label_intel} Quantization Avg", color='#33d17a', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_int['min'], width=bar_width, label=f"{label_intel} Quantization Min", color='#26a269', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple['max'], width=bar_width, label=f"{label_apple} Max", color='#f9f06b', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple['avg'], width=bar_width, label=f"{label_apple} Avg", color='#f6d32d', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple['min'], width=bar_width, label=f"{label_apple} Min", color='#e5a50a', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_int['max'], width=bar_width, label=f"{label_apple} Quantization Max", color='#ffbe6f', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_int['avg'], width=bar_width, label=f"{label_apple} Quantization Avg", color='#ff7800', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_int['min'], width=bar_width, label=f"{label_apple} Quantization Min", color='#c64600', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd['max'], width=bar_width, label=f"{label_amd} Max", color='#f66151', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd['avg'], width=bar_width, label=f"{label_amd} Avg", color='#e01b24', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd['min'], width=bar_width, label=f"{label_amd} Min", color='#a51d2d', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_int['max'], width=bar_width, label=f"{label_amd} Quantization Max", color='#dc8add', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_int['avg'], width=bar_width, label=f"{label_amd} Quantization Avg", color='#9141ac', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_int['min'], width=bar_width, label=f"{label_amd} Quantization Min", color='#613583', edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("Quantization", fontsize=12, color=foreground_color, loc='center')
        case "int_xl":
            group_width = bar_width * 2 # Space occupied by one group of bars
            group_gap = 0.15 # Gap between groups
            indices = np.arange(len(threads_omp)) * (group_width + group_gap) # Indices for each group

            ax.bar(indices - 0.5 * bar_width, df_intel_xl['max'], width=bar_width, label=f"{label_intel} XL Max", color='#99c1f1', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_xl['avg'], width=bar_width, label=f"{label_intel} XL Avg", color='#3584e4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_xl['min'], width=bar_width, label=f"{label_intel} XL Min", color='#1a5fb4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_int_xl['max'], width=bar_width, label=f"{label_intel} Quantization XL Max", color='#8ff0a4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_int_xl['avg'], width=bar_width, label=f"{label_intel} Quantization XL Avg", color='#33d17a', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_int_xl['min'], width=bar_width, label=f"{label_intel} Quantization XL Min", color='#26a269', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_xl['max'], width=bar_width, label=f"{label_apple} XL Max", color='#f9f06b', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_xl['avg'], width=bar_width, label=f"{label_apple} XL Avg", color='#f6d32d', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_xl['min'], width=bar_width, label=f"{label_apple} XL Min", color='#e5a50a', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_int_xl['max'], width=bar_width, label=f"{label_apple} Quantization XL Max", color='#ffbe6f', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_int_xl['avg'], width=bar_width, label=f"{label_apple} Quantization XL Avg", color='#ff7800', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_int_xl['min'], width=bar_width, label=f"{label_apple} Quantization XL Min", color='#c64600', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_xl['max'], width=bar_width, label=f"{label_amd} XL Max", color='#f66151', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_xl['avg'], width=bar_width, label=f"{label_amd} XL Avg", color='#e01b24', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_xl['min'], width=bar_width, label=f"{label_amd} XL Min", color='#a51d2d', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_int_xl['max'], width=bar_width, label=f"{label_amd} Quantization XL Max", color='#dc8add', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_int_xl['avg'], width=bar_width, label=f"{label_amd} Quantization XL Avg", color='#9141ac', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_int_xl['min'], width=bar_width, label=f"{label_amd} Quantization XL Min", color='#613583', edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("Quantization XL", fontsize=12, color=foreground_color, loc='center')
        case "simd":
            group_width = bar_width * 2 # Space occupied by one group of bars
            group_gap = 0.15 # Gap between groups
            indices = np.arange(len(threads_omp)) * (group_width + group_gap) # Indices for each group

            ax.bar(indices - 0.5 * bar_width, df_intel_old['max'], width=bar_width, label=f"{label_intel} Old Max", color='#99c1f1', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_old['avg'], width=bar_width, label=f"{label_intel} Old Avg", color='#3584e4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_old['min'], width=bar_width, label=f"{label_intel} Old Min", color='#1a5fb4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel['max'], width=bar_width, label=f"{label_intel} Max", color='#8ff0a4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel['avg'], width=bar_width, label=f"{label_intel} Avg", color='#33d17a', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel['min'], width=bar_width, label=f"{label_intel} Min", color='#26a269', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_old['max'], width=bar_width, label=f"{label_apple} Old Max", color='#f9f06b', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_old['avg'], width=bar_width, label=f"{label_apple} Old Avg", color='#f6d32d', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_old['min'], width=bar_width, label=f"{label_apple} Old Min", color='#e5a50a', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple['max'], width=bar_width, label=f"{label_apple} Max", color='#ffbe6f', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple['avg'], width=bar_width, label=f"{label_apple} Avg", color='#ff7800', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple['min'], width=bar_width, label=f"{label_apple} Min", color='#c64600', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_old['max'], width=bar_width, label=f"{label_amd} Old Max", color='#f66151', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_old['avg'], width=bar_width, label=f"{label_amd} Old Avg", color='#e01b24', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_old['min'], width=bar_width, label=f"{label_amd} Old Min", color='#a51d2d', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd['max'], width=bar_width, label=f"{label_amd} Max", color='#dc8add', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd['avg'], width=bar_width, label=f"{label_amd} Avg", color='#9141ac', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd['min'], width=bar_width, label=f"{label_amd} Min", color='#613583', edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("SIMD", fontsize=12, color=foreground_color, loc='center')
        case "simd_xl":
            group_width = bar_width * 2 # Space occupied by one group of bars
            group_gap = 0.15 # Gap between groups
            indices = np.arange(len(threads_omp)) * (group_width + group_gap) # Indices for each group

            ax.bar(indices - 0.5 * bar_width, df_intel_xl_old['max'], width=bar_width, label=f"{label_intel} Old XL Max", color='#99c1f1', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_xl_old['avg'], width=bar_width, label=f"{label_intel} Old XL Avg", color='#3584e4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_xl_old['min'], width=bar_width, label=f"{label_intel} Old XL Min", color='#1a5fb4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_xl['max'], width=bar_width, label=f"{label_intel} XL Max", color='#8ff0a4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_xl['avg'], width=bar_width, label=f"{label_intel} XL Avg", color='#33d17a', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_xl['min'], width=bar_width, label=f"{label_intel} XL Min", color='#26a269', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_xl_old['max'], width=bar_width, label=f"{label_apple} Old XL Max", color='#f9f06b', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_xl_old['avg'], width=bar_width, label=f"{label_apple} Old XL Avg", color='#f6d32d', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_xl_old['min'], width=bar_width, label=f"{label_apple} Old XL Min", color='#e5a50a', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_xl['max'], width=bar_width, label=f"{label_apple} XL Max", color='#ffbe6f', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_xl['avg'], width=bar_width, label=f"{label_apple} XL Avg", color='#ff7800', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_xl['min'], width=bar_width, label=f"{label_apple} XL Min", color='#c64600', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_xl_old['max'], width=bar_width, label=f"{label_amd} Old XL Max", color='#f66151', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_xl_old['avg'], width=bar_width, label=f"{label_amd} Old XL Avg", color='#e01b24', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_xl_old['min'], width=bar_width, label=f"{label_amd} Old XL Min", color='#a51d2d', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_xl['max'], width=bar_width, label=f"{label_amd} XL Max", color='#dc8add', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_xl['avg'], width=bar_width, label=f"{label_amd} XL Avg", color='#9141ac', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_xl['min'], width=bar_width, label=f"{label_amd} XL Min", color='#613583', edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("SIMD XL", fontsize=12, color=foreground_color, loc='center')
        case "omp":
            group_width = bar_width * 2 # Space occupied by one group of bars
            group_gap = 0.15 # Gap between groups
            indices = np.arange(len(threads_omp)) * (group_width + group_gap) # Indices for each group

            ax.bar(indices - 0.5 * bar_width, df_intel['max'], width=bar_width, label=f"{label_intel} Max", color='#99c1f1', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel['avg'], width=bar_width, label=f"{label_intel} Avg", color='#3584e4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel['min'], width=bar_width, label=f"{label_intel} Min", color='#1a5fb4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_omp['max'], width=bar_width, label=f"{label_intel} OpenMP Max", color='#8ff0a4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_omp['avg'], width=bar_width, label=f"{label_intel} OpenMP Avg", color='#33d17a', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_omp['min'], width=bar_width, label=f"{label_intel} OpenMP Min", color='#26a269', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple['max'], width=bar_width, label=f"{label_apple} Max", color='#f9f06b', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple['avg'], width=bar_width, label=f"{label_apple} Avg", color='#f6d32d', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple['min'], width=bar_width, label=f"{label_apple} Min", color='#e5a50a', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_omp['max'], width=bar_width, label=f"{label_apple} OpenMP Max", color='#ffbe6f', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_omp['avg'], width=bar_width, label=f"{label_apple} OpenMP Avg", color='#ff7800', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_omp['min'], width=bar_width, label=f"{label_apple} OpenMP Min", color='#c64600', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd['max'], width=bar_width, label=f"{label_amd} Max", color='#f66151', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd['avg'], width=bar_width, label=f"{label_amd} Avg", color='#e01b24', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd['min'], width=bar_width, label=f"{label_amd} Min", color='#a51d2d', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_omp['max'], width=bar_width, label=f"{label_amd} OpenMP Max", color='#dc8add', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_omp['avg'], width=bar_width, label=f"{label_amd} OpenMP Avg", color='#9141ac', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_omp['min'], width=bar_width, label=f"{label_amd} OpenMP Min", color='#613583', edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("OpenMP", fontsize=12, color=foreground_color, loc='center')
        case "omp_xl":
            group_width = bar_width * 2 # Space occupied by one group of bars
            group_gap = 0.15 # Gap between groups
            indices = np.arange(len(threads_omp)) * (group_width + group_gap) # Indices for each group

            ax.bar(indices - 0.5 * bar_width, df_intel_xl['max'], width=bar_width, label=f"{label_intel} XL Max", color='#99c1f1', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_xl['avg'], width=bar_width, label=f"{label_intel} XL Avg", color='#3584e4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_xl['min'], width=bar_width, label=f"{label_intel} XL Min", color='#1a5fb4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_omp_xl['max'], width=bar_width, label=f"{label_intel} OpenMP XL Max", color='#8ff0a4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_omp_xl['avg'], width=bar_width, label=f"{label_intel} OpenMP XL Avg", color='#33d17a', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_omp_xl['min'], width=bar_width, label=f"{label_intel} OpenMP XL Min", color='#26a269', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_xl['max'], width=bar_width, label=f"{label_apple} XL Max", color='#f9f06b', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_xl['avg'], width=bar_width, label=f"{label_apple} XL Avg", color='#f6d32d', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_xl['min'], width=bar_width, label=f"{label_apple} XL Min", color='#e5a50a', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_omp_xl['max'], width=bar_width, label=f"{label_apple} OpenMP XL Max", color='#ffbe6f', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_omp_xl['avg'], width=bar_width, label=f"{label_apple} OpenMP XL Avg", color='#ff7800', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_omp_xl['min'], width=bar_width, label=f"{label_apple} OpenMP XL Min", color='#c64600', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_xl['max'], width=bar_width, label=f"{label_amd} XL Max", color='#f66151', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_xl['avg'], width=bar_width, label=f"{label_amd} XL Avg", color='#e01b24', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_xl['min'], width=bar_width, label=f"{label_amd} XL Min", color='#a51d2d', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_omp_xl['max'], width=bar_width, label=f"{label_amd} OpenMP XL Max", color='#dc8add', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_omp_xl['avg'], width=bar_width, label=f"{label_amd} OpenMP XL Avg", color='#9141ac', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_omp_xl['min'], width=bar_width, label=f"{label_amd} OpenMP XL Min", color='#613583', edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("OpenMP XL", fontsize=12, color=foreground_color, loc='center')
        case "simd_intel":
            threads_omp.pop(1)
            indices = np.delete(indices, 2)

            bar_width = 0.075

            group_width = bar_width * 2 # Space occupied by one group of bars
            group_gap = 0.15 # Gap between groups
            indices = np.arange(len(threads_omp)) * (group_width + group_gap) # Indices for each group

            df_intel_intel_old.drop(df_intel_intel_old.index[1], inplace=True)
            df_intel_intel_old.reset_index(drop=True, inplace=True)
            df_intel_intel.drop(df_intel_intel.index[1], inplace=True)
            df_intel_intel.reset_index(drop=True, inplace=True)
            df_amd_intel_old.drop(df_amd_intel_old.index[1], inplace=True)
            df_amd_intel_old.reset_index(drop=True, inplace=True)
            df_amd_intel.drop(df_amd_intel.index[1], inplace=True)
            df_amd_intel.reset_index(drop=True, inplace=True)

            ax.bar(indices - 0.5 * bar_width, df_intel_intel_old['max'], width=bar_width, label=f"{label_intel} Old ICPX Max", color='#99c1f1', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_intel_old['avg'], width=bar_width, label=f"{label_intel} Old ICPX Avg", color='#3584e4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_intel_old['min'], width=bar_width, label=f"{label_intel} Old ICPX Min", color='#1a5fb4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_intel['max'], width=bar_width, label=f"{label_intel} ICPX Max", color='#8ff0a4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_intel['avg'], width=bar_width, label=f"{label_intel} ICPX Avg", color='#33d17a', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_intel['min'], width=bar_width, label=f"{label_intel} ICPX Min", color='#26a269', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_intel_old['max'], width=bar_width, label=f"{label_amd} Old ICPX Max", color='#f66151', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_intel_old['avg'], width=bar_width, label=f"{label_amd} Old ICPX Avg", color='#e01b24', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_intel_old['min'], width=bar_width, label=f"{label_amd} Old ICPX Min", color='#a51d2d', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_intel['max'], width=bar_width, label=f"{label_amd} ICPX Max", color='#dc8add', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_intel['avg'], width=bar_width, label=f"{label_amd} ICPX Avg", color='#9141ac', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_intel['min'], width=bar_width, label=f"{label_amd} ICPX Min", color='#613583', edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("ICPX SIMD", fontsize=12, color=foreground_color, loc='center')
        case "simd_xl_intel":
            bar_width = 0.075

            group_width = bar_width * 2 # Space occupied by one group of bars
            group_gap = 0.15 # Gap between groups
            indices = np.arange(len(threads_omp)) * (group_width + group_gap) # Indices for each group

            df_intel_xl_intel_old.drop(df_intel_xl_intel_old.index[1], inplace=True)
            df_intel_xl_intel_old.reset_index(drop=True, inplace=True)
            df_intel_xl_intel.drop(df_intel_xl_intel.index[1], inplace=True)
            df_intel_xl_intel.reset_index(drop=True, inplace=True)
            df_amd_xl_intel_old.drop(df_amd_xl_intel_old.index[1], inplace=True)
            df_amd_xl_intel_old.reset_index(drop=True, inplace=True)
            df_amd_xl_intel.drop(df_amd_xl_intel.index[1], inplace=True)
            df_amd_xl_intel.reset_index(drop=True, inplace=True)

            ax.bar(indices - 0.5 * bar_width, df_intel_xl_intel_old['max'], width=bar_width, label=f"{label_intel} Old ICPX XL Max", color='#99c1f1', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_xl_intel_old['avg'], width=bar_width, label=f"{label_intel} Old ICPX XL Avg", color='#3584e4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_xl_intel_old['min'], width=bar_width, label=f"{label_intel} Old ICPX XL Min", color='#1a5fb4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_xl_intel['max'], width=bar_width, label=f"{label_intel} ICPX XL Max", color='#8ff0a4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_xl_intel['avg'], width=bar_width, label=f"{label_intel} ICPX XL Avg", color='#33d17a', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_xl_intel['min'], width=bar_width, label=f"{label_intel} ICPX XL Min", color='#26a269', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_xl_intel_old['max'], width=bar_width, label=f"{label_amd} Old ICPX XL Max", color='#f66151', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_xl_intel_old['avg'], width=bar_width, label=f"{label_amd} Old ICPX XL Avg", color='#e01b24', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_xl_intel_old['min'], width=bar_width, label=f"{label_amd} Old ICPX XL Min", color='#a51d2d', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_xl_intel['max'], width=bar_width, label=f"{label_amd} ICPX XL Max", color='#dc8add', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_xl_intel['avg'], width=bar_width, label=f"{label_amd} ICPX XL Avg", color='#9141ac', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_xl_intel['min'], width=bar_width, label=f"{label_amd} ICPX XL Min", color='#613583', edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("ICPX SIMD XL", fontsize=12, color=foreground_color, loc='center')
        case "omp_intel":
            bar_width = 0.075

            group_width = bar_width * 2 # Space occupied by one group of bars
            group_gap = 0.15 # Gap between groups
            indices = np.arange(len(threads_omp)) * (group_width + group_gap) # Indices for each group

            df_intel_omp.drop(df_intel_omp.index[1], inplace=True)
            df_intel_omp.reset_index(drop=True, inplace=True)
            df_intel_omp_intel.drop(df_intel_omp_intel.index[1], inplace=True)
            df_intel_omp_intel.reset_index(drop=True, inplace=True)
            df_amd_omp.drop(df_amd_omp.index[1], inplace=True)
            df_amd_omp.reset_index(drop=True, inplace=True)
            df_amd_omp_intel.drop(df_amd_omp_intel.index[1], inplace=True)
            df_amd_omp_intel.reset_index(drop=True, inplace=True)

            ax.bar(indices - 0.5 * bar_width, df_intel_omp['max'], width=bar_width, label=f"{label_intel} OpenMP Max", color='#99c1f1', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_omp['avg'], width=bar_width, label=f"{label_intel} OpenMP Avg", color='#3584e4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_omp['min'], width=bar_width, label=f"{label_intel} OpenMP Min", color='#1a5fb4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_omp_intel['max'], width=bar_width, label=f"{label_intel} ICPX OpenMP Max", color='#8ff0a4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_omp_intel['avg'], width=bar_width, label=f"{label_intel} ICPX OpenMP Avg", color='#33d17a', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_omp_intel['min'], width=bar_width, label=f"{label_intel} ICPX OpenMP Min", color='#26a269', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_omp['max'], width=bar_width, label=f"{label_amd} OpenMP Max", color='#f66151', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_omp['avg'], width=bar_width, label=f"{label_amd} OpenMP Avg", color='#e01b24', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_omp['min'], width=bar_width, label=f"{label_amd} OpenMP Min", color='#a51d2d', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_omp_intel['max'], width=bar_width, label=f"{label_amd} ICPX OpenMP Max", color='#dc8add', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_omp_intel['avg'], width=bar_width, label=f"{label_amd} ICPX OpenMP Avg", color='#9141ac', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_omp_intel['min'], width=bar_width, label=f"{label_amd} ICPX OpenMP Min", color='#613583', edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("ICPX OpenMP", fontsize=12, color=foreground_color, loc='center')
        case "omp_xl_intel":
            bar_width = 0.075

            group_width = bar_width * 2 # Space occupied by one group of bars
            group_gap = 0.15 # Gap between groups
            indices = np.arange(len(threads_omp)) * (group_width + group_gap) # Indices for each group

            df_intel_omp_xl.drop(df_intel_omp_xl.index[1], inplace=True)
            df_intel_omp.reset_index(drop=True, inplace=True)
            df_intel_omp_xl_intel.drop(df_intel_omp_xl_intel.index[1], inplace=True)
            df_intel_omp_xl_intel.reset_index(drop=True, inplace=True)
            df_amd_omp_xl.drop(df_amd_omp_xl.index[1], inplace=True)
            df_amd_omp_xl.reset_index(drop=True, inplace=True)
            df_amd_omp_xl_intel.drop(df_amd_omp_xl_intel.index[1], inplace=True)
            df_amd_omp_xl_intel.reset_index(drop=True, inplace=True)

            ax.bar(indices - 0.5 * bar_width, df_intel_omp_xl['max'], width=bar_width, label=f"{label_intel} OpenMP XL Max", color='#99c1f1', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_omp_xl['avg'], width=bar_width, label=f"{label_intel} OpenMP XL Avg", color='#3584e4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_omp_xl['min'], width=bar_width, label=f"{label_intel} OpenMP XL Min", color='#1a5fb4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_omp_xl_intel['max'], width=bar_width, label=f"{label_intel} ICPX OpenMP XL Max", color='#8ff0a4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_omp_xl_intel['avg'], width=bar_width, label=f"{label_intel} ICPX OpenMP XL Avg", color='#33d17a', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_omp_xl_intel['min'], width=bar_width, label=f"{label_intel} ICPX OpenMP XL Min", color='#26a269', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_omp_xl['max'], width=bar_width, label=f"{label_amd} OpenMP XL Max", color='#f66151', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_omp_xl['avg'], width=bar_width, label=f"{label_amd} OpenMP XL Avg", color='#e01b24', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_omp_xl['min'], width=bar_width, label=f"{label_amd} OpenMP XL Min", color='#a51d2d', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_omp_xl_intel['max'], width=bar_width, label=f"{label_amd} ICPX OpenMP XL Max", color='#dc8add', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_omp_xl_intel['avg'], width=bar_width, label=f"{label_amd} ICPX OpenMP XL Avg", color='#9141ac', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_omp_xl_intel['min'], width=bar_width, label=f"{label_amd} ICPX OpenMP XL Min", color='#613583', edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("ICPX OpenMP XL", fontsize=12, color=foreground_color, loc='center')
        case "cuda":
            bar_width = 0.075

            group_width = bar_width * 2 # Space occupied by one group of bars
            group_gap = 0.15 # Gap between groups
            indices = np.arange(len(threads_omp)) * (group_width + group_gap) # Indices for each group

            df_intel_nvidia_old.drop(df_intel_nvidia_old.index[1], inplace=True)
            df_intel_nvidia_old.reset_index(drop=True, inplace=True)
            df_intel_nvidia.drop(df_intel_nvidia.index[1], inplace=True)
            df_intel_nvidia.reset_index(drop=True, inplace=True)
            df_amd_nvidia_old.drop(df_amd_nvidia_old.index[1], inplace=True)
            df_amd_nvidia_old.reset_index(drop=True, inplace=True)
            df_amd_nvidia.drop(df_amd_nvidia.index[1], inplace=True)
            df_amd_nvidia.reset_index(drop=True, inplace=True)

            ax.bar(indices - 0.5 * bar_width, df_intel_nvidia_old['max'], width=bar_width, label=f"{label_intel_nvidia} Old Max", color='#99c1f1', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_nvidia_old['avg'], width=bar_width, label=f"{label_intel_nvidia} Old Avg", color='#3584e4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_nvidia_old['min'], width=bar_width, label=f"{label_intel_nvidia} Old Min", color='#1a5fb4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_nvidia['max'], width=bar_width, label=f"{label_intel_nvidia} Max", color='#8ff0a4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_nvidia['avg'], width=bar_width, label=f"{label_intel_nvidia} Avg", color='#33d17a', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_nvidia['min'], width=bar_width, label=f"{label_intel_nvidia} Min", color='#26a269', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_nvidia_old['max'], width=bar_width, label=f"{label_amd_nvidia} Old Max", color='#f66151', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_nvidia_old['avg'], width=bar_width, label=f"{label_amd_nvidia} Old Avg", color='#e01b24', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_nvidia_old['min'], width=bar_width, label=f"{label_amd_nvidia} Old Min", color='#a51d2d', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_nvidia['max'], width=bar_width, label=f"{label_amd_nvidia} Max", color='#dc8add', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_nvidia['avg'], width=bar_width, label=f"{label_amd_nvidia} Avg", color='#9141ac', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_nvidia['min'], width=bar_width, label=f"{label_amd_nvidia} Min", color='#613583', edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("CUDA", fontsize=12, color=foreground_color, loc='center')

            b1 = df_intel_nvidia_old['avg'][0] + df_amd_nvidia_old['avg'][1]
            b2 = df_intel_nvidia["avg"][0] + df_amd_nvidia['avg'][1]

            print(f"cuda: {np.round((b2 - b1) / b1 * 100, 1)}")
            print(f'intel: {np.round((df_intel_nvidia["avg"][0] - df_intel_nvidia_old["avg"][0]) / df_intel_nvidia_old["avg"][0] * 100, 1)}')
            print(f'amd: {np.round((df_amd_nvidia["avg"][1] - df_amd_nvidia_old["avg"][1]) / df_amd_nvidia_old["avg"][1] * 100, 1)}')
            print(f"{label_intel_nvidia} \\\\")
            print(f'\\hspace{{0.5cm}}Old & {df_intel_nvidia_old["min"][0]} & {df_intel_nvidia_old["avg"][0]} & {df_intel_nvidia_old["max"][0]} & {df_intel_nvidia_old["std"][0]} \\\\')
            print(f'\\hspace{{0.5cm}}New & {df_intel_nvidia["min"][0]} & {df_intel_nvidia["avg"][0]} & {df_intel_nvidia["max"][0]} & {df_intel_nvidia["std"][0]} \\\\')
            print(f"{label_amd_nvidia} \\\\")
            print(f'\\hspace{{0.5cm}}Old & {df_amd_nvidia_old["min"][1]} & {df_amd_nvidia_old["avg"][1]} & {df_amd_nvidia_old["max"][1]} & {df_amd_nvidia_old["std"][1]} \\\\')
            print(f'\\hspace{{0.5cm}}New & {df_amd_nvidia["min"][1]} & {df_amd_nvidia["avg"][1]} & {df_amd_nvidia["max"][1]} & {df_amd_nvidia["std"][1]} \\\\')
        case "cuda_xl":
            bar_width = 0.075

            group_width = bar_width * 2 # Space occupied by one group of bars
            group_gap = 0.15 # Gap between groups
            indices = np.arange(len(threads_omp)) * (group_width + group_gap) # Indices for each group

            df_intel_xl_nvidia_old.drop(df_intel_xl_nvidia_old.index[1], inplace=True)
            df_intel_xl_nvidia_old.reset_index(drop=True, inplace=True)
            df_intel_xl_nvidia.drop(df_intel_xl_nvidia.index[1], inplace=True)
            df_intel_xl_nvidia.reset_index(drop=True, inplace=True)
            df_amd_xl_nvidia_old.drop(df_amd_xl_nvidia_old.index[1], inplace=True)
            df_amd_xl_nvidia_old.reset_index(drop=True, inplace=True)
            df_amd_xl_nvidia.drop(df_amd_xl_nvidia.index[1], inplace=True)
            df_amd_xl_nvidia.reset_index(drop=True, inplace=True)

            ax.bar(indices - 0.5 * bar_width, df_intel_xl_nvidia_old['max'], width=bar_width, label=f"{label_intel_nvidia} Old XL Max", color='#99c1f1', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_xl_nvidia_old['avg'], width=bar_width, label=f"{label_intel_nvidia} Old XL Avg", color='#3584e4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_xl_nvidia_old['min'], width=bar_width, label=f"{label_intel_nvidia} Old XL Min", color='#1a5fb4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_xl_nvidia['max'], width=bar_width, label=f"{label_intel_nvidia} XL Max", color='#8ff0a4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_xl_nvidia['avg'], width=bar_width, label=f"{label_intel_nvidia} XL Avg", color='#33d17a', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_xl_nvidia['min'], width=bar_width, label=f"{label_intel_nvidia} XL Min", color='#26a269', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_xl_nvidia_old['max'], width=bar_width, label=f"{label_amd_nvidia} Old XL Max", color='#f66151', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_xl_nvidia_old['avg'], width=bar_width, label=f"{label_amd_nvidia} Old XL Avg", color='#e01b24', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_xl_nvidia_old['min'], width=bar_width, label=f"{label_amd_nvidia} Old XL Min", color='#a51d2d', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_xl_nvidia['max'], width=bar_width, label=f"{label_amd_nvidia} XL Max", color='#dc8add', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_xl_nvidia['avg'], width=bar_width, label=f"{label_amd_nvidia} XL Avg", color='#9141ac', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_xl_nvidia['min'], width=bar_width, label=f"{label_amd_nvidia} XL Min", color='#613583', edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("CUDA XL", fontsize=12, color=foreground_color, loc='center')

            b1 = df_intel_xl_nvidia_old['avg'][0] + df_amd_xl_nvidia_old['avg'][1]
            b2 = df_intel_xl_nvidia["avg"][0] + df_amd_xl_nvidia['avg'][1]

            print(f"cuda_xl: {np.round((b2 - b1) / b1 * 100, 1)}")
            print(f'intel: {np.round((df_intel_xl_nvidia["avg"][0] - df_intel_xl_nvidia_old["avg"][0]) / df_intel_xl_nvidia_old["avg"][0] * 100, 1)}')
            print(f'amd: {np.round((df_amd_xl_nvidia["avg"][1] - df_amd_xl_nvidia_old["avg"][1]) / df_amd_xl_nvidia_old["avg"][1] * 100, 1)}')
            print(f"{label_intel_nvidia} \\\\")
            print(f'\\hspace{{0.5cm}}Old XL & {df_intel_xl_nvidia_old["min"][0]} & {df_intel_xl_nvidia_old["avg"][0]} & {df_intel_xl_nvidia_old["max"][0]} & {df_intel_xl_nvidia_old["std"][0]} \\\\')
            print(f'\\hspace{{0.5cm}}New XL & {df_intel_xl_nvidia["min"][0]} & {df_intel_xl_nvidia["avg"][0]} & {df_intel_xl_nvidia["max"][0]} & {df_intel_xl_nvidia["std"][0]} \\\\')
            print(f"{label_amd_nvidia} \\\\")
            print(f'\\hspace{{0.5cm}}Old XL & {df_amd_xl_nvidia_old["min"][1]} & {df_amd_xl_nvidia_old["avg"][1]} & {df_amd_xl_nvidia_old["max"][1]} & {df_amd_xl_nvidia_old["std"][1]} \\\\')
            print(f'\\hspace{{0.5cm}}New XL & {df_amd_xl_nvidia["min"][1]} & {df_amd_xl_nvidia["avg"][1]} & {df_amd_xl_nvidia["max"][1]} & {df_amd_xl_nvidia["std"][1]} \\\\')
        case "cpu_gpu":
            bar_width = 0.075

            group_width = bar_width * 2 # Space occupied by one group of bars
            group_gap = 0.15 # Gap between groups
            indices = np.arange(len(threads_omp)) * (group_width + group_gap) # Indices for each group

            df_intel.drop(df_intel.index[1], inplace=True)
            df_intel.reset_index(drop=True, inplace=True)
            df_amd.drop(df_amd.index[1], inplace=True)
            df_amd.reset_index(drop=True, inplace=True)

            ax.bar(indices - 0.5 * bar_width, df_intel['max'], width=bar_width, label=f"{label_intel} Max", color='#99c1f1', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel['avg'], width=bar_width, label=f"{label_intel} Avg", color='#3584e4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel['min'], width=bar_width, label=f"{label_intel} Min", color='#1a5fb4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_nvidia['max'], width=bar_width, label=f"{label_intel_nvidia} Max", color='#8ff0a4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_nvidia['avg'], width=bar_width, label=f"{label_intel_nvidia} Avg", color='#33d17a', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_nvidia['min'], width=bar_width, label=f"{label_intel_nvidia} Min", color='#26a269', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd['max'], width=bar_width, label=f"{label_amd} Max", color='#f66151', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd['avg'], width=bar_width, label=f"{label_amd} Avg", color='#e01b24', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd['min'], width=bar_width, label=f"{label_amd} Min", color='#a51d2d', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_nvidia['max'], width=bar_width, label=f"{label_amd_nvidia} Max", color='#dc8add', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_nvidia['avg'], width=bar_width, label=f"{label_amd_nvidia} Avg", color='#9141ac', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_nvidia['min'], width=bar_width, label=f"{label_amd_nvidia} Min", color='#613583', edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("CPU vs. GPU", fontsize=12, color=foreground_color, loc='center')
        case "cpu_gpu_xl":
            bar_width = 0.075

            group_width = bar_width * 2 # Space occupied by one group of bars
            group_gap = 0.15 # Gap between groups
            indices = np.arange(len(threads_omp)) * (group_width + group_gap) # Indices for each group

            df_intel_xl.drop(df_intel_xl.index[1], inplace=True)
            df_intel_xl.reset_index(drop=True, inplace=True)
            df_amd_xl.drop(df_amd_xl.index[1], inplace=True)
            df_amd_xl.reset_index(drop=True, inplace=True)

            ax.bar(indices - 0.5 * bar_width, df_intel_xl['max'], width=bar_width, label=f"{label_intel} XL Max", color='#99c1f1', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_xl['avg'], width=bar_width, label=f"{label_intel} XL Avg", color='#3584e4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_xl['min'], width=bar_width, label=f"{label_intel} XL Min", color='#1a5fb4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_xl_nvidia['max'], width=bar_width, label=f"{label_intel_nvidia} XL Max", color='#8ff0a4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_xl_nvidia['avg'], width=bar_width, label=f"{label_intel_nvidia} XL Avg", color='#33d17a', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_xl_nvidia['min'], width=bar_width, label=f"{label_intel_nvidia} XL Min", color='#26a269', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_xl['max'], width=bar_width, label=f"{label_amd} XL Max", color='#f66151', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_xl['avg'], width=bar_width, label=f"{label_amd} XL Avg", color='#e01b24', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_xl['min'], width=bar_width, label=f"{label_amd} XL Min", color='#a51d2d', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_xl_nvidia['max'], width=bar_width, label=f"{label_amd_nvidia} XL Max", color='#dc8add', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_xl_nvidia['avg'], width=bar_width, label=f"{label_amd_nvidia} XL Avg", color='#9141ac', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_xl_nvidia['min'], width=bar_width, label=f"{label_amd_nvidia} XL Min", color='#613583', edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("CPU vs. GPU XL", fontsize=12, color=foreground_color, loc='center')
        case "amx":
            bar_width = 0.075

            group_width = bar_width * 2 # Space occupied by one group of bars
            group_gap = 0.15 # Gap between groups
            indices = np.arange(len(threads_omp)) * (group_width + group_gap) # Indices for each group

            df_apple_amx.drop(df_apple_amx.index[2], inplace=True)
            df_apple_amx.reset_index(drop=True, inplace=True)
            df_apple_neon.drop(df_apple_neon.index[2], inplace=True)
            df_apple_neon.reset_index(drop=True, inplace=True)

            ax.bar(indices - 0.5 * bar_width, df_apple_amx['max'], width=bar_width, label=f"{label_apple} AMX Max", color='#99c1f1', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_amx['avg'], width=bar_width, label=f"{label_apple} AMX Avg", color='#3584e4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_amx['min'], width=bar_width, label=f"{label_apple} AMX Min", color='#1a5fb4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_neon['max'], width=bar_width, label=f"{label_apple} Neon Max", color='#8ff0a4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_neon['avg'], width=bar_width, label=f"{label_apple} Neon Avg", color='#33d17a', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_neon['min'], width=bar_width, label=f"{label_apple} Neon Min", color='#26a269', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_amx['max'], width=bar_width, label=f"{label_apple} AMX Max", color='#99c1f1', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_amx['avg'], width=bar_width, label=f"{label_apple} AMX Avg", color='#3584e4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_amx['min'], width=bar_width, label=f"{label_apple} AMX Min", color='#1a5fb4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_neon['max'], width=bar_width, label=f"{label_apple} Neon Max", color='#8ff0a4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_neon['avg'], width=bar_width, label=f"{label_apple} Neon Avg", color='#33d17a', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_neon['min'], width=bar_width, label=f"{label_apple} Neon Min", color='#26a269', edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("AMX vs. Neon", fontsize=12, color=foreground_color, loc='center')
        case "amx_xl":
            bar_width = 0.075

            group_width = bar_width * 2 # Space occupied by one group of bars
            group_gap = 0.15 # Gap between groups
            indices = np.arange(len(threads_omp)) * (group_width + group_gap) # Indices for each group

            df_apple_amx_xl.drop(df_apple_amx_xl.index[2], inplace=True)
            df_apple_amx_xl.reset_index(drop=True, inplace=True)
            df_apple_neon_xl.drop(df_apple_neon_xl.index[2], inplace=True)
            df_apple_neon_xl.reset_index(drop=True, inplace=True)

            ax.bar(indices - 0.5 * bar_width, df_apple_amx_xl['max'], width=bar_width, label=f"{label_apple} AMX XL Max", color='#99c1f1', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_amx_xl['avg'], width=bar_width, label=f"{label_apple} AMX XL Avg", color='#3584e4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_amx_xl['min'], width=bar_width, label=f"{label_apple} AMX XL Min", color='#1a5fb4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_neon_xl['max'], width=bar_width, label=f"{label_apple} Neon XL Max", color='#8ff0a4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_neon_xl['avg'], width=bar_width, label=f"{label_apple} Neon XL Avg", color='#33d17a', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_neon_xl['min'], width=bar_width, label=f"{label_apple} Neon XL Min", color='#26a269', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_amx_xl['max'], width=bar_width, label=f"{label_apple} AMX XL Max", color='#99c1f1', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_amx_xl['avg'], width=bar_width, label=f"{label_apple} AMX XL Avg", color='#3584e4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_amx_xl['min'], width=bar_width, label=f"{label_apple} AMX XL Min", color='#1a5fb4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_neon_xl['max'], width=bar_width, label=f"{label_apple} Neon XL Max", color='#8ff0a4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_neon_xl['avg'], width=bar_width, label=f"{label_apple} Neon XL Avg", color='#33d17a', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_neon_xl['min'], width=bar_width, label=f"{label_apple} Neon XL Min", color='#26a269', edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("AMX vs. Neon XL", fontsize=12, color=foreground_color, loc='center')

    ax.set_xlabel("Threads", fontsize=12, color=foreground_color)
    if("cuda" in f):
        ax.set_xlabel("Device", fontsize=12, color=foreground_color)
    ax.set_ylabel("Total time (µs)", fontsize=12, color=foreground_color)
    ax.set_xticks(indices)
    ax.set_xticklabels(threads_omp, color=foreground_color)
    if("cuda" in f):
        ax.set_xticklabels(["Laptop", "Desktop"], color=foreground_color)
    if("amx" in f):
        ax.set_xlabel("Threads", fontsize=12, color=foreground_color)
        ax.set_xticklabels(["1", "11"], color=foreground_color)
    ax.tick_params(axis='x', colors=foreground_color)  # X-axis tick marks and labels
    ax.tick_params(axis='y', colors=foreground_color)  # Y-axis tick marks and labels
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), facecolor=background_color, edgecolor=foreground_color, labelcolor=foreground_color)

    # Make spines (box edges)
    for spine in ax.spines.values():
        spine.set_edgecolor(foreground_color)

    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    plt.tight_layout()

    # ensure the directory exists
    try:
        os.mkdir("../Graphs/5. Presentation")
    except:
        pass

    match f:
        case "int":
            plt.savefig("../Graphs/5. Presentation/Quantization.png", dpi=300)
        case "int_xl":
            plt.savefig("../Graphs/5. Presentation/Quantization XL.png", dpi=300)
        case "simd":
            plt.savefig("../Graphs/5. Presentation/SIMD.png", dpi=300)
        case "simd_xl":
            plt.savefig("../Graphs/5. Presentation/SIMD XL.png", dpi=300)
        case "omp":
            plt.savefig("../Graphs/5. Presentation/OpenMP.png", dpi=300)
        case "omp_xl":
            plt.savefig("../Graphs/5. Presentation/OpenMP XL.png", dpi=300)
        case "simd_intel":
            plt.savefig("../Graphs/5. Presentation/ICPX SIMD.png", dpi=300)
        case "simd_xl_intel":
            plt.savefig("../Graphs/5. Presentation/ICPX SIMD XL.png", dpi=300)
        case "omp_intel":
            plt.savefig("../Graphs/5. Presentation/ICPX OpenMP.png", dpi=300)
        case "omp_xl_intel":
            plt.savefig("../Graphs/5. Presentation/ICPX OpenMP XL.png", dpi=300)
        case "cuda":
            plt.savefig("../Graphs/5. Presentation/CUDA.png", dpi=300)
        case "cuda_xl":
            plt.savefig("../Graphs/5. Presentation/CUDA XL.png", dpi=300)
        case "cpu_gpu":
            plt.savefig("../Graphs/5. Presentation/CPU vs GPU.png", dpi=300)
        case "cpu_gpu_xl":
            plt.savefig("../Graphs/5. Presentation/CPU vs GPU XL.png", dpi=300)
        case "amx":
            plt.savefig("../Graphs/5. Presentation/AMX vs Neon.png", dpi=300)
        case "amx_xl":
            plt.savefig("../Graphs/5. Presentation/AMX vs Neon XL.png", dpi=300)

    # Show the plot
    # plt.show()
    plt.close(fig)
