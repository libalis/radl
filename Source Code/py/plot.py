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

foreground_color = "#242424"
background_color = "#ffffff"

threads = [1, 4, 5, 8, 11, 16]
threads_omp = [8, 11, 16]

file_paths = {
    "amd": {
        "b_1": "../Archive/1. Presentation/csv/amd/benchmark.csv",
        "intel_1": "../Archive/1. Presentation/csv/amd/benchmark_intel.csv",
        "xl_1": "../Archive/1. Presentation/csv/amd/benchmark_xl.csv",
        "xl_intel_1": "../Archive/1. Presentation/csv/amd/benchmark_xl_intel.csv",
        "b_2": "../Archive/2. Presentation/csv/amd/benchmark.csv",
        "intel_2": "../Archive/2. Presentation/csv/amd/benchmark_intel.csv",
        "xl_2": "../Archive/2. Presentation/csv/amd/benchmark_xl.csv",
        "xl_intel_2": "../Archive/2. Presentation/csv/amd/benchmark_xl_intel.csv",
        "b_3": "./csv/3. Presentation/amd/benchmark.csv",
        "intel_3": "./csv/3. Presentation/amd/benchmark_intel.csv",
        "nvidia_3": "./csv/3. Presentation/amd/benchmark_nvidia.csv",
        "omp_3": "./csv/3. Presentation/amd/benchmark_omp.csv",
        "omp_intel_3": "./csv/3. Presentation/amd/benchmark_omp_intel.csv",
        "omp_xl_3": "./csv/3. Presentation/amd/benchmark_omp_xl.csv",
        "omp_xl_intel_3": "./csv/3. Presentation/amd/benchmark_omp_xl_intel.csv",
        "xl_3": "./csv/3. Presentation/amd/benchmark_xl.csv",
        "xl_intel_3": "./csv/3. Presentation/amd/benchmark_xl_intel.csv",
        "xl_nvidia_3": "./csv/3. Presentation/amd/benchmark_xl_nvidia.csv",
        "b_4": "./csv/4. Presentation/amd/benchmark.csv",
        "intel_4": "./csv/4. Presentation/amd/benchmark_intel.csv",
        "nvidia_4": "./csv/4. Presentation/amd/benchmark_nvidia.csv",
        "omp_4": "./csv/4. Presentation/amd/benchmark_omp.csv",
        "omp_intel_4": "./csv/4. Presentation/amd/benchmark_omp_intel.csv",
        "omp_xl_4": "./csv/4. Presentation/amd/benchmark_omp_xl.csv",
        "omp_xl_intel_4": "./csv/4. Presentation/amd/benchmark_omp_xl_intel.csv",
        "xl_4": "./csv/4. Presentation/amd/benchmark_xl.csv",
        "xl_intel_4": "./csv/4. Presentation/amd/benchmark_xl_intel.csv",
        "xl_nvidia_4": "./csv/4. Presentation/amd/benchmark_xl_nvidia.csv",
        "b_6": "./csv/6. Presentation/amd/benchmark.csv",
        "int_6": "./csv/6. Presentation/amd/benchmark_int.csv",
        "int_xl_6": "./csv/6. Presentation/amd/benchmark_int_xl.csv",
        "intel_6": "./csv/6. Presentation/amd/benchmark_intel.csv",
        "nvidia_6": "./csv/6. Presentation/amd/benchmark_nvidia.csv",
        "omp_6": "./csv/6. Presentation/amd/benchmark_omp.csv",
        "omp_intel_6": "./csv/6. Presentation/amd/benchmark_omp_intel.csv",
        "omp_xl_6": "./csv/6. Presentation/amd/benchmark_omp_xl.csv",
        "omp_xl_intel_6": "./csv/6. Presentation/amd/benchmark_omp_xl_intel.csv",
        "xl_6": "./csv/6. Presentation/amd/benchmark_xl.csv",
        "xl_intel_6": "./csv/6. Presentation/amd/benchmark_xl_intel.csv",
        "xl_nvidia_6": "./csv/6. Presentation/amd/benchmark_xl_nvidia.csv",
        "b_7": "./csv/7. Presentation/amd/benchmark.csv",
        "int_7": "./csv/7. Presentation/amd/benchmark_int.csv",
        "int_xl_7": "./csv/7. Presentation/amd/benchmark_int_xl.csv",
        "intel_7": "./csv/7. Presentation/amd/benchmark_intel.csv",
        "no_simd_7": "./csv/7. Presentation/amd/benchmark_no_simd.csv",
        "no_simd_int_7": "./csv/7. Presentation/amd/benchmark_no_simd_int.csv",
        "no_simd_int_xl_7": "./csv/7. Presentation/amd/benchmark_no_simd_int_xl.csv",
        "no_simd_xl_7": "./csv/7. Presentation/amd/benchmark_no_simd_xl.csv",
        "nvidia_7": "./csv/7. Presentation/amd/benchmark_nvidia.csv",
        "omp_7": "./csv/7. Presentation/amd/benchmark_omp.csv",
        "omp_intel_7": "./csv/7. Presentation/amd/benchmark_omp_intel.csv",
        "omp_xl_7": "./csv/7. Presentation/amd/benchmark_omp_xl.csv",
        "omp_xl_intel_7": "./csv/7. Presentation/amd/benchmark_omp_xl_intel.csv",
        "xl_7": "./csv/7. Presentation/amd/benchmark_xl.csv",
        "xl_intel_7": "./csv/7. Presentation/intel/benchmark_xl_intel.csv",
        "xl_nvidia_7": "./csv/7. Presentation/amd/benchmark_xl_nvidia.csv"
    },
    "apple": {
        "b_1": "../Archive/1. Presentation/csv/apple/benchmark.csv",
        "xl_1": "../Archive/1. Presentation/csv/apple/benchmark_xl.csv",
        "b_2": "../Archive/2. Presentation/csv/apple/benchmark.csv",
        "xl_2": "../Archive/2. Presentation/csv/apple/benchmark_xl.csv",
        "b_3": "./csv/3. Presentation/apple/benchmark.csv",
        "xl_3": "./csv/3. Presentation/apple/benchmark_xl.csv",
        "omp_3": "./csv/3. Presentation/apple/benchmark_omp.csv",
        "omp_xl_3": "./csv/3. Presentation/apple/benchmark_omp_xl.csv",
        "b_4": "./csv/4. Presentation/apple/benchmark.csv",
        "xl_4": "./csv/4. Presentation/apple/benchmark_xl.csv",
        "omp_4": "./csv/4. Presentation/apple/benchmark_omp.csv",
        "omp_xl_4": "./csv/4. Presentation/apple/benchmark_omp_xl.csv",
        "b_6": "./csv/6. Presentation/apple/benchmark.csv",
        "int_6": "./csv/6. Presentation/apple/benchmark_int.csv",
        "int_xl_6": "./csv/6. Presentation/apple/benchmark_int_xl.csv",
        "omp_6": "./csv/6. Presentation/apple/benchmark_omp.csv",
        "omp_xl_6": "./csv/6. Presentation/apple/benchmark_omp_xl.csv",
        "xl_6": "./csv/6. Presentation/apple/benchmark_xl.csv",
        "amx_6": "./csv/6. Presentation/apple/benchmark_amx.csv",
        "amx_int_6": "./csv/6. Presentation/apple/benchmark_amx_int.csv",
        "amx_int_xl_6": "./csv/6. Presentation/apple/benchmark_amx_int_xl.csv",
        "amx_xl_6": "./csv/6. Presentation/apple/benchmark_amx_xl.csv",
        "amx_xxl_6": "./csv/6. Presentation/apple/benchmark_amx_xxl.csv",
        "neon_6": "./csv/6. Presentation/apple/benchmark.csv",
        "neon_xl_6": "./csv/6. Presentation/apple/benchmark_xl.csv",
        "neon_xxl_6": "./csv/6. Presentation/apple/benchmark_neon_xxl.csv",
        "b_7": "./csv/7. Presentation/apple/benchmark.csv",
        "int_7": "./csv/7. Presentation/apple/benchmark_int.csv",
        "int_xl_7": "./csv/7. Presentation/apple/benchmark_int_xl.csv",
        "omp_7": "./csv/7. Presentation/apple/benchmark_omp.csv",
        "omp_xl_7": "./csv/7. Presentation/apple/benchmark_omp_xl.csv",
        "xl_7": "./csv/7. Presentation/apple/benchmark_xl.csv",
        "amx_7": "./csv/7. Presentation/apple/benchmark_amx.csv",
        "amx_int_7": "./csv/7. Presentation/apple/benchmark_amx_int.csv",
        "amx_int_xl_7": "./csv/7. Presentation/apple/benchmark_amx_int_xl.csv",
        "amx_xl_7": "./csv/7. Presentation/apple/benchmark_amx_xl.csv",
        "amx_xxl_7": "./csv/7. Presentation/apple/benchmark_amx_xxl.csv",
        "neon_7": "./csv/7. Presentation/apple/benchmark.csv",
        "neon_xl_7": "./csv/7. Presentation/apple/benchmark_xl.csv",
        "neon_xxl_7": "./csv/7. Presentation/apple/benchmark_neon_xxl.csv",
        "no_neon_7": "./csv/7. Presentation/apple/benchmark_no_simd.csv",
        "no_neon_xl_7": "./csv/7. Presentation/apple/benchmark_no_simd_xl.csv",
        "no_simd_7": "./csv/7. Presentation/apple/benchmark_no_simd.csv",
        "no_simd_int_7": "./csv/7. Presentation/apple/benchmark_no_simd_int.csv",
        "no_simd_int_xl_7": "./csv/7. Presentation/apple/benchmark_no_simd_int_xl.csv",
        "no_simd_xl_7": "./csv/7. Presentation/apple/benchmark_no_simd_xl.csv"
    },
    "intel": {
        "b_1": "../Archive/1. Presentation/csv/intel/benchmark.csv",
        "intel_1": "../Archive/1. Presentation/csv/intel/benchmark_intel.csv",
        "xl_1": "../Archive/1. Presentation/csv/intel/benchmark_xl.csv",
        "xl_intel_1": "../Archive/1. Presentation/csv/intel/benchmark_xl_intel.csv",
        "b_2": "../Archive/2. Presentation/csv/intel/benchmark.csv",
        "intel_2": "../Archive/2. Presentation/csv/intel/benchmark_intel.csv",
        "xl_2": "../Archive/2. Presentation/csv/intel/benchmark_xl.csv",
        "xl_intel_2": "../Archive/2. Presentation/csv/intel/benchmark_xl_intel.csv",
        "b_3": "./csv/3. Presentation/intel/benchmark.csv",
        "intel_3": "./csv/3. Presentation/intel/benchmark_intel.csv",
        "nvidia_3": "./csv/3. Presentation/intel/benchmark_nvidia.csv",
        "omp_3": "./csv/3. Presentation/intel/benchmark_omp.csv",
        "omp_intel_3": "./csv/3. Presentation/intel/benchmark_omp_intel.csv",
        "omp_xl_3": "./csv/3. Presentation/intel/benchmark_omp_xl.csv",
        "omp_xl_intel_3": "./csv/3. Presentation/intel/benchmark_omp_xl_intel.csv",
        "xl_3": "./csv/3. Presentation/intel/benchmark_xl.csv",
        "xl_intel_3": "./csv/3. Presentation/intel/benchmark_xl_intel.csv",
        "xl_nvidia_3": "./csv/3. Presentation/intel/benchmark_xl_nvidia.csv",
        "b_4": "./csv/4. Presentation/intel/benchmark.csv",
        "intel_4": "./csv/4. Presentation/intel/benchmark_intel.csv",
        "nvidia_4": "./csv/4. Presentation/intel/benchmark_nvidia.csv",
        "omp_4": "./csv/4. Presentation/intel/benchmark_omp.csv",
        "omp_intel_4": "./csv/4. Presentation/intel/benchmark_omp_intel.csv",
        "omp_xl_4": "./csv/4. Presentation/intel/benchmark_omp_xl.csv",
        "omp_xl_intel_4": "./csv/4. Presentation/intel/benchmark_omp_xl_intel.csv",
        "xl_4": "./csv/4. Presentation/intel/benchmark_xl.csv",
        "xl_intel_4": "./csv/4. Presentation/intel/benchmark_xl_intel.csv",
        "xl_nvidia_4": "./csv/4. Presentation/intel/benchmark_xl_nvidia.csv",
        "b_6": "./csv/6. Presentation/intel/benchmark.csv",
        "int_6": "./csv/6. Presentation/intel/benchmark_int.csv",
        "int_xl_6": "./csv/6. Presentation/intel/benchmark_int_xl.csv",
        "intel_6": "./csv/6. Presentation/intel/benchmark_intel.csv",
        "nvidia_6": "./csv/6. Presentation/intel/benchmark_nvidia.csv",
        "omp_6": "./csv/6. Presentation/intel/benchmark_omp.csv",
        "omp_intel_6": "./csv/6. Presentation/intel/benchmark_omp_intel.csv",
        "omp_xl_6": "./csv/6. Presentation/intel/benchmark_omp_xl.csv",
        "omp_xl_intel_6": "./csv/6. Presentation/intel/benchmark_omp_xl_intel.csv",
        "xl_6": "./csv/6. Presentation/intel/benchmark_xl.csv",
        "xl_intel_6": "./csv/6. Presentation/intel/benchmark_xl_intel.csv",
        "xl_nvidia_6": "./csv/6. Presentation/intel/benchmark_xl_nvidia.csv",
        "b_7": "./csv/7. Presentation/intel/benchmark.csv",
        "int_7": "./csv/7. Presentation/intel/benchmark_int.csv",
        "int_xl_7": "./csv/7. Presentation/intel/benchmark_int_xl.csv",
        "intel_7": "./csv/7. Presentation/intel/benchmark_intel.csv",
        "no_simd_7": "./csv/7. Presentation/intel/benchmark_no_simd.csv",
        "no_simd_int_7": "./csv/7. Presentation/intel/benchmark_no_simd_int.csv",
        "no_simd_int_xl_7": "./csv/7. Presentation/intel/benchmark_no_simd_int_xl.csv",
        "no_simd_xl_7": "./csv/7. Presentation/intel/benchmark_no_simd_xl.csv",
        "nvidia_7": "./csv/7. Presentation/intel/benchmark_nvidia.csv",
        "omp_7": "./csv/7. Presentation/intel/benchmark_omp.csv",
        "omp_intel_7": "./csv/7. Presentation/intel/benchmark_omp_intel.csv",
        "omp_xl_7": "./csv/7. Presentation/intel/benchmark_omp_xl.csv",
        "omp_xl_intel_7": "./csv/7. Presentation/intel/benchmark_omp_xl_intel.csv",
        "xl_7": "./csv/7. Presentation/intel/benchmark_xl.csv",
        "xl_intel_7": "./csv/7. Presentation/intel/benchmark_xl_intel.csv",
        "xl_nvidia_7": "./csv/7. Presentation/intel/benchmark_xl_nvidia.csv"
    }
}

df_structure = pd.DataFrame({
    "min": [0.0] * len(threads_omp),
    "avg": [0.0] * len(threads_omp),
    "max": [0.0] * len(threads_omp),
    "std": [0.0] * len(threads_omp)
})

df_dict = {}
tt_dict = {}
for group, files in file_paths.items():
    df_dict[group] = {}
    tt_dict[group] = {}
    for key, file_path in files.items():
        df_dict[group][key] = pd.read_csv(file_path)
        tt_dict[group][key] = df_dict[group][key]["total_time_us"].values
        df_dict[group][key] = df_structure.copy(deep=True)

for i in range(len(threads)):
    for group in df_dict.keys():
        for key in df_dict[group].keys():
            if (group == "amd" and threads[i] == 16) or \
               (group == "apple" and threads[i] == 11) or \
               (group == "intel" and threads[i] == 8):
                if "omp" in key or "nvidia" in key:
                    df_dict[group][key].loc[threads_omp.index(threads[i]), "min"] = np.min(tt_dict[group][key][0:10])
                    df_dict[group][key].loc[threads_omp.index(threads[i]), "avg"] = np.average(tt_dict[group][key][0:10])
                    df_dict[group][key].loc[threads_omp.index(threads[i]), "max"] = np.max(tt_dict[group][key][0:10])
                    df_dict[group][key].loc[threads_omp.index(threads[i]), "std"] = np.round(np.std(tt_dict[group][key][0:10]), 1)
                else:
                    df_dict[group][key].loc[threads_omp.index(threads[i]), "min"] = np.min(tt_dict[group][key][10*i:10*(i+1)])
                    df_dict[group][key].loc[threads_omp.index(threads[i]), "avg"] = np.average(tt_dict[group][key][10*i:10*(i+1)])
                    df_dict[group][key].loc[threads_omp.index(threads[i]), "max"] = np.max(tt_dict[group][key][10*i:10*(i+1)])
                    df_dict[group][key].loc[threads_omp.index(threads[i]), "std"] = np.round(np.std(tt_dict[group][key][10*i:10*(i+1)]), 1)
            elif group == "apple" and ("amx" in key or "neon" in key) and threads[i] == 1:
                df_dict[group][key].loc[threads.index(threads[i]), "min"] = np.min(tt_dict[group][key][10*i:10*(i+1)])
                df_dict[group][key].loc[threads.index(threads[i]), "avg"] = np.average(tt_dict[group][key][10*i:10*(i+1)])
                df_dict[group][key].loc[threads.index(threads[i]), "max"] = np.max(tt_dict[group][key][10*i:10*(i+1)])
                df_dict[group][key].loc[threads.index(threads[i]), "std"] = np.round(np.std(tt_dict[group][key][10*i:10*(i+1)]), 1)


try:
    os.mkdir("../Graphs/7. Presentation")
except:
    pass
try:
    os.mkdir("../Graphs/7. Presentation/csv")
except:
    pass

for group, files in file_paths.items():
    for key, file_path in files.items():
        df_dict[group][key].to_csv(f"../Graphs/7. Presentation/csv/{key}_{group}.csv", index=False)

figures = ["naive", "naive_xl", "mt_init", "mt_init_xl", "mt", "mt_xl", "omp", "omp_xl", "sse", "sse_xl", "avx", "avx_xl",\
            "neon", "neon_xl", "amx", "amx_xxl", "int", "int_xl", "icpx", "icpx_xl", "icpx_omp", "icpx_omp_xl", "final", "final_xl",\
            "cuda", "cuda_xl", "cuda_final", "cuda_final_xl", "cpu_gpu", "cpu_gpu_xl"]
for f in figures:
    bar_width = 0.15

    indices = np.arange(len(threads_omp))
    fig, ax = plt.subplots(figsize=(10, 6))

    match f:
        case "naive":
            group_width = bar_width * 2
            group_gap = 0.15
            indices = np.arange(len(threads_omp)) * (group_width + group_gap)

            ax.bar(indices, df_dict["intel"]["b_1"]["max"], width=bar_width, label=f"{label_intel} Naive Max", color="#99c1f1", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_dict["intel"]["b_1"]["avg"], width=bar_width, label=f"{label_intel} Naive Avg", color="#3584e4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_dict["intel"]["b_1"]["min"], width=bar_width, label=f"{label_intel} Naive Min", color="#1a5fb4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_dict["apple"]["b_1"]["max"], width=bar_width, label=f"{label_apple} Naive Max", color="#f9f06b", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_dict["apple"]["b_1"]["avg"], width=bar_width, label=f"{label_apple} Naive Avg", color="#f6d32d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_dict["apple"]["b_1"]["min"], width=bar_width, label=f"{label_apple} Naive Min", color="#e5a50a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_dict["amd"]["b_1"]["max"], width=bar_width, label=f"{label_amd} Naive Max", color="#f66151", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_dict["amd"]["b_1"]["avg"], width=bar_width, label=f"{label_amd} Naive Avg", color="#e01b24", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_dict["amd"]["b_1"]["min"], width=bar_width, label=f"{label_amd} Naive Min", color="#a51d2d", edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("Naive", fontsize=12, color=foreground_color, loc="center")
        case "naive_xl":
            group_width = bar_width * 2
            group_gap = 0.15
            indices = np.arange(len(threads_omp)) * (group_width + group_gap)

            ax.bar(indices, df_dict["intel"]["xl_1"]["max"], width=bar_width, label=f"{label_intel} Naive XL Max", color="#99c1f1", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_dict["intel"]["xl_1"]["avg"], width=bar_width, label=f"{label_intel} Naive XL Avg", color="#3584e4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_dict["intel"]["xl_1"]["min"], width=bar_width, label=f"{label_intel} Naive XL Min", color="#1a5fb4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_dict["apple"]["xl_1"]["max"], width=bar_width, label=f"{label_apple} Naive XL Max", color="#f9f06b", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_dict["apple"]["xl_1"]["avg"], width=bar_width, label=f"{label_apple} Naive XL Avg", color="#f6d32d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_dict["apple"]["xl_1"]["min"], width=bar_width, label=f"{label_apple} Naive XL Min", color="#e5a50a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_dict["amd"]["xl_1"]["max"], width=bar_width, label=f"{label_amd} Naive XL Max", color="#f66151", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_dict["amd"]["xl_1"]["avg"], width=bar_width, label=f"{label_amd} Naive XL Avg", color="#e01b24", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_dict["amd"]["xl_1"]["min"], width=bar_width, label=f"{label_amd} Naive XL Min", color="#a51d2d", edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("Naive XL", fontsize=12, color=foreground_color, loc="center")
        case "mt_init":
            group_width = bar_width * 2
            group_gap = 0.15
            indices = np.arange(len(threads_omp)) * (group_width + group_gap)

            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["b_1"]["max"], width=bar_width, label=f"{label_intel} Naive Max", color="#99c1f1", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["b_1"]["avg"], width=bar_width, label=f"{label_intel} Naive Avg", color="#3584e4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["b_1"]["min"], width=bar_width, label=f"{label_intel} Naive Min", color="#1a5fb4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["b_2"]["max"], width=bar_width, label=f"{label_intel} Initial Multithreading Max", color="#8ff0a4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["b_2"]["avg"], width=bar_width, label=f"{label_intel} Initial Multithreading Avg", color="#33d17a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["b_2"]["min"], width=bar_width, label=f"{label_intel} Initial Multithreading Min", color="#26a269", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["apple"]["b_1"]["max"], width=bar_width, label=f"{label_apple} Naive Max", color="#f9f06b", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["apple"]["b_1"]["avg"], width=bar_width, label=f"{label_apple} Naive Avg", color="#f6d32d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["apple"]["b_1"]["min"], width=bar_width, label=f"{label_apple} Naive Min", color="#e5a50a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["apple"]["b_2"]["max"], width=bar_width, label=f"{label_apple} Initial Multithreading Max", color="#ffbe6f", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["apple"]["b_2"]["avg"], width=bar_width, label=f"{label_apple} Initial Multithreading Avg", color="#ff7800", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["apple"]["b_2"]["min"], width=bar_width, label=f"{label_apple} Initial Multithreading Min", color="#c64600", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["b_1"]["max"], width=bar_width, label=f"{label_amd} Naive Max", color="#f66151", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["b_1"]["avg"], width=bar_width, label=f"{label_amd} Naive Avg", color="#e01b24", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["b_1"]["min"], width=bar_width, label=f"{label_amd} Naive Min", color="#a51d2d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["b_2"]["max"], width=bar_width, label=f"{label_amd} Initial Multithreading Max", color="#dc8add", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["b_2"]["avg"], width=bar_width, label=f"{label_amd} Initial Multithreading Avg", color="#9141ac", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["b_2"]["min"], width=bar_width, label=f"{label_amd} Initial Multithreading Min", color="#613583", edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("Initial Multithreading", fontsize=12, color=foreground_color, loc="center")

            b1 = df_dict["intel"]["b_1"]["avg"][0] + df_dict["apple"]["b_1"]["avg"][1] + df_dict["amd"]["b_1"]["avg"][2]
            b2 = df_dict["intel"]["b_2"]["avg"][0] + df_dict["apple"]["b_2"]["avg"][1] + df_dict["amd"]["b_2"]["avg"][2]

            print(f"mt_init: {np.round((b2 - b1) / b1 * 100, 1)}")
            print(f'intel: {np.round((df_dict["intel"]["b_2"]["avg"][0] - df_dict["intel"]["b_1"]["avg"][0]) / df_dict["intel"]["b_1"]["avg"][0] * 100, 1)}')
            print(f'apple: {np.round((df_dict["apple"]["b_2"]["avg"][1] - df_dict["apple"]["b_1"]["avg"][1]) / df_dict["apple"]["b_1"]["avg"][1] * 100, 1)}')
            print(f'amd: {np.round((df_dict["amd"]["b_2"]["avg"][2] - df_dict["amd"]["b_1"]["avg"][2]) / df_dict["amd"]["b_1"]["avg"][2] * 100, 1)}')
            print(f"{label_intel} \\\\")
            print(f'\\hspace{{0.5cm}}Naive & {df_dict["intel"]["b_1"]["min"][0]} & {df_dict["intel"]["b_1"]["avg"][0]} & {df_dict["intel"]["b_1"]["max"][0]} & {df_dict["intel"]["b_1"]["std"][0]} \\\\')
            print(f'\\hspace{{0.5cm}}Initial Multithreading & {df_dict["intel"]["b_2"]["min"][0]} & {df_dict["intel"]["b_2"]["avg"][0]} & {df_dict["intel"]["b_2"]["max"][0]} & {df_dict["intel"]["b_2"]["std"][0]} \\\\')
            print(f"{label_apple} \\\\")
            print(f'\\hspace{{0.5cm}}Naive & {df_dict["apple"]["b_1"]["min"][1]} & {df_dict["apple"]["b_1"]["avg"][1]} & {df_dict["apple"]["b_1"]["max"][1]} & {df_dict["apple"]["b_1"]["std"][1]} \\\\')
            print(f'\\hspace{{0.5cm}}Initial Multithreading & {df_dict["apple"]["b_2"]["min"][1]} & {df_dict["apple"]["b_2"]["avg"][1]} & {df_dict["apple"]["b_2"]["max"][1]} & {df_dict["apple"]["b_2"]["std"][1]} \\\\')
            print(f"{label_amd} \\\\")
            print(f'\\hspace{{0.5cm}}Naive & {df_dict["amd"]["b_1"]["min"][2]} & {df_dict["amd"]["b_1"]["avg"][2]} & {df_dict["amd"]["b_1"]["max"][2]} & {df_dict["amd"]["b_1"]["std"][2]} \\\\')
            print(f'\\hspace{{0.5cm}}Initial Multithreading & {df_dict["amd"]["b_2"]["min"][2]} & {df_dict["amd"]["b_2"]["avg"][2]} & {df_dict["amd"]["b_2"]["max"][2]} & {df_dict["amd"]["b_2"]["std"][2]} \\\\')
        case "mt_init_xl":
            group_width = bar_width * 2
            group_gap = 0.15
            indices = np.arange(len(threads_omp)) * (group_width + group_gap)

            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["xl_1"]["max"], width=bar_width, label=f"{label_intel} Naive XL Max", color="#99c1f1", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["xl_1"]["avg"], width=bar_width, label=f"{label_intel} Naive XL Avg", color="#3584e4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["xl_1"]["min"], width=bar_width, label=f"{label_intel} Naive XL Min", color="#1a5fb4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["xl_2"]["max"], width=bar_width, label=f"{label_intel} Initial Multithreading XL Max", color="#8ff0a4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["xl_2"]["avg"], width=bar_width, label=f"{label_intel} Initial Multithreading XL Avg", color="#33d17a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["xl_2"]["min"], width=bar_width, label=f"{label_intel} Initial Multithreading XL Min", color="#26a269", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["apple"]["xl_1"]["max"], width=bar_width, label=f"{label_apple} Naive XL Max", color="#f9f06b", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["apple"]["xl_1"]["avg"], width=bar_width, label=f"{label_apple} Naive XL Avg", color="#f6d32d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["apple"]["xl_1"]["min"], width=bar_width, label=f"{label_apple} Naive XL Min", color="#e5a50a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["apple"]["xl_2"]["max"], width=bar_width, label=f"{label_apple} Initial Multithreading XL Max", color="#ffbe6f", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["apple"]["xl_2"]["avg"], width=bar_width, label=f"{label_apple} Initial Multithreading XL Avg", color="#ff7800", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["apple"]["xl_2"]["min"], width=bar_width, label=f"{label_apple} Initial Multithreading XL Min", color="#c64600", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["xl_1"]["max"], width=bar_width, label=f"{label_amd} Naive XL Max", color="#f66151", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["xl_1"]["avg"], width=bar_width, label=f"{label_amd} Naive XL Avg", color="#e01b24", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["xl_1"]["min"], width=bar_width, label=f"{label_amd} Naive XL Min", color="#a51d2d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["xl_2"]["max"], width=bar_width, label=f"{label_amd} Initial Multithreading XL Max", color="#dc8add", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["xl_2"]["avg"], width=bar_width, label=f"{label_amd} Initial Multithreading XL Avg", color="#9141ac", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["xl_2"]["min"], width=bar_width, label=f"{label_amd} Initial Multithreading XL Min", color="#613583", edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("Initial Multithreading XL", fontsize=12, color=foreground_color, loc="center")

            b1 = df_dict["intel"]["xl_1"]["avg"][0] + df_dict["apple"]["xl_1"]["avg"][1] + df_dict["amd"]["xl_1"]["avg"][2]
            b2 = df_dict["intel"]["xl_2"]["avg"][0] + df_dict["apple"]["xl_2"]["avg"][1] + df_dict["amd"]["xl_2"]["avg"][2]

            print(f"mt_init_xl: {np.round((b2 - b1) / b1 * 100, 1)}")
            print(f'intel: {np.round((df_dict["intel"]["xl_2"]["avg"][0] - df_dict["intel"]["xl_1"]["avg"][0]) / df_dict["intel"]["xl_1"]["avg"][0] * 100, 1)}')
            print(f'apple: {np.round((df_dict["apple"]["xl_2"]["avg"][1] - df_dict["apple"]["xl_1"]["avg"][1]) / df_dict["apple"]["xl_1"]["avg"][1] * 100, 1)}')
            print(f'amd: {np.round((df_dict["amd"]["xl_2"]["avg"][2] - df_dict["amd"]["xl_1"]["avg"][2]) / df_dict["amd"]["xl_1"]["avg"][2] * 100, 1)}')
            print(f"{label_intel} \\\\")
            print(f'\\hspace{{0.5cm}}Naive XL & {df_dict["intel"]["xl_1"]["min"][0]} & {df_dict["intel"]["xl_1"]["avg"][0]} & {df_dict["intel"]["xl_1"]["max"][0]} & {df_dict["intel"]["xl_1"]["std"][0]} \\\\')
            print(f'\\hspace{{0.5cm}}Initial Multithreading XL & {df_dict["intel"]["xl_2"]["min"][0]} & {df_dict["intel"]["xl_2"]["avg"][0]} & {df_dict["intel"]["xl_2"]["max"][0]} & {df_dict["intel"]["xl_2"]["std"][0]} \\\\')
            print(f"{label_apple} \\\\")
            print(f'\\hspace{{0.5cm}}Naive XL & {df_dict["apple"]["xl_1"]["min"][1]} & {df_dict["apple"]["xl_1"]["avg"][1]} & {df_dict["apple"]["xl_1"]["max"][1]} & {df_dict["apple"]["xl_1"]["std"][1]} \\\\')
            print(f'\\hspace{{0.5cm}}Initial Multithreading XL & {df_dict["apple"]["xl_2"]["min"][1]} & {df_dict["apple"]["xl_2"]["avg"][1]} & {df_dict["apple"]["xl_2"]["max"][1]} & {df_dict["apple"]["xl_2"]["std"][1]} \\\\')
            print(f"{label_amd} \\\\")
            print(f'\\hspace{{0.5cm}}Naive XL & {df_dict["amd"]["xl_1"]["min"][2]} & {df_dict["amd"]["xl_1"]["avg"][2]} & {df_dict["amd"]["xl_1"]["max"][2]} & {df_dict["amd"]["xl_1"]["std"][2]} \\\\')
            print(f'\\hspace{{0.5cm}}Initial Multithreading XL & {df_dict["amd"]["xl_2"]["min"][2]} & {df_dict["amd"]["xl_2"]["avg"][2]} & {df_dict["amd"]["xl_2"]["max"][2]} & {df_dict["amd"]["xl_2"]["std"][2]} \\\\')
        case "mt":
            group_width = bar_width * 2
            group_gap = 0.15
            indices = np.arange(len(threads_omp)) * (group_width + group_gap)

            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["b_2"]["max"], width=bar_width, label=f"{label_intel} Initial Multithreading Max", color="#99c1f1", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["b_2"]["avg"], width=bar_width, label=f"{label_intel} Initial Multithreading Avg", color="#3584e4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["b_2"]["min"], width=bar_width, label=f"{label_intel} Initial Multithreading Min", color="#1a5fb4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["no_simd_7"]["max"], width=bar_width, label=f"{label_intel} Multithreading Max", color="#8ff0a4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["no_simd_7"]["avg"], width=bar_width, label=f"{label_intel} Multithreading Avg", color="#33d17a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["no_simd_7"]["min"], width=bar_width, label=f"{label_intel} Multithreading Min", color="#26a269", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["apple"]["b_2"]["max"], width=bar_width, label=f"{label_apple} Initial Multithreading Max", color="#f9f06b", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["apple"]["b_2"]["avg"], width=bar_width, label=f"{label_apple} Initial Multithreading Avg", color="#f6d32d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["apple"]["b_2"]["min"], width=bar_width, label=f"{label_apple} Initial Multithreading Min", color="#e5a50a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["apple"]["no_simd_7"]["max"], width=bar_width, label=f"{label_apple} Multithreading Max", color="#ffbe6f", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["apple"]["no_simd_7"]["avg"], width=bar_width, label=f"{label_apple} Multithreading Avg", color="#ff7800", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["apple"]["no_simd_7"]["min"], width=bar_width, label=f"{label_apple} Multithreading Min", color="#c64600", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["b_2"]["max"], width=bar_width, label=f"{label_amd} Initial Multithreading Max", color="#f66151", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["b_2"]["avg"], width=bar_width, label=f"{label_amd} Initial Multithreading Avg", color="#e01b24", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["b_2"]["min"], width=bar_width, label=f"{label_amd} Initial Multithreading Min", color="#a51d2d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["no_simd_7"]["max"], width=bar_width, label=f"{label_amd} Multithreading Max", color="#dc8add", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["no_simd_7"]["avg"], width=bar_width, label=f"{label_amd} Multithreading Avg", color="#9141ac", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["no_simd_7"]["min"], width=bar_width, label=f"{label_amd} Multithreading Min", color="#613583", edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("Multithreading", fontsize=12, color=foreground_color, loc="center")

            b1 = df_dict["intel"]["b_2"]["avg"][0] + df_dict["apple"]["b_2"]["avg"][1] + df_dict["amd"]["b_2"]["avg"][2]
            b2 = df_dict["intel"]["no_simd_7"]["avg"][0] + df_dict["apple"]["no_simd_7"]["avg"][1] + df_dict["amd"]["no_simd_7"]["avg"][2]

            print(f"mt: {np.round((b2 - b1) / b1 * 100, 1)}")
            print(f'intel: {np.round((df_dict["intel"]["no_simd_7"]["avg"][0] - df_dict["intel"]["b_2"]["avg"][0]) / df_dict["intel"]["b_2"]["avg"][0] * 100, 1)}')
            print(f'apple: {np.round((df_dict["apple"]["no_simd_7"]["avg"][1] - df_dict["apple"]["b_2"]["avg"][1]) / df_dict["apple"]["b_2"]["avg"][1] * 100, 1)}')
            print(f'amd: {np.round((df_dict["amd"]["no_simd_7"]["avg"][2] - df_dict["amd"]["b_2"]["avg"][2]) / df_dict["amd"]["b_2"]["avg"][2] * 100, 1)}')
            print(f"{label_intel} \\\\")
            print(f'\\hspace{{0.5cm}}Initial Multithreading & {df_dict["intel"]["b_2"]["min"][0]} & {df_dict["intel"]["b_2"]["avg"][0]} & {df_dict["intel"]["b_2"]["max"][0]} & {df_dict["intel"]["b_2"]["std"][0]} \\\\')
            print(f'\\hspace{{0.5cm}}Multithreading & {df_dict["intel"]["no_simd_7"]["min"][0]} & {df_dict["intel"]["no_simd_7"]["avg"][0]} & {df_dict["intel"]["no_simd_7"]["max"][0]} & {df_dict["intel"]["no_simd_7"]["std"][0]} \\\\')
            print(f"{label_apple} \\\\")
            print(f'\\hspace{{0.5cm}}Initial Multithreading & {df_dict["apple"]["b_2"]["min"][1]} & {df_dict["apple"]["b_2"]["avg"][1]} & {df_dict["apple"]["b_2"]["max"][1]} & {df_dict["apple"]["b_2"]["std"][1]} \\\\')
            print(f'\\hspace{{0.5cm}}Multithreading & {df_dict["apple"]["no_simd_7"]["min"][1]} & {df_dict["apple"]["no_simd_7"]["avg"][1]} & {df_dict["apple"]["no_simd_7"]["max"][1]} & {df_dict["apple"]["no_simd_7"]["std"][1]} \\\\')
            print(f"{label_amd} \\\\")
            print(f'\\hspace{{0.5cm}}Initial Multithreading & {df_dict["amd"]["b_2"]["min"][2]} & {df_dict["amd"]["b_2"]["avg"][2]} & {df_dict["amd"]["b_2"]["max"][2]} & {df_dict["amd"]["b_2"]["std"][2]} \\\\')
            print(f'\\hspace{{0.5cm}}Multithreading & {df_dict["amd"]["no_simd_7"]["min"][2]} & {df_dict["amd"]["no_simd_7"]["avg"][2]} & {df_dict["amd"]["no_simd_7"]["max"][2]} & {df_dict["amd"]["no_simd_7"]["std"][2]} \\\\')
        case "mt_xl":
            group_width = bar_width * 2
            group_gap = 0.15
            indices = np.arange(len(threads_omp)) * (group_width + group_gap)

            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["xl_2"]["max"], width=bar_width, label=f"{label_intel} Initial Multithreading XL Max", color="#99c1f1", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["xl_2"]["avg"], width=bar_width, label=f"{label_intel} Initial Multithreading XL Avg", color="#3584e4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["xl_2"]["min"], width=bar_width, label=f"{label_intel} Initial Multithreading XL Min", color="#1a5fb4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["no_simd_xl_7"]["max"], width=bar_width, label=f"{label_intel} Multithreading XL Max", color="#8ff0a4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["no_simd_xl_7"]["avg"], width=bar_width, label=f"{label_intel} Multithreading XL Avg", color="#33d17a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["no_simd_xl_7"]["min"], width=bar_width, label=f"{label_intel} Multithreading XL Min", color="#26a269", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["apple"]["xl_2"]["max"], width=bar_width, label=f"{label_apple} Initial Multithreading XL Max", color="#f9f06b", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["apple"]["xl_2"]["avg"], width=bar_width, label=f"{label_apple} Initial Multithreading XL Avg", color="#f6d32d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["apple"]["xl_2"]["min"], width=bar_width, label=f"{label_apple} Initial Multithreading XL Min", color="#e5a50a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["apple"]["no_simd_xl_7"]["max"], width=bar_width, label=f"{label_apple} Multithreading XL Max", color="#ffbe6f", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["apple"]["no_simd_xl_7"]["avg"], width=bar_width, label=f"{label_apple} Multithreading XL Avg", color="#ff7800", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["apple"]["no_simd_xl_7"]["min"], width=bar_width, label=f"{label_apple} Multithreading XL Min", color="#c64600", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["xl_2"]["max"], width=bar_width, label=f"{label_amd} Initial Multithreading XL Max", color="#f66151", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["xl_2"]["avg"], width=bar_width, label=f"{label_amd} Initial Multithreading XL Avg", color="#e01b24", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["xl_2"]["min"], width=bar_width, label=f"{label_amd} Initial Multithreading XL Min", color="#a51d2d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["no_simd_xl_7"]["max"], width=bar_width, label=f"{label_amd} Multithreading XL Max", color="#dc8add", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["no_simd_xl_7"]["avg"], width=bar_width, label=f"{label_amd} Multithreading XL Avg", color="#9141ac", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["no_simd_xl_7"]["min"], width=bar_width, label=f"{label_amd} Multithreading XL Min", color="#613583", edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("Multithreading XL", fontsize=12, color=foreground_color, loc="center")

            b1 = df_dict["intel"]["xl_2"]["avg"][0] + df_dict["apple"]["xl_2"]["avg"][1] + df_dict["amd"]["xl_2"]["avg"][2]
            b2 = df_dict["intel"]["no_simd_xl_7"]["avg"][0] + df_dict["apple"]["no_simd_xl_7"]["avg"][1] + df_dict["amd"]["no_simd_xl_7"]["avg"][2]

            print(f"mt_xl: {np.round((b2 - b1) / b1 * 100, 1)}")
            print(f'intel: {np.round((df_dict["intel"]["no_simd_xl_7"]["avg"][0] - df_dict["intel"]["xl_2"]["avg"][0]) / df_dict["intel"]["xl_2"]["avg"][0] * 100, 1)}')
            print(f'apple: {np.round((df_dict["apple"]["no_simd_xl_7"]["avg"][1] - df_dict["apple"]["xl_2"]["avg"][1]) / df_dict["apple"]["xl_2"]["avg"][1] * 100, 1)}')
            print(f'amd: {np.round((df_dict["amd"]["no_simd_xl_7"]["avg"][2] - df_dict["amd"]["xl_2"]["avg"][2]) / df_dict["amd"]["xl_2"]["avg"][2] * 100, 1)}')
            print(f"{label_intel} \\\\")
            print(f'\\hspace{{0.5cm}}Initial Multithreading XL & {df_dict["intel"]["xl_2"]["min"][0]} & {df_dict["intel"]["xl_2"]["avg"][0]} & {df_dict["intel"]["xl_2"]["max"][0]} & {df_dict["intel"]["xl_2"]["std"][0]} \\\\')
            print(f'\\hspace{{0.5cm}}Multithreading XL & {df_dict["intel"]["no_simd_xl_7"]["min"][0]} & {df_dict["intel"]["no_simd_xl_7"]["avg"][0]} & {df_dict["intel"]["no_simd_xl_7"]["max"][0]} & {df_dict["intel"]["no_simd_xl_7"]["std"][0]} \\\\')
            print(f"{label_apple} \\\\")
            print(f'\\hspace{{0.5cm}}Initial Multithreading XL & {df_dict["apple"]["xl_2"]["min"][1]} & {df_dict["apple"]["xl_2"]["avg"][1]} & {df_dict["apple"]["xl_2"]["max"][1]} & {df_dict["apple"]["xl_2"]["std"][1]} \\\\')
            print(f'\\hspace{{0.5cm}}Multithreading XL & {df_dict["apple"]["no_simd_xl_7"]["min"][1]} & {df_dict["apple"]["no_simd_xl_7"]["avg"][1]} & {df_dict["apple"]["no_simd_xl_7"]["max"][1]} & {df_dict["apple"]["no_simd_xl_7"]["std"][1]} \\\\')
            print(f"{label_amd} \\\\")
            print(f'\\hspace{{0.5cm}}Initial Multithreading XL & {df_dict["amd"]["xl_2"]["min"][2]} & {df_dict["amd"]["xl_2"]["avg"][2]} & {df_dict["amd"]["xl_2"]["max"][2]} & {df_dict["amd"]["xl_2"]["std"][2]} \\\\')
            print(f'\\hspace{{0.5cm}}Multithreading XL & {df_dict["amd"]["no_simd_xl_7"]["min"][2]} & {df_dict["amd"]["no_simd_xl_7"]["avg"][2]} & {df_dict["amd"]["no_simd_xl_7"]["max"][2]} & {df_dict["amd"]["no_simd_xl_7"]["std"][2]} \\\\')
        case "omp":
            group_width = bar_width * 2
            group_gap = 0.15
            indices = np.arange(len(threads_omp)) * (group_width + group_gap)

            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["no_simd_7"]["max"], width=bar_width, label=f"{label_intel} Multithreading Max", color="#99c1f1", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["no_simd_7"]["avg"], width=bar_width, label=f"{label_intel} Multithreading Avg", color="#3584e4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["no_simd_7"]["min"], width=bar_width, label=f"{label_intel} Multithreading Min", color="#1a5fb4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["omp_4"]["max"], width=bar_width, label=f"{label_intel} OpenMP Max", color="#8ff0a4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["omp_4"]["avg"], width=bar_width, label=f"{label_intel} OpenMP Avg", color="#33d17a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["omp_4"]["min"], width=bar_width, label=f"{label_intel} OpenMP Min", color="#26a269", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["apple"]["no_simd_7"]["max"], width=bar_width, label=f"{label_apple} Multithreading Max", color="#f9f06b", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["apple"]["no_simd_7"]["avg"], width=bar_width, label=f"{label_apple} Multithreading Avg", color="#f6d32d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["apple"]["no_simd_7"]["min"], width=bar_width, label=f"{label_apple} Multithreading Min", color="#e5a50a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["apple"]["omp_4"]["max"], width=bar_width, label=f"{label_apple} OpenMP Max", color="#ffbe6f", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["apple"]["omp_4"]["avg"], width=bar_width, label=f"{label_apple} OpenMP Avg", color="#ff7800", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["apple"]["omp_4"]["min"], width=bar_width, label=f"{label_apple} OpenMP Min", color="#c64600", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["no_simd_7"]["max"], width=bar_width, label=f"{label_amd} Multithreading Max", color="#f66151", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["no_simd_7"]["avg"], width=bar_width, label=f"{label_amd} Multithreading Avg", color="#e01b24", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["no_simd_7"]["min"], width=bar_width, label=f"{label_amd} Multithreading Min", color="#a51d2d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["omp_4"]["max"], width=bar_width, label=f"{label_amd} OpenMP Max", color="#dc8add", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["omp_4"]["avg"], width=bar_width, label=f"{label_amd} OpenMP Avg", color="#9141ac", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["omp_4"]["min"], width=bar_width, label=f"{label_amd} OpenMP Min", color="#613583", edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("OpenMP", fontsize=12, color=foreground_color, loc="center")

            b1 = df_dict["intel"]["no_simd_7"]["avg"][0] + df_dict["apple"]["no_simd_7"]["avg"][1] + df_dict["amd"]["no_simd_7"]["avg"][2]
            b2 = df_dict["intel"]["omp_4"]["avg"][0] + df_dict["apple"]["omp_4"]["avg"][1] + df_dict["amd"]["omp_4"]["avg"][2]

            print(f"omp: {np.round((b2 - b1) / b1 * 100, 1)}")
            print(f'intel: {np.round((df_dict["intel"]["omp_4"]["avg"][0] - df_dict["intel"]["no_simd_7"]["avg"][0]) / df_dict["intel"]["no_simd_7"]["avg"][0] * 100, 1)}')
            print(f'apple: {np.round((df_dict["apple"]["omp_4"]["avg"][1] - df_dict["apple"]["no_simd_7"]["avg"][1]) / df_dict["apple"]["no_simd_7"]["avg"][1] * 100, 1)}')
            print(f'amd: {np.round((df_dict["amd"]["omp_4"]["avg"][2] - df_dict["amd"]["no_simd_7"]["avg"][2]) / df_dict["amd"]["no_simd_7"]["avg"][2] * 100, 1)}')
            print(f"{label_intel} \\\\")
            print(f'\\hspace{{0.5cm}}Multithreading & {df_dict["intel"]["no_simd_7"]["min"][0]} & {df_dict["intel"]["no_simd_7"]["avg"][0]} & {df_dict["intel"]["no_simd_7"]["max"][0]} & {df_dict["intel"]["no_simd_7"]["std"][0]} \\\\')
            print(f'\\hspace{{0.5cm}}OpenMP & {df_dict["intel"]["omp_4"]["min"][0]} & {df_dict["intel"]["omp_4"]["avg"][0]} & {df_dict["intel"]["omp_4"]["max"][0]} & {df_dict["intel"]["omp_4"]["std"][0]} \\\\')
            print(f"{label_apple} \\\\")
            print(f'\\hspace{{0.5cm}}Multithreading & {df_dict["apple"]["no_simd_7"]["min"][1]} & {df_dict["apple"]["no_simd_7"]["avg"][1]} & {df_dict["apple"]["no_simd_7"]["max"][1]} & {df_dict["apple"]["no_simd_7"]["std"][1]} \\\\')
            print(f'\\hspace{{0.5cm}}OpenMP & {df_dict["apple"]["omp_4"]["min"][1]} & {df_dict["apple"]["omp_4"]["avg"][1]} & {df_dict["apple"]["omp_4"]["max"][1]} & {df_dict["apple"]["omp_4"]["std"][1]} \\\\')
            print(f"{label_amd} \\\\")
            print(f'\\hspace{{0.5cm}}Multithreading & {df_dict["amd"]["no_simd_7"]["min"][2]} & {df_dict["amd"]["no_simd_7"]["avg"][2]} & {df_dict["amd"]["no_simd_7"]["max"][2]} & {df_dict["amd"]["no_simd_7"]["std"][2]} \\\\')
            print(f'\\hspace{{0.5cm}}OpenMP & {df_dict["amd"]["omp_4"]["min"][2]} & {df_dict["amd"]["omp_4"]["avg"][2]} & {df_dict["amd"]["omp_4"]["max"][2]} & {df_dict["amd"]["omp_4"]["std"][2]} \\\\')
        case "omp_xl":
            group_width = bar_width * 2
            group_gap = 0.15
            indices = np.arange(len(threads_omp)) * (group_width + group_gap)

            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["no_simd_xl_7"]["max"], width=bar_width, label=f"{label_intel} Multithreading XL Max", color="#99c1f1", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["no_simd_xl_7"]["avg"], width=bar_width, label=f"{label_intel} Multithreading XL Avg", color="#3584e4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["no_simd_xl_7"]["min"], width=bar_width, label=f"{label_intel} Multithreading XL Min", color="#1a5fb4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["omp_xl_4"]["max"], width=bar_width, label=f"{label_intel} OpenMP XL Max", color="#8ff0a4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["omp_xl_4"]["avg"], width=bar_width, label=f"{label_intel} OpenMP XL Avg", color="#33d17a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["omp_xl_4"]["min"], width=bar_width, label=f"{label_intel} OpenMP XL Min", color="#26a269", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["apple"]["no_simd_xl_7"]["max"], width=bar_width, label=f"{label_apple} Multithreading XL Max", color="#f9f06b", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["apple"]["no_simd_xl_7"]["avg"], width=bar_width, label=f"{label_apple} Multithreading XL Avg", color="#f6d32d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["apple"]["no_simd_xl_7"]["min"], width=bar_width, label=f"{label_apple} Multithreading XL Min", color="#e5a50a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["apple"]["omp_xl_4"]["max"], width=bar_width, label=f"{label_apple} OpenMP XL Max", color="#ffbe6f", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["apple"]["omp_xl_4"]["avg"], width=bar_width, label=f"{label_apple} OpenMP XL Avg", color="#ff7800", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["apple"]["omp_xl_4"]["min"], width=bar_width, label=f"{label_apple} OpenMP XL Min", color="#c64600", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["no_simd_xl_7"]["max"], width=bar_width, label=f"{label_amd} Multithreading XL Max", color="#f66151", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["no_simd_xl_7"]["avg"], width=bar_width, label=f"{label_amd} Multithreading XL Avg", color="#e01b24", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["no_simd_xl_7"]["min"], width=bar_width, label=f"{label_amd} Multithreading XL Min", color="#a51d2d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["omp_xl_4"]["max"], width=bar_width, label=f"{label_amd} OpenMP XL Max", color="#dc8add", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["omp_xl_4"]["avg"], width=bar_width, label=f"{label_amd} OpenMP XL Avg", color="#9141ac", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["omp_xl_4"]["min"], width=bar_width, label=f"{label_amd} OpenMP XL Min", color="#613583", edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("OpenMP XL", fontsize=12, color=foreground_color, loc="center")

            b1 = df_dict["intel"]["no_simd_xl_7"]["avg"][0] + df_dict["apple"]["no_simd_xl_7"]["avg"][1] + df_dict["amd"]["no_simd_xl_7"]["avg"][2]
            b2 = df_dict["intel"]["omp_xl_4"]["avg"][0] + df_dict["apple"]["omp_xl_4"]["avg"][1] + df_dict["amd"]["omp_xl_4"]["avg"][2]

            print(f"omp_xl: {np.round((b2 - b1) / b1 * 100, 1)}")
            print(f'intel: {np.round((df_dict["intel"]["omp_xl_4"]["avg"][0] - df_dict["intel"]["no_simd_xl_7"]["avg"][0]) / df_dict["intel"]["no_simd_xl_7"]["avg"][0] * 100, 1)}')
            print(f'apple: {np.round((df_dict["apple"]["omp_xl_4"]["avg"][1] - df_dict["apple"]["no_simd_xl_7"]["avg"][1]) / df_dict["apple"]["no_simd_xl_7"]["avg"][1] * 100, 1)}')
            print(f'amd: {np.round((df_dict["amd"]["omp_xl_4"]["avg"][2] - df_dict["amd"]["no_simd_xl_7"]["avg"][2]) / df_dict["amd"]["no_simd_xl_7"]["avg"][2] * 100, 1)}')
            print(f"{label_intel} \\\\")
            print(f'\\hspace{{0.5cm}}Multithreading XL & {df_dict["intel"]["no_simd_xl_7"]["min"][0]} & {df_dict["intel"]["no_simd_xl_7"]["avg"][0]} & {df_dict["intel"]["no_simd_xl_7"]["max"][0]} & {df_dict["intel"]["no_simd_xl_7"]["std"][0]} \\\\')
            print(f'\\hspace{{0.5cm}}OpenMP XL & {df_dict["intel"]["omp_xl_4"]["min"][0]} & {df_dict["intel"]["omp_xl_4"]["avg"][0]} & {df_dict["intel"]["omp_xl_4"]["max"][0]} & {df_dict["intel"]["omp_xl_4"]["std"][0]} \\\\')
            print(f"{label_apple} \\\\")
            print(f'\\hspace{{0.5cm}}Multithreading XL & {df_dict["apple"]["no_simd_xl_7"]["min"][1]} & {df_dict["apple"]["no_simd_xl_7"]["avg"][1]} & {df_dict["apple"]["no_simd_xl_7"]["max"][1]} & {df_dict["apple"]["no_simd_xl_7"]["std"][1]} \\\\')
            print(f'\\hspace{{0.5cm}}OpenMP XL & {df_dict["apple"]["omp_xl_4"]["min"][1]} & {df_dict["apple"]["omp_xl_4"]["avg"][1]} & {df_dict["apple"]["omp_xl_4"]["max"][1]} & {df_dict["apple"]["omp_xl_4"]["std"][1]} \\\\')
            print(f"{label_amd} \\\\")
            print(f'\\hspace{{0.5cm}}Multithreading XL & {df_dict["amd"]["no_simd_xl_7"]["min"][2]} & {df_dict["amd"]["no_simd_xl_7"]["avg"][2]} & {df_dict["amd"]["no_simd_xl_7"]["max"][2]} & {df_dict["amd"]["no_simd_xl_7"]["std"][2]} \\\\')
            print(f'\\hspace{{0.5cm}}OpenMP XL & {df_dict["amd"]["omp_xl_4"]["min"][2]} & {df_dict["amd"]["omp_xl_4"]["avg"][2]} & {df_dict["amd"]["omp_xl_4"]["max"][2]} & {df_dict["amd"]["omp_xl_4"]["std"][2]} \\\\')
        case "sse":
            group_width = bar_width * 2
            group_gap = 0.15
            indices = np.arange(len([8, 16])) * (group_width + group_gap)

            df_intel_b_2 = df_dict["intel"]["b_2"].drop(index=1).reset_index(drop=True)
            df_intel_b_3 = df_dict["intel"]["b_3"].drop(index=1).reset_index(drop=True)
            df_amd_b_2 = df_dict["amd"]["b_2"].drop(index=1).reset_index(drop=True)
            df_amd_b_3 = df_dict["amd"]["b_3"].drop(index=1).reset_index(drop=True)

            ax.bar(indices - 0.5 * bar_width, df_intel_b_2["max"], width=bar_width, label=f"{label_intel} Initial Multithreading Max", color="#99c1f1", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_b_2["avg"], width=bar_width, label=f"{label_intel} Initial Multithreading Avg", color="#3584e4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_b_2["min"], width=bar_width, label=f"{label_intel} Initial Multithreading Min", color="#1a5fb4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_b_3["max"], width=bar_width, label=f"{label_intel} Initial Multithreading + SSE Max", color="#8ff0a4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_b_3["avg"], width=bar_width, label=f"{label_intel} Initial Multithreading + SSE Avg", color="#33d17a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_b_3["min"], width=bar_width, label=f"{label_intel} Initial Multithreading + SSE Min", color="#26a269", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_b_2["max"], width=bar_width, label=f"{label_amd} Initial Multithreading Max", color="#f66151", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_b_2["avg"], width=bar_width, label=f"{label_amd} Initial Multithreading Avg", color="#e01b24", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_b_2["min"], width=bar_width, label=f"{label_amd} Initial Multithreading Min", color="#a51d2d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_b_3["max"], width=bar_width, label=f"{label_amd} Initial Multithreading + SSE Max", color="#dc8add", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_b_3["avg"], width=bar_width, label=f"{label_amd} Initial Multithreading + SSE Avg", color="#9141ac", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_b_3["min"], width=bar_width, label=f"{label_amd} Initial Multithreading + SSE Min", color="#613583", edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("SSE", fontsize=12, color=foreground_color, loc="center")

            b1 = df_intel_b_2["avg"][0] + df_amd_b_2["avg"][1]
            b2 = df_intel_b_3["avg"][0] + df_amd_b_3["avg"][1]

            print(f"sse: {np.round((b2 - b1) / b1 * 100, 0)}")
        case "sse_xl":
            group_width = bar_width * 2
            group_gap = 0.15
            indices = np.arange(len([8, 16])) * (group_width + group_gap)

            df_intel_xl_2 = df_dict["intel"]["xl_2"].drop(index=1).reset_index(drop=True)
            df_intel_xl_3 = df_dict["intel"]["xl_3"].drop(index=1).reset_index(drop=True)
            df_amd_xl_2 = df_dict["amd"]["xl_2"].drop(index=1).reset_index(drop=True)
            df_amd_xl_3 = df_dict["amd"]["xl_3"].drop(index=1).reset_index(drop=True)

            ax.bar(indices - 0.5 * bar_width, df_intel_xl_2["max"], width=bar_width, label=f"{label_intel} Initial Multithreading XL Max", color="#99c1f1", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_xl_2["avg"], width=bar_width, label=f"{label_intel} Initial Multithreading XL Avg", color="#3584e4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_xl_2["min"], width=bar_width, label=f"{label_intel} Initial Multithreading XL Min", color="#1a5fb4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_xl_3["max"], width=bar_width, label=f"{label_intel} Initial Multithreading + SSE XL Max", color="#8ff0a4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_xl_3["avg"], width=bar_width, label=f"{label_intel} Initial Multithreading + SSE XL Avg", color="#33d17a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_xl_3["min"], width=bar_width, label=f"{label_intel} Initial Multithreading + SSE XL Min", color="#26a269", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_xl_2["max"], width=bar_width, label=f"{label_amd} Initial Multithreading XL Max", color="#f66151", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_xl_2["avg"], width=bar_width, label=f"{label_amd} Initial Multithreading XL Avg", color="#e01b24", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_xl_2["min"], width=bar_width, label=f"{label_amd} Initial Multithreading XL Min", color="#a51d2d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_xl_3["max"], width=bar_width, label=f"{label_amd} Initial Multithreading + SSE XL Max", color="#dc8add", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_xl_3["avg"], width=bar_width, label=f"{label_amd} Initial Multithreading + SSE XL Avg", color="#9141ac", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_xl_3["min"], width=bar_width, label=f"{label_amd} Initial Multithreading + SSE XL Min", color="#613583", edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("SSE XL", fontsize=12, color=foreground_color, loc="center")

            b1 = df_intel_xl_2["avg"][0] + df_amd_xl_2["avg"][1]
            b2 = df_intel_xl_3["avg"][0] + df_amd_xl_3["avg"][1]

            print(f"sse_xl: {np.round((b2 - b1) / b1 * 100, 0)}")
        case "avx":
            group_width = bar_width * 2
            group_gap = 0.15
            indices = np.arange(len([8, 16])) * (group_width + group_gap)

            df_intel_no_simd_7 = df_dict["intel"]["no_simd_7"].drop(index=1).reset_index(drop=True)
            df_intel_b_6 = df_dict["intel"]["b_6"].drop(index=1).reset_index(drop=True)
            df_amd_no_simd_7 = df_dict["amd"]["no_simd_7"].drop(index=1).reset_index(drop=True)
            df_amd_b_6 = df_dict["amd"]["b_6"].drop(index=1).reset_index(drop=True)

            ax.bar(indices - 0.5 * bar_width, df_intel_no_simd_7["max"], width=bar_width, label=f"{label_intel} Multithreading Max", color="#99c1f1", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_no_simd_7["avg"], width=bar_width, label=f"{label_intel} Multithreading Avg", color="#3584e4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_no_simd_7["min"], width=bar_width, label=f"{label_intel} Multithreading Min", color="#1a5fb4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_b_6["max"], width=bar_width, label=f"{label_intel} Multithreading + AVX-512 Max", color="#8ff0a4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_b_6["avg"], width=bar_width, label=f"{label_intel} Multithreading + AVX-512 Avg", color="#33d17a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_b_6["min"], width=bar_width, label=f"{label_intel} Multithreading + AVX-512 Min", color="#26a269", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_no_simd_7["max"], width=bar_width, label=f"{label_amd} Multithreading Max", color="#f66151", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_no_simd_7["avg"], width=bar_width, label=f"{label_amd} Multithreading Avg", color="#e01b24", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_no_simd_7["min"], width=bar_width, label=f"{label_amd} Multithreading Min", color="#a51d2d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_b_6["max"], width=bar_width, label=f"{label_amd} Multithreading + AVX2 Max", color="#dc8add", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_b_6["avg"], width=bar_width, label=f"{label_amd} Multithreading + AVX2 Avg", color="#9141ac", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_b_6["min"], width=bar_width, label=f"{label_amd} Multithreading + AVX2 Min", color="#613583", edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("AVX", fontsize=12, color=foreground_color, loc="center")

            b1 = df_intel_no_simd_7["avg"][0] + df_amd_no_simd_7["avg"][1]
            b2 = df_intel_b_6["avg"][0] + df_amd_b_6["avg"][1]

            print(f"avx: {np.round((b2 - b1) / b1 * 100, 0)}")
        case "avx_xl":
            group_width = bar_width * 2
            group_gap = 0.15
            indices = np.arange(len([8, 16])) * (group_width + group_gap)

            df_intel_no_simd_xl_7 = df_dict["intel"]["no_simd_xl_7"].drop(index=1).reset_index(drop=True)
            df_intel_xl_6 = df_dict["intel"]["xl_6"].drop(index=1).reset_index(drop=True)
            df_amd_no_simd_xl_7 = df_dict["amd"]["no_simd_xl_7"].drop(index=1).reset_index(drop=True)
            df_amd_xl_6 = df_dict["amd"]["xl_6"].drop(index=1).reset_index(drop=True)

            ax.bar(indices - 0.5 * bar_width, df_intel_no_simd_xl_7["max"], width=bar_width, label=f"{label_intel} Multithreading XL Max", color="#99c1f1", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_no_simd_xl_7["avg"], width=bar_width, label=f"{label_intel} Multithreading XL Avg", color="#3584e4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_no_simd_xl_7["min"], width=bar_width, label=f"{label_intel} Multithreading XL Min", color="#1a5fb4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_xl_6["max"], width=bar_width, label=f"{label_intel} Multithreading + AVX-512 XL Max", color="#8ff0a4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_xl_6["avg"], width=bar_width, label=f"{label_intel} Multithreading + AVX-512 XL Avg", color="#33d17a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_xl_6["min"], width=bar_width, label=f"{label_intel} Multithreading + AVX-512 XL Min", color="#26a269", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_no_simd_xl_7["max"], width=bar_width, label=f"{label_amd} Multithreading XL Max", color="#f66151", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_no_simd_xl_7["avg"], width=bar_width, label=f"{label_amd} Multithreading XL Avg", color="#e01b24", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_no_simd_xl_7["min"], width=bar_width, label=f"{label_amd} Multithreading XL Min", color="#a51d2d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_xl_6["max"], width=bar_width, label=f"{label_amd} Multithreading + AVX2 XL Max", color="#dc8add", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_xl_6["avg"], width=bar_width, label=f"{label_amd} Multithreading + AVX2 XL Avg", color="#9141ac", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_xl_6["min"], width=bar_width, label=f"{label_amd} Multithreading + AVX2 XL Min", color="#613583", edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("AVX XL", fontsize=12, color=foreground_color, loc="center")

            b1 = df_intel_no_simd_xl_7["avg"][0] + df_amd_no_simd_xl_7["avg"][1]
            b2 = df_intel_xl_6["avg"][0] + df_amd_xl_6["avg"][1]

            print(f"avx_xl: {np.round((b2 - b1) / b1 * 100, 0)}")
        case "neon":
            group_width = bar_width * 2
            group_gap = 0.15
            indices = np.arange(len([1, 11])) * (group_width + group_gap)

            df_apple_no_neon_7 = df_dict["apple"]["no_neon_7"].drop(index=2).reset_index(drop=True)
            df_apple_neon_7 = df_dict["apple"]["neon_7"].drop(index=2).reset_index(drop=True)

            ax.bar(indices - 0.5 * bar_width, df_apple_no_neon_7["max"], width=bar_width, label=f"{label_apple} Multithreading Max", color="#99c1f1", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_no_neon_7["avg"], width=bar_width, label=f"{label_apple} Multithreading Avg", color="#3584e4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_no_neon_7["min"], width=bar_width, label=f"{label_apple} Multithreading Min", color="#1a5fb4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_neon_7["max"], width=bar_width, label=f"{label_apple} Multithreading + Neon Max", color="#8ff0a4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_neon_7["avg"], width=bar_width, label=f"{label_apple} Multithreading + Neon Avg", color="#33d17a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_neon_7["min"], width=bar_width, label=f"{label_apple} Multithreading + Neon Min", color="#26a269", edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("Neon", fontsize=12, color=foreground_color, loc="center")

            b1 = df_apple_no_neon_7["avg"][0] + df_apple_no_neon_7["avg"][1]
            b2 = df_apple_neon_7["avg"][0] + df_apple_neon_7["avg"][1]

            print(f"neon: {np.round((b2 - b1) / b1 * 100, 1)}")
            print(f"{label_apple} \\\\")
            print(f'\\hspace{{0.5cm}}Multithreading & {df_apple_no_neon_7["min"][1]} & {df_apple_no_neon_7["avg"][1]} & {df_apple_no_neon_7["max"][1]} & {df_apple_no_neon_7["std"][1]} \\\\')
            print(f'\\hspace{{0.5cm}}Multithreading + Neon & {df_apple_neon_7["min"][1]} & {df_apple_neon_7["avg"][1]} & {df_apple_neon_7["max"][1]} & {df_apple_neon_7["std"][1]} \\\\')
        case "neon_xl":
            group_width = bar_width * 2
            group_gap = 0.15
            indices = np.arange(len([1, 11])) * (group_width + group_gap)

            df_apple_no_neon_xl_7 = df_dict["apple"]["no_neon_xl_7"].drop(index=2).reset_index(drop=True)
            df_apple_neon_xl_7 = df_dict["apple"]["neon_xl_7"].drop(index=2).reset_index(drop=True)

            ax.bar(indices - 0.5 * bar_width, df_apple_no_neon_xl_7["max"], width=bar_width, label=f"{label_apple} Multithreading XL Max", color="#99c1f1", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_no_neon_xl_7["avg"], width=bar_width, label=f"{label_apple} Multithreading XL Avg", color="#3584e4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_no_neon_xl_7["min"], width=bar_width, label=f"{label_apple} Multithreading XL Min", color="#1a5fb4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_neon_xl_7["max"], width=bar_width, label=f"{label_apple} Multithreading + Neon XL Max", color="#8ff0a4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_neon_xl_7["avg"], width=bar_width, label=f"{label_apple} Multithreading + Neon XL Avg", color="#33d17a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_neon_xl_7["min"], width=bar_width, label=f"{label_apple} Multithreading + Neon XL Min", color="#26a269", edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("Neon XL", fontsize=12, color=foreground_color, loc="center")

            b1 = df_apple_no_neon_xl_7["avg"][0] + df_apple_no_neon_xl_7["avg"][1]
            b2 = df_apple_neon_xl_7["avg"][0] + df_apple_neon_xl_7["avg"][1]

            print(f"neon_xl: {np.round((b2 - b1) / b1 * 100, 0)}")
            print(f"{label_apple} \\\\")
            print(f'\\hspace{{0.5cm}}Multithreading XL & {df_apple_no_neon_xl_7["min"][1]} & {df_apple_no_neon_xl_7["avg"][1]} & {df_apple_no_neon_xl_7["max"][1]} & {df_apple_no_neon_xl_7["std"][1]} \\\\')
            print(f'\\hspace{{0.5cm}}Multithreading + Neon XL & {df_apple_neon_xl_7["min"][1]} & {df_apple_neon_xl_7["avg"][1]} & {df_apple_neon_xl_7["max"][1]} & {df_apple_neon_xl_7["std"][1]} \\\\')
        case "amx":
            group_width = bar_width * 2
            group_gap = 0.15
            indices = np.arange(len([1, 11])) * (group_width + group_gap)

            df_apple_neon_6 = df_dict["apple"]["neon_6"].drop(index=2).reset_index(drop=True)
            df_apple_amx_6 = df_dict["apple"]["amx_6"].drop(index=2).reset_index(drop=True)

            ax.bar(indices - 0.5 * bar_width, df_apple_neon_6["max"], width=bar_width, label=f"{label_apple} Multithreading + Neon Max", color="#99c1f1", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_neon_6["avg"], width=bar_width, label=f"{label_apple} Multithreading + Neon Avg", color="#3584e4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_neon_6["min"], width=bar_width, label=f"{label_apple} Multithreading + Neon Min", color="#1a5fb4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_amx_6["max"], width=bar_width, label=f"{label_apple} Multithreading + AMX Max", color="#8ff0a4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_amx_6["avg"], width=bar_width, label=f"{label_apple} Multithreading + AMX Avg", color="#33d17a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_amx_6["min"], width=bar_width, label=f"{label_apple} Multithreading + AMX Min", color="#26a269", edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("AMX", fontsize=12, color=foreground_color, loc="center")

            b1 = df_apple_neon_6["avg"][0] + df_apple_neon_6["avg"][1]
            b2 = df_apple_amx_6["avg"][0] + df_apple_amx_6["avg"][1]

            print(f"amx: {np.round((b2 - b1) / b1 * 100, 1)}")
            print(f"{label_apple} \\\\")
            print(f'\\hspace{{0.5cm}}Multithreading + Neon & {df_apple_neon_6["min"][1]} & {df_apple_neon_6["avg"][1]} & {df_apple_neon_6["max"][1]} & {df_apple_neon_6["std"][1]} \\\\')
            print(f'\\hspace{{0.5cm}}Multithreading + AMX & {df_apple_amx_6["min"][1]} & {df_apple_amx_6["avg"][1]} & {df_apple_amx_6["max"][1]} & {df_apple_amx_6["std"][1]} \\\\')
        case "amx_xxl":
            group_width = bar_width * 2
            group_gap = 0.15
            indices = np.arange(len([1, 11])) * (group_width + group_gap)

            df_apple_neon_xxl_6 = df_dict["apple"]["neon_xxl_6"].drop(index=2).reset_index(drop=True)
            df_apple_amx_xxl_6 = df_dict["apple"]["amx_xxl_6"].drop(index=2).reset_index(drop=True)

            ax.bar(indices - 0.5 * bar_width, df_apple_neon_xxl_6["max"], width=bar_width, label=f"{label_apple} Multithreading + Neon XXL Max", color="#99c1f1", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_neon_xxl_6["avg"], width=bar_width, label=f"{label_apple} Multithreading + Neon XXL Avg", color="#3584e4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_neon_xxl_6["min"], width=bar_width, label=f"{label_apple} Multithreading + Neon XXL Min", color="#1a5fb4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_amx_xxl_6["max"], width=bar_width, label=f"{label_apple} Multithreading + AMX XXL Max", color="#8ff0a4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_amx_xxl_6["avg"], width=bar_width, label=f"{label_apple} Multithreading + AMX XXL Avg", color="#33d17a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_amx_xxl_6["min"], width=bar_width, label=f"{label_apple} Multithreading + AMX XXL Min", color="#26a269", edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("AMX XXL", fontsize=12, color=foreground_color, loc="center")

            b1 = df_apple_neon_xxl_6["avg"][0] + df_apple_neon_xxl_6["avg"][1]
            b2 = df_apple_amx_xxl_6["avg"][0] + df_apple_amx_xxl_6["avg"][1]

            print(f"amx_xxl: {np.round((b2 - b1) / b1 * 100, 1)}")
            print(f"{label_apple} \\\\")
            print(f'\\hspace{{0.5cm}}Multithreading + Neon XXL & {df_apple_neon_xxl_6["min"][1]} & {df_apple_neon_xxl_6["avg"][1]} & {df_apple_neon_xxl_6["max"][1]} & {df_apple_neon_xxl_6["std"][1]} \\\\')
            print(f'\\hspace{{0.5cm}}Multithreading + AMX XXL & {df_apple_amx_xxl_6["min"][1]} & {df_apple_amx_xxl_6["avg"][1]} & {df_apple_amx_xxl_6["max"][1]} & {df_apple_amx_xxl_6["std"][1]} \\\\')
        case "int":
            group_width = bar_width * 2
            group_gap = 0.15
            indices = np.arange(len(threads_omp)) * (group_width + group_gap)

            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["b_7"]["max"], width=bar_width, label=f"{label_intel} AVX-512 Max", color="#99c1f1", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["b_7"]["avg"], width=bar_width, label=f"{label_intel} AVX-512 Avg", color="#3584e4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["b_7"]["min"], width=bar_width, label=f"{label_intel} AVX-512 Min", color="#1a5fb4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["int_7"]["max"], width=bar_width, label=f"{label_intel} AVX-512 + Quantization Max", color="#8ff0a4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["int_7"]["avg"], width=bar_width, label=f"{label_intel} AVX-512 + Quantization Avg", color="#33d17a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["int_7"]["min"], width=bar_width, label=f"{label_intel} AVX-512 + Quantization Min", color="#26a269", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["apple"]["b_7"]["max"], width=bar_width, label=f"{label_apple} Neon Max", color="#f9f06b", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["apple"]["b_7"]["avg"], width=bar_width, label=f"{label_apple} Neon Avg", color="#f6d32d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["apple"]["b_7"]["min"], width=bar_width, label=f"{label_apple} Neon Min", color="#e5a50a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["apple"]["int_7"]["max"], width=bar_width, label=f"{label_apple} Neon + Quantization Max", color="#ffbe6f", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["apple"]["int_7"]["avg"], width=bar_width, label=f"{label_apple} Neon + Quantization Avg", color="#ff7800", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["apple"]["int_7"]["min"], width=bar_width, label=f"{label_apple} Neon + Quantization Min", color="#c64600", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["b_7"]["max"], width=bar_width, label=f"{label_amd} AVX2 Max", color="#f66151", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["b_7"]["avg"], width=bar_width, label=f"{label_amd} AVX2 Avg", color="#e01b24", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["b_7"]["min"], width=bar_width, label=f"{label_amd} AVX2 Min", color="#a51d2d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["int_7"]["max"], width=bar_width, label=f"{label_amd} AVX2 + Quantization Max", color="#dc8add", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["int_7"]["avg"], width=bar_width, label=f"{label_amd} AVX2 + Quantization Avg", color="#9141ac", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["int_7"]["min"], width=bar_width, label=f"{label_amd} AVX2 + Quantization Min", color="#613583", edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("Quantization", fontsize=12, color=foreground_color, loc="center")

            b1 = df_dict["intel"]["b_7"]["avg"][0] + df_dict["apple"]["b_7"]["avg"][1] + df_dict["amd"]["b_7"]["avg"][2]
            b2 = df_dict["intel"]["int_7"]["avg"][0] + df_dict["apple"]["int_7"]["avg"][1] + df_dict["amd"]["int_7"]["avg"][2]

            print(f"int: {np.round((b2 - b1) / b1 * 100, 1)}")
            print(f'intel: {np.round((df_dict["intel"]["b_7"]["avg"][0] - df_dict["intel"]["int_7"]["avg"][0]) / df_dict["intel"]["int_7"]["avg"][0] * 100, 1)}')
            print(f'apple: {np.round((df_dict["apple"]["b_7"]["avg"][1] - df_dict["apple"]["int_7"]["avg"][1]) / df_dict["apple"]["int_7"]["avg"][1] * 100, 1)}')
            print(f'amd: {np.round((df_dict["amd"]["b_7"]["avg"][2] - df_dict["amd"]["int_7"]["avg"][2]) / df_dict["amd"]["int_7"]["avg"][2] * 100, 1)}')
            print(f"{label_intel} \\\\")
            print(f'\\hspace{{0.5cm}}AVX-512 & {df_dict["intel"]["b_7"]["min"][0]} & {df_dict["intel"]["b_7"]["avg"][0]} & {df_dict["intel"]["b_7"]["max"][0]} & {df_dict["intel"]["b_7"]["std"][0]} \\\\')
            print(f'\\hspace{{0.5cm}}AVX-512 + Quantization & {df_dict["intel"]["int_7"]["min"][0]} & {df_dict["intel"]["int_7"]["avg"][0]} & {df_dict["intel"]["int_7"]["max"][0]} & {df_dict["intel"]["int_7"]["std"][0]} \\\\')
            print(f"{label_apple} \\\\")
            print(f'\\hspace{{0.5cm}}Neon & {df_dict["apple"]["b_7"]["min"][1]} & {df_dict["apple"]["b_7"]["avg"][1]} & {df_dict["apple"]["b_7"]["max"][1]} & {df_dict["apple"]["b_7"]["std"][1]} \\\\')
            print(f'\\hspace{{0.5cm}}Neon + Quantization & {df_dict["apple"]["int_7"]["min"][1]} & {df_dict["apple"]["int_7"]["avg"][1]} & {df_dict["apple"]["int_7"]["max"][1]} & {df_dict["apple"]["int_7"]["std"][1]} \\\\')
            print(f"{label_amd} \\\\")
            print(f'\\hspace{{0.5cm}}AVX2 & {df_dict["amd"]["b_7"]["min"][2]} & {df_dict["amd"]["b_7"]["avg"][2]} & {df_dict["amd"]["b_7"]["max"][2]} & {df_dict["amd"]["b_7"]["std"][2]} \\\\')
            print(f'\\hspace{{0.5cm}}AVX2 + Quantization & {df_dict["amd"]["int_7"]["min"][2]} & {df_dict["amd"]["int_7"]["avg"][2]} & {df_dict["amd"]["int_7"]["max"][2]} & {df_dict["amd"]["int_7"]["std"][2]} \\\\')
        case "int_xl":
            group_width = bar_width * 2
            group_gap = 0.15
            indices = np.arange(len(threads_omp)) * (group_width + group_gap)

            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["xl_7"]["max"], width=bar_width, label=f"{label_intel} AVX-512 XL Max", color="#99c1f1", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["xl_7"]["avg"], width=bar_width, label=f"{label_intel} AVX-512 XL Avg", color="#3584e4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["xl_7"]["min"], width=bar_width, label=f"{label_intel} AVX-512 XL Min", color="#1a5fb4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["int_xl_7"]["max"], width=bar_width, label=f"{label_intel} AVX-512 + Quantization XL Max", color="#8ff0a4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["int_xl_7"]["avg"], width=bar_width, label=f"{label_intel} AVX-512 + Quantization XL Avg", color="#33d17a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["int_xl_7"]["min"], width=bar_width, label=f"{label_intel} AVX-512 + Quantization XL Min", color="#26a269", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["apple"]["xl_7"]["max"], width=bar_width, label=f"{label_apple} Neon XL Max", color="#f9f06b", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["apple"]["xl_7"]["avg"], width=bar_width, label=f"{label_apple} Neon XL Avg", color="#f6d32d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["apple"]["xl_7"]["min"], width=bar_width, label=f"{label_apple} Neon XL Min", color="#e5a50a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["apple"]["int_xl_7"]["max"], width=bar_width, label=f"{label_apple} Neon + Quantization XL Max", color="#ffbe6f", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["apple"]["int_xl_7"]["avg"], width=bar_width, label=f"{label_apple} Neon + Quantization XL Avg", color="#ff7800", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["apple"]["int_xl_7"]["min"], width=bar_width, label=f"{label_apple} Neon + Quantization XL Min", color="#c64600", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["xl_7"]["max"], width=bar_width, label=f"{label_amd} AVX2 XL Max", color="#f66151", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["xl_7"]["avg"], width=bar_width, label=f"{label_amd} AVX2 XL Avg", color="#e01b24", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["xl_7"]["min"], width=bar_width, label=f"{label_amd} AVX2 XL Min", color="#a51d2d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["int_xl_7"]["max"], width=bar_width, label=f"{label_amd} AVX2 + Quantization XL Max", color="#dc8add", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["int_xl_7"]["avg"], width=bar_width, label=f"{label_amd} AVX2 + Quantization XL Avg", color="#9141ac", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["int_xl_7"]["min"], width=bar_width, label=f"{label_amd} AVX2 + Quantization XL Min", color="#613583", edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("Quantization XL", fontsize=12, color=foreground_color, loc="center")

            b1 = df_dict["intel"]["xl_7"]["avg"][0] + df_dict["apple"]["xl_7"]["avg"][1] + df_dict["amd"]["xl_7"]["avg"][2]
            b2 = df_dict["intel"]["int_xl_7"]["avg"][0] + df_dict["apple"]["int_xl_7"]["avg"][1] + df_dict["amd"]["int_xl_7"]["avg"][2]

            print(f"int_xl: {np.round((b2 - b1) / b1 * 100, 1)}")
            print(f'intel: {np.round((df_dict["intel"]["xl_7"]["avg"][0] - df_dict["intel"]["int_xl_7"]["avg"][0]) / df_dict["intel"]["int_xl_7"]["avg"][0] * 100, 1)}')
            print(f'apple: {np.round((df_dict["apple"]["xl_7"]["avg"][1] - df_dict["apple"]["int_xl_7"]["avg"][1]) / df_dict["apple"]["int_xl_7"]["avg"][1] * 100, 1)}')
            print(f'amd: {np.round((df_dict["amd"]["xl_7"]["avg"][2] - df_dict["amd"]["int_xl_7"]["avg"][2]) / df_dict["amd"]["int_xl_7"]["avg"][2] * 100, 1)}')
            print(f"{label_intel} \\\\")
            print(f'\\hspace{{0.5cm}}AVX-512 XL & {df_dict["intel"]["xl_7"]["min"][0]} & {df_dict["intel"]["xl_7"]["avg"][0]} & {df_dict["intel"]["xl_7"]["max"][0]} & {df_dict["intel"]["xl_7"]["std"][0]} \\\\')
            print(f'\\hspace{{0.5cm}}AVX-512 + Quantization XL & {df_dict["intel"]["int_xl_7"]["min"][0]} & {df_dict["intel"]["int_xl_7"]["avg"][0]} & {df_dict["intel"]["int_xl_7"]["max"][0]} & {df_dict["intel"]["int_xl_7"]["std"][0]} \\\\')
            print(f"{label_apple} \\\\")
            print(f'\\hspace{{0.5cm}}Neon XL & {df_dict["apple"]["xl_7"]["min"][1]} & {df_dict["apple"]["xl_7"]["avg"][1]} & {df_dict["apple"]["xl_7"]["max"][1]} & {df_dict["apple"]["xl_7"]["std"][1]} \\\\')
            print(f'\\hspace{{0.5cm}}Neon + Quantization XL & {df_dict["apple"]["int_xl_7"]["min"][1]} & {df_dict["apple"]["int_xl_7"]["avg"][1]} & {df_dict["apple"]["int_xl_7"]["max"][1]} & {df_dict["apple"]["int_xl_7"]["std"][1]} \\\\')
            print(f"{label_amd} \\\\")
            print(f'\\hspace{{0.5cm}}AVX2 XL & {df_dict["amd"]["xl_7"]["min"][2]} & {df_dict["amd"]["xl_7"]["avg"][2]} & {df_dict["amd"]["xl_7"]["max"][2]} & {df_dict["amd"]["xl_7"]["std"][2]} \\\\')
            print(f'\\hspace{{0.5cm}}AVX2 + Quantization XL & {df_dict["amd"]["int_xl_7"]["min"][2]} & {df_dict["amd"]["int_xl_7"]["avg"][2]} & {df_dict["amd"]["int_xl_7"]["max"][2]} & {df_dict["amd"]["int_xl_7"]["std"][2]} \\\\')
        case "icpx":
            group_width = bar_width * 2
            group_gap = 0.15
            indices = np.arange(len([8, 16])) * (group_width + group_gap)

            df_intel_b_1 = df_dict["intel"]["b_1"].drop(index=1).reset_index(drop=True)
            df_intel_intel_1 = df_dict["intel"]["intel_1"].drop(index=1).reset_index(drop=True)
            df_amd_b_1 = df_dict["amd"]["b_1"].drop(index=1).reset_index(drop=True)
            df_amd_intel_1 = df_dict["amd"]["intel_1"].drop(index=1).reset_index(drop=True)

            ax.bar(indices - 0.5 * bar_width, df_intel_b_1["max"], width=bar_width, label=f"{label_intel} Clang Max", color="#99c1f1", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_b_1["avg"], width=bar_width, label=f"{label_intel} Clang Avg", color="#3584e4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_b_1["min"], width=bar_width, label=f"{label_intel} Clang Min", color="#1a5fb4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_intel_1["max"], width=bar_width, label=f"{label_intel} ICPX Max", color="#8ff0a4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_intel_1["avg"], width=bar_width, label=f"{label_intel} ICPX Avg", color="#33d17a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_intel_1["min"], width=bar_width, label=f"{label_intel} ICPX Min", color="#26a269", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_b_1["max"], width=bar_width, label=f"{label_amd} Clang Max", color="#f66151", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_b_1["avg"], width=bar_width, label=f"{label_amd} Clang Avg", color="#e01b24", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_b_1["min"], width=bar_width, label=f"{label_amd} Clang Min", color="#a51d2d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_intel_1["max"], width=bar_width, label=f"{label_amd} ICPX Max", color="#dc8add", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_intel_1["avg"], width=bar_width, label=f"{label_amd} ICPX Avg", color="#9141ac", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_intel_1["min"], width=bar_width, label=f"{label_amd} ICPX Min", color="#613583", edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("ICPX", fontsize=12, color=foreground_color, loc="center")

            b1 = df_intel_b_1["avg"][0] + df_amd_b_1["avg"][1]
            b2 = df_intel_intel_1["avg"][0] + df_amd_intel_1["avg"][1]

            print(f"icpx: {np.round((b2 - b1) / b1 * 100, 1)}")
            print(f'intel: {np.round((df_dict["intel"]["intel_1"]["avg"][0] - df_dict["intel"]["b_1"]["avg"][0]) / df_dict["intel"]["b_1"]["avg"][0] * 100, 1)}')
            print(f'amd: {np.round((df_dict["amd"]["intel_1"]["avg"][2] - df_dict["amd"]["b_1"]["avg"][2]) / df_dict["amd"]["b_1"]["avg"][2] * 100, 1)}')
            print(f"{label_intel} \\\\")
            print(f'\\hspace{{0.5cm}}Clang & {df_dict["intel"]["b_1"]["min"][0]} & {df_dict["intel"]["b_1"]["avg"][0]} & {df_dict["intel"]["b_1"]["max"][0]} & {df_dict["intel"]["b_1"]["std"][0]} \\\\')
            print(f'\\hspace{{0.5cm}}ICPX & {df_dict["intel"]["intel_1"]["min"][0]} & {df_dict["intel"]["intel_1"]["avg"][0]} & {df_dict["intel"]["intel_1"]["max"][0]} & {df_dict["intel"]["intel_1"]["std"][0]} \\\\')
            print(f"{label_amd} \\\\")
            print(f'\\hspace{{0.5cm}}Clang & {df_dict["amd"]["b_1"]["min"][2]} & {df_dict["amd"]["b_1"]["avg"][2]} & {df_dict["amd"]["b_1"]["max"][2]} & {df_dict["amd"]["b_1"]["std"][2]} \\\\')
            print(f'\\hspace{{0.5cm}}ICPX & {df_dict["amd"]["intel_1"]["min"][2]} & {df_dict["amd"]["intel_1"]["avg"][2]} & {df_dict["amd"]["intel_1"]["max"][2]} & {df_dict["amd"]["intel_1"]["std"][2]} \\\\')
        case "icpx_xl":
            group_width = bar_width * 2
            group_gap = 0.15
            indices = np.arange(len([8, 16])) * (group_width + group_gap)

            df_intel_xl_1 = df_dict["intel"]["xl_1"].drop(index=1).reset_index(drop=True)
            df_intel_xl_intel_1 = df_dict["intel"]["xl_intel_1"].drop(index=1).reset_index(drop=True)
            df_amd_xl_1 = df_dict["amd"]["xl_1"].drop(index=1).reset_index(drop=True)
            df_amd_xl_intel_1 = df_dict["amd"]["xl_intel_1"].drop(index=1).reset_index(drop=True)

            ax.bar(indices - 0.5 * bar_width, df_intel_xl_1["max"], width=bar_width, label=f"{label_intel} Clang XL Max", color="#99c1f1", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_xl_1["avg"], width=bar_width, label=f"{label_intel} Clang XL Avg", color="#3584e4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_xl_1["min"], width=bar_width, label=f"{label_intel} Clang XL Min", color="#1a5fb4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_xl_intel_1["max"], width=bar_width, label=f"{label_intel} ICPX XL Max", color="#8ff0a4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_xl_intel_1["avg"], width=bar_width, label=f"{label_intel} ICPX XL Avg", color="#33d17a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_xl_intel_1["min"], width=bar_width, label=f"{label_intel} ICPX XL Min", color="#26a269", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_xl_1["max"], width=bar_width, label=f"{label_amd} Clang XL Max", color="#f66151", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_xl_1["avg"], width=bar_width, label=f"{label_amd} Clang XL Avg", color="#e01b24", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_xl_1["min"], width=bar_width, label=f"{label_amd} Clang XL Min", color="#a51d2d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_xl_intel_1["max"], width=bar_width, label=f"{label_amd} ICPX XL Max", color="#dc8add", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_xl_intel_1["avg"], width=bar_width, label=f"{label_amd} ICPX XL Avg", color="#9141ac", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_xl_intel_1["min"], width=bar_width, label=f"{label_amd} ICPX XL Min", color="#613583", edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("ICPX XL", fontsize=12, color=foreground_color, loc="center")

            b1 = df_intel_xl_1["avg"][0] + df_amd_xl_1["avg"][1]
            b2 = df_intel_xl_intel_1["avg"][0] + df_amd_xl_intel_1["avg"][1]

            print(f"icpx_xl: {np.round((b2 - b1) / b1 * 100, 1)}")
            print(f'intel: {np.round((df_dict["intel"]["xl_intel_1"]["avg"][0] - df_dict["intel"]["xl_1"]["avg"][0]) / df_dict["intel"]["xl_1"]["avg"][0] * 100, 1)}')
            print(f'amd: {np.round((df_dict["amd"]["xl_intel_1"]["avg"][2] - df_dict["amd"]["xl_1"]["avg"][2]) / df_dict["amd"]["xl_1"]["avg"][2] * 100, 1)}')
            print(f"{label_intel} \\\\")
            print(f'\\hspace{{0.5cm}}Clang XL & {df_dict["intel"]["xl_1"]["min"][0]} & {df_dict["intel"]["xl_1"]["avg"][0]} & {df_dict["intel"]["xl_1"]["max"][0]} & {df_dict["intel"]["xl_1"]["std"][0]} \\\\')
            print(f'\\hspace{{0.5cm}}ICPX XL & {df_dict["intel"]["xl_intel_1"]["min"][0]} & {df_dict["intel"]["xl_intel_1"]["avg"][0]} & {df_dict["intel"]["xl_intel_1"]["max"][0]} & {df_dict["intel"]["xl_intel_1"]["std"][0]} \\\\')
            print(f"{label_amd} \\\\")
            print(f'\\hspace{{0.5cm}}Clang XL & {df_dict["amd"]["xl_1"]["min"][2]} & {df_dict["amd"]["xl_1"]["avg"][2]} & {df_dict["amd"]["xl_1"]["max"][2]} & {df_dict["amd"]["xl_1"]["std"][2]} \\\\')
            print(f'\\hspace{{0.5cm}}ICPX XL & {df_dict["amd"]["xl_intel_1"]["min"][2]} & {df_dict["amd"]["xl_intel_1"]["avg"][2]} & {df_dict["amd"]["xl_intel_1"]["max"][2]} & {df_dict["amd"]["xl_intel_1"]["std"][2]} \\\\')
        case "icpx_omp":
            group_width = bar_width * 2
            group_gap = 0.15
            indices = np.arange(len([8, 16])) * (group_width + group_gap)

            df_intel_omp_4 = df_dict["intel"]["omp_4"].drop(index=1).reset_index(drop=True)
            df_intel_omp_intel_4 = df_dict["intel"]["omp_intel_4"].drop(index=1).reset_index(drop=True)
            df_amd_omp_4 = df_dict["amd"]["omp_4"].drop(index=1).reset_index(drop=True)
            df_amd_omp_intel_4 = df_dict["amd"]["omp_intel_4"].drop(index=1).reset_index(drop=True)

            ax.bar(indices - 0.5 * bar_width, df_intel_omp_4["max"], width=bar_width, label=f"{label_intel} Clang OpenMP Max", color="#99c1f1", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_omp_4["avg"], width=bar_width, label=f"{label_intel} Clang OpenMP Avg", color="#3584e4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_omp_4["min"], width=bar_width, label=f"{label_intel} Clang OpenMP Min", color="#1a5fb4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_omp_intel_4["max"], width=bar_width, label=f"{label_intel} ICPX OpenMP Max", color="#8ff0a4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_omp_intel_4["avg"], width=bar_width, label=f"{label_intel} ICPX OpenMP Avg", color="#33d17a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_omp_intel_4["min"], width=bar_width, label=f"{label_intel} ICPX OpenMP Min", color="#26a269", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_omp_4["max"], width=bar_width, label=f"{label_amd} Clang OpenMP Max", color="#f66151", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_omp_4["avg"], width=bar_width, label=f"{label_amd} Clang OpenMP Avg", color="#e01b24", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_omp_4["min"], width=bar_width, label=f"{label_amd} Clang OpenMP Min", color="#a51d2d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_omp_intel_4["max"], width=bar_width, label=f"{label_amd} ICPX OpenMP Max", color="#dc8add", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_omp_intel_4["avg"], width=bar_width, label=f"{label_amd} ICPX OpenMP Avg", color="#9141ac", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_omp_intel_4["min"], width=bar_width, label=f"{label_amd} ICPX OpenMP Min", color="#613583", edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("ICPX OpenMP", fontsize=12, color=foreground_color, loc="center")

            b1 = df_intel_omp_4["avg"][0] + df_amd_omp_4["avg"][1]
            b2 = df_intel_omp_intel_4["avg"][0] + df_amd_omp_intel_4["avg"][1]

            print(f"icpx_omp: {np.round((b2 - b1) / b1 * 100, 1)}")
            print(f'intel: {np.round((df_dict["intel"]["omp_intel_4"]["avg"][0] - df_dict["intel"]["omp_4"]["avg"][0]) / df_dict["intel"]["omp_4"]["avg"][0] * 100, 1)}')
            print(f'amd: {np.round((df_dict["amd"]["omp_intel_4"]["avg"][2] - df_dict["amd"]["omp_4"]["avg"][2]) / df_dict["amd"]["omp_4"]["avg"][2] * 100, 1)}')
            print(f"{label_intel} \\\\")
            print(f'\\hspace{{0.5cm}}Clang OpenMP & {df_dict["intel"]["omp_4"]["min"][0]} & {df_dict["intel"]["omp_4"]["avg"][0]} & {df_dict["intel"]["omp_4"]["max"][0]} & {df_dict["intel"]["omp_4"]["std"][0]} \\\\')
            print(f'\\hspace{{0.5cm}}ICPX OpenMP & {df_dict["intel"]["omp_intel_4"]["min"][0]} & {df_dict["intel"]["omp_intel_4"]["avg"][0]} & {df_dict["intel"]["omp_intel_4"]["max"][0]} & {df_dict["intel"]["omp_intel_4"]["std"][0]} \\\\')
            print(f"{label_amd} \\\\")
            print(f'\\hspace{{0.5cm}}Clang OpenMP & {df_dict["amd"]["omp_4"]["min"][2]} & {df_dict["amd"]["omp_4"]["avg"][2]} & {df_dict["amd"]["omp_4"]["max"][2]} & {df_dict["amd"]["omp_4"]["std"][2]} \\\\')
            print(f'\\hspace{{0.5cm}}ICPX OpenMP & {df_dict["amd"]["omp_intel_4"]["min"][2]} & {df_dict["amd"]["omp_intel_4"]["avg"][2]} & {df_dict["amd"]["omp_intel_4"]["max"][2]} & {df_dict["amd"]["omp_intel_4"]["std"][2]} \\\\')
        case "icpx_omp_xl":
            group_width = bar_width * 2
            group_gap = 0.15
            indices = np.arange(len([8, 16])) * (group_width + group_gap)

            df_intel_omp_xl_4 = df_dict["intel"]["omp_xl_4"].drop(index=1).reset_index(drop=True)
            df_intel_omp_xl_intel_4 = df_dict["intel"]["omp_xl_intel_4"].drop(index=1).reset_index(drop=True)
            df_amd_omp_xl_4 = df_dict["amd"]["omp_xl_4"].drop(index=1).reset_index(drop=True)
            df_amd_omp_xl_intel_4 = df_dict["amd"]["omp_xl_intel_4"].drop(index=1).reset_index(drop=True)

            ax.bar(indices - 0.5 * bar_width, df_intel_omp_xl_4["max"], width=bar_width, label=f"{label_intel} Clang OpenMP XL Max", color="#99c1f1", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_omp_xl_4["avg"], width=bar_width, label=f"{label_intel} Clang OpenMP XL Avg", color="#3584e4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_omp_xl_4["min"], width=bar_width, label=f"{label_intel} Clang OpenMP XL Min", color="#1a5fb4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_omp_xl_intel_4["max"], width=bar_width, label=f"{label_intel} ICPX OpenMP XL Max", color="#8ff0a4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_omp_xl_intel_4["avg"], width=bar_width, label=f"{label_intel} ICPX OpenMP XL Avg", color="#33d17a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_omp_xl_intel_4["min"], width=bar_width, label=f"{label_intel} ICPX OpenMP XL Min", color="#26a269", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_omp_xl_4["max"], width=bar_width, label=f"{label_amd} Clang OpenMP XL Max", color="#f66151", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_omp_xl_4["avg"], width=bar_width, label=f"{label_amd} Clang OpenMP XL Avg", color="#e01b24", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_omp_xl_4["min"], width=bar_width, label=f"{label_amd} Clang OpenMP XL Min", color="#a51d2d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_omp_xl_intel_4["max"], width=bar_width, label=f"{label_amd} ICPX OpenMP XL Max", color="#dc8add", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_omp_xl_intel_4["avg"], width=bar_width, label=f"{label_amd} ICPX OpenMP XL Avg", color="#9141ac", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_omp_xl_intel_4["min"], width=bar_width, label=f"{label_amd} ICPX OpenMP XL Min", color="#613583", edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("ICPX OpenMP XL", fontsize=12, color=foreground_color, loc="center")

            b1 = df_intel_omp_xl_4["avg"][0] + df_amd_omp_xl_4["avg"][1]
            b2 = df_intel_omp_xl_intel_4["avg"][0] + df_amd_omp_xl_intel_4["avg"][1]

            print(f"icpx_omp_xl: {np.round((b2 - b1) / b1 * 100, 1)}")
            print(f'intel: {np.round((df_dict["intel"]["omp_xl_intel_4"]["avg"][0] - df_dict["intel"]["omp_xl_4"]["avg"][0]) / df_dict["intel"]["omp_xl_4"]["avg"][0] * 100, 1)}')
            print(f'amd: {np.round((df_dict["amd"]["omp_xl_intel_4"]["avg"][2] - df_dict["amd"]["omp_xl_4"]["avg"][2]) / df_dict["amd"]["omp_xl_4"]["avg"][2] * 100, 1)}')
            print(f"{label_intel} \\\\")
            print(f'\\hspace{{0.5cm}}Clang OpenMP XL & {df_dict["intel"]["omp_xl_4"]["min"][0]} & {df_dict["intel"]["omp_xl_4"]["avg"][0]} & {df_dict["intel"]["omp_xl_4"]["max"][0]} & {df_dict["intel"]["omp_xl_4"]["std"][0]} \\\\')
            print(f'\\hspace{{0.5cm}}ICPX OpenMP XL & {df_dict["intel"]["omp_xl_intel_4"]["min"][0]} & {df_dict["intel"]["omp_xl_intel_4"]["avg"][0]} & {df_dict["intel"]["omp_xl_intel_4"]["max"][0]} & {df_dict["intel"]["omp_xl_intel_4"]["std"][0]} \\\\')
            print(f"{label_amd} \\\\")
            print(f'\\hspace{{0.5cm}}Clang OpenMP XL & {df_dict["amd"]["omp_xl_4"]["min"][2]} & {df_dict["amd"]["omp_xl_4"]["avg"][2]} & {df_dict["amd"]["omp_xl_4"]["max"][2]} & {df_dict["amd"]["omp_xl_4"]["std"][2]} \\\\')
            print(f'\\hspace{{0.5cm}}ICPX OpenMP XL & {df_dict["amd"]["omp_xl_intel_4"]["min"][2]} & {df_dict["amd"]["omp_xl_intel_4"]["avg"][2]} & {df_dict["amd"]["omp_xl_intel_4"]["max"][2]} & {df_dict["amd"]["omp_xl_intel_4"]["std"][2]} \\\\')
        case "final":
            group_width = bar_width * 2
            group_gap = 0.15
            indices = np.arange(len(threads_omp)) * (group_width + group_gap)

            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["b_1"]["max"], width=bar_width, label=f"{label_intel} Naive Max", color="#99c1f1", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["b_1"]["avg"], width=bar_width, label=f"{label_intel} Naive Avg", color="#3584e4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["b_1"]["min"], width=bar_width, label=f"{label_intel} Naive Min", color="#1a5fb4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["b_7"]["max"], width=bar_width, label=f"{label_intel} Final Max", color="#8ff0a4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["b_7"]["avg"], width=bar_width, label=f"{label_intel} Final Avg", color="#33d17a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["b_7"]["min"], width=bar_width, label=f"{label_intel} Final Min", color="#26a269", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["apple"]["b_1"]["max"], width=bar_width, label=f"{label_apple} Naive Max", color="#f9f06b", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["apple"]["b_1"]["avg"], width=bar_width, label=f"{label_apple} Naive Avg", color="#f6d32d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["apple"]["b_1"]["min"], width=bar_width, label=f"{label_apple} Naive Min", color="#e5a50a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["apple"]["b_7"]["max"], width=bar_width, label=f"{label_apple} Final Max", color="#ffbe6f", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["apple"]["b_7"]["avg"], width=bar_width, label=f"{label_apple} Final Avg", color="#ff7800", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["apple"]["b_7"]["min"], width=bar_width, label=f"{label_apple} Final Min", color="#c64600", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["b_1"]["max"], width=bar_width, label=f"{label_amd} Naive Max", color="#f66151", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["b_1"]["avg"], width=bar_width, label=f"{label_amd} Naive Avg", color="#e01b24", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["b_1"]["min"], width=bar_width, label=f"{label_amd} Naive Min", color="#a51d2d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["b_7"]["max"], width=bar_width, label=f"{label_amd} Final Max", color="#dc8add", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["b_7"]["avg"], width=bar_width, label=f"{label_amd} Final Avg", color="#9141ac", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["b_7"]["min"], width=bar_width, label=f"{label_amd} Final Min", color="#613583", edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("Final", fontsize=12, color=foreground_color, loc="center")

            b1 = df_dict["intel"]["b_1"]["avg"][0] + df_dict["apple"]["b_1"]["avg"][1] + df_dict["amd"]["b_1"]["avg"][2]
            b2 = df_dict["intel"]["b_7"]["avg"][0] + df_dict["apple"]["b_7"]["avg"][1] + df_dict["amd"]["b_7"]["avg"][2]

            print(f"final: {np.round((b2 - b1) / b1 * 100, 0)}")
        case "final_xl":
            group_width = bar_width * 2
            group_gap = 0.15
            indices = np.arange(len(threads_omp)) * (group_width + group_gap)

            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["xl_1"]["max"], width=bar_width, label=f"{label_intel} Naive Max", color="#99c1f1", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["xl_1"]["avg"], width=bar_width, label=f"{label_intel} Naive Avg", color="#3584e4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["xl_1"]["min"], width=bar_width, label=f"{label_intel} Naive Min", color="#1a5fb4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["xl_7"]["max"], width=bar_width, label=f"{label_intel} Final Max", color="#8ff0a4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["xl_7"]["avg"], width=bar_width, label=f"{label_intel} Final Avg", color="#33d17a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["xl_7"]["min"], width=bar_width, label=f"{label_intel} Final Min", color="#26a269", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["apple"]["xl_1"]["max"], width=bar_width, label=f"{label_apple} Naive Max", color="#f9f06b", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["apple"]["xl_1"]["avg"], width=bar_width, label=f"{label_apple} Naive Avg", color="#f6d32d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["apple"]["xl_1"]["min"], width=bar_width, label=f"{label_apple} Naive Min", color="#e5a50a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["apple"]["xl_7"]["max"], width=bar_width, label=f"{label_apple} Final Max", color="#ffbe6f", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["apple"]["xl_7"]["avg"], width=bar_width, label=f"{label_apple} Final Avg", color="#ff7800", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["apple"]["xl_7"]["min"], width=bar_width, label=f"{label_apple} Final Min", color="#c64600", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["xl_1"]["max"], width=bar_width, label=f"{label_amd} Naive Max", color="#f66151", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["xl_1"]["avg"], width=bar_width, label=f"{label_amd} Naive Avg", color="#e01b24", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["xl_1"]["min"], width=bar_width, label=f"{label_amd} Naive Min", color="#a51d2d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["xl_7"]["max"], width=bar_width, label=f"{label_amd} Final Max", color="#dc8add", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["xl_7"]["avg"], width=bar_width, label=f"{label_amd} Final Avg", color="#9141ac", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["xl_7"]["min"], width=bar_width, label=f"{label_amd} Final Min", color="#613583", edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("Final XL", fontsize=12, color=foreground_color, loc="center")

            b1 = df_dict["intel"]["xl_1"]["avg"][0] + df_dict["apple"]["xl_1"]["avg"][1] + df_dict["amd"]["xl_1"]["avg"][2]
            b2 = df_dict["intel"]["xl_7"]["avg"][0] + df_dict["apple"]["xl_7"]["avg"][1] + df_dict["amd"]["xl_7"]["avg"][2]

            print(f"final_xl: {np.round((b2 - b1) / b1 * 100, 0)}")
        case "cuda":
            group_width = bar_width * 2
            group_gap = 0.15
            indices = np.arange(len([8, 16])) * (group_width + group_gap)

            df_intel_nvidia_3 = df_dict["intel"]["nvidia_3"].drop(index=1).reset_index(drop=True)
            df_amd_nvidia_3 = df_dict["amd"]["nvidia_3"].drop(index=1).reset_index(drop=True)

            ax.bar(indices, df_intel_nvidia_3["max"], width=bar_width, label=f"{label_intel_nvidia} Naive Max", color="#99c1f1", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_intel_nvidia_3["avg"], width=bar_width, label=f"{label_intel_nvidia} Naive Avg", color="#3584e4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_intel_nvidia_3["min"], width=bar_width, label=f"{label_intel_nvidia} Naive Min", color="#1a5fb4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_amd_nvidia_3["max"], width=bar_width, label=f"{label_amd_nvidia} Naive Max", color="#f66151", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_amd_nvidia_3["avg"], width=bar_width, label=f"{label_amd_nvidia} Naive Avg", color="#e01b24", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_amd_nvidia_3["min"], width=bar_width, label=f"{label_amd_nvidia} Naive Min", color="#a51d2d", edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("CUDA Naive", fontsize=12, color=foreground_color, loc="center")
        case "cuda_xl":
            group_width = bar_width * 2
            group_gap = 0.15
            indices = np.arange(len([8, 16])) * (group_width + group_gap)

            df_intel_xl_nvidia_3 = df_dict["intel"]["xl_nvidia_3"].drop(index=1).reset_index(drop=True)
            df_amd_xl_nvidia_3 = df_dict["amd"]["xl_nvidia_3"].drop(index=1).reset_index(drop=True)

            ax.bar(indices, df_intel_xl_nvidia_3["max"], width=bar_width, label=f"{label_intel_nvidia} Naive XL Max", color="#99c1f1", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_intel_xl_nvidia_3["avg"], width=bar_width, label=f"{label_intel_nvidia} Naive XL Avg", color="#3584e4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_intel_xl_nvidia_3["min"], width=bar_width, label=f"{label_intel_nvidia} Naive XL Min", color="#1a5fb4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_amd_xl_nvidia_3["max"], width=bar_width, label=f"{label_amd_nvidia} Naive XL Max", color="#f66151", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_amd_xl_nvidia_3["avg"], width=bar_width, label=f"{label_amd_nvidia} Naive XL Avg", color="#e01b24", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_amd_xl_nvidia_3["min"], width=bar_width, label=f"{label_amd_nvidia} Naive XL Min", color="#a51d2d", edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("CUDA Naive XL", fontsize=12, color=foreground_color, loc="center")
        case "cuda_final":
            group_width = bar_width * 2
            group_gap = 0.15
            indices = np.arange(len([8, 16])) * (group_width + group_gap)

            df_intel_nvidia_3 = df_dict["intel"]["nvidia_3"].drop(index=1).reset_index(drop=True)
            df_intel_nvidia_7 = df_dict["intel"]["nvidia_7"].drop(index=1).reset_index(drop=True)
            df_amd_nvidia_3 = df_dict["amd"]["nvidia_3"].drop(index=1).reset_index(drop=True)
            df_amd_nvidia_7 = df_dict["amd"]["nvidia_7"].drop(index=1).reset_index(drop=True)

            ax.bar(indices - 0.5 * bar_width, df_intel_nvidia_3["max"], width=bar_width, label=f"{label_intel_nvidia} Naive Max", color="#99c1f1", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_nvidia_3["avg"], width=bar_width, label=f"{label_intel_nvidia} Naive Avg", color="#3584e4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_nvidia_3["min"], width=bar_width, label=f"{label_intel_nvidia} Naive Min", color="#1a5fb4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_nvidia_7["max"], width=bar_width, label=f"{label_intel_nvidia} Final Max", color="#8ff0a4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_nvidia_7["avg"], width=bar_width, label=f"{label_intel_nvidia} Final Avg", color="#33d17a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_nvidia_7["min"], width=bar_width, label=f"{label_intel_nvidia} Final Min", color="#26a269", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_nvidia_3["max"], width=bar_width, label=f"{label_amd_nvidia} Naive Max", color="#f66151", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_nvidia_3["avg"], width=bar_width, label=f"{label_amd_nvidia} Naive Avg", color="#e01b24", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_nvidia_3["min"], width=bar_width, label=f"{label_amd_nvidia} Naive Min", color="#a51d2d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_nvidia_7["max"], width=bar_width, label=f"{label_amd_nvidia} Final Max", color="#dc8add", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_nvidia_7["avg"], width=bar_width, label=f"{label_amd_nvidia} Final Avg", color="#9141ac", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_nvidia_7["min"], width=bar_width, label=f"{label_amd_nvidia} Final Min", color="#613583", edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("CUDA Final", fontsize=12, color=foreground_color, loc="center")

            b1 = df_intel_nvidia_3["avg"][0] + df_amd_nvidia_3["avg"][0]
            b2 = df_intel_nvidia_7["avg"][0] + df_amd_nvidia_7["avg"][0]

            print(f"cuda_final: {np.round((b2 - b1) / b1 * 100, 0)}")
        case "cuda_final_xl":
            group_width = bar_width * 2
            group_gap = 0.15
            indices = np.arange(len([8, 16])) * (group_width + group_gap)

            df_intel_xl_nvidia_3 = df_dict["intel"]["xl_nvidia_3"].drop(index=1).reset_index(drop=True)
            df_intel_xl_nvidia_7 = df_dict["intel"]["xl_nvidia_7"].drop(index=1).reset_index(drop=True)
            df_amd_xl_nvidia_3 = df_dict["amd"]["xl_nvidia_3"].drop(index=1).reset_index(drop=True)
            df_amd_xl_nvidia_7 = df_dict["amd"]["xl_nvidia_7"].drop(index=1).reset_index(drop=True)

            ax.bar(indices - 0.5 * bar_width, df_intel_xl_nvidia_3["max"], width=bar_width, label=f"{label_intel_nvidia} Naive XL Max", color="#99c1f1", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_xl_nvidia_3["avg"], width=bar_width, label=f"{label_intel_nvidia} Naive XL Avg", color="#3584e4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_intel_xl_nvidia_3["min"], width=bar_width, label=f"{label_intel_nvidia} Naive XL Min", color="#1a5fb4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_xl_nvidia_7["max"], width=bar_width, label=f"{label_intel_nvidia} Final XL Max", color="#8ff0a4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_xl_nvidia_7["avg"], width=bar_width, label=f"{label_intel_nvidia} Final XL Avg", color="#33d17a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_intel_xl_nvidia_7["min"], width=bar_width, label=f"{label_intel_nvidia} Final XL Min", color="#26a269", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_xl_nvidia_3["max"], width=bar_width, label=f"{label_amd_nvidia} Naive XL Max", color="#f66151", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_xl_nvidia_3["avg"], width=bar_width, label=f"{label_amd_nvidia} Naive XL Avg", color="#e01b24", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_amd_xl_nvidia_3["min"], width=bar_width, label=f"{label_amd_nvidia} Naive XL Min", color="#a51d2d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_xl_nvidia_7["max"], width=bar_width, label=f"{label_amd_nvidia} Final XL Max", color="#dc8add", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_xl_nvidia_7["avg"], width=bar_width, label=f"{label_amd_nvidia} Final XL Avg", color="#9141ac", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_amd_xl_nvidia_7["min"], width=bar_width, label=f"{label_amd_nvidia} Final XL Min", color="#613583", edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("CUDA Final XL", fontsize=12, color=foreground_color, loc="center")

            b1 = df_intel_xl_nvidia_3["avg"][0] + df_amd_xl_nvidia_3["avg"][0]
            b2 = df_intel_xl_nvidia_7["avg"][0] + df_amd_xl_nvidia_7["avg"][0]

            print(f"cuda_final_xl: {np.round((b2 - b1) / b1 * 100, 0)}")
        case "cpu_gpu":
            group_width = bar_width * 2
            group_gap = 0.15
            indices = np.arange(len(threads_omp)) * (group_width + group_gap)

            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["b_7"]["max"], width=bar_width, label=f"{label_intel} Max", color="#99c1f1", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["b_7"]["avg"], width=bar_width, label=f"{label_intel} Avg", color="#3584e4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["b_7"]["min"], width=bar_width, label=f"{label_intel} Min", color="#1a5fb4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["nvidia_7"]["max"], width=bar_width, label=f"{label_intel_nvidia} Max", color="#8ff0a4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["nvidia_7"]["avg"], width=bar_width, label=f"{label_intel_nvidia} Avg", color="#33d17a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["nvidia_7"]["min"], width=bar_width, label=f"{label_intel_nvidia} Min", color="#26a269", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_dict["apple"]["b_7"]["max"], width=bar_width, label=f"{label_apple} Max", color="#f9f06b", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_dict["apple"]["b_7"]["avg"], width=bar_width, label=f"{label_apple} Avg", color="#f6d32d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_dict["apple"]["b_7"]["min"], width=bar_width, label=f"{label_apple} Min", color="#e5a50a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["b_7"]["max"], width=bar_width, label=f"{label_amd} Max", color="#f66151", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["b_7"]["avg"], width=bar_width, label=f"{label_amd} Avg", color="#e01b24", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["b_7"]["min"], width=bar_width, label=f"{label_amd} Min", color="#a51d2d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["nvidia_7"]["max"], width=bar_width, label=f"{label_amd_nvidia} Max", color="#dc8add", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["nvidia_7"]["avg"], width=bar_width, label=f"{label_amd_nvidia} Avg", color="#9141ac", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["nvidia_7"]["min"], width=bar_width, label=f"{label_amd_nvidia} Min", color="#613583", edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("CPU vs. GPU", fontsize=12, color=foreground_color, loc="center")

            b1 = df_dict["intel"]["b_7"]["avg"][0] + df_dict["apple"]["b_7"]["avg"][1] + df_dict["amd"]["b_7"]["avg"][2]
            b2 = df_dict["intel"]["nvidia_7"]["avg"][0] + df_dict["apple"]["b_7"]["avg"][1] + df_dict["amd"]["nvidia_7"]["avg"][2]

            print(f"cpu_gpu: {np.round((b2 - b1) / b1 * 100, 0)}")
        case "cpu_gpu_xl":
            group_width = bar_width * 2
            group_gap = 0.15
            indices = np.arange(len(threads_omp)) * (group_width + group_gap)

            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["xl_7"]["max"], width=bar_width, label=f"{label_intel} XL Max", color="#99c1f1", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["xl_7"]["avg"], width=bar_width, label=f"{label_intel} XL Avg", color="#3584e4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["intel"]["xl_7"]["min"], width=bar_width, label=f"{label_intel} XL Min", color="#1a5fb4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["xl_nvidia_7"]["max"], width=bar_width, label=f"{label_intel_nvidia} XL Max", color="#8ff0a4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["xl_nvidia_7"]["avg"], width=bar_width, label=f"{label_intel_nvidia} XL Avg", color="#33d17a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["intel"]["xl_nvidia_7"]["min"], width=bar_width, label=f"{label_intel_nvidia} XL Min", color="#26a269", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_dict["apple"]["xl_7"]["max"], width=bar_width, label=f"{label_apple} XL Max", color="#f9f06b", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_dict["apple"]["xl_7"]["avg"], width=bar_width, label=f"{label_apple} XL Avg", color="#f6d32d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_dict["apple"]["xl_7"]["min"], width=bar_width, label=f"{label_apple} XL Min", color="#e5a50a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["xl_7"]["max"], width=bar_width, label=f"{label_amd} XL Max", color="#f66151", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["xl_7"]["avg"], width=bar_width, label=f"{label_amd} XL Avg", color="#e01b24", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_dict["amd"]["xl_7"]["min"], width=bar_width, label=f"{label_amd} XL Min", color="#a51d2d", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["xl_nvidia_7"]["max"], width=bar_width, label=f"{label_amd_nvidia} XL Max", color="#dc8add", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["xl_nvidia_7"]["avg"], width=bar_width, label=f"{label_amd_nvidia} XL Avg", color="#9141ac", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_dict["amd"]["xl_nvidia_7"]["min"], width=bar_width, label=f"{label_amd_nvidia} XL Min", color="#613583", edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("CPU vs. GPU XL", fontsize=12, color=foreground_color, loc="center")

            b1 = df_dict["intel"]["xl_7"]["avg"][0] + df_dict["apple"]["xl_7"]["avg"][1] + df_dict["amd"]["xl_7"]["avg"][2]
            b2 = df_dict["intel"]["xl_nvidia_7"]["avg"][0] + df_dict["apple"]["xl_7"]["avg"][1] + df_dict["amd"]["xl_nvidia_7"]["avg"][2]

            print(f"cpu_gpu_xl: {np.round((b2 - b1) / b1 * 100, 0)}")

    #ax.set_xlabel("Threads", fontsize=12, color=foreground_color)
    ax.set_ylabel("Total time (µs)", fontsize=12, color=foreground_color)
    ax.set_xticks(indices)
    if "sse" in f or "avx" in f or "icpx" in f:
        ax.set_xticklabels([label_intel, label_amd], color=foreground_color)
    elif "neon" in f or "amx" in f:
        ax.set_xticklabels(["w/o Multithreading", "w/ Multithreading"], color=foreground_color)
    elif "cuda" in f:
        ax.set_xticklabels([label_intel_nvidia, label_amd_nvidia], color=foreground_color)
    elif "cpu" in f:
        ax.set_xticklabels([f"{label_intel}\nvs.\n{label_intel_nvidia}", label_apple, f"{label_amd}\nvs.\n{label_amd_nvidia}"], color=foreground_color)
    else:
        ax.set_xticklabels([label_intel, label_apple, label_amd], color=foreground_color)
    ax.tick_params(axis="x", colors=foreground_color)
    ax.tick_params(axis="y", colors=foreground_color)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), facecolor=background_color, edgecolor=foreground_color, labelcolor=foreground_color)

    for spine in ax.spines.values():
        spine.set_edgecolor(foreground_color)

    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    plt.tight_layout()

    match f:
        case "naive":
            plt.savefig("../Graphs/7. Presentation/Naive.png", dpi=300)
        case "naive_xl":
            plt.savefig("../Graphs/7. Presentation/Naive XL.png", dpi=300)
        case "mt_init":
            plt.savefig("../Graphs/7. Presentation/Initial Multithreading.png", dpi=300)
        case "mt_init_xl":
            plt.savefig("../Graphs/7. Presentation/Initial Multithreading XL.png", dpi=300)
        case "mt":
            plt.savefig("../Graphs/7. Presentation/Multithreading.png", dpi=300)
        case "mt_xl":
            plt.savefig("../Graphs/7. Presentation/Multithreading XL.png", dpi=300)
        case "omp":
            plt.savefig("../Graphs/7. Presentation/OpenMP.png", dpi=300)
        case "omp_xl":
            plt.savefig("../Graphs/7. Presentation/OpenMP XL.png", dpi=300)
        case "sse":
            plt.savefig("../Graphs/7. Presentation/SSE.png", dpi=300)
        case "sse_xl":
            plt.savefig("../Graphs/7. Presentation/SSE XL.png", dpi=300)
        case "avx":
            plt.savefig("../Graphs/7. Presentation/AVX.png", dpi=300)
        case "avx_xl":
            plt.savefig("../Graphs/7. Presentation/AVX XL.png", dpi=300)
        case "neon":
            plt.savefig("../Graphs/7. Presentation/Neon.png", dpi=300)
        case "neon_xl":
            plt.savefig("../Graphs/7. Presentation/Neon XL.png", dpi=300)
        case "amx":
            plt.savefig("../Graphs/7. Presentation/AMX.png", dpi=300)
        case "amx_xxl":
            plt.savefig("../Graphs/7. Presentation/AMX XXL.png", dpi=300)
        case "int":
            plt.savefig("../Graphs/7. Presentation/Quantization.png", dpi=300)
        case "int_xl":
            plt.savefig("../Graphs/7. Presentation/Quantization XL.png", dpi=300)
        case "icpx":
            plt.savefig("../Graphs/7. Presentation/ICPX.png", dpi=300)
        case "icpx_xl":
            plt.savefig("../Graphs/7. Presentation/ICPX XL.png", dpi=300)
        case "icpx_omp":
            plt.savefig("../Graphs/7. Presentation/ICPX OpenMP.png", dpi=300)
        case "icpx_omp_xl":
            plt.savefig("../Graphs/7. Presentation/ICPX OpenMP XL.png", dpi=300)
        case "final":
            plt.savefig("../Graphs/7. Presentation/Final.png", dpi=300)
        case "final_xl":
            plt.savefig("../Graphs/7. Presentation/Final XL.png", dpi=300)
        case "cuda":
            plt.savefig("../Graphs/7. Presentation/CUDA.png", dpi=300)
        case "cuda_xl":
            plt.savefig("../Graphs/7. Presentation/CUDA XL.png", dpi=300)
        case "cuda_final":
            plt.savefig("../Graphs/7. Presentation/CUDA Final.png", dpi=300)
        case "cuda_final_xl":
            plt.savefig("../Graphs/7. Presentation/CUDA Final XL.png", dpi=300)
        case "cpu_gpu":
            plt.savefig("../Graphs/7. Presentation/CPU vs. GPU.png", dpi=300)
        case "cpu_gpu_xl":
            plt.savefig("../Graphs/7. Presentation/CPU vs. GPU XL.png", dpi=300)

    # plt.show()
    plt.close(fig)
