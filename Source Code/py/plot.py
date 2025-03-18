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
        "neon_xxl": "./csv/7. Presentation/apple/benchmark_neon_xxl.csv",
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
    "max": [0.0] * len(threads_omp)
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
                else:
                    df_dict[group][key].loc[threads_omp.index(threads[i]), "min"] = np.min(tt_dict[group][key][10*i:10*(i+1)])
                    df_dict[group][key].loc[threads_omp.index(threads[i]), "avg"] = np.average(tt_dict[group][key][10*i:10*(i+1)])
                    df_dict[group][key].loc[threads_omp.index(threads[i]), "max"] = np.max(tt_dict[group][key][10*i:10*(i+1)])
            elif group == "apple" and ("amx" in key or "neon" in key) and threads[i] == 1:
                df_dict[group][key].loc[threads.index(threads[i]), "min"] = np.min(tt_dict[group][key][10*i:10*(i+1)])
                df_dict[group][key].loc[threads.index(threads[i]), "avg"] = np.average(tt_dict[group][key][10*i:10*(i+1)])
                df_dict[group][key].loc[threads.index(threads[i]), "max"] = np.max(tt_dict[group][key][10*i:10*(i+1)])

figures = ["naive", "naive_xl", "mt_init", "mt_init_xl", "mt", "mt_xl", "omp", "omp_xl", "sse", "sse_xl", "avx", "avx_xl", "neon", "neon_xl"]
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
        case "neon":
            group_width = bar_width * 2
            group_gap = 0.15
            indices = np.arange(len([1, 11])) * (group_width + group_gap)

            df_apple_no_neon_7 = df_dict["apple"]["no_neon_7"].drop(index=2).reset_index(drop=True)
            df_apple_neon_7 = df_dict["apple"]["neon_7"].drop(index=2).reset_index(drop=True)

            ax.bar(indices - 0.5 * bar_width, df_apple_no_neon_7["max"], width=bar_width, label=f"{label_intel} Multithreading Max", color="#99c1f1", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_no_neon_7["avg"], width=bar_width, label=f"{label_intel} Multithreading Avg", color="#3584e4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_no_neon_7["min"], width=bar_width, label=f"{label_intel} Multithreading Min", color="#1a5fb4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_neon_7["max"], width=bar_width, label=f"{label_intel} Multithreading + Neon Max", color="#8ff0a4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_neon_7["avg"], width=bar_width, label=f"{label_intel} Multithreading + Neon Avg", color="#33d17a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_neon_7["min"], width=bar_width, label=f"{label_intel} Multithreading + Neon Min", color="#26a269", edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("Neon", fontsize=12, color=foreground_color, loc="center")
        case "neon_xl":
            group_width = bar_width * 2
            group_gap = 0.15
            indices = np.arange(len([1, 11])) * (group_width + group_gap)

            df_apple_no_neon_xl_7 = df_dict["apple"]["no_neon_xl_7"].drop(index=2).reset_index(drop=True)
            df_apple_xl_7 = df_dict["apple"]["neon_xl_7"].drop(index=2).reset_index(drop=True)

            ax.bar(indices - 0.5 * bar_width, df_apple_no_neon_xl_7["max"], width=bar_width, label=f"{label_intel} Multithreading XL Max", color="#99c1f1", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_no_neon_xl_7["avg"], width=bar_width, label=f"{label_intel} Multithreading XL Avg", color="#3584e4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 0.5 * bar_width, df_apple_no_neon_xl_7["min"], width=bar_width, label=f"{label_intel} Multithreading XL Min", color="#1a5fb4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_xl_7["max"], width=bar_width, label=f"{label_intel} Multithreading + Neon XL Max", color="#8ff0a4", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_xl_7["avg"], width=bar_width, label=f"{label_intel} Multithreading + Neon XL Avg", color="#33d17a", edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + 0.5 * bar_width, df_apple_xl_7["min"], width=bar_width, label=f"{label_intel} Multithreading + Neon XL Min", color="#26a269", edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("Neon XL", fontsize=12, color=foreground_color, loc="center")

    if "naive" in f:
        pass
    elif "cuda" in f or "cpu" in f:
        ax.set_xlabel("Device", fontsize=12, color=foreground_color)
    else:
        ax.set_xlabel("Threads", fontsize=12, color=foreground_color)
    ax.set_ylabel("Total time (µs)", fontsize=12, color=foreground_color)
    ax.set_xticks(indices)
    if "naive" in f:
        ax.set_xticklabels([label_intel, label_apple, label_amd], color=foreground_color)
    elif "sse" in f or "avx" in f:
        ax.set_xticklabels([8, 16], color=foreground_color)
    elif "neon" in f:
        ax.set_xticklabels([1, 11], color=foreground_color)
    elif "cuda" in f or "cpu" in f:
        ax.set_xticklabels(["Laptop", "Desktop"], color=foreground_color)
    else:
        ax.set_xticklabels(threads_omp, color=foreground_color)
    ax.tick_params(axis="x", colors=foreground_color)
    ax.tick_params(axis="y", colors=foreground_color)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), facecolor=background_color, edgecolor=foreground_color, labelcolor=foreground_color)

    for spine in ax.spines.values():
        spine.set_edgecolor(foreground_color)

    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    plt.tight_layout()

    try:
        os.rmdir("../Graphs/7. Presentation")
        os.mkdir("../Graphs/7. Presentation")
    except:
        pass

    match f:
        case "naive":
            plt.savefig("../Graphs/7. Presentation/Naive.png")
        case "naive_xl":
            plt.savefig("../Graphs/7. Presentation/Naive XL.png")
        case "mt_init":
            plt.savefig("../Graphs/7. Presentation/Initial Multithreading.png")
        case "mt_init_xl":
            plt.savefig("../Graphs/7. Presentation/Initial Multithreading XL.png")
        case "mt":
            plt.savefig("../Graphs/7. Presentation/Multithreading.png")
        case "mt_xl":
            plt.savefig("../Graphs/7. Presentation/Multithreading XL.png")
        case "omp":
            plt.savefig("../Graphs/7. Presentation/OpenMP.png")
        case "omp_xl":
            plt.savefig("../Graphs/7. Presentation/OpenMP XL.png")
        case "sse":
            plt.savefig("../Graphs/7. Presentation/SSE.png")
        case "sse_xl":
            plt.savefig("../Graphs/7. Presentation/SSE XL.png")
        case "avx":
            plt.savefig("../Graphs/7. Presentation/AVX.png")
        case "avx_xl":
            plt.savefig("../Graphs/7. Presentation/AVX XL.png")
        case "neon":
            plt.savefig("../Graphs/7. Presentation/Neon.png")
        case "neon_xl":
            plt.savefig("../Graphs/7. Presentation/Neon XL.png")

    # plt.show()
    plt.close(fig)
