#!/usr/bin/env python
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

label_amd = "AMD Ryzen 7 3800XT"
label_apple = "Apple M3 Pro 11-Core"
label_intel = "Intel Core i7 1065G7"

foreground_color = '#242424'
background_color = '#ffffff'

threads = [1, 4, 5, 8, 11, 16]
tdp = [105, 27, 15]

"""
READ BENCHMARKS
"""
# amd
df_amd_raw = pd.read_csv("./csv/mt/amd/benchmark.csv")
df_amd_xl_raw = pd.read_csv("./csv/mt/amd/benchmark_xl.csv")
df_amd_intel_raw = pd.read_csv("./csv/mt/amd/benchmark_intel.csv")
df_amd_xl_intel_raw = pd.read_csv("./csv/mt/amd/benchmark_xl_intel.csv")

tt_amd = df_amd_raw['total_time_us'].values
create_mt_amd = df_amd_raw['create_mt_time_us'].values
join_mt_amd = df_amd_raw['join_mt_time_us'].values

tt_amd_xl = df_amd_xl_raw['total_time_us'].values
create_mt_amd_xl = df_amd_xl_raw['create_mt_time_us'].values
join_mt_amd_xl = df_amd_xl_raw['join_mt_time_us'].values

tt_amd_intel = df_amd_intel_raw['total_time_us'].values
create_mt_amd_intel = df_amd_intel_raw['create_mt_time_us'].values
join_mt_amd_intel = df_amd_intel_raw['join_mt_time_us'].values

tt_amd_xl_intel = df_amd_xl_intel_raw['total_time_us'].values
create_mt_amd_xl_intel = df_amd_xl_intel_raw['create_mt_time_us'].values
join_mt_amd_xl_intel = df_amd_xl_intel_raw['join_mt_time_us'].values

# apple
df_apple_raw = pd.read_csv("./csv/mt/apple/benchmark.csv")
df_apple_xl_raw = pd.read_csv("./csv/mt/apple/benchmark_xl.csv")

tt_apple = df_apple_raw['total_time_us'].values
create_mt_apple = df_apple_raw['create_mt_time_us'].values
join_mt_apple = df_apple_raw['join_mt_time_us'].values

create_mt_apple_xl = df_apple_xl_raw['create_mt_time_us'].values
tt_apple_xl = df_apple_xl_raw['total_time_us'].values
join_mt_apple_xl = df_apple_xl_raw['join_mt_time_us'].values

# intel
df_intel_raw = pd.read_csv("./csv/mt/intel/benchmark.csv")
df_intel_xl_raw = pd.read_csv("./csv/mt/intel/benchmark_xl.csv")
df_intel_intel_raw = pd.read_csv("./csv/mt/intel/benchmark_intel.csv")
df_intel_xl_intel_raw = pd.read_csv("./csv/mt/intel/benchmark_xl_intel.csv")

tt_intel = df_intel_raw['total_time_us'].values
create_mt_intel= df_intel_raw['create_mt_time_us'].values
join_mt_intel= df_intel_raw['join_mt_time_us'].values

tt_intel_xl = df_intel_xl_raw['total_time_us'].values
create_mt_intel_xl = df_intel_xl_raw['create_mt_time_us'].values
join_mt_intel_xl = df_intel_xl_raw['join_mt_time_us'].values

tt_intel_intel = df_intel_intel_raw['total_time_us'].values
create_mt_intel_intel= df_intel_intel_raw['create_mt_time_us'].values
join_mt_intel_intel = df_intel_intel_raw['join_mt_time_us'].values

tt_intel_xl_intel = df_intel_xl_intel_raw['total_time_us'].values
create_mt_intel_xl_intel= df_intel_xl_intel_raw['create_mt_time_us'].values
join_mt_intel_xl_intel = df_intel_xl_intel_raw['join_mt_time_us'].values

"""
CREATE DATA FRAMES
"""
# amd
df_amd = pd.DataFrame({
    'min': [0.0] * len(threads),
    'avg': [0.0] * len(threads),
    'max': [0.0] * len(threads)
})
df_amd_overhead = df_amd.copy(deep=True)
df_amd_xl = df_amd.copy(deep=True)
df_amd_xl_overhead = df_amd.copy(deep=True)
df_amd_intel = df_amd.copy(deep=True)
df_amd_intel_overhead = df_amd.copy(deep=True)
df_amd_xl_intel = df_amd.copy(deep=True)
df_amd_xl_intel_overhead = df_amd.copy(deep=True)

# apple
df_apple = df_amd.copy(deep=True)
df_apple_overhead = df_amd.copy(deep=True)
df_apple_xl = df_apple.copy(deep=True)
df_apple_xl_overhead = df_amd.copy(deep=True)

# intel
df_intel = df_amd.copy(deep=True)
df_intel_overhead = df_amd.copy(deep=True)
df_intel_xl = df_amd.copy(deep=True)
df_intel_xl_overhead = df_amd.copy(deep=True)
df_intel_intel = df_amd.copy(deep=True)
df_intel_intel_overhead = df_amd.copy(deep=True)
df_intel_xl_intel = df_amd.copy(deep=True)
df_intel_xl_intel_overhead = df_amd.copy(deep=True)

# Fill DFs with Minimum, Average, Maximum values
for i in range(len(threads)):
    # amd
    df_amd.loc[i, 'min'] = np.min(tt_amd[10*i:10*(i+1)])
    df_amd.loc[i, 'avg'] = np.average(tt_amd[10*i:10*(i+1)])
    df_amd.loc[i, 'max'] = np.max(tt_amd[10*i:10*(i+1)])
    df_amd_overhead.loc[i, 'min'] = np.min(create_mt_amd[10*i:10*(i+1)] + join_mt_amd[10*i:10*(i+1)])
    df_amd_overhead.loc[i, 'avg'] = np.average(create_mt_amd[10*i:10*(i+1)] + join_mt_amd[10*i:10*(i+1)])
    df_amd_overhead.loc[i, 'max'] = np.max(create_mt_amd[10*i:10*(i+1)] + join_mt_amd[10*i:10*(i+1)])
    df_amd_xl.loc[i, 'min'] = np.min(tt_amd_xl[10*i:10*(i+1)])
    df_amd_xl.loc[i, 'avg'] = np.average(tt_amd_xl[10*i:10*(i+1)])
    df_amd_xl.loc[i, 'max'] = np.max(tt_amd_xl[10*i:10*(i+1)])
    df_amd_xl_overhead.loc[i, 'min'] = np.min(create_mt_amd_xl[10*i:10*(i+1)] + join_mt_amd_xl[10*i:10*(i+1)])
    df_amd_xl_overhead.loc[i, 'avg'] = np.average(create_mt_amd_xl[10*i:10*(i+1)] + join_mt_amd_xl[10*i:10*(i+1)])
    df_amd_xl_overhead.loc[i, 'max'] = np.max(create_mt_amd_xl[10*i:10*(i+1)] + join_mt_amd_xl[10*i:10*(i+1)])
    df_amd_intel.loc[i, 'min'] = np.min(tt_amd_intel[10*i:10*(i+1)])
    df_amd_intel.loc[i, 'avg'] = np.average(tt_amd_intel[10*i:10*(i+1)])
    df_amd_intel.loc[i, 'max'] = np.max(tt_amd_intel[10*i:10*(i+1)])
    df_amd_intel_overhead.loc[i, 'min'] = np.min(create_mt_amd_intel[10*i:10*(i+1)] + join_mt_amd_intel[10*i:10*(i+1)])
    df_amd_intel_overhead.loc[i, 'avg'] = np.average(create_mt_amd_intel[10*i:10*(i+1)] + join_mt_amd_intel[10*i:10*(i+1)])
    df_amd_intel_overhead.loc[i, 'max'] = np.max(create_mt_amd_intel[10*i:10*(i+1)] + join_mt_amd_intel[10*i:10*(i+1)])
    df_amd_xl_intel.loc[i, 'min'] = np.min(tt_amd_xl_intel[10*i:10*(i+1)])
    df_amd_xl_intel.loc[i, 'avg'] = np.average(tt_amd_xl_intel[10*i:10*(i+1)])
    df_amd_xl_intel.loc[i, 'max'] = np.max(tt_amd_xl_intel[10*i:10*(i+1)])
    df_amd_xl_intel_overhead.loc[i, 'min'] = np.min(create_mt_amd_xl_intel[10*i:10*(i+1)] + join_mt_amd_xl_intel[10*i:10*(i+1)])
    df_amd_xl_intel_overhead.loc[i, 'avg'] = np.average(create_mt_amd_xl_intel[10*i:10*(i+1)] + join_mt_amd_xl_intel[10*i:10*(i+1)])
    df_amd_xl_intel_overhead.loc[i, 'max'] = np.max(create_mt_amd_xl_intel[10*i:10*(i+1)] + join_mt_amd_xl_intel[10*i:10*(i+1)])

    # apple
    df_apple.loc[i, 'min'] = np.min(tt_apple[10*i:10*(i+1)])
    df_apple.loc[i, 'avg'] = np.average(tt_apple[10*i:10*(i+1)])
    df_apple.loc[i, 'max'] = np.max(tt_apple[10*i:10*(i+1)])
    df_apple_overhead.loc[i, 'min'] = np.min(create_mt_apple[10*i:10*(i+1)] + join_mt_apple[10*i:10*(i+1)])
    df_apple_overhead.loc[i, 'avg'] = np.average(create_mt_apple[10*i:10*(i+1)] + join_mt_apple[10*i:10*(i+1)])
    df_apple_overhead.loc[i, 'max'] = np.max(create_mt_apple[10*i:10*(i+1)] + join_mt_apple[10*i:10*(i+1)])
    df_apple_xl.loc[i, 'min'] = np.min(tt_apple_xl[10*i:10*(i+1)])
    df_apple_xl.loc[i, 'avg'] = np.average(tt_apple_xl[10*i:10*(i+1)])
    df_apple_xl.loc[i, 'max'] = np.max(tt_apple_xl[10*i:10*(i+1)])
    df_apple_xl_overhead.loc[i, 'min'] = np.min(create_mt_apple_xl[10*i:10*(i+1)] + join_mt_apple_xl[10*i:10*(i+1)])
    df_apple_xl_overhead.loc[i, 'avg'] = np.average(create_mt_apple_xl[10*i:10*(i+1)] + join_mt_apple_xl[10*i:10*(i+1)])
    df_apple_xl_overhead.loc[i, 'max'] = np.max(create_mt_apple_xl[10*i:10*(i+1)] + join_mt_apple_xl[10*i:10*(i+1)])

    # intel
    df_intel.loc[i, 'min'] = np.min(tt_intel[10*i:10*(i+1)])
    df_intel.loc[i, 'avg'] = np.average(tt_intel[10*i:10*(i+1)])
    df_intel.loc[i, 'max'] = np.max(tt_intel[10*i:10*(i+1)])
    df_intel_overhead.loc[i, 'min'] = np.min(create_mt_intel[10*i:10*(i+1)] + join_mt_intel[10*i:10*(i+1)])
    df_intel_overhead.loc[i, 'avg'] = np.average(create_mt_intel[10*i:10*(i+1)] + join_mt_intel[10*i:10*(i+1)])
    df_intel_overhead.loc[i, 'max'] = np.max(create_mt_intel[10*i:10*(i+1)] + join_mt_intel[10*i:10*(i+1)])
    df_intel_xl.loc[i, 'min'] = np.min(tt_intel_xl[10*i:10*(i+1)])
    df_intel_xl.loc[i, 'avg'] = np.average(tt_intel_xl[10*i:10*(i+1)])
    df_intel_xl.loc[i, 'max'] = np.max(tt_intel_xl[10*i:10*(i+1)])
    df_intel_xl_overhead.loc[i, 'min'] = np.min(create_mt_intel_xl[10*i:10*(i+1)] + join_mt_intel_xl[10*i:10*(i+1)])
    df_intel_xl_overhead.loc[i, 'avg'] = np.average(create_mt_intel_xl[10*i:10*(i+1)] + join_mt_intel_xl[10*i:10*(i+1)])
    df_intel_xl_overhead.loc[i, 'max'] = np.max(create_mt_intel_xl[10*i:10*(i+1)] + join_mt_intel_xl[10*i:10*(i+1)])
    df_intel_intel.loc[i, 'min'] = np.min(tt_intel_intel[10*i:10*(i+1)])
    df_intel_intel.loc[i, 'avg'] = np.average(tt_intel_intel[10*i:10*(i+1)])
    df_intel_intel.loc[i, 'max'] = np.max(tt_intel_intel[10*i:10*(i+1)])
    df_intel_intel_overhead.loc[i, 'min'] = np.min(create_mt_intel_intel[10*i:10*(i+1)] + join_mt_intel_intel[10*i:10*(i+1)])
    df_intel_intel_overhead.loc[i, 'avg'] = np.average(create_mt_intel_intel[10*i:10*(i+1)] + join_mt_intel_intel[10*i:10*(i+1)])
    df_intel_intel_overhead.loc[i, 'max'] = np.max(create_mt_intel_intel[10*i:10*(i+1)] + join_mt_intel_intel[10*i:10*(i+1)])
    df_intel_xl_intel.loc[i, 'min'] = np.min(tt_intel_xl_intel[10*i:10*(i+1)])
    df_intel_xl_intel.loc[i, 'avg'] = np.average(tt_intel_xl_intel[10*i:10*(i+1)])
    df_intel_xl_intel.loc[i, 'max'] = np.max(tt_intel_xl_intel[10*i:10*(i+1)])
    df_intel_xl_intel_overhead.loc[i, 'min'] = np.min(create_mt_intel_xl_intel[10*i:10*(i+1)] + join_mt_intel_xl_intel[10*i:10*(i+1)])
    df_intel_xl_intel_overhead.loc[i, 'avg'] = np.average(create_mt_intel_xl_intel[10*i:10*(i+1)] + join_mt_intel_xl_intel[10*i:10*(i+1)])
    df_intel_xl_intel_overhead.loc[i, 'max'] = np.max(create_mt_intel_xl_intel[10*i:10*(i+1)] + join_mt_intel_xl_intel[10*i:10*(i+1)])

"""
CREATE FIGURES
 0 performance
 1 performance xl
 2 overhead
 3 overhead xl
 4 icx
 5 icx xl
"""


figures = ["performance", "performance_xl", "overhead", "overhead_xl", "icx", "icx_xl"]
for f in figures:
    bar_width = 0.25

    if f in ["icx", "icx_xl"]:
        bar_width = 0.2

    indices = np.arange(len(threads))

    fig, ax = plt.subplots(figsize=(10, 6))

    match f:
        case "performance":
            ax.bar(indices - bar_width, df_amd['max'], width=bar_width, label=f"{label_amd} Max", color='#99c1f1', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - bar_width, df_amd['avg'], width=bar_width, label=f"{label_amd} Avg", color='#3584e4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - bar_width, df_amd['min'], width=bar_width, label=f"{label_amd} Min", color='#1a5fb4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_apple['max'], width=bar_width, label=f"{label_apple} Max", color='#8ff0a4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_apple['avg'], width=bar_width, label=f"{label_apple} Avg", color='#33d17a', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_apple['min'], width=bar_width, label=f"{label_apple} Min", color='#26a269', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + bar_width, df_intel['max'], width=bar_width, label=f"{label_intel} Max", color='#f66151', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + bar_width, df_intel['avg'], width=bar_width, label=f"{label_intel} Avg", color='#e01b24', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + bar_width, df_intel['min'], width=bar_width, label=f"{label_intel} Min", color='#a51d2d', edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("Performance", fontsize=12, color=foreground_color, loc='center')
        case "performance_xl":
            ax.bar(indices - bar_width, df_amd_xl['max'], width=bar_width, label=f"{label_amd} Max", color='#99c1f1', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - bar_width, df_amd_xl['avg'], width=bar_width, label=f"{label_amd} Avg", color='#3584e4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - bar_width, df_amd_xl['min'], width=bar_width, label=f"{label_amd} Min", color='#1a5fb4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_apple_xl['max'], width=bar_width, label=f"{label_apple} Max", color='#8ff0a4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_apple_xl['avg'], width=bar_width, label=f"{label_apple} Avg", color='#33d17a', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_apple_xl['min'], width=bar_width, label=f"{label_apple} Min", color='#26a269', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + bar_width, df_intel_xl['max'], width=bar_width, label=f"{label_intel} Max", color='#f66151', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + bar_width, df_intel_xl['avg'], width=bar_width, label=f"{label_intel} Avg", color='#e01b24', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + bar_width, df_intel_xl['min'], width=bar_width, label=f"{label_intel} Min", color='#a51d2d', edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("Performance XL", fontsize=12, color=foreground_color, loc='center')
        case "overhead":
            ax.bar(indices - bar_width, df_amd_overhead['max'], width=bar_width, label=f"{label_amd} Overhead Max", color='#99c1f1', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - bar_width, df_amd_overhead['avg'], width=bar_width, label=f"{label_amd} Overhead Avg", color='#3584e4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - bar_width, df_amd_overhead['min'], width=bar_width, label=f"{label_amd} Overhead Min", color='#1a5fb4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_apple_overhead['max'], width=bar_width, label=f"{label_apple} Overhead Max", color='#8ff0a4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_apple_overhead['avg'], width=bar_width, label=f"{label_apple} Overhead Avg", color='#33d17a', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_apple_overhead['min'], width=bar_width, label=f"{label_apple} Overhead Min", color='#26a269', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + bar_width, df_intel_overhead['max'], width=bar_width, label=f"{label_intel} Overhead Max", color='#f66151', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + bar_width, df_intel_overhead['avg'], width=bar_width, label=f"{label_intel} Overhead Avg", color='#e01b24', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + bar_width, df_intel_overhead['min'], width=bar_width, label=f"{label_intel} Overhead Min", color='#a51d2d', edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("Overhead", fontsize=12, color=foreground_color, loc='center')
        case "overhead_xl":
            ax.bar(indices - bar_width, df_amd_xl_overhead['max'], width=bar_width, label=f"{label_amd} Overhead Max", color='#99c1f1', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - bar_width, df_amd_xl_overhead['avg'], width=bar_width, label=f"{label_amd} Overhead Avg", color='#3584e4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - bar_width, df_amd_xl_overhead['min'], width=bar_width, label=f"{label_amd} Overhead Min", color='#1a5fb4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_apple_xl_overhead['max'], width=bar_width, label=f"{label_apple} Overhead Max", color='#8ff0a4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_apple_xl_overhead['avg'], width=bar_width, label=f"{label_apple} Overhead Avg", color='#33d17a', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_apple_xl_overhead['min'], width=bar_width, label=f"{label_apple} Overhead Min", color='#26a269', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + bar_width, df_intel_xl_overhead['max'], width=bar_width, label=f"{label_intel} Overhead Max", color='#f66151', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + bar_width, df_intel_xl_overhead['avg'], width=bar_width, label=f"{label_intel} Overhead Avg", color='#e01b24', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + bar_width, df_intel_xl_overhead['min'], width=bar_width, label=f"{label_intel} Overhead Min", color='#a51d2d', edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("Overhead XL", fontsize=12, color=foreground_color, loc='center')
        case "icx":
            ax.bar(indices - 2 * bar_width, df_amd['max'], width=bar_width, label=f"{label_amd} Max", color='#99c1f1', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 2 * bar_width, df_amd['avg'], width=bar_width, label=f"{label_amd} Avg", color='#3584e4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 2 * bar_width, df_amd['min'], width=bar_width, label=f"{label_amd} Min", color='#1a5fb4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - bar_width, df_amd_intel['max'], width=bar_width, label=f"{label_amd} ICX Max", color='#8ff0a4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - bar_width, df_amd_intel['avg'], width=bar_width, label=f"{label_amd} ICX Avg", color='#33d17a', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - bar_width, df_amd_intel['min'], width=bar_width, label=f"{label_amd} ICX Min", color='#26a269', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_intel['max'], width=bar_width, label=f"{label_amd} Max", color='#f66151', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_intel['avg'], width=bar_width, label=f"{label_amd} Avg", color='#e01b24', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_intel['min'], width=bar_width, label=f"{label_amd} Min", color='#a51d2d', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + bar_width, df_intel_intel['max'], width=bar_width, label=f"{label_amd} ICX Max", color='#dc8add', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + bar_width, df_intel_intel['avg'], width=bar_width, label=f"{label_amd} ICX Avg", color='#9141ac', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + bar_width, df_intel_intel['min'], width=bar_width, label=f"{label_amd} ICX Min", color='#613583', edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("ICX vs. GCC", fontsize=12, color=foreground_color, loc='center')
        case "icx_xl":
            ax.bar(indices - 2 * bar_width, df_amd_xl['max'], width=bar_width, label=f"{label_amd} Max", color='#99c1f1', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 2 * bar_width, df_amd_xl['avg'], width=bar_width, label=f"{label_amd} Avg", color='#3584e4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - 2 * bar_width, df_amd_xl['min'], width=bar_width, label=f"{label_amd} Min", color='#1a5fb4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - bar_width, df_amd_xl_intel['max'], width=bar_width, label=f"{label_amd} ICX Max", color='#8ff0a4', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - bar_width, df_amd_xl_intel['avg'], width=bar_width, label=f"{label_amd} ICX Avg", color='#33d17a', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices - bar_width, df_amd_xl_intel['min'], width=bar_width, label=f"{label_amd} ICX Min", color='#26a269', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_intel_xl_intel['max'], width=bar_width, label=f"{label_intel} Max", color='#f66151', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_intel_xl_intel['avg'], width=bar_width, label=f"{label_intel} Avg", color='#e01b24', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices, df_intel_xl_intel['min'], width=bar_width, label=f"{label_intel} Min", color='#a51d2d', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + bar_width, df_intel_xl_intel['max'], width=bar_width, label=f"{label_intel} ICX Max", color='#dc8add', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + bar_width, df_intel_xl_intel['avg'], width=bar_width, label=f"{label_intel} ICX Avg", color='#9141ac', edgecolor=foreground_color, linewidth=0.75)
            ax.bar(indices + bar_width, df_intel_xl_intel['min'], width=bar_width, label=f"{label_intel} ICX Min", color='#613583', edgecolor=foreground_color, linewidth=0.75)

            ax.set_title("ICX vs. GCC XL", fontsize=12, color=foreground_color, loc='center')

    ax.set_xlabel("Threads", fontsize=12, color=foreground_color)
    ax.set_ylabel("Total time (µs)", fontsize=12, color=foreground_color)
    ax.set_xticks(indices)
    ax.set_xticklabels(threads, color=foreground_color)
    ax.tick_params(axis='x', colors=foreground_color)  # X-axis tick marks and labels
    ax.tick_params(axis='y', colors=foreground_color)  # Y-axis tick marks and labels
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), facecolor=background_color, edgecolor=foreground_color, labelcolor=foreground_color)

    # Make spines (box edges)
    for spine in ax.spines.values():
        spine.set_edgecolor(foreground_color)

    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    # Show the plot
    plt.tight_layout()

    match f:
        case "performance":
            plt.savefig("../Graphs/mt/Performance.png")
        case "performance_xl":
            plt.savefig("../Graphs/mt/Performance XL.png")
        case "overhead":
            plt.savefig("../Graphs/mt/Overhead.png")
        case "overhead_xl":
            plt.savefig("../Graphs/mt/Overhead XL.png")
        case "icx":
            plt.savefig("../Graphs/mt/ICX vs. GCC.png")
        case "icx_xl":
            plt.savefig("../Graphs/mt/ICX vs. GCC XL.png")

    # plt.show()
    plt.close(fig)
