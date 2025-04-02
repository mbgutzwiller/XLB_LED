import matplotlib.pyplot as plt
import numpy as np

# device = "RTX2060s"
# grid_sizes       = np.array([50, 100, 200, 400, 800, 1600, 3200, 4000], dtype=float)
# gpu_utilization  = np.array([16, 13, 47, 100, 100, 100, 100, 100], dtype=float)           # %, this is how much the compute units are used
# mem_utilization  = np.array([0, 0, 22, 55, 60, 60, 60, 60], dtype=float)                  # % memory bandwidth, stays in cache/l1/l2 first probably
# power_usage      = np.array([37.417, 37.753, 86.236, 131.858, 138.097,
#                     139.593, 141.327, 145.512])                     # Watts
# vram_usage       = np.array([536.6, 536.6, 536.6, 568.6, 760.6,
#                     1144.6, 2616.6, 3832.6])                         # MB vram, preallocated in the beginning, 

# gpu_utilization_plot = gpu_utilization / np.max(gpu_utilization)
# mem_utilization_plot = mem_utilization / np.max(mem_utilization)
# power_usage_plot = power_usage / np.max(power_usage)
# vram_usage_plot = vram_usage / np.max(vram_usage)

# fig1, axs1 = plt.subplots(1, 1, figsize = (10, 5))
# axs1.semilogx(grid_sizes, gpu_utilization_plot, markersize=8, label=f"GPU Compute Util. (max = {max(gpu_utilization)}%)")
# axs1.semilogx(grid_sizes, mem_utilization_plot, markersize=8, label=f"GPU Mem. Bandw. (max = {max(mem_utilization)}%)")
# axs1.semilogx(grid_sizes, power_usage_plot, markersize=8, label=f"GPU Power (max = {np.round(max(power_usage), 2)} W)")
# axs1.semilogx(grid_sizes, vram_usage_plot, markersize=8, label=f"GPU VRAM Util. (max = {np.round(max(vram_usage/1000), 2)} Gb)")
# axs1.set_xlabel("Grid size [-]")
# axs1.set_ylabel("Relative [-]")
# axs1.grid(True, which="both")
# fig1.suptitle(f"GPU Load vs. Problem Size ({device})")
# axs1.legend()
# # plt.savefig(f"/home/merrill/Documents/ETH/LBM for Linear Elastodynamics/Code/xlb/XLB/examples/led/figures/gpu_load")
# plt.savefig(f"/home/merrillg/XLB_LED/examples/led/figures/gpu_load_{device}")
# plt.show()

device = "A100"
grid_sizes = np.array([50, 100, 200, 300, 400, 500, 600, 700, 800, 900,
                       1000, 1600, 3200, 4800, 8000, 12000])

runtimes = np.array([0.031522274017333984, 0.06381678581237793, 0.12872052192687988,
                     0.1918048858642578, 0.25765299797058105, 0.32117319107055664,
                     0.387068510055542, 0.45124101638793945, 0.5203630924224854,
                     0.5810229778289795, 0.6436975002288818, 2.6614675521850586,
                     22.71507692337036, 78.36191320419312, 375.69121837615967,
                     1285.8478937149048])
# runtimes_scaling_study = np.array([166.17, 96.32, 54.89, 39.85, 31.72, 26.50, 22.78, 20.14, 18.16, 16.88, 18.68, 30.11, 67.61, 109.40, 199.62, 319.51])
gpu_utilization = np.array([10, 10, 9, 17, 27, 35, 48, 62, 77, 100, 100, 100, 100, 100, 100, 100])
mem_utilization = np.array([0, 0, 0, 0, 7, 22, 34, 46, 61, 75, 76, 83, 84, 83, 83, 84])
power_usage = np.array([66.085, 67.275, 72.255, 82.468, 104.42, 142.689, 178.663, 219.857, 267.907, 295.723, 298.863, 297.184, 300.935, 295.345, 298.352, 301.898])
vram_usage = np.array([1271.2, 1271.2, 1271.2, 1303.2, 1335.2, 1399.2, 1431.2, 1463.2, 1495.2, 1527.2, 1559.2, 1847.2, 3383.2, 5911.2, 13847.2, 29463.2])  # MB

# device = "RTX2080ti"
# grid_sizes = np.array([50, 100, 200, 300, 400, 500, 600, 700, 800])
# runtimes = np.array([0.12873291969299316,
#                      0.7808837890625,
#                      5.682956218719482,
#                      20.92768430709839,
#                      46.91501474380493,
#                      100.96249389648438,
#                      175.29467725753784,
#                      277.6062626838684,
#                      435.90000557899475])


gpu_utilization_plot = gpu_utilization / np.max(gpu_utilization)
mem_utilization_plot = mem_utilization / np.max(mem_utilization)
power_usage_plot = power_usage / np.max(power_usage)
vram_usage_plot = vram_usage / np.max(vram_usage)

_index = 10
C_loglog_3 = runtimes[_index] / (grid_sizes[_index])**3  # reference line for 2nd order convergence
C_loglog_4 = runtimes[_index] / (grid_sizes[_index])**4  # reference line for 2nd order convergence
C_loglog_1 = runtimes[0] / (grid_sizes[0])  # reference line for 2nd order convergence

fig1, axs1 = plt.subplots(1, 1, figsize = (8, 5))
axs1.semilogx(grid_sizes, gpu_utilization_plot, markersize=8, label=f"GPU Compute Util. (max = {max(gpu_utilization)}%)")
axs1.semilogx(grid_sizes, mem_utilization_plot, markersize=8, label=f"GPU Mem. Bandw. (max = {max(mem_utilization)}%)")
axs1.semilogx(grid_sizes, power_usage_plot, markersize=8, label=f"GPU Power (max = {np.round(max(power_usage), 2)} W)")
axs1.semilogx(grid_sizes, vram_usage_plot, markersize=8, label=f"GPU VRAM Util. (max = {np.round(max(vram_usage/1000), 2)} Gb)")
axs1.set_xlabel("Grid size [-]")
axs1.set_ylabel("Relative [-]")
axs1.grid(True, which="both")
# # Second y-axis for runtimes
axs2 = axs1.twinx()
axs2.loglog(grid_sizes, runtimes, 'k--o', markersize=8, label=f"Runtime [s]")
axs2.loglog(grid_sizes[:-4], C_loglog_1 * np.array(grid_sizes)[:-4], "--", label="Slope = 1", alpha=0.5, color="black")
axs2.loglog(grid_sizes[_index-3:], C_loglog_3 * np.array(grid_sizes[_index-3:])**3, label="Slope = 3", alpha=0.5, color="black")
# axs2.loglog(grid_sizes[_index:], C_loglog_4 * np.array(grid_sizes[_index:])**4, ":", label="Slope = 4", alpha=0.5, color="black")
axs2.set_ylabel("Runtime [s]")

# # Combine legends from both axes
lines_1, labels_1 = axs1.get_legend_handles_labels()
lines_2, labels_2 = axs2.get_legend_handles_labels()
axs1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

# Title and layout
fig1.suptitle(f"GPU Load vs. Problem Size ({device})")
fig1.tight_layout()
# fig1.subplots_adjust(top=0.9)

# Save and show
plt.savefig(f"/home/merrillg/XLB_LED/examples/led/figures/gpu_load_{device}_runtime", dpi=600)
plt.show()