import matplotlib.pyplot as plt
import numpy as np

grid_sizes       = np.array([50, 100, 200, 400, 800, 1600, 3200, 4000], dtype=float)
gpu_utilization  = np.array([16, 13, 47, 100, 100, 100, 100, 100], dtype=float)           # %, this is how much the compute units are used
mem_utilization  = np.array([0, 0, 22, 55, 60, 60, 60, 60], dtype=float)                  # % memory bandwidth, stays in cache/l1/l2 first probably
power_usage      = np.array([37.417, 37.753, 86.236, 131.858, 138.097,
                    139.593, 141.327, 145.512])                     # Watts
vram_usage       = np.array([536.6, 536.6, 536.6, 568.6, 760.6,
                    1144.6, 2616.6, 3832.6])                         # MB vram, preallocated in the beginning, 

gpu_utilization /= np.max(gpu_utilization)
mem_utilization /= np.max(mem_utilization)
power_usage /= np.max(power_usage)
vram_usage /= np.max(vram_usage)

# Plotting L2 errors
# C_loglog_2 = runtimes[-1] / (grid_size_run[-1]**3)  # reference line for 2nd order convergence
# C_loglog_1 = runtimes[0] / (grid_size_run[0])  # reference line for 2nd order convergence
fig1, axs1 = plt.subplots(1, 1, figsize = (10, 5))
axs1.semilogx(grid_sizes, gpu_utilization, markersize=8, label="GPU Compute Utilization")
axs1.semilogx(grid_sizes, mem_utilization, markersize=8, label="GPU Memory Bandwidth")
axs1.semilogx(grid_sizes, power_usage, markersize=8, label="GPU Power Draw")
axs1.semilogx(grid_sizes, vram_usage, markersize=8, label="GPU VRAM Utilization")
axs1.set_xlabel("Grid size")
axs1.set_ylabel("Relative [-]")
axs1.grid(True, which="both")
# axs1.loglog(grid_sizes, C_loglog_2 * np.array(grid_sizes)**3, "--", label="Slope = 3", alpha=0.5, color="black")
# axs1.loglog(grid_sizes, C_loglog_1 * np.array(grid_sizes), "--", label="Slope = 1", alpha=0.5, color="black")
axs1.legend()
plt.savefig(f"/home/merrill/Documents/ETH/LBM for Linear Elastodynamics/Code/xlb/XLB/examples/led/figures/gpu_load")
# plt.savefig(f"/home/merrillg/XLB_LED/examples/led/figures/f_paper_dirBC_u_sig_L2_ck_{int(np.round(c_k_led**2, 1)*10)}_16_2_n_{len(grid_sizes)}_test_run{run_i}")
plt.show()