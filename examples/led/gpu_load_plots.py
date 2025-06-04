import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['axes.titlesize'] = 16      # Title
plt.rcParams['axes.labelsize'] = 14      # X and Y labels

script_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(script_dir, "figures_final_scaling")
os.makedirs(figures_dir, exist_ok=True)

grid_sizes = np.array([50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 800, 1600, 2400, 3200, 4000])
# This is f0bound
# gpu_compute_util = np.array([20, 23, 48, 100, 100, 100, 100, 100, 100, 100, 100])
# gpu_mem_util = np.array([0, 0, 25, 67, 72, 73, 78, 78, 78, 79, 78])
# gpu_power = np.array([43.892, 45.393, 55.74, 74.692, 76.398, 78.013, 80.601, 80.626, 82.489, 81.335, 81.265])
# gpu_vram = np.array([492.5, 492.5, 492.5, 524.5, 556.5, 620.5, 748.5, 1100.5, 1740.5, 2668.5, 3724.5])

# this is pulse
# gpu_compute_util = np.array([12, 14, 33, 100, 100, 100, 100, 100, 100])
# gpu_mem_util = np.array( [0, 0, 23, 91, 97, 100, 100, 100, 99])
# gpu_vram = np.array([492.5, 492.5, 492.5, 524.5, 716.5, 1100.5, 1740.5, 2572.5, 3788.5])
# gpu_power = np.array( [45.336, 46.417, 54.346, 80.225, 84.007, 87.032, 88.212, 88.737, 87.631])
# runtimes: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
gpu_compute_util = np.array([20, 23, 26, 53, 74, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
gpu_mem_util = np.array([0, 0, 9, 25, 49, 69, 70, 70, 74, 76, 75, 77, 78, 77, 82, 82])
gpu_vram = np.array([492.5, 492.5, 492.5, 492.5, 524.5, 524.5, 556.5, 524.5, 588.5, 556.5, 620.5, 684.5, 1100.5, 1708.5, 2700.5, 3724.5])
gpu_power = np.array([40.704, 42.476, 46.23, 53.048, 63.322, 70.223, 71.928, 71.122, 74.106, 74.528, 75.691, 75.003, 76.599, 77.042, 88.792, 89.776])

# grid_sizes = np.array([50, 100, 150, 200, 250, 300, 350, 400, 800, 1600, 2400, 3200, 4000])
# gpu_compute_util = np.array([21, 22, 29, 48, 70, 100, 100, 100, 100, 100, 100, 100, 100])
# gpu_mem_util = np.array([0, 0, 9, 25, 49, 70, 71, 70, 75, 77, 77, 78, 76])
# gpu_vram = np.array([492.5, 492.5, 492.5, 492.5, 524.5, 524.5, 556.5, 524.5, 652.5, 1196.5, 1772.5, 2604.5, 3788.5])
# gpu_power = np.array([46.099, 47.189, 50.523, 58.238, 68.311, 77.009, 78.217, 78.137, 79.543, 83.227, 82.54, 83.425, 82.078])


max_compute = 100./100.
max_mem_util = 100./100.
max_power = 125./100.
max_vram = 8192./100.

plt.figure(figsize=(8, 5))
plt.semilogx(grid_sizes, gpu_mem_util/max_mem_util, marker="o", label="GPU Mem. Util.", color="blue")
plt.semilogx(grid_sizes, gpu_power/max_power, marker="o", label="GPU Power", color="black")
plt.semilogx(grid_sizes, gpu_vram/max_vram, marker="o", label="GPU Vram Usage", color="orange")
plt.semilogx(grid_sizes, gpu_compute_util/max_compute, marker="o", label="GPU Compute Util.", color="green")
plt.xlabel("Grid Size [-]")
plt.ylabel("GPU Metrics [%]")
plt.legend()
plt.grid(visible=True, which="both")
plt.savefig(os.path.join(figures_dir, f"gpu_utils_f0bound_rerun_new"), dpi=600)
plt.close()