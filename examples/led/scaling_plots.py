import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['axes.titlesize'] = 16      # Title
plt.rcParams['axes.labelsize'] = 14      # X and Y labels

script_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(script_dir, "figures_final_scaling")
os.makedirs(figures_dir, exist_ok=True)

# grid_sizes = [40, 80, 160, 320, 640, 1280, 2560, 3840]
# runtimes_f0bound_no_comp = np.array([0.016037464141845703, 0.031234264373779297, 0.061258554458618164, 0.1215505599975586, 0.6975667476654053, 6.217107534408569, 53.538559913635254, 190.36338782310486])
# runtimes_f0bound_comp = np.array([10.478154420852661, 10.544797658920288, 10.502889156341553, 10.666243076324463, 11.208498239517212, 16.556936979293823, 64.23121881484985, 200.984365940094])
grid_sizes = np.array([50, 100, 200, 400, 800, 1600, 2400, 3200, 4000])
runtimes_f0bound_no_comp = np.array([0.018832921981811523, 0.038495779037475586, 0.07565593719482422, 0.1499497890472412, 1.421764850616455, 12.71289324760437, 45.0375542640686, 107.83751463890076, 217.93051481246948])
runtimes_f0bound_comp = np.array([10.339588642120361, 10.35711407661438, 10.53181791305542, 10.534061908721924, 11.962241411209106, 23.02586841583252, 55.40644884109497, 118.28580951690674, 230.0424611568451])

runtimes_fpaper_no_comp = np.array([0.017750024795532227, 0.03725457191467285, 0.07348299026489258, 0.19050264358520508, 2.155949354171753, 19.72943687438965, 69.55024290084839, 167.87471508979797, 334.1596167087555])
runtimes_fpaper_comp = np.array([24.695094108581543, 25.186437845230103, 25.091731548309326, 24.97805643081665, 26.81846523284912, 44.416804790496826, 94.5247917175293, 192.75134301185608, 359.12451338768005])

C_1 = runtimes_f0bound_no_comp[0] / grid_sizes[0]  # scale to match first point
C_3 = runtimes_f0bound_no_comp[3] / grid_sizes[3]**3  # scale to match first point
C_4 = runtimes_f0bound_no_comp[3] / grid_sizes[3]**4  # scale to match first point

plt.figure(figsize=(8, 5))
# plt.loglog(grid_sizes, C_1 * grid_sizes, "--", label="Slope = 1", color="black", alpha=0.8)
# plt.loglog(grid_sizes, C_3 * grid_sizes**3, "-.", label="Slope = 3", color="black", alpha=0.8)
# plt.loglog(grid_sizes, C_4 * grid_sizes**4, ":", label="Slope = 4", color="black", alpha=0.8)
plt.loglog(grid_sizes, runtimes_f0bound_no_comp, marker="o", label="Runtime only", color="blue")
plt.loglog(grid_sizes, runtimes_fpaper_no_comp, "o:", label="Runtime only", color="blue")
plt.loglog(grid_sizes, runtimes_f0bound_comp, marker="o", label="Runtime + compilation time", color="green")
plt.loglog(grid_sizes, runtimes_fpaper_comp, "o:", label="Runtime + compilation time", color="green")
plt.ylim(0.5*np.min(runtimes_f0bound_no_comp), 2*np.max(runtimes_f0bound_comp))
plt.xlabel("Grid size [-]")
plt.ylabel("Runtime [s]")
plt.legend()
plt.grid(visible=True, which="both")
plt.savefig(os.path.join(figures_dir, f"scaling_compil_compar"), dpi=600)
plt.close()