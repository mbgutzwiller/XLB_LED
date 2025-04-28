import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['axes.titlesize'] = 16      # Title
plt.rcParams['axes.labelsize'] = 14      # X and Y labels

script_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(script_dir, "figures_final_scaling")
os.makedirs(figures_dir, exist_ok=True)

# grid_sizes = np.array([40, 80, 160, 320, 640, 1280, 2560, 3840])
# runtimes_f0bound_no_comp = np.array([0.016037464141845703, 0.031234264373779297, 0.061258554458618164, 0.1215505599975586, 0.6975667476654053, 6.217107534408569, 53.538559913635254, 190.36338782310486])
# runtimes_f0bound_comp = np.array([10.478154420852661, 10.544797658920288, 10.502889156341553, 10.666243076324463, 11.208498239517212, 16.556936979293823, 64.23121881484985, 200.984365940094])
grid_sizes = np.array([50, 100, 200, 400, 800, 1600, 2400, 3200, 4000])
grid_sizes_2 = np.array([50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 800, 1600, 2400, 3200, 4000])
runtimes_f0bound_comp = np.array([10.339588642120361, 10.35711407661438, 10.53181791305542, 10.534061908721924, 11.962241411209106, 23.02586841583252, 55.40644884109497, 118.28580951690674, 230.0424611568451])


C_3 = runtimes_f0bound_comp[-1] / grid_sizes_2[-1]**3  # scale to match first point

plt.figure(figsize=(8, 5))
plt.loglog(grid_sizes[-6:], C_3 * grid_sizes[-6:]**3, "-.", label="Slope = 3", color="black", alpha=0.8)
plt.loglog(grid_sizes, runtimes_f0bound_comp, marker="o", label="Runtime only", color="green")
# plt.loglog(grid_sizes, runtimes_f0bound_comp, marker="o", label="Runtime + compilation time", color="green")
plt.ylim(0.5*np.min(runtimes_f0bound_comp), 2*np.max(runtimes_f0bound_comp))
plt.xlabel("Grid size [-]")
plt.ylabel("Runtime [s]")
plt.legend()
plt.grid(visible=True, which="both")
plt.savefig(os.path.join(figures_dir, f"scaling_compil_compar_temp"), dpi=600)
plt.close()