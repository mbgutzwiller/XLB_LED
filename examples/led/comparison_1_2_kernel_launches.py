import numpy as np
import matplotlib.pyplot as plt

# Raw runtimes
one_kernel = [0.016743898391723633, 0.030529260635375977, 0.0560758113861084, 0.10651731491088867, 0.9175777435302734, 10.913824319839478, 40.83757400512695]
two_kernel = [0.024507999420166016, 0.05144524574279785, 0.09547281265258789, 0.19022488594055176, 1.7953689098358154, 16.255247831344604, 57.64478087425232]
grid_sizes = [50, 100, 200, 400, 800, 1600, 2400]
# Calculate relative speed increase: (one_kernel - two_kernel) / one_kernel
speed_increase = np.array(two_kernel) / np.array(one_kernel)

# Create two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

# Top plot: raw runtimes on a log scale
ax1.loglog(grid_sizes, one_kernel, 'o-', label='One Kernel')
ax1.loglog(grid_sizes, two_kernel, 's-', label='Two Kernels')
# ax1.set_yscale('log')
ax1.set_ylabel('Raw Runtime [s]')
ax1.set_title('Raw Runtimes and Speed Increase')
ax1.legend()
ax1.grid(True, which="both")

# Bottom plot: relative speed increase
ax2.semilogx(grid_sizes, speed_increase, '^-', color='green')
ax2.set_xlabel('Grid Size')
ax2.set_ylabel('Relative Speed Increase [-]')
ax2.grid(True, which="both")

plt.tight_layout()
plt.show()
