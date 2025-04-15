import matplotlib.pyplot as plt
import numpy as np

errors_perBC = np.array([np.float32(0.014712143), np.float32(0.0036837147), np.float32(0.0009140503), np.float32(0.00020173939), np.float32(2.7621358e-05), np.float32(2.4414062e-06), np.float32(3.0517577e-07)])
grid_sizes_perBC = np.array([40, 80, 160, 320, 640, 1280, 2560])
errors_dirBC = np.array([np.float32(0.005702435), np.float32(0.003683714), np.float32(0.0009140494), np.float32(0.00020173939), np.float32(9.931171e-06), np.float32(2.4414062e-06), np.float32(2.1579186e-07)])

plt.figure()
plt.loglog(grid_sizes_perBC, errors_perBC, marker="o", label='Numerical Error Per. BC.')
plt.loglog(grid_sizes_perBC, errors_dirBC, marker="o", label='Numerical Error Dir. BC.')

# Add second-order convergence reference line
# Pick a reference point (e.g., first one) and scale with (h/h0)^2
ref_index = 0
ref_error = errors_perBC[ref_index]
ref_grid = grid_sizes_perBC[ref_index]
reference_line = ref_error * (grid_sizes_perBC / ref_grid)**-2

plt.loglog(grid_sizes_perBC, reference_line, 'k--', label='Second-order slope')

plt.xlabel('Grid size [-]')
plt.ylabel(r'L$_2$ Error of Displacement [-]')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig("temp_convergence", dpi=600)
plt.show()
