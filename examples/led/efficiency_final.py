import matplotlib.pyplot as plt
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(script_dir, "figures_final")

grid_sizes = [50, 100, 200, 400, 800, 1600, 3200, 4000]
total_runtimes_led = np.array([5.435685157775879, 4.452804088592529, 4.6209397315979, 4.630028247833252, 5.644485712051392, 14.834207773208618, 91.37700748443604, 186.40812468528748])
l2_errors_led = np.array([0.022124830500582292, 0.005521168932786559, 0.0013830086677426208, 0.00034603700517274206, 0.00034603700517274206/3.9997, 0.00034603700517274206/3.9997**2, 0.00034603700517274206/3.9997**3, 0.00034603700517274206/3.9997**3 /(4000/3200)**2 ])

total_runtimes_fem_8 =       np.array([0.18093156814575195, 0.19237661361694336, 0.744234561920166,   4.600370645523071,  37.01590156555176,  291.7129862308502, 2583.3561222553253, 4877.020454645157])
total_runtimes_fem_8_tilde = np.array([0.11204075813293457, 0.11629104614257812, 0.33806347846984863, 1.7251503467559814, 12.040383100509644, 101.41140651702881, 921.3175673484802, 1533])
l2_errors_fem =       np.array([0.010794387359063452, 0.0027124405440420536, 0.0006795322335664068, 0.0001700401216448107, 4.252363813238472e-05, 4.252363813238472e-05/3.997, 4.252363813238472e-05/3.997**2, 4.252363813238472e-05/3.997**2 /(4000/3200)**(2)])
l2_errors_fem_tilde = np.array([0.02191428273941109, 0.0055242669157295815, 0.0013856111056867525, 0.000346886783114532, 8.67626684356439e-05, 8.67626684356439e-05/4, 8.67626684356439e-05/4**2, 8.67626684356439e-05/4**2/(4/3.2)**2])

print(total_runtimes_fem_8_tilde/total_runtimes_led)
print(l2_errors_fem_tilde/l2_errors_led)

print(np.mean(l2_errors_led[:2] / l2_errors_led[1:3]))
print(np.mean(l2_errors_fem / l2_errors_led))
print(f"speedup: {(total_runtimes_fem_8/total_runtimes_led)[-3]}")

x_ref_led = l2_errors_led[-1]
y_ref_led = total_runtimes_led[-1]
ref_line_led = y_ref_led * (l2_errors_led / x_ref_led)**-1.5

x_ref_fem = l2_errors_fem[3]
y_ref_fem = total_runtimes_fem_8[3]
ref_line_fem = y_ref_fem * (l2_errors_fem / x_ref_fem)**-1.5

plt.figure(figsize=(8, 5))

plt.loglog(l2_errors_fem, total_runtimes_fem_8, marker="o", label="Runtimes FEM, Grid spacings h", markersize=8, color="blue")
# plt.loglog(l2_errors_fem_tilde, total_runtimes_fem_8_tilde, marker="o", label=r"Runtimes FEM, Grid spacings h$\cdot \sqrt{2.036}$", markersize=8, color="Green")
plt.loglog(l2_errors_fem, ref_line_fem, "--", label="Slope = 1.5", color="black", alpha=0.5)

plt.loglog(l2_errors_led, total_runtimes_led, marker="o", label="Runtimes XLB-LED, Grid spacings h", markersize=8, color="black")
plt.loglog(l2_errors_led, ref_line_led, "--", label="Slope = 1.5", color="blue", alpha=0.5)



plt.grid(True, which="both")
plt.xlabel("Rel. L2 Error of Displacement [-]")
plt.ylabel("Total Runtime [s]")
plt.legend(loc="upper left")
plt.gca().invert_xaxis()
# plt.title("Efficiency: Runtime vs. Rel. L2 Error of Displacement")
plt.ylim(bottom=0.5*np.min(total_runtimes_fem_8), top=2*np.max(total_runtimes_fem_8))
plt.savefig(os.path.join(figures_dir, f"efficiency_led_vs_fenicsx"), dpi=600)
plt.close()

plt.figure(figsize=(8, 5))

plt.loglog(grid_sizes, total_runtimes_fem_8, marker="o", label="Runtimes FEM", markersize=8, color="black")
# plt.loglog(l2_errors_fem, ref_line_fem, "--", label="Slope = 1.5", color="black", alpha=0.5)

plt.loglog(grid_sizes, total_runtimes_led, marker="o", label="Runtimes XLB-LED", markersize=8, color="blue")
# plt.loglog(l2_errors_led, ref_line_led, "--", label="Slope = 1.5", color="blue", alpha=0.5)
plt.grid(True, which="both")
plt.xlabel("Grid Size [-]")
plt.ylabel("Total Runtime [s]")
plt.legend(loc="upper left")
# plt.title("Efficiency: Runtime vs. Rel. L2 Error of Displacement")
plt.ylim(bottom=0.5*np.min(total_runtimes_fem_8), top=2*np.max(total_runtimes_fem_8))
plt.savefig(os.path.join(figures_dir, f"runtimes_led_vs_fenicsx"), dpi=600)
plt.close()


plt.figure(figsize=(8, 5))

plt.semilogx(grid_sizes, total_runtimes_fem_8/total_runtimes_led, marker="o",markersize=8, color="black")
plt.semilogx(grid_sizes, total_runtimes_fem_8/total_runtimes_fem_8, color="orange", alpha=0.5, label="Reference line")
# plt.loglog(l2_errors_fem, ref_line_fem, "--", label="Slope = 1.5", color="black", alpha=0.5)

# plt.loglog(grid_sizes, total_runtimes_led, marker="o", label="Runtimes XLB-LED", markersize=8, color="blue")
# plt.loglog(l2_errors_led, ref_line_led, "--", label="Slope = 1.5", color="blue", alpha=0.5)
plt.grid(True, which="both")
plt.xlabel("Grid Size [-]")
plt.ylabel("Total Speedup using LBM over FEM [-]")
plt.legend()
# plt.title("Efficiency: Runtime vs. Rel. L2 Error of Displacement")
plt.savefig(os.path.join(figures_dir, f"relative_runtimes_led_vs_fenicsx"), dpi=600)
plt.close()