import matplotlib.pyplot as plt
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(script_dir, "figures_final")


runtimes_led = np.array([5.435685157775879, 4.452804088592529, 4.6209397315979, 4.630028247833252, 5.644485712051392, 14.834207773208618, 91.37700748443604, 186.40812468528748])
l2_errors_led = np.array([0.022124830500582292, 0.005521168932786559, 0.0013830086677426208, 0.00034603700517274206, 0.00034603700517274206/3.9997, 0.00034603700517274206/3.9997**2, 0.00034603700517274206/3.9997**3, 0.00034603700517274206/3.9997**3 /(4000/3200)**2 ])

runtimes_fem_8 = np.array([0.18093156814575195, 0.19237661361694336, 0.744234561920166, 4.600370645523071, 37.01590156555176, 291.7129862308502, 2703.3561222553253, 4877.020454645157])
l2_errors_fem = np.array([0.010794387359063452, 0.0027124405440420536, 0.0006795322335664068, 0.0001700401216448107, 4.252363813238472e-05, 4.252363813238472e-05/3.989, 4.252363813238472e-05/3.989**2, 4.252363813238472e-05/3.989**2 /(4000/3200)**2])

print(np.mean(l2_errors_led[:2] / l2_errors_led[1:3]))

x_ref_led = l2_errors_led[-1]
y_ref_led = runtimes_led[-1]
ref_line_led = y_ref_led * (l2_errors_led / x_ref_led)**-1.5

x_ref_fem = l2_errors_fem[3]
y_ref_fem = runtimes_fem_8[3]
ref_line_fem = y_ref_fem * (l2_errors_fem / x_ref_fem)**-1.5

plt.figure(figsize=(8, 5))

plt.loglog(l2_errors_fem, runtimes_fem_8, marker="o", label="Runtimes FEM", markersize=8, color="black")
plt.loglog(l2_errors_fem, ref_line_fem, "--", label="Slope = 1.5", color="black", alpha=0.5)

plt.loglog(l2_errors_led, runtimes_led, marker="o", label="Runtimes XLB-LED", markersize=8, color="blue")
plt.loglog(l2_errors_led, ref_line_led, "--", label="Slope = 1.5", color="blue", alpha=0.5)

plt.grid(True, which="both")
plt.xlabel("Rel. L2 Error of Displacement [-]")
plt.ylabel("Time [s]")
plt.legend(loc="upper left")
plt.gca().invert_xaxis()
# plt.title("Efficiency: Runtime vs. Rel. L2 Error of Displacement")
plt.ylim(bottom=0.5*np.min(runtimes_fem_8), top=2*np.max(runtimes_fem_8))
plt.savefig(os.path.join(figures_dir, f"efficiency_led_vs_fenicsx"), dpi=600)
plt.close()
