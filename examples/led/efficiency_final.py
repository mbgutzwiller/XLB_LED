import matplotlib.pyplot as plt
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(script_dir, "figures_final")

runtimes_led = np.array([5.435685157775879, 4.452804088592529, 4.6209397315979, 4.630028247833252, 5.644485712051392, 14.834207773208618, 91.37700748443604, 186.40812468528748])
l2_errors_led = np.array([0.022124830500582292, 0.005521168932786559, 0.0013830086677426208, 0.00034603700517274206, 0.00034603700517274206/4, 0.00034603700517274206/4**2, 0.00034603700517274206/4**3, 0.00034603700517274206/4**3 /(4000/3200)**2 ])

l2_errors_fem = np.array([0.010794387359063452, 0.0027124405440420536, 0.0006795322335664068, 0.0001700401216448107, 4.252363813238472e-05, 4.252363813238472e-05/3.9956, 4.252363813238472e-05/3.9956**2, 4.252363813238472e-05/3.9956**2 /(4000/3200*4/3.9956)**2])
runtimes_fem = np.array([0.055315256118774414, 0.14597415924072266, 0.6111440658569336, 4.10930871963501, 33.86465096473694, 282.5657756328583, 2588.7438769340515, 4863.604686260223])

# print(np.mean(l2_errors_fem[2:] / l2_errors_fem[1:-1]))

x_ref_led = l2_errors_led[-1]
y_ref_led = runtimes_led[-1]
ref_line_led = y_ref_led * (l2_errors_led / x_ref_led)**-1.5

x_ref_fem = l2_errors_fem[-1]
y_ref_fem = runtimes_fem[-1]
ref_line_fem = y_ref_fem * (l2_errors_fem / x_ref_fem)**-1.5

plt.figure(figsize=(8, 5))

plt.loglog(l2_errors_fem, runtimes_fem, marker="o", label="Runtimes FEM", markersize=8, color="black")
plt.loglog(l2_errors_fem, ref_line_fem, "--", label="Slope = 1.5", color="black", alpha=0.5)

plt.loglog(l2_errors_led, runtimes_led, marker="o", label="Runtimes LED", markersize=8, color="blue")
plt.loglog(l2_errors_led, ref_line_led, "--", label="Slope = 1.5", color="blue", alpha=0.5)

plt.grid(True, which="both")
plt.xlabel("Rel. L2 Error of Displacement [-]")
plt.ylabel("Time [s]")
plt.legend(loc="upper left")
plt.gca().invert_xaxis()
plt.title("Efficiency: Runtime vs. Rel. L2 Error of Displacement")
plt.ylim(bottom=0.5*np.min(runtimes_fem), top=2*np.max(runtimes_fem))
plt.savefig(os.path.join(figures_dir, f"efficiency_led_vs_fenicsx"), dpi=600)
plt.close()
