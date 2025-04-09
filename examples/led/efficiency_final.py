import matplotlib.pyplot as plt
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(script_dir, "figures_final")

runtimes_led = np.array([5.435685157775879, 4.452804088592529, 4.6209397315979, 4.630028247833252, 5.644485712051392, 14.834207773208618, 91.37700748443604, 186.40812468528748])
l2_errors_led = np.array([0.022124830500582292, 0.005521168932786559, 0.0013830086677426208, 0.00034603700517274206, 0.00034603700517274206/4, 0.00034603700517274206/4**2, 0.00034603700517274206/4**3, 0.00034603700517274206/4**3 /(4/3.2)**2 ])


plt.figure()
plt.loglog(l2_errors_led, runtimes_led, marker="o", markersize=8, color="blue")
plt.gca().invert_xaxis()
plt.grid(True, which="both")
plt.xlabel("Rel. L2 Error of Displacement [-]")
plt.ylabel("Time [s]")
plt.title("Efficiency: Runtime vs. Rel. L2 Error of Displacement")
plt.savefig(os.path.join(figures_dir, f"efficiency_led_vs_fenicsx"))
plt.close()

# print(np.mean(l2_errors_led[1:] / l2_errors_led[:-1]))