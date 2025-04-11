import os
# os.environ["JAX_PLATFORMS"] = "cpu"  # Sometimes it wont automatically run on cpu if no gpu is available.


import warp as wp
import numpy as np
import matplotlib.pyplot as plt
import time

import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from pulse_2d_linear_elastodynamics import Pulse2D_LED
wp.build.clear_kernel_cache()

plt.rcParams['axes.titlesize'] = 16      # Title
plt.rcParams['axes.labelsize'] = 14      # X and Y labels


if __name__ == "__main__":
    domain_size = 1
    total_time = 1
    c_k = [0.8]
    c_mu =[0.7]


    # grid_sizes = [int(16 * 2**n) for n in range(4)]
    # grid_sizes = [40, 80, 160, 320, 640, 1280, 2560, 3840]
    grid_sizes = [50, 100, 200, 400, 800, 1600, 2400, 3200, 4000]
    num_stepss = [int(grid_size * 2.5) for grid_size in grid_sizes]
    print(c_k)
    for c_k_led, c_mu_led in zip(c_k, c_mu):
        compute_backend = ComputeBackend.WARP
        precision_policy = PrecisionPolicy.FP32FP32

        velocity_set = xlb.velocity_set.D2Q4(precision_policy=precision_policy, compute_backend=compute_backend)

        l2_errors_u = []
        l2_errors_sigma = []
        linf_errors_u = []
        linf_errors_sigma = []
        delta_xs = []
        runtimes = []
        pure_runtimes = []

        for grid_size, num_steps, i in zip(grid_sizes, num_stepss, range(len(grid_sizes))):

            import xlb
            from xlb.compute_backend import ComputeBackend
            from xlb.precision_policy import PrecisionPolicy
            from pulse_2d_linear_elastodynamics import Pulse2D_LED
            wp.build.clear_kernel_cache()
            print(f"Starting run {i + 1} of {len(grid_sizes)}")
            grid_shape = (grid_size, grid_size)
            
            delta_x_led = domain_size/grid_size
            delta_t_led = total_time/num_steps
            c_led = delta_x_led/delta_t_led

            stability_factor = 2.0*np.sqrt(c_k_led**2+c_mu_led**2)/c_led
            print(f"Stability factor: {stability_factor}")
            assert stability_factor < 1, "Unstable"

            wp.c_mu_led = wp.constant(c_mu_led)
            wp.c_k_led = wp.constant(c_k_led)
            wp.delta_t_led = wp.constant(delta_t_led)
            wp.delta_x_led = wp.constant(delta_x_led)
            wp.c_led = wp.constant(c_led)
            wp.grid_size = wp.constant(grid_size)
            wp.S_pulse = 5e-3

            simulation = Pulse2D_LED(grid_shape, velocity_set, compute_backend, precision_policy)
            wp.build.clear_kernel_cache()
            stime_run = time.time()
            runtime = simulation.run(num_steps=num_steps, post_process_interval=num_steps+1)
            runtime_w_comp = time.time() - stime_run
            delta_xs.append(delta_x_led)
            runtimes.append(runtime_w_comp)
            pure_runtimes.append(runtime)

            script_dir = os.path.dirname(os.path.abspath(__file__))
            figures_dir = os.path.join(script_dir, "figures_final")
            os.makedirs(figures_dir, exist_ok=True)
            wp.build.clear_kernel_cache()

            print(pure_runtimes)
            print(runtimes)
        print(f"Finished runs for ck = {c_k_led}, cmu = {c_mu_led}.")
    print("Finished all runs.")






