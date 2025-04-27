import os
# os.environ["JAX_PLATFORMS"] = "cpu"  # Sometimes it wont automatically run on cpu if no gpu is available.


import warp as wp
import numpy as np
import matplotlib.pyplot as plt
import time

import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from sine_wave_2d_linear_elastodynamics_v2 import SineWave2D_LED
wp.build.clear_kernel_cache()

plt.rcParams['axes.titlesize'] = 16      # Title
plt.rcParams['axes.labelsize'] = 14      # X and Y labels


if __name__ == "__main__":
    domain_size = 1
    total_time = 1
    c_k = [0.8]
    c_mu =[0.7]


    # grid_sizes = [int(16 * 2**n) for n in range(4)]
    grid_sizes = [40, 80, 120, 160, 240]
    # grid_sizes = [30, 50, 80, 100, 120]# 200, 300, 400, 600, 800, 1600, 2400, 3200, 4000]
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

        for grid_size, num_steps, i in zip(grid_sizes, num_stepss, range(len(grid_sizes))):
            import xlb
            from xlb.compute_backend import ComputeBackend
            from xlb.precision_policy import PrecisionPolicy
            from sine_wave_2d_linear_elastodynamics_v2 import SineWave2D_LED
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

            simulation = SineWave2D_LED(grid_shape, velocity_set, compute_backend, precision_policy)
            wp.build.clear_kernel_cache()
            error_u, error_sigma, linf_error_u, linf_error_sigma, _ = simulation.run(num_steps=num_steps, post_process_interval=1)
            print(np.float32(wp.c_k_led)**2)
            l2_errors_u.append(error_u)
            l2_errors_sigma.append(error_sigma)
            linf_errors_u.append(linf_error_u)
            linf_errors_sigma.append(linf_error_sigma)
            delta_xs.append(delta_x_led)

            script_dir = os.path.dirname(os.path.abspath(__file__))
            figures_dir = os.path.join(script_dir, "figures_final")
            os.makedirs(figures_dir, exist_ok=True)
            wp.build.clear_kernel_cache()

            # Plotting L2 errors
            C_u = l2_errors_u[0] / (delta_xs[0] ** 2)  # reference line for 2nd order convergence
            C_sigma = l2_errors_sigma[0] / (delta_xs[0] ** 2)
            fig1, axs1 = plt.subplots(1, 2, figsize = (12, 5), gridspec_kw={'wspace': 0.3})
            axs1[0].loglog(delta_xs, l2_errors_u, marker="o", markersize=8)
            axs1[0].set_xlabel("Grid Spacing [m]")
            axs1[0].set_ylabel("Rel. Error Displacement [-]")
            axs1[0].grid(True, which="both")
            axs1[0].loglog(delta_xs, C_u * np.array(delta_xs)**2, "--", label="Slope = 2", alpha=1, color="black")
            axs1[0].legend()
            axs1[1].loglog(delta_xs, l2_errors_sigma, marker="o", markersize=8)
            axs1[1].set_xlabel("Grid Spacing [m]")
            axs1[1].set_ylabel("Rel. Error Stress [-]")
            axs1[1].grid(True, which="both")
            axs1[1].loglog(delta_xs, C_sigma * np.array(delta_xs)**2, "--", label="Slope = 2", alpha=1, color="black")
            axs1[1].legend()
            # fig1.suptitle(r"Relative $L_{2}$ Error of Displacement and Stress", fontsize=16)
            plt.savefig(os.path.join(figures_dir, f"dirBC_L2_errors_f0bound_ck_{int(np.round(c_k_led, 1)*10)}_temp"), dpi=600)
            plt.show(block=False)

            # Plotting L2 errors
            C_u = linf_errors_u[0] / (delta_xs[0] ** 2)  # reference line for 2nd order convergence
            C_sigma = linf_errors_sigma[0] / (delta_xs[0] ** 2)
            C_sigma_linear = linf_errors_sigma[0] / (delta_xs[0])
            fig2, axs2 = plt.subplots(1, 2, figsize = (12, 5), gridspec_kw={'wspace': 0.3})
            axs2[0].loglog(delta_xs, linf_errors_u, marker="o", markersize=8)
            axs2[0].set_xlabel("Grid Spacing [m]")
            axs2[0].set_ylabel("Rel. Error Displacement [-]")
            axs2[0].grid(True, which="both")
            axs2[0].loglog(delta_xs, C_u * np.array(delta_xs)**2, "--", label="Slope = 2", alpha=1, color="black")
            axs2[0].legend()
            axs2[1].loglog(delta_xs, linf_errors_sigma, marker="o", markersize=8)
            axs2[1].set_xlabel("Grid Spacing [m]")
            axs2[1].set_ylabel("Rel. Error Stress [-]")
            axs2[1].grid(True, which="both")
            axs2[1].loglog(delta_xs, C_sigma_linear * np.array(delta_xs), "--", label="Slope = 1", alpha=1, color="orange")
            axs2[1].loglog(delta_xs, C_sigma * np.array(delta_xs)**2, "--", label="Slope = 2", alpha=1, color="black")
            axs2[1].legend()
            # fig2.suptitle(r"Relative $L_{\infty}$ Error of Displacement and Stress", fontsize=16)
            plt.savefig(os.path.join(figures_dir, f"dirBC_LINF_errors_f0bound_ck_{int(np.round(c_k_led, 1)*10)}_temp"), dpi=600)
            plt.show(block=False)
            print(f"l2 errors u: {l2_errors_u}")
            print(f"l2 errors sigma: {l2_errors_sigma}")
            print(f"linf errors u: {linf_errors_u}")
            print(f"linf errors sigma: {linf_errors_sigma}")
            print(f"Finished runs for ck = {c_k_led}, cmu = {c_mu_led}.")
    print("Finished all runs.")






