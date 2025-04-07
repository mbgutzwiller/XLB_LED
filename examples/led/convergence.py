import numpy as np
import matplotlib.pyplot as plt
import time
import os

import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
import warp as wp

from examples.led.sine_wave_2d_linear_elastodynamics import SineWave2D_LED


# To save figures when on remote desktop using ssh which makes
#  showing plots directly using matplotlib pretty much impossible.
script_dir = os.path.dirname(os.path.abspath(__file__))
# Create 'figures' directory next to the script
figures_dir = os.path.join(script_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)


if __name__ == "__main__":
    _run_id = int(time.time())
    domain_size = 1
    total_time = 1
    c_k = np.array([0.8])
    c_mu = np.array([0.7])
    for c_k_led, c_mu_led in zip(c_k, c_mu):
        compute_backend = ComputeBackend.WARP
        precision_policy = PrecisionPolicy.FP32FP32

        velocity_set = xlb.velocity_set.D2Q4(precision_policy=precision_policy, compute_backend=compute_backend)

        grid_sizes = [80, 120, 160, 240, 320, 400]
        num_stepss = [int(grid_size * 2.5) for grid_size in grid_sizes]

        l2_errors_u = []
        l2_errors_sigma = []
        linf_errors_u = []
        linf_errors_sigma = []
        delta_xs = []

        for grid_size, num_steps, i in zip(grid_sizes, num_stepss, range(len(grid_sizes))):
            import xlb
            from xlb.compute_backend import ComputeBackend
            from xlb.precision_policy import PrecisionPolicy
            from examples.led.sine_wave_2d_linear_elastodynamics import SineWave2D_LED
            wp.build.clear_kernel_cache()
            print(f"Starting run {i + 1} of {len(grid_sizes)}")
            grid_shape = (grid_size, grid_size)
            
            delta_x_led = domain_size/grid_size
            delta_t_led = total_time/num_steps
            c_led = delta_x_led/delta_t_led

            stability_factor = 2.0*np.sqrt(c_k_led+c_mu_led)/c_led
            print(f"Stability factor: {stability_factor}")
            assert stability_factor < 1, "Unstable"

            wp.c_mu_led = wp.constant(float(c_mu_led))
            wp.c_k_led = wp.constant(float(c_k_led))
            wp.delta_t_led = wp.constant(float(delta_t_led))
            wp.delta_x_led = wp.constant(float(delta_x_led))
            wp.c_led = wp.constant(float(c_led))
            wp.grid_size = wp.constant(float(grid_size))

            print("Compiling...")
            simulation = SineWave2D_LED(grid_shape, velocity_set, compute_backend, precision_policy)
            print("... has finished.")
            n_pp_steps = np.array([20, 5])
            pp_intervals = num_steps / n_pp_steps
            for pp_interval in pp_intervals:
                error_u, error_sigma, linf_error_u, linf_error_sigma =  simulation.run(num_steps=num_steps, post_process_interval=pp_interval)
                l2_errors_u.append(error_u)
                l2_errors_sigma.append(error_sigma)
                linf_errors_u.append(linf_error_u)
                linf_errors_sigma.append(linf_error_sigma)
                delta_xs.append(delta_x_led)

                # Plotting L2 errors
                C_u = l2_errors_u[0] / (delta_xs[0] ** 2)  # reference line for 2nd order convergence
                C_sigma = l2_errors_sigma[0] / (delta_xs[0] ** 2)
                fig1, axs1 = plt.subplots(1, 2, figsize = (10, 5))
                axs1[0].loglog(delta_xs[::2], l2_errors_u[::2], marker="o", markersize=8, color="black", label=f"N pp. steps = {n_pp_steps[0]}")
                axs1[0].loglog(delta_xs[1::2], l2_errors_u[1::2], marker="o", markersize=8, color="green", label=f"N pp. steps = {n_pp_steps[1]}")
                axs1[0].set_xlabel("delta_x [m]")
                axs1[0].set_ylabel("error [m]")
                axs1[0].grid(True, which="both")
                axs1[0].loglog(delta_xs, C_u * np.array(delta_xs)**2, ":", label="Slope = 2", alpha=1, color="grey")
                axs1[0].legend()
                axs1[1].loglog(delta_xs[::2], l2_errors_sigma[::2], marker="o", markersize=8, color="black", label=f"N pp. steps = {n_pp_steps[0]}")
                axs1[1].loglog(delta_xs[1::2], l2_errors_sigma[1::2], marker="o", markersize=8, color="green", label=f"N pp. steps = {n_pp_steps[1]}")
                axs1[1].set_xlabel("delta_x [m]")
                axs1[1].set_ylabel("error [N/m^2]")
                axs1[1].grid(True, which="both")
                axs1[1].loglog(delta_xs, C_sigma * np.array(delta_xs)**2, ":", label="Slope = 2", alpha=1, color="grey")
                axs1[1].legend()
                fig1.suptitle(f"L2 Error of Displacement and Stress")
                plt.savefig(f"{figures_dir}/f_0bound_dirBC_u_sig_L2_ck_ppinterval_test_N_pp_{n_pp_steps[0]}_{n_pp_steps[1]}")
                plt.show(block=False)

                # Plotting Linf errors
                C_u = linf_errors_u[0] / (delta_xs[0] ** 2)  # reference line for 2nd order convergence
                C_sigma = linf_errors_sigma[0] / (delta_xs[0] ** 2)
                C_sigma_linear = linf_errors_sigma[0] / (delta_xs[0])
                fig2, axs2 = plt.subplots(1, 2, figsize = (10, 5))
                axs2[0].loglog(delta_xs[::2], linf_errors_u[::2], marker="o", markersize=8, color="black", label=f"N pp. steps = {n_pp_steps[0]}")
                axs2[0].loglog(delta_xs[1::2], linf_errors_u[1::2], marker="o", markersize=8, color="green", label=f"N pp. steps = {n_pp_steps[1]}")
                axs2[0].set_xlabel("delta_x [m]")
                axs2[0].set_ylabel("error [m]")
                axs2[0].grid(True, which="both")
                axs2[0].loglog(delta_xs, C_u * np.array(delta_xs)**2, ":", label="Slope = 2", alpha=1, color="grey")
                axs2[0].legend()
                axs2[1].loglog(delta_xs[::2], linf_errors_sigma[::2], marker="o", markersize=8, color="black", label=f"N pp. steps = {n_pp_steps[0]}")
                axs2[1].loglog(delta_xs[1::2], linf_errors_sigma[1::2], marker="o", markersize=8, color="green", label=f"N pp. steps = {n_pp_steps[1]}")
                axs2[1].set_xlabel("delta_x [m]")
                axs2[1].set_ylabel("error [N/m^2]")
                axs2[1].grid(True, which="both")
                axs2[1].loglog(delta_xs, C_sigma_linear * np.array(delta_xs), ":", label="Slope = 1", alpha=1, color="blue")
                axs2[1].loglog(delta_xs, C_sigma * np.array(delta_xs)**2, ":", label="Slope = 2", alpha=1, color="grey")
                axs2[1].legend()
                fig2.suptitle("LINF Error of Displacement and Stress")
                plt.savefig(f"{figures_dir}/f_0bound_dirBC_u_sig_LINF_ck_ppinterval_test_N_pp_{n_pp_steps[0]}_{n_pp_steps[1]}")
                plt.show(block=False)
        print(f"Finished runs for ck = {c_k_led}, cmu = {c_mu_led}.")
    print("Finished all runs.")






