import warp as wp
import numpy as np
import matplotlib.pyplot as plt

import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy

from sine_wave_2d_linear_elastodynamics_v2 import SineWave2D_LED


if __name__ == "__main__":
    domain_size = 1  # Size of domain in meters
    total_time = 1  # Total real world time
    c_k_led = 1.1**0.5
    c_mu_led = 0.4**0.5

    compute_backend = ComputeBackend.WARP
    precision_policy = PrecisionPolicy.FP32FP32

    velocity_set = xlb.velocity_set.D2Q4(precision_policy=precision_policy, compute_backend=compute_backend)

    grid_sizes = [80, 120, 160, 240, 320]
    num_stepss = [int(grid_size * 2.5) for grid_size in grid_sizes]

    l2_errors_u = []
    l2_errors_sigma = []
    linf_errors_u = []
    linf_errors_sigma = []
    delta_xs = []

    for grid_size, num_steps, i in zip(grid_sizes, num_stepss, range(len(grid_sizes))):
        print(f"Starting run {i + 1} of {len(grid_sizes)}")
        grid_shape = (grid_size, grid_size)
        
        delta_x_led = domain_size/grid_size
        delta_t_led = total_time/num_steps
        c_led = delta_x_led/delta_t_led

        wp.c_mu_led = wp.constant(c_mu_led)
        wp.c_k_led = wp.constant(c_k_led)
        wp.delta_t_led = wp.constant(delta_t_led)
        wp.delta_x_led = wp.constant(delta_x_led)
        wp.c_led = wp.constant(c_led)
        wp.grid_size = wp.constant(grid_size)

        stability_factor = 2.0*np.sqrt(c_k_led**2+c_mu_led**2.0)/c_led
        print(f"Stability factor: {stability_factor}")
        assert stability_factor < 1, "Unstable"

        simulation = SineWave2D_LED(grid_shape, velocity_set, compute_backend, precision_policy)
        error_u, error_sigma, linf_error_u, linf_error_sigma =  simulation.run(num_steps=num_steps, post_process_interval=1)
        l2_errors_u.append(error_u)
        l2_errors_sigma.append(error_sigma)
        linf_errors_u.append(linf_error_u)
        linf_errors_sigma.append(linf_error_sigma)
        delta_xs.append(delta_x_led)
    print("Finished all runs.")

    # Plotting L2 errors
    C_u = l2_errors_u[0] / (delta_xs[0] ** 2)  # reference line for 2nd order convergence
    C_sigma = l2_errors_sigma[0] / (delta_xs[0] ** 2)
    fig, axs = plt.subplots(1, 2, figsize = (10, 5))
    axs[0].loglog(delta_xs, l2_errors_u, marker="o", markersize=8)
    axs[0].set_xlabel("delta_x [m]")
    axs[0].set_ylabel("error [m]")
    axs[0].grid(True, which="both")
    axs[0].loglog(delta_xs, C_u * np.array(delta_xs)**2, "--", label="Slope = 2", alpha=0.5, color="black")
    axs[0].legend()
    axs[1].loglog(delta_xs, l2_errors_sigma, marker="o", markersize=8)
    axs[1].set_xlabel("delta_x [m]")
    axs[1].set_ylabel("error [N/m^2]")
    axs[1].grid(True, which="both")
    axs[1].loglog(delta_xs, C_sigma * np.array(delta_xs)**2, "--", label="Slope = 2", alpha=0.5, color="black")
    axs[1].legend()
    fig.suptitle("Approximate L2 Error of Displacement and Stress")
    # plt.savefig("/home/merrill/Documents/ETH/LBM for Linear Elastodynamics/Code/xlb/XLB/examples/led/figures/dirichlet_u_x sigma_xy convergence plot L2")
    plt.savefig("/home/merrillg/XLB_LED/examples/led/figures/dirichlet_u_x sigma_xy convergence plot L2")
    plt.show(block=True)
    plt.clf()

    # Plotting L2 errors
    C_u = linf_errors_u[0] / (delta_xs[0] ** 2)  # reference line for 2nd order convergence
    C_sigma = linf_errors_sigma[0] / (delta_xs[0] ** 2)
    fig, axs = plt.subplots(1, 2, figsize = (10, 5))
    axs[0].loglog(delta_xs, linf_errors_u, marker="o", markersize=8)
    axs[0].set_xlabel("delta_x [m]")
    axs[0].set_ylabel("error [m]")
    axs[0].grid(True, which="both")
    axs[0].loglog(delta_xs, C_u * np.array(delta_xs)**2, "--", label="Slope = 2", alpha=0.5, color="black")
    axs[0].legend()
    axs[1].loglog(delta_xs, linf_errors_sigma, marker="o", markersize=8)
    axs[1].set_xlabel("delta_x [m]")
    axs[1].set_ylabel("error [N/m^2]")
    axs[1].grid(True, which="both")
    axs[1].loglog(delta_xs, C_sigma * np.array(delta_xs)**2, "--", label="Slope = 2", alpha=0.5, color="black")
    axs[1].legend()
    fig.suptitle("Approximate LINF Error of Displacement and Stress")
    # plt.savefig("/home/merrill/Documents/ETH/LBM for Linear Elastodynamics/Code/xlb/XLB/examples/led/figures/dirichlet_u_x sigma_xy convergence plot LINF")
    plt.savefig("/home/merrillg/XLB_LED/examples/led/figures/dirichlet_u_x sigma_xy convergence plot LINF")
    plt.show(block=True)
    plt.clf()




