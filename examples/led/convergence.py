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

    errors = []
    delta_xs = []

    for grid_size, num_steps in zip(grid_sizes, num_stepss):
        grid_shape = (grid_size, grid_size)
        
        delta_x_led = domain_size/grid_size
        delta_t_led = total_time/num_steps
        c_led = delta_x_led/delta_t_led

        wp.c_mu_led = wp.constant(c_mu_led)
        wp.c_k_led = wp.constant(c_k_led)
        wp.delta_t_led = wp.constant(delta_t_led)
        wp.delta_x_led = wp.constant(delta_x_led)
        wp.c_led = wp.constant(c_led)

        stability_factor = 2.0*np.sqrt(c_k_led**2+c_mu_led**2.0)/c_led
        print(f"Stability factor: {stability_factor}")
        assert stability_factor < 1, "Unstable"

        simulation = SineWave2D_LED(grid_shape, velocity_set, compute_backend, precision_policy)
        errors.append(simulation.run(num_steps=num_steps, post_process_interval=1))
        delta_xs.append(delta_x_led)

    plt.figure()
    plt.loglog(delta_xs, errors, marker="o", markersize=8)
    plt.show()

