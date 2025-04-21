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
# wp.build.clear_kernel_cache()

from pynvml import *
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)



if __name__ == "__main__":
    domain_size = 1
    total_time = 1
    _c_k = [0.8]
    _c_mu =[0.7]

    c_k = [_**0.5 for _ in _c_k]
    c_mu = [_**0.5 for _ in _c_mu]

    grid_sizes = [50, 100, 150, 200, 250, 300, 350, 400, 800, 1600, 2400, 3200, 4000]
    grid_sizes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 800, 1600, 2400, 3200, 4000]
    num_stepss = [int(grid_size * 2.5) for grid_size in grid_sizes]
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
        grid_size_run = []

        gpu_vrams = []
        gpu_powers = []
        gpu_compute_utils = []
        gpu_mem_utils = []

        for grid_size, num_steps, i in zip(grid_sizes, num_stepss, range(len(grid_sizes))):
            time.sleep(5)
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
            assert stability_factor < 1, f"Unstable: is {stability_factor}"

            wp.c_mu_led = wp.constant(c_mu_led)
            wp.c_k_led = wp.constant(c_k_led)
            wp.delta_t_led = wp.constant(delta_t_led)
            wp.delta_x_led = wp.constant(delta_x_led)
            wp.c_led = wp.constant(c_led)
            wp.grid_size = wp.constant(grid_size)

            simulation = SineWave2D_LED(grid_shape, velocity_set, compute_backend, precision_policy)
            wp.build.clear_kernel_cache()
            num_steps_gpu_load = int(3e7/num_steps**0.8)
            error_u, error_sigma, linf_error_u, linf_error_sigma, runtime = simulation.run(num_steps=num_steps_gpu_load, post_process_interval=num_steps_gpu_load)
            # Get GPU utilization
            runtimes.append(runtime)
            utilization = nvmlDeviceGetUtilizationRates(handle)
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            power = nvmlDeviceGetPowerUsage(handle) / 1000  # in watts
            power_limit = nvmlDeviceGetEnforcedPowerLimit(handle) / 1000  # in watts
            # Get memory usage
            vram_used = mem_info.used / (1024 ** 2)   # in MB
            vram_total = mem_info.total / (1024 ** 2) # in MB

            # Get utilization rates
            gpu_util = utilization.gpu  # in percent
            mem_util = utilization.memory  # in percent

            gpu_powers.append(power)
            gpu_vrams.append(vram_used)
            gpu_mem_utils.append(mem_util)
            gpu_compute_utils.append(gpu_util)

            # Print results
            print(f"runtime {runtime}")
            print(f"grid size {grid_size}")
            print(f"GPU Utilization     : {gpu_util}%")
            print(f"Memory Utilization  : {mem_util}%")
            print(f"Power Usage         : {power} W (Limit: {power_limit} W)")
            print(f"VRAM Usage          : {vram_used:.1f} MB / {vram_total:.1f} MB")

            print(f"runtimes: {runtimes}")
            print(f"gpu comp util: {gpu_compute_utils}")
            print(f"gpu mem util: {gpu_mem_utils}")
            print(f"gpu vram: {gpu_vrams}")
            print(f"gpu power: {gpu_powers}")

        print(f"Finished runs for ck = {c_k_led}, cmu = {c_mu_led}.")
    print("Finished all runs.")






