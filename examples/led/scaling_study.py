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
    _c_k = [1.4]
    _c_mu =[0.1]

    c_k = [_**0.5 for _ in _c_k]
    c_mu = [_**0.5 for _ in _c_mu]

    grid_sizes = [50, 100, 200, 400, 800, 1600, 3200, 4000]
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
            error_u, error_sigma, linf_error_u, linf_error_sigma, runtime = simulation.run(num_steps=num_steps, post_process_interval=num_steps)
            # Get GPU utilization
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

            # Print results
            print(f"runtime {runtime}")
            print(f"grid size {grid_size}")
            print(f"GPU Utilization     : {gpu_util}%")
            print(f"Memory Utilization  : {mem_util}%")
            print(f"Power Usage         : {power} W (Limit: {power_limit} W)")
            print(f"VRAM Usage          : {vram_used:.1f} MB / {vram_total:.1f} MB")
            
            print(f"grid size {grid_size} fits on gpu")
            print(np.float32(wp.c_k_led)**2)
            l2_errors_u.append(error_u)
            l2_errors_sigma.append(error_sigma)
            linf_errors_u.append(linf_error_u)
            linf_errors_sigma.append(linf_error_sigma)
            delta_xs.append(delta_x_led)
            runtimes.append(runtime)
            grid_size_run.append(grid_size)
            _index = 3 if len(grid_size_run) >=4 else -1
            C_loglog_3 = runtimes[_index] / (grid_size_run[_index]**3)  # reference line for 2nd order convergence
            C_loglog_4 = runtimes[_index] / (grid_size_run[_index]**4)  # reference line for 2nd order convergence
            C_loglog_1 = runtimes[0] / (grid_size_run[0])  # reference line for 2nd order convergence
            fig1, axs1 = plt.subplots(1, 1, figsize = (10, 5))
            axs1.loglog(grid_size_run, runtimes, marker="o", markersize=8, color="black")
            axs1.set_xlabel("Grid size")
            axs1.set_ylabel("Runtime [s]")
            axs1.grid(True, which="both")
            axs1.loglog(grid_size_run, C_loglog_1 * np.array(grid_size_run), ":", label="Slope = 1", alpha=0.5, color="black")
            axs1.loglog(grid_size_run, C_loglog_3 * np.array(grid_size_run)**3, "--", label="Slope = 3", alpha=0.5, color="black")
            axs1.loglog(grid_size_run, C_loglog_4 * np.array(grid_size_run)**4, "-.", label="Slope = 4", alpha=0.5, color="black")
            axs1.legend()
            plt.savefig(f"/home/merrill/Documents/ETH/LBM for Linear Elastodynamics/Code/xlb/XLB/examples/led/figures/scaling_ck_{int(np.round(c_k_led**2, 1)*10)}_134")
            # plt.savefig(f"/home/merrillg/XLB_LED/examples/led/figures/f_paper_dirBC_u_sig_L2_ck_{int(np.round(c_k_led**2, 1)*10)}_16_2_n_{len(grid_sizes)}_test_run{run_i}")
            plt.show(block=False)


        print(f"Finished runs for ck = {c_k_led}, cmu = {c_mu_led}.")
    print("Finished all runs.")






