import time
import numpy as np
import importlib
# os.environ["JAX_PLATFORMS"] = "cpu"  # Sometimes it wont automatically run on cpu if no gpu is available.
import warp as wp
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from examples.led.pulse_2d_linear_elastodynamics import Pulse2D_LED
import matplotlib.pyplot as plt
wp.build.clear_kernel_cache()

if __name__ == "__main__":
    domain_size = 1
    total_time = 1
    c_k = [1.4]
    c_mu =[0.1]

    grid_sizes = [50, 100, 200, 400, 800, 1600, 3200, 4000]
    num_stepss = [int(_ * 2.5) for _ in grid_sizes]
    for c_k_led, c_mu_led in zip(c_k, c_mu):
        grid_size_run = []
        total_runtimes = []
        
        for grid_size, num_steps, i in zip(grid_sizes, num_stepss, range(len(grid_sizes))):
            stime = time.time()
            grid_size_run.append(grid_size)
            compute_backend = ComputeBackend.WARP
            precision_policy = PrecisionPolicy.FP32FP32
            velocity_set = xlb.velocity_set.D2Q4(precision_policy=precision_policy, compute_backend=compute_backend)
            
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
            wp.S_pulse = wp.constant(5e-3)

            stability_factor = 2.0*np.sqrt(c_k_led+c_mu_led)/c_led
            print(f"Stability factor: {stability_factor}")
            assert stability_factor < 1, f"Unstable: is {stability_factor}"

            simulation = Pulse2D_LED(grid_shape, velocity_set, compute_backend, precision_policy)
            # num_steps_gpu_load = int(1e7/num_steps**0.8)
            num_steps_gpu_load = num_steps
            runtime = simulation.run(num_steps=num_steps_gpu_load, post_process_interval=num_steps_gpu_load)
            total_runtime = time.time() - stime
            total_runtimes.append(total_runtime)
            # Get GPU utilization
            print(f"runtime {runtime}")
            print(f"total runtime {total_runtime}")
            print(f"grid size {grid_size}")
            print(f"grid size {grid_size} fits on gpu")
            print(np.float32(wp.c_k_led)**2)
            # C_loglog_3 = runtimes[_index] / (grid_size_run[_index]**3)  # reference line for 2nd order convergence
            # C_loglog_4 = runtimes[_index] / (grid_size_run[_index]**4)  # reference line for 2nd order convergence
            # C_loglog_1 = runtimes[0] / (grid_size_run[0])  # reference line for 2nd order convergence

            fig1, axs1 = plt.subplots(1, 1, figsize = (10, 5))
            axs1.loglog(grid_size_run, total_runtimes, marker="o", markersize=8, color="black", label="Runtime")
            axs1.set_xlabel("Grid size [-]")
            axs1.set_ylabel("Runtime [s]")
            axs1.grid(True, which="both")
            # axs1.loglog(grid_size_run, C_loglog_1 * np.array(grid_size_run), ":", label="Slope = 1", alpha=0.5, color="black")
            # axs1.loglog(grid_size_run, C_loglog_3 * np.array(grid_size_run)**3, "--", label="Slope = 3", alpha=0.5, color="black")
            # axs1.loglog(grid_size_run, C_loglog_4 * np.array(grid_size_run)**4, "-.", label="Slope = 4", alpha=0.5, color="black")
            axs1.set_ylim(0.5 * np.min(total_runtimes), 2*np.max(total_runtimes))
            fig1.suptitle(f"Runtime vs. Problem Size")
            axs1.legend()

            # Save plot using absolute path
            import os
            # Get path to current script
            script_dir = os.path.dirname(os.path.abspath(__file__))

            # Full path to 'figures/' folder next to this script
            figures_dir = os.path.join(script_dir, "figures_final")
            os.makedirs(figures_dir, exist_ok=True)
            plt.savefig(os.path.join(figures_dir, f"total_runtime_ck_{int(np.round(c_k_led**2, 1)*10)}_pulse"))
            # plt.savefig(f"figures/scaling_ck_{int(np.round(c_k_led**2, 1)*10)}_134_test")
            # plt.savefig(f"/home/merrillg/XLB_LED/examples/led/figures/f_paper_dirBC_u_sig_L2_ck_{int(np.round(c_k_led**2, 1)*10)}_16_2_n_{len(grid_sizes)}_test_run{run_i}")
            plt.show(block=False)

            wp.build.clear_kernel_cache()



        print(f"Finished runs for ck = {c_k_led}, cmu = {c_mu_led}.")
        print(f"total runtimes were {total_runtimes}")
    print("Finished all runs.")







