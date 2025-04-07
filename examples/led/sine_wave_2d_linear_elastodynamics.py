import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.operator.stepper import LinearElastodynamicsStepperStream, LinearElastodynamicsStepperCollide
from xlb.operator.boundary_condition import DirichletBC_LED
from xlb.operator.macroscopic import Macroscopic_LED
from xlb.utils import save_fields_vtk, save_image
import xlb.velocity_set
import warp as wp
import jax.numpy as jnp
import numpy as np
import time
from xlb.helper.initializers_v2 import Initializer_LED
import matplotlib.pyplot as plt
from xlb.operator.stream import Stream_LED
from tqdm import tqdm
import os


class SineWave2D_LED:
    def __init__(self, grid_shape, velocity_set, compute_backend, precision_policy):
        xlb.init(
            velocity_set=velocity_set,
            default_backend=compute_backend,
            default_precision_policy=precision_policy,
        )

        self.grid_shape = grid_shape
        self.velocity_set = velocity_set
        self.compute_backend = compute_backend
        self.precision_policy = precision_policy
        self.omega = 2.
        self.boundary_conditions = []

        # Create grid using factory
        self.grid = grid_factory(grid_shape, compute_backend=compute_backend)

        # Setup the simulation BC and stepper
        self._setup()

    def _setup(self):
        self.setup_boundary_conditions()
        self.setup_stepper()
        # Initialize fields using the stepper
        self.f_0, self.f_1, self.f_1, self.bc_mask, self.missing_mask, self.U_num_tilde, self.u_num_displ_0, self.u_num_displ_0 = self.stepper_collide.prepare_fields()

    def define_boundary_indices(self):
        box = self.grid.bounding_box_indices()  # For interior nodes
        # lid = box_no_edge["top"]
        walls = [box["bottom"][i] + box["left"][i] + box["right"][i] + box["top"][i] for i in range(self.velocity_set.d)]
        walls = np.unique(np.array(walls), axis=-1).tolist()
        return walls  # Return as many different indices sets as you need.

    def setup_boundary_conditions(self):
        # TODO: Adjust BCs here.
        walls = self.define_boundary_indices()
        bc_walls = DirichletBC_LED(indices=walls, velocity_set=self.velocity_set)
        self.boundary_conditions = [bc_walls]
        # self.boundary_conditions = []

    def setup_stepper(self):
        self.stepper_stream = LinearElastodynamicsStepperStream(
            grid=self.grid,
            boundary_conditions=self.boundary_conditions,
            collision_type="BGK_LED",
        )
        self.stepper_collide = LinearElastodynamicsStepperCollide(
            grid=self.grid,
            boundary_conditions=self.boundary_conditions,
            collision_type="BGK_LED",
        )

    def run(self, num_steps, post_process_interval=np.inf, show_plot=False):
        # TODO: initialize U_num_here
        self.stream_LED = Stream_LED(self.velocity_set, self.precision_policy, self.compute_backend)
        initializer = Initializer_LED(velocity_set=self.velocity_set,
                                      precision_policy=self.precision_policy,
                                      compute_backend=self.compute_backend)
        # plt.figure()
        self.f_0, self.U_num_tilde, self.u_num_displ_0 = initializer(self.f_0, self.U_num_tilde, self.u_num_displ_0)

        # Initialize error calculation over whole run
        self.cumulative_error_u = 0
        self.cumulative_u = 0
        self.cumulative_error_sigma = 0
        self.cumulative_sigma = 0

        self.max_error_u = 0
        self.max_error_sigma = 0

        if show_plot:
            # To save figures when working on remote desktop using ssh which makes
            #  showing plots directly using matplotlib pretty much impossible.
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Create 'figures' directory next to the script
            figures_dir = os.path.join(script_dir, "figures")
            os.makedirs(figures_dir, exist_ok=True)
        else:
            figures_dir = None

        wp.synchronize()
        wp.synchronize_device()
        for timestep in tqdm(range(num_steps)):
            # Collision
            self.f_1, self.f_0, self.U_num_tilde, self.u_num_displ_1 = self.stepper_collide(self.f_0, self.f_1, self.bc_mask, self.omega, timestep, self.U_num_tilde, self.u_num_displ_0, self.u_num_displ_0)

            # Postprocessing, happens only if post_process_interval is smaller than num_steps.
            #  -> set pp interval > numsteps for performance analysis.
            if (timestep % post_process_interval == 0 or timestep == num_steps - 1) and (num_steps > post_process_interval):
                wp.synchronize()
                wp.synchronize_device()
                self.post_process(timestep, show_plot, figures_dir)
            
            # Streaming
            self.f_1, self.f_0, self.u_num_displ_0, self.u_num_displ_0 = self.stepper_stream(self.f_0, self.f_1, self.bc_mask, self.missing_mask, self.omega, timestep, self.U_num_tilde, self.u_num_displ_1, self.u_num_displ_0)

        if post_process_interval < num_steps:  # probably only accurate for small post_process_intervals
            # Final calculation for relative L2 and LINF error like in paper by Oliver.
            final_error_norm_u = np.sqrt(self.cumulative_error_u * np.float32(wp.delta_x_led)**2 * np.float32(wp.delta_t_led))
            final_norm_u = np.sqrt(self.cumulative_u * np.float32(wp.delta_x_led)**2 * np.float32(wp.delta_t_led))
            final_error_norm_sigma = np.sqrt(self.cumulative_error_sigma * np.float32(wp.delta_x_led)**2 * np.float32(wp.delta_t_led))
            final_norm_sigma = np.sqrt(self.cumulative_sigma * np.float32(wp.delta_x_led)**2 * np.float32(wp.delta_t_led))

            try:
                return final_error_norm_u/final_norm_u, final_error_norm_sigma/final_norm_sigma, self.max_error_u/(final_norm_u * (post_process_interval**0.5)), self.max_error_sigma/(final_norm_sigma * (post_process_interval**0.5))
            except:
                # in case no manufactured solution is available for error calculation.
                return None, None, None, None
        else:
            return None, None, None, None



    def u_num_exact_x(self, x, y, t):
        return np.sin(4.*np.pi*x) * np.sin(2.*np.pi*y) * np.sin(4.*np.pi*(t-0.1))
        # return np.sin(4.*np.pi*(x-0.3*t)) * np.cos(2.*np.pi*(y-0.8*t)) * np.sin(4.*np.pi*(t-0.1))
    
    def u_num_exact_y(self, x, y, t):
        return np.sin(4.*np.pi*x) * np.sin(2.*np.pi*y) * np.sin(4.*np.pi*(t+0.3))
        # return np.cos(4.*np.pi*(x-0.7*t)) * np.sin(2.*np.pi*(y-0.1*t)) * np.cos(4.*np.pi*(t+0.4))
    
    def U_vx(self, x, y, t):
        return 4.*np.pi*np.sin(4.*np.pi*x)*np.sin(2.*np.pi*y)*np.cos(4.*np.pi*(t - 1./10.))
        # return 1.6*np.pi*np.sin(np.pi*(-1.6*t + 2.0*y))*np.sin(np.pi*(-1.2*t + 4.0*x))*np.sin(np.pi*(4.0*t - 0.4)) + 4.0*np.pi*np.sin(np.pi*(-1.2*t + 4.0*x))*np.cos(np.pi*(-1.6*t + 2.0*y))*np.cos(np.pi*(4.0*t - 0.4)) - 1.2*np.pi*np.sin(np.pi*(4.0*t - 0.4))*np.cos(np.pi*(-1.6*t + 2.0*y))*np.cos(np.pi*(-1.2*t + 4.0*x))
    
    def U_vy(self, x, y, t):
        return 4.*np.pi*np.sin(4.*np.pi*x)*np.sin(2.*np.pi*y)*np.cos(4.*np.pi*(t + 3./10.))
        # return 2.8*np.pi*np.sin(np.pi*(-2.8*t + 4.0*x))*np.sin(np.pi*(-0.2*t + 2.0*y))*np.cos(np.pi*(4.0*t + 1.6)) - 4.0*np.pi*np.sin(np.pi*(-0.2*t + 2.0*y))*np.sin(np.pi*(4.0*t + 1.6))*np.cos(np.pi*(-2.8*t + 4.0*x)) - 0.2*np.pi*np.cos(np.pi*(-2.8*t + 4.0*x))*np.cos(np.pi*(-0.2*t + 2.0*y))*np.cos(np.pi*(4.0*t + 1.6))
    
    def U_js(self, x, y, t):
        return -wp.c_k_led**(1./2.)*(4.*np.pi*np.cos(4.*np.pi*x)*np.sin(2.*np.pi*y)*np.sin(4.*np.pi*(t - 1./10.)) + 2.*np.pi*np.cos(2.*np.pi*y)*np.sin(4.*np.pi*x)*np.sin(4.*np.pi*(t + 3./10.)))
        # return -np.c_k_led*(4.0*np.pi*np.sin(np.pi*(4.0*t - 0.4))*np.cos(np.pi*(-1.6*t + 2.0*y))*np.cos(np.pi*(-1.2*t + 4.0*x))+2.0*np.pi*np.cos(np.pi*(-2.8*t + 4.0*x))*np.cos(np.pi*(-0.2*t + 2.0*y))*np.cos(np.pi*(4.0*t + 1.6)))
    
    def U_jd(self, x, y, t):
        return -wp.c_mu_led**(1./2.)*(4.*np.pi*np.cos(4.*np.pi*x)*np.sin(2.*np.pi*y)*np.sin(4.*np.pi*(t - 1./10.)) - 2.*np.pi*np.cos(2.*np.pi*y)*np.sin(4.*np.pi*x)*np.sin(4.*np.pi*(t + 3./10.)))
        # return -np.c_mu_led*(4.0*np.pi*np.sin(np.pi*(4.0*t - 0.4))*np.cos(np.pi*(-1.6*t + 2.0*y))*np.cos(np.pi*(-1.2*t + 4.0*x))-2.0*np.pi*np.cos(np.pi*(-2.8*t + 4.0*x))*np.cos(np.pi*(-0.2*t + 2.0*y))*np.cos(np.pi*(4.0*t + 1.6)))
    
    def U_jxy(self,x, y, t):
        return -wp.c_mu_led**(1./2.)*(2.*np.pi*np.cos(2.*np.pi*y)*np.sin(4.*np.pi*x)*np.sin(4.*np.pi*(t - 1./10.)) + 4.*np.pi*np.cos(4.*np.pi*x)*np.sin(2.*np.pi*y)*np.sin(4.*np.pi*(t + 3./10.)))
        # return -wp.c_mu_led*(-2.0*np.pi*np.sin(np.pi*(-1.6*t + 2.0*y))*np.sin(np.pi*(-1.2*t + 4.0*x))*np.sin(np.pi*(4.0*t - 0.4))-4.0*np.pi*np.sin(np.pi*(-2.8*t + 4.0*x))*np.sin(np.pi*(-0.2*t + 2.0*y))*np.cos(np.pi*(4.0*t + 1.6)))

    def post_process(self, i, show_plot=False, figures_dir=None):
        # Write the results, using JAX compute_backend for the post-processing
        if not isinstance(self.f_0, jnp.ndarray):
            # If the compute_backend is warp, we need to drop the last dimension added by warp for 2D simulations
            U_num_tilde = wp.to_jax(self.U_num_tilde)[..., 0]
            u_num_displ = wp.to_jax(self.u_num_displ_1)[..., 0]

        # For postprocessing e.g. in paraview
        fields = {"u_x": u_num_displ[0],
                  "u_y": u_num_displ[1],
                  "abs_u": np.sqrt(np.square(u_num_displ[0]) + np.square(u_num_displ[1])),
                  "sigma_xx": -(wp.c_k_led * U_num_tilde[2] + wp.c_mu_led * U_num_tilde[3]),
                  "sigma_yy": -(wp.c_k_led * U_num_tilde[2] - wp.c_mu_led * U_num_tilde[3]),
                  "sigma_xy": -(wp.c_mu_led * U_num_tilde[4])}
        save_fields_vtk(fields, timestep=i, prefix="results/pulse")
        # save_image(fields["sigma_xx"], timestep=i, prefix="results/pulse")

        # # Compare solutions on cuts through 2d plane
        # t = np.float32(i * wp.delta_t_led)
        # grid_size = self.grid_shape[0]
        # plot_index = int(0.57 * (grid_size-1))  # set relative cut position here
        # assert (plot_index >=0) and (plot_index <= grid_size-1), "Plotting position/index invalid"
        # plot_index_num = plot_index
        # domain_size = 1
        # delta_x = domain_size/grid_size
        # cut_position = delta_x * (plot_index + 0.5)
        # x_axis_num = np.linspace(delta_x/2, domain_size-delta_x/2, num=grid_size)

        # if show_plot:
        #     plt.clf()
        #     plt.plot(x_axis_num, self.u_num_exact_x(x=x_axis_num, y=cut_position, t=t), label="y = const, u_ex_x", color="blue")
        #     plt.plot(x_axis_num, self.u_num_exact_y(x=cut_position, y=x_axis_num, t=t), label="x = const, u_ex_y", color="green")
        #     plt.plot(x_axis_num, u_num_displ[0, :, plot_index_num], label="y = const, u_num_x", linestyle=":", color="red")
        #     plt.plot(x_axis_num, u_num_displ[1, plot_index_num, :], label="y = const, u_num_y", linestyle=":", color="orange")
        #     plt.grid()
        #     plt.ylim(-1, 1)
        #     plt.title(f"t = {t:.6f}s, interval {i}")
        #     plt.legend()
        #     plt.draw()
        #     plt.savefig(f"{figures_dir}/00_ux_uy_figure", dpi=300)
        #     plt.pause(1)


        # """
        # Approximate error calculation
        # """
        # # Calculation for L2 norm
        # u_ex_x = np.array([np.array([self.u_num_exact_x(x=_y, y=_x, t=t) for _x in x_axis_num]) for _y in x_axis_num])
        # u_ex_y = np.array([np.array([self.u_num_exact_y(x=_y, y=_x, t=t) for _x in x_axis_num]) for _y in x_axis_num])

        # u_num_x = u_num_displ[0, :, :]
        # u_num_y = u_num_displ[1, :, :]

        # # norm_error_u = np.linalg.norm(np.stack([(u_ex_x - u_num_x), (u_ex_y - u_num_y)], axis=0))
        # norm_error_u = np.sum((u_ex_x - u_num_x) ** 2 + (u_ex_y - u_num_y) ** 2)
        # max_error_u = np.max(np.abs(np.stack([u_ex_x - u_num_x,
        #                                       u_ex_y - u_num_y,
        #                                       ], axis=0)))
        # # norm_u = np.linalg.norm(np.stack([u_ex_x, u_ex_y], axis=0))
        # norm_u = np.sum((u_ex_x) ** 2 + (u_ex_y) ** 2)

        # self.cumulative_error_u += norm_error_u
        # self.cumulative_u += norm_u
        # if max_error_u > self.max_error_u:
        #     self.max_error_u = max_error_u

        # _U_js_ex = np.array([np.array([self.U_js(x=_y, y=_x, t=t) for _x in x_axis_num]) for _y in x_axis_num])
        # _U_jd_ex = np.array([np.array([self.U_jd(x=_y, y=_x, t=t) for _x in x_axis_num]) for _y in x_axis_num])
        # _U_jxy_ex = np.array([np.array([self.U_jxy(x=_y, y=_x, t=t) for _x in x_axis_num]) for _y in x_axis_num])

        # sigma_xx_ex = -(wp.c_k_led ** 0.5 * _U_js_ex + wp.c_mu_led ** 0.5 * _U_jd_ex)
        # sigma_yy_ex = -(wp.c_k_led ** 0.5 * _U_js_ex - wp.c_mu_led ** 0.5 * _U_jd_ex)
        # sigma_xy_ex = -(wp.c_mu_led ** 0.5 * _U_jxy_ex)

        # sigma_xx_num = -(wp.c_k_led ** 0.5 * U_num_tilde[2, :, :] + wp.c_mu_led ** 0.5 * U_num_tilde[3, :, :])
        # sigma_yy_num = -(wp.c_k_led ** 0.5 * U_num_tilde[2, :, :] - wp.c_mu_led ** 0.5 * U_num_tilde[3, :, :])
        # sigma_xy_num = -(wp.c_mu_led ** 0.5 * U_num_tilde[4, :, :])

        # norm_error_sigma = np.sum((sigma_xx_ex - sigma_xx_num) ** 2 + (sigma_yy_ex - sigma_yy_num) ** 2 + (sigma_xy_ex - sigma_xy_num) ** 2)
        # norm_sigma = np.sum((sigma_xx_ex) ** 2 + (sigma_yy_ex) ** 2 + (sigma_xy_ex) ** 2)
        # max_error_sigma = np.max(np.abs(np.stack([sigma_xx_ex - sigma_xx_num,
        #                                           sigma_yy_ex - sigma_yy_num,
        #                                           sigma_xy_ex - sigma_xy_num
        #                                           ], axis=0)))
                
        # self.cumulative_error_sigma += norm_error_sigma
        # self.cumulative_sigma += norm_sigma
        # if max_error_sigma > self.max_error_sigma:
        #     self.max_error_sigma = max_error_sigma


if __name__ == "__main__":
    # # Running the simulation
    grid_size = 200  # Number of grid cells along one dimension
    grid_shape = (grid_size, grid_size)
    num_steps = int(2.5 * grid_size)  # Number of collision/streaming steps
    pp_interval = int(1)  # Post process interval
    domain_size = 1  # Size of domain in meters
    delta_x_led = domain_size/grid_size
    total_time = 1  # Total real world time
    delta_t_led = total_time/num_steps
    c_led = delta_x_led/delta_t_led
    c_k_led = 0.8  
    c_mu_led = 0.7
    stability_factor = 2.0*np.sqrt(c_k_led**2+c_mu_led**2.0)/c_led

    # Setting problem parameters as constants to warp for ease of use and performance
    wp.grid_size = wp.constant(grid_size)
    wp.c_mu_led = wp.constant(c_mu_led)
    wp.c_k_led = wp.constant(c_k_led)
    wp.delta_t_led = wp.constant(delta_t_led)
    wp.delta_x_led = wp.constant(delta_x_led)
    wp.c_led = wp.constant(c_led)
    # Setting sharpness of pulse:
    wp.S_pulse = wp.constant(5e-3)

    print(f"Stability factor: {stability_factor}")
    assert stability_factor < 1, "Unstable!"

    compute_backend = ComputeBackend.WARP
    precision_policy = PrecisionPolicy.FP32FP32

    velocity_set = xlb.velocity_set.D2Q4(precision_policy=precision_policy, compute_backend=compute_backend)

    stime = time.time()
    simulation = SineWave2D_LED(grid_shape, velocity_set, compute_backend, precision_policy)
    simulation.run(num_steps=num_steps, post_process_interval=pp_interval, show_plot=False)
    print(f"took {time.time() - stime:.2} seconds")

