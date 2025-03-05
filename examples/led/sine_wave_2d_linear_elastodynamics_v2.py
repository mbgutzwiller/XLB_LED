import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.operator.stepper import LinearElastodynamicsStepperStream, LinearElastodynamicsStepperCollide
from xlb.operator.equilibrium import Equilibrium_LED
from xlb.operator.boundary_condition import DirichletBC_LED  # TODO: Add periodic and Dirichlet BCs.
from xlb.operator.macroscopic import Macroscopic_LED
from xlb.utils import save_fields_vtk, save_image
import xlb.velocity_set  # Done.
import warp as wp
import jax.numpy as jnp
import numpy as np
import time
from xlb.helper.initializers import Initializer_LED
import matplotlib.pyplot as plt
plt.ion()

# wp.config.print_launches = False
# wp.config.mode = "release"


class SineWave2D_LED:
    def __init__(self, omega, grid_shape, velocity_set, compute_backend, precision_policy):
        # initialize compute_backend
        xlb.init(
            velocity_set=velocity_set,
            default_backend=compute_backend,
            default_precision_policy=precision_policy,
        )

        self.grid_shape = grid_shape
        self.velocity_set = velocity_set
        self.compute_backend = compute_backend
        self.precision_policy = precision_policy
        self.omega = omega
        self.boundary_conditions = []
        # self.prescribed_vel = prescribed_vel

        # Create grid using factory
        self.grid = grid_factory(grid_shape, compute_backend=compute_backend)

        # Setup the simulation BC and stepper
        self._setup()

    def _setup(self):
        self.setup_boundary_conditions()
        self.setup_stepper()
        # Initialize fields using the stepper
        self.f_0, self.f_1, self.bc_mask, self.missing_mask, self.U_num_tilde, self.u_num_displ_0, self.u_num_displ_1 = self.stepper_collide.prepare_fields()

    def define_boundary_indices(self):
        box = self.grid.bounding_box_indices()  # For interior nodes
        box_no_edge = self.grid.bounding_box_indices(remove_edges=True)  # For boundary nodes
        # lid = box_no_edge["top"]
        walls = [box["bottom"][i] + box["left"][i] + box["right"][i] + box["top"][i] for i in range(self.velocity_set.d)]
        walls = np.unique(np.array(walls), axis=-1).tolist()
        return walls  # Return as many different indices sets as you need.

    def setup_boundary_conditions(self):
        # # TODO: Adjust BCs here.
        # walls = self.define_boundary_indices()
        # bc_walls = DirichletBC_LED(indices=walls)
        # self.boundary_conditions = [bc_walls]
        self.boundary_conditions = []

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

    def run(self, num_steps, post_process_interval=100):
        # TODO: initialize U_num_here
        initializer = Initializer_LED(velocity_set=self.velocity_set,
                                      precision_policy=self.precision_policy,
                                      compute_backend=self.compute_backend)
        plt.figure()
        self.f_0, self.U_num_tilde, self.u_num_displ_0 = initializer(self.f_0, self.U_num_tilde, self.u_num_displ_0)
        wp.copy(dest=self.f_1, src=self.f_0)
        # wp.copy(self.U_num_tilde_1, self.U_num_tilde_0)
        wp.copy(self.u_num_displ_1 , self.u_num_displ_0)

        for timestep in range(num_steps):
            # Collision first:
            self.f_0, self.f_1, self.U_num_tilde, self.u_num_displ_0, self.u_num_displ_1 = self.stepper_collide(self.f_0, self.f_1, self.bc_mask, self.omega, timestep, self.U_num_tilde, self.u_num_displ_0, self.u_num_displ_1)
            # Streaming second:
            # self.f_0, self.f_1 = self.f_1, self.f_0
            self.f_0, self.f_1, self.u_num_displ_0, self.u_num_displ_1 = self.stepper_stream(self.f_0, self.f_1, self.bc_mask, self.missing_mask, self.omega, timestep, self.U_num_tilde, self.u_num_displ_0, self.u_num_displ_1)
            # Setup new step.
            # self.f_0, self.f_1 = self.f_1, self.f_0
            # self.U_num_tilde_0, self.U_num_tilde_1 = self.U_num_tilde_1, self.U_num_tilde_0
            # self.u_num_displ_0, self.u_num_displ_1 = self.u_num_displ_1, self.u_num_displ_0
            # # f0 is just a copy of the old state here
            # # TODO: get rid of in place updates of u_num_displ and U_num_tilde...
            # self.f_0, self.f_1, self.U_num_tilde_0, self.U_num_tilde_1, self.u_num_displ_1, self.u_num_displ_1 = self.stepper(self.f_0, self.f_1, self.bc_mask, self.missing_mask, self.omega, i, self.U_num_tilde_0, self.U_num_tilde_1, self.u_num_displ_0, self.u_num_displ_1)
            # # f0 is assigned the new state f1.
            # # Now assign the old state to the variable which holds the new state after computation.
            # # f1 is not used in postprocessing, only f0. This allows for maintaining correct time evolution and
            # # memory efficiency. This is needed because in place updates are slow on gpu and jax needs immutability
            # # and efficient computation for warp without any synchronization issues. [chatgpt...]
            
            if timestep % post_process_interval == 0 or timestep == num_steps - 1:
                self.post_process(timestep)
    
    def u_num_exact_x(self, x, y, t):
        return np.sin(4.*np.pi*(x-0.3*t)) * np.cos(2.*np.pi*(y-0.8*t)) * np.sin(4.*np.pi*(t-0.1))
    
    def u_num_exact_y(self, x, y, t):
        return np.cos(4.*np.pi*(x-0.7*t)) * np.sin(2.*np.pi*(y-0.1*t)) * np.cos(4.*np.pi*(t+0.4))
    
    def U_vx(self, x, y, t):
        return 1.6*np.pi*np.sin(np.pi*(-1.6*t + 2.0*y))*np.sin(np.pi*(-1.2*t + 4.0*x))*np.sin(np.pi*(4.0*t - 0.4)) + 4.0*np.pi*np.sin(np.pi*(-1.2*t + 4.0*x))*np.cos(np.pi*(-1.6*t + 2.0*y))*np.cos(np.pi*(4.0*t - 0.4)) - 1.2*np.pi*np.sin(np.pi*(4.0*t - 0.4))*np.cos(np.pi*(-1.6*t + 2.0*y))*np.cos(np.pi*(-1.2*t + 4.0*x))
    
    def U_vy(self, x, y, t):
        return 2.8*np.pi*np.sin(np.pi*(-2.8*t + 4.0*x))*np.sin(np.pi*(-0.2*t + 2.0*y))*np.cos(np.pi*(4.0*t + 1.6)) - 4.0*np.pi*np.sin(np.pi*(-0.2*t + 2.0*y))*np.sin(np.pi*(4.0*t + 1.6))*np.cos(np.pi*(-2.8*t + 4.0*x)) - 0.2*np.pi*np.cos(np.pi*(-2.8*t + 4.0*x))*np.cos(np.pi*(-0.2*t + 2.0*y))*np.cos(np.pi*(4.0*t + 1.6))
    
    def U_js(self, x, y, t):
        return -wp.c_k_led*(4.0*np.pi*np.sin(np.pi*(4.0*t - 0.4))*np.cos(np.pi*(-1.6*t + 2.0*y))*np.cos(np.pi*(-1.2*t + 4.0*x))+2.0*np.pi*np.cos(np.pi*(-2.8*t + 4.0*x))*np.cos(np.pi*(-0.2*t + 2.0*y))*np.cos(np.pi*(4.0*t + 1.6)))
    
    def U_jd(self, x, y, t):
        return -wp.c_mu_led*(4.0*np.pi*np.sin(np.pi*(4.0*t - 0.4))*np.cos(np.pi*(-1.6*t + 2.0*y))*np.cos(np.pi*(-1.2*t + 4.0*x))-2.0*np.pi*np.cos(np.pi*(-2.8*t + 4.0*x))*np.cos(np.pi*(-0.2*t + 2.0*y))*np.cos(np.pi*(4.0*t + 1.6)))
    
    def U_jxy(self,x, y, t):
        return -wp.c_mu_led*(-2.0*np.pi*np.sin(np.pi*(-1.6*t + 2.0*y))*np.sin(np.pi*(-1.2*t + 4.0*x))*np.sin(np.pi*(4.0*t - 0.4))-4.0*np.pi*np.sin(np.pi*(-2.8*t + 4.0*x))*np.sin(np.pi*(-0.2*t + 2.0*y))*np.cos(np.pi*(4.0*t + 1.6)))

    def post_process(self, i):
        # Write the results. We'll use JAX compute_backend for the post-processing
        if not isinstance(self.f_0, jnp.ndarray):
            # If the compute_backend is warp, we need to drop the last dimension added by warp for 2D simulations
            f_0 = wp.to_jax(self.f_0)[..., 0]
            U_num_tilde = wp.to_jax(self.U_num_tilde)[..., 0]
            u_num_displ = wp.to_jax(self.u_num_displ_0)[..., 0]
        else:
            f_0 = self.f_0
            U_num_tilde = self.U_num_tilde
            u_num_displ = self.u_num_displ

        
        # remove boundary cells
        # u_num_displ = u_num_displ[:, 1:-1, 1:-1]
        # U_num_tilde = U_num_tilde[:, 1:-1, 1:-1]

        print("Contains NaNs:", np.isnan(U_num_tilde).any())
        print("Contains Inf:", np.isinf(U_num_tilde).any())

        fields = {"u_x": u_num_displ[0],
                  "u_y": u_num_displ[1],
                  "abs_u": np.sqrt(np.square(u_num_displ[0]) + np.square(u_num_displ[1])),
                  "sigma_xx": -(wp.c_k_led * U_num_tilde[2] + wp.c_mu_led * U_num_tilde[3]),
                  "sigma_yy": -(wp.c_k_led * U_num_tilde[2] - wp.c_mu_led * U_num_tilde[3]),
                  "sigma_xy": -(wp.c_mu_led * U_num_tilde[4])}
        
        # save_fields_vtk(fields, timestep=i, prefix="2d_sine_wave")
        # save_image(fields["sigma_xx"], timestep=i, prefix="2d_sine_wave")

        # Compare solutions on cuts through 2d plane
        t = np.float32((i) * wp.delta_t_led)  # TODO investigate this
        grid_size = self.grid_shape[0]
        plot_index = int(0.379 * grid_size)
        plot_index_num = plot_index
        domain_size = 1
        delta_x = domain_size/grid_size
        x_cut = delta_x * (plot_index + 0.5)
        y_cut = delta_x * (plot_index + 0.5)
        
        x_axis = np.linspace(0, domain_size, num=grid_size)
        y_axis = x_axis
        # Plot cut of u_num_x, u_num_y for constant y, x_axis
        plt.plot(x_axis, self.u_num_exact_x(x=x_axis, y=y_cut, t=t), label="y = const, u_ex", color="green")
        plt.plot(y_axis, self.u_num_exact_y(x=x_axis, y=y_cut, t=t), label="x = const, u_ex", color="orange")
        plt.plot(x_axis, u_num_displ[0, :, plot_index_num], label="y = const, u_num_x", linestyle="--", color="green")
        plt.plot(y_axis, u_num_displ[1, :, plot_index_num], label="y = const, u_num_y", linestyle="--", color="orange")
        plt.legend()
        plt.draw()
        # plt.savefig("/home/merrillg/XLB_LED/examples/led/figures/00_ux_uy_figure", dpi=300)
        plt.savefig("/home/merrill/Documents/ETH/LBM for Linear Elastodynamics/Code/xlb/XLB/examples/led/figures/00_ux_uy_figure", dpi=300)
        plt.pause(0.001)
        plt.clf()

        # Plot cut of vx, vy for constant y, x_axis
        plt.plot(x_axis, self.U_vx(x=x_axis, y=y_cut, t=t), label="y = const, vx_ex", color="green")
        plt.plot(y_axis, self.U_vy(x=x_axis, y=y_cut, t=t), label="y = const, vy_ex", color="orange")
        plt.plot(x_axis, U_num_tilde[0, :, plot_index_num], label="y = const, vx_num", linestyle="--", color="green")
        plt.plot(y_axis, U_num_tilde[1, :, plot_index_num], label="y = const, vy_num", linestyle="--", color="orange")
        plt.title(f"t = {t:.6f}s, interval {i}")
        plt.legend()
        plt.draw()
        # plt.savefig("/home/merrillg/XLB_LED/examples/led/figures/00_vx_vy_figure", dpi=300)
        plt.savefig("/home/merrill/Documents/ETH/LBM for Linear Elastodynamics/Code/xlb/XLB/examples/led/figures/00_vx_vy_figure", dpi=300)
        plt.pause(0.001)
        plt.clf()


        """
        Exact error calculation
        """

        # u_exact_x = np.array([np.array(self.u_num_exact_x(x=x_axis, y=_y, t=t)) for _y in x_axis])
        # u_exact_y = np.array([np.array(self.u_num_exact_y(x=x_axis, y=_y, t=t)) for _y in x_axis])
        # # print(u_exact_x.shape)
        # error_x = u_num_displ[0] - u_exact_x
        # error_sum = 0
        # for i in range(error_x.shape[0]):
        #     for j in range(error_x.shape[0]):
        #         error_sum += error_x[i, j]**2
        # error_sum *= wp.delta_t_led * wp.delta_x_led ** 2 * 40000
        # error_sum = np.sqrt(error_sum)
        # norm_sol = 0
        # for i in range(error_x.shape[0]):
        #     for j in range(error_x.shape[0]):
        #         norm_sol += u_exact_x[i, j]**2
        # norm_sol *= wp.delta_t_led * wp.delta_x_led ** 2 * 40000
        # norm_sol = np.sqrt(norm_sol)
        # print(error_sum/norm_sol)


        # l2_error_step = 


if __name__ == "__main__":
    # # Running the simulation
    grid_size = 80  # Number of grid cells along one dimension
    grid_shape = (grid_size, grid_size)
    num_steps = 200  # Number of collision/streaming steps
    pp_interval = 1  # Post process interval
    domain_size = 1  # Size of domain in meters
    delta_x_led = domain_size/grid_size
    total_time = 1  # Total real world time
    delta_t_led = total_time/num_steps
    c_led = delta_x_led/delta_t_led
    c_k_led = 1.1**0.5
    c_mu_led = 0.4**0.5
    stability_factor = 2.0*np.sqrt(c_k_led**2+c_mu_led**2.0)/c_led

    wp.c_mu_led = wp.constant(c_mu_led)
    wp.c_k_led = wp.constant(c_k_led)
    wp.delta_t_led = wp.constant(delta_t_led)
    wp.delta_x_led = wp.constant(delta_x_led)
    wp.c_led = wp.constant(c_led)

    print(f"Stability factor: {stability_factor}")
    assert stability_factor < 1, "Unstable!"
    


    print(f"delta_x: {delta_x_led:.21f}")
    print(f"delta_t: {delta_t_led:.21f}")
    print(f"c_led: {c_led:.21f}")
    print(f"c_k_led: {c_k_led:.21f}")
    print(f"c_mu_led: {c_mu_led:.21f}")
    

    compute_backend = ComputeBackend.WARP
    precision_policy = PrecisionPolicy.FP32FP32

    # Change velocity set to D2Q4
    velocity_set = xlb.velocity_set.D2Q4(precision_policy=precision_policy, compute_backend=compute_backend)

    omega = 2
    stime = time.time()
    simulation = SineWave2D_LED(omega, grid_shape, velocity_set, compute_backend, precision_policy)
    simulation.run(num_steps=num_steps, post_process_interval=pp_interval)
    print(f"took {time.time() - stime:.2} seconds")

