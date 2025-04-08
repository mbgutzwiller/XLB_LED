import time
stime_glob = time.time()
ftime_glob = time.time()

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
# import time
from xlb.helper.initializers_v2 import Initializer_LED
import matplotlib.pyplot as plt
from xlb.operator.stream import Stream_LED
from tqdm import tqdm
import os


class Pulse2D_LED:
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
        self.f_0, self.U_num_tilde, self.u_num_displ_0 = initializer(self.f_0, self.U_num_tilde, self.u_num_displ_0)

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
        ftime_glob = time.time()
        print(ftime_glob - stime_glob)
        for timestep in tqdm(range(num_steps)):
            # Collision
            self.f_1, self.f_0, self.U_num_tilde, self.u_num_displ_1 = self.stepper_collide(self.f_0, self.f_1, self.bc_mask, self.omega, timestep, self.U_num_tilde, self.u_num_displ_0, self.u_num_displ_0)

            # Postprocessing, happens only if post_process_interval is smaller than num_steps.
            #  -> set pp interval > numsteps for performance analysis.
            if timestep == 2:  # make sure all compilation has finished.
                stime = time.time()
            if (timestep % post_process_interval == 0 or timestep == num_steps - 1) and (num_steps > post_process_interval):
                wp.synchronize()
                wp.synchronize_device()
                self.post_process(timestep, show_plot, figures_dir)
            
            # Streaming
            self.f_1, self.f_0, self.u_num_displ_0, self.u_num_displ_0 = self.stepper_stream(self.f_0, self.f_1, self.bc_mask, self.missing_mask, self.omega, timestep, self.U_num_tilde, self.u_num_displ_1, self.u_num_displ_0)

        return time.time() - stime


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
      

if __name__ == "__main__":
    # # Running the simulation
    grid_size = 1000  # Number of grid cells along one dimension
    grid_shape = (grid_size, grid_size)
    num_steps = int(2.5 * grid_size)  # Number of collision/streaming steps
    pp_interval = num_steps + 1  # Post process interval
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
    simulation = Pulse2D_LED(grid_shape, velocity_set, compute_backend, precision_policy)
    runtime = simulation.run(num_steps=num_steps, post_process_interval=pp_interval, show_plot=False)
    print(f"took {runtime:.4} seconds")

