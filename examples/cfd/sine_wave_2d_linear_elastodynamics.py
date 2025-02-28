import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.operator.stepper import LinearElastodynamicsStepper
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

wp.config.print_launches = False
wp.config.mode = "release"


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
        self.f_0, self.f_1, self.bc_mask, self.missing_mask, self.U_num_tilde, self.u_num_displ = self.stepper.prepare_fields()

    def define_boundary_indices(self):
        box = self.grid.bounding_box_indices()  # For interior nodes
        box_no_edge = self.grid.bounding_box_indices(remove_edges=True)  # For boundary nodes
        # lid = box_no_edge["top"]
        walls = [box["bottom"][i] + box["left"][i] + box["right"][i] + box["top"][i] for i in range(self.velocity_set.d)]
        walls = np.unique(np.array(walls), axis=-1).tolist()
        return walls  # Return as many different indices sets as you need.

    def setup_boundary_conditions(self):
        # # TODO: Adjust BCs here.
        walls = self.define_boundary_indices()
        # bc_walls = DirichletBC_LED(indices=walls)
        # self.boundary_conditions = [bc_walls]
        self.boundary_conditions = []

    def setup_stepper(self):
        self.stepper = LinearElastodynamicsStepper(
            grid=self.grid,
            boundary_conditions=self.boundary_conditions,
            collision_type="BGK_LED",
        )

    def run(self, num_steps, post_process_interval=100):
        # TODO: initialize U_num_here
        initializer = Initializer_LED(velocity_set=self.velocity_set,
                                      precision_policy=self.precision_policy,
                                      compute_backend=self.compute_backend)
        self.f_0, self.U_num_tilde, self.u_num_displ = initializer(self.f_0, self.U_num_tilde, self.u_num_displ)

        for i in range(num_steps):
            # f0 is just a copy of the old state here
            # TODO: get rid of in place updates of u_num_displ and U_num_tilde...
            self.f_0, self.f_1, self.U_num_tilde, self.u_num_displ = self.stepper(self.f_0, self.f_1, self.bc_mask, self.missing_mask, self.omega, i, self.U_num_tilde, self.u_num_displ)
            # f0 is assigned the new state f1.
            # Now assign the old state to the variable which holds the new state after computation.
            # f1 is not used in postprocessing, only f0. This allows for maintaining correct time evolution and
            # memory efficiency. This is needed because in place updates are slow on gpu and jax needs immutability
            # and efficient computation for warp without any synchronization issues. [chatgpt...]
            self.f_0, self.f_1 = self.f_1, self.f_0

            if i % post_process_interval == 0 or i == num_steps - 1:
                self.post_process(i)

    def post_process(self, i):
        # Write the results. We'll use JAX compute_backend for the post-processing
        if not isinstance(self.f_0, jnp.ndarray):
            # If the compute_backend is warp, we need to drop the last dimension added by warp for 2D simulations
            f_0 = wp.to_jax(self.f_0)[..., 0]
            U_num_tilde = wp.to_jax(self.U_num_tilde)[..., 0]
            u_num_displ = wp.to_jax(self.u_num_displ)[..., 0]
        else:
            f_0 = self.f_0
            U_num_tilde = self.U_num_tilde
            u_num_displ = self.u_num_displ

        macro = Macroscopic_LED(
            compute_backend=ComputeBackend.JAX,
            precision_policy=self.precision_policy,
            velocity_set=xlb.velocity_set.D2Q4(precision_policy=self.precision_policy, compute_backend=ComputeBackend.JAX),
        )
        # t = i * wp.delta_t_led
        # U_num_tilde = macro(f_0)
        # remove boundary cells
        u_num_displ = u_num_displ[:, 1:-1, 1:-1]
        U_num_tilde = U_num_tilde[:, 1:-1, 1:-1]
        # print(U_num_tilde)
        # U_num_tilde = np.array(U_num_tilde)
        # U_num_tilde = U_num_tilde.astype(np.float64)
        print("Contains NaNs:", np.isnan(U_num_tilde).any())
        print("Contains Inf:", np.isinf(U_num_tilde).any())

        # print(np.mean(U_num_tilde[0, -1]))
        # print(np.max(np.abs(U_num_tilde)))
        # print(U_num_tilde.shape)
        # print(np.sqrt(np.square(u_num_displ[0]) + np.square(u_num_displ[1])).shape)

        fields = {"u_x": u_num_displ[0],
                  "u_y": u_num_displ[1],
                  "abs_u": np.sqrt(np.square(u_num_displ[0]) + np.square(u_num_displ[1])),
                  "sigma_xx": -(wp.c_k_led * U_num_tilde[2] + wp.c_mu_led * U_num_tilde[3]),
                  "sigma_yy": -(wp.c_k_led * U_num_tilde[2] - wp.c_mu_led * U_num_tilde[3]),
                  "sigma_xy": -(wp.c_mu_led * U_num_tilde[4])}
        
        # print(fields)
        save_fields_vtk(fields, timestep=i, prefix="2d_sine_wave")
        save_image(fields["sigma_xx"], timestep=i, prefix="2d_sine_wave")


if __name__ == "__main__":
    # # Running the simulation
    grid_size = 500  # Number of grid cells along one dimension
    grid_shape = (grid_size, grid_size)
    num_steps = 50000  # Number of collision/streaming steps
    pp_interval = 100  # Post process interval
    domain_size = 1  # Size of domain in meters
    delta_x_led = domain_size/grid_size
    total_time = 1  # Total real world time
    delta_t_led = total_time/num_steps
    c_led = delta_x_led/delta_t_led
    c_k_led = 1.1**0.5
    c_mu_led = 0.4**0.5
    stability_factor = 2.0*np.sqrt(c_k_led**2+c_mu_led**2.0)/c_led

    print(f"Stability factor: {stability_factor}")
    assert stability_factor, "Unstable!"
    


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

