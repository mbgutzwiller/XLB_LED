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
from xlb.helper.initializers_v2 import Initializer_LED
import matplotlib.pyplot as plt
from xlb.operator.stream import Stream_LED
from tqdm import tqdm
import jax
import jax.numpy as jnp

wp.build.clear_kernel_cache()
plt.ion()

# wp.config.print_launches = False
# wp.config.mode = "release"


class SineWave2D_LED_jaxpp:
    def __init__(self, grid_shape, velocity_set, compute_backend, precision_policy):
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
        self.omega = 2.
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
        self.f_0, self.f_1, self.f_1, self.bc_mask, self.missing_mask, self.U_num_tilde, self.u_num_displ_0, self.u_num_displ_0 = self.stepper_collide.prepare_fields()

    def define_boundary_indices(self):
        box = self.grid.bounding_box_indices()  # For interior nodes
        box_no_edge = self.grid.bounding_box_indices(remove_edges=True)  # For boundary nodes
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

    def run(self, num_steps, post_process_interval=100, show_plot=False):
        # TODO: initialize U_num_here
        self.stream_LED = Stream_LED(self.velocity_set, self.precision_policy, self.compute_backend)
        initializer = Initializer_LED(velocity_set=self.velocity_set,
                                      precision_policy=self.precision_policy,
                                      compute_backend=self.compute_backend)
        # plt.figure()
        self.f_0, self.U_num_tilde, self.u_num_displ_0 = initializer(self.f_0, self.U_num_tilde, self.u_num_displ_0)
        wp.synchronize()
        wp.synchronize_device()
        self.cumulative_error_u = 0
        self.cumulative_u = 0
        self.cumulative_error_sigma = 0
        self.cumulative_sigma = 0

        self.max_error_u = 0
        self.max_error_sigma = 0

        for timestep in tqdm(range(num_steps)):
            # Collision
            self.f_1, self.f_0, self.U_num_tilde, self.u_num_displ_1 = self.stepper_collide(self.f_0, self.f_1, self.bc_mask, self.omega, timestep, self.U_num_tilde, self.u_num_displ_0, self.u_num_displ_0)

            # Postprocessing: Show plot or calculate error
            if timestep % post_process_interval == 0 or timestep == num_steps - 1:
                self.post_process(timestep, show_plot)
            
            # Streaming
            self.f_1, self.f_0, self.u_num_displ_0, self.u_num_displ_0 = self.stepper_stream(self.f_0, self.f_1, self.bc_mask, self.missing_mask, self.omega, timestep, self.U_num_tilde, self.u_num_displ_1, self.u_num_displ_0)

        # Calculation for L2 norm
        final_error_norm_u = np.sqrt(self.cumulative_error_u * np.float32(wp.delta_x_led)**2 * np.float32(wp.delta_t_led))
        final_norm_u = np.sqrt(self.cumulative_u * np.float32(wp.delta_x_led)**2 * np.float32(wp.delta_t_led))
        final_error_norm_sigma = np.sqrt(self.cumulative_error_sigma * np.float32(wp.delta_x_led)**2 * np.float32(wp.delta_t_led))
        final_norm_sigma = np.sqrt(self.cumulative_sigma * np.float32(wp.delta_x_led)**2 * np.float32(wp.delta_t_led))
        return final_error_norm_u/final_norm_u, final_error_norm_sigma/final_norm_sigma, self.max_error_u/final_norm_u, self.max_error_sigma/final_norm_sigma


    def u_num_exact_x(self, x, y, t):
        return np.sin(4.*np.pi*x) * np.sin(2.*np.pi*y) * np.sin(4.*np.pi*(t-0.1))
        # return np.sin(4.*np.pi*(x-0.3*t)) * np.cos(2.*np.pi*(y-0.8*t)) * np.sin(4.*np.pi*(t-0.1))
    
    def u_num_exact_x_jax(x, y, t):
        return jnp.sin(4.*jnp.pi*x) * jnp.sin(2.*jnp.pi*y) * jnp.sin(4.*jnp.pi*(t-0.1))

    def u_num_exact_y(self, x, y, t):
        return np.sin(4.*np.pi*x) * np.sin(2.*np.pi*y) * np.sin(4.*np.pi*(t+0.3))
        # return np.cos(4.*np.pi*(x-0.7*t)) * np.sin(2.*np.pi*(y-0.1*t)) * np.cos(4.*np.pi*(t+0.4))
    
    def u_num_exact_y_jax(x, y, t):
        return jnp.sin(4.*jnp.pi*x) * jnp.sin(2.*jnp.pi*y) * jnp.sin(4.*jnp.pi*(t+0.3))
    
    def U_vx(self, x, y, t):
        return 4.*np.pi*np.sin(4.*np.pi*x)*np.sin(2.*np.pi*y)*np.cos(4.*np.pi*(t - 1./10.))
        # return 1.6*np.pi*np.sin(np.pi*(-1.6*t + 2.0*y))*np.sin(np.pi*(-1.2*t + 4.0*x))*np.sin(np.pi*(4.0*t - 0.4)) + 4.0*np.pi*np.sin(np.pi*(-1.2*t + 4.0*x))*np.cos(np.pi*(-1.6*t + 2.0*y))*np.cos(np.pi*(4.0*t - 0.4)) - 1.2*np.pi*np.sin(np.pi*(4.0*t - 0.4))*np.cos(np.pi*(-1.6*t + 2.0*y))*np.cos(np.pi*(-1.2*t + 4.0*x))

    def U_vx_jax(x, y, t):
            return 4.*jnp.pi*jnp.sin(4.*jnp.pi*x)*jnp.sin(2.*jnp.pi*y)*jnp.cos(4.*jnp.pi*(t - 1./10.))

    def U_vy(self, x, y, t):
        return 4.*np.pi*np.sin(4.*np.pi*x)*np.sin(2.*np.pi*y)*np.cos(4.*np.pi*(t + 3./10.))
        # return 2.8*np.pi*np.sin(np.pi*(-2.8*t + 4.0*x))*np.sin(np.pi*(-0.2*t + 2.0*y))*np.cos(np.pi*(4.0*t + 1.6)) - 4.0*np.pi*np.sin(np.pi*(-0.2*t + 2.0*y))*np.sin(np.pi*(4.0*t + 1.6))*np.cos(np.pi*(-2.8*t + 4.0*x)) - 0.2*np.pi*np.cos(np.pi*(-2.8*t + 4.0*x))*np.cos(np.pi*(-0.2*t + 2.0*y))*np.cos(np.pi*(4.0*t + 1.6))

    def U_vy_jax(x, y, t):
        return 4.*jnp.pi*jnp.sin(4.*jnp.pi*x)*jnp.sin(2.*jnp.pi*y)*jnp.cos(4.*jnp.pi*(t + 3./10.))

    def U_js(self, x, y, t):
        return -wp.c_k_led**(1./2.)*(4.*np.pi*np.cos(4.*np.pi*x)*np.sin(2.*np.pi*y)*np.sin(4.*np.pi*(t - 1./10.)) + 2.*np.pi*np.cos(2.*np.pi*y)*np.sin(4.*np.pi*x)*np.sin(4.*np.pi*(t + 3./10.)))
        # return -np.c_k_led*(4.0*np.pi*np.sin(np.pi*(4.0*t - 0.4))*np.cos(np.pi*(-1.6*t + 2.0*y))*np.cos(np.pi*(-1.2*t + 4.0*x))+2.0*np.pi*np.cos(np.pi*(-2.8*t + 4.0*x))*np.cos(np.pi*(-0.2*t + 2.0*y))*np.cos(np.pi*(4.0*t + 1.6)))
    
    def U_js_jax(x, y, t):
        return -wp.c_k_led**(1./2.)*(4.*jnp.pi*jnp.cos(4.*jnp.pi*x)*jnp.sin(2.*jnp.pi*y)*jnp.sin(4.*jnp.pi*(t - 1./10.)) + 2.*jnp.pi*jnp.cos(2.*jnp.pi*y)*jnp.sin(4.*jnp.pi*x)*jnp.sin(4.*jnp.pi*(t + 3./10.)))

    def U_jd(self, x, y, t):
        return -wp.c_mu_led**(1./2.)*(4.*np.pi*np.cos(4.*np.pi*x)*np.sin(2.*np.pi*y)*np.sin(4.*np.pi*(t - 1./10.)) - 2.*np.pi*np.cos(2.*np.pi*y)*np.sin(4.*np.pi*x)*np.sin(4.*np.pi*(t + 3./10.)))
        # return -np.c_mu_led*(4.0*np.pi*np.sin(np.pi*(4.0*t - 0.4))*np.cos(np.pi*(-1.6*t + 2.0*y))*np.cos(np.pi*(-1.2*t + 4.0*x))-2.0*np.pi*np.cos(np.pi*(-2.8*t + 4.0*x))*np.cos(np.pi*(-0.2*t + 2.0*y))*np.cos(np.pi*(4.0*t + 1.6)))

    def U_jd_jax(x, y, t):
        return -wp.c_mu_led**(1./2.)*(4.*jnp.pi*jnp.cos(4.*jnp.pi*x)*jnp.sin(2.*jnp.pi*y)*jnp.sin(4.*jnp.pi*(t - 1./10.)) - 2.*jnp.pi*jnp.cos(2.*jnp.pi*y)*jnp.sin(4.*jnp.pi*x)*jnp.sin(4.*jnp.pi*(t + 3./10.)))

    def U_jxy(self,x, y, t):
        return -wp.c_mu_led**(1./2.)*(2.*np.pi*np.cos(2.*np.pi*y)*np.sin(4.*np.pi*x)*np.sin(4.*np.pi*(t - 1./10.)) + 4.*np.pi*np.cos(4.*np.pi*x)*np.sin(2.*np.pi*y)*np.sin(4.*np.pi*(t + 3./10.)))
        # return -wp.c_mu_led*(-2.0*np.pi*np.sin(np.pi*(-1.6*t + 2.0*y))*np.sin(np.pi*(-1.2*t + 4.0*x))*np.sin(np.pi*(4.0*t - 0.4))-4.0*np.pi*np.sin(np.pi*(-2.8*t + 4.0*x))*np.sin(np.pi*(-0.2*t + 2.0*y))*np.cos(np.pi*(4.0*t + 1.6)))
    
    def U_jxy_jax(x, y, t):
        return -wp.c_mu_led**(1./2.)*(2.*jnp.pi*jnp.cos(2.*jnp.pi*y)*jnp.sin(4.*jnp.pi*x)*jnp.sin(4.*jnp.pi*(t - 1./10.)) + 4.*jnp.pi*jnp.cos(4.*jnp.pi*x)*jnp.sin(2.*jnp.pi*y)*jnp.sin(4.*jnp.pi*(t + 3./10.)))
    
    @jax.jit(static_argnums=(3, 5, 6, 7, 8, 9))
    def post_process_jax(
        U_num_tilde,         # jnp.array; assumed shape (n_vars, grid, grid)
        u_num_displ,         # jnp.array; assumed shape (2, grid, grid)
        grid_size,          # e.g. (grid, grid)
        t,                   # current time step (integer)
        u_num_exact_x,       # function: (x, y, t) -> float (JAX compatible)
        u_num_exact_y,       # function: (x, y, t) -> float (JAX compatible)
        U_js, U_jd, U_jxy,   # functions for stress components (JAX compatible)
        c_k_led, c_mu_led,   # constants for stress calculations
        cumulative_error_u, cumulative_u, max_error_u,
        cumulative_error_sigma, cumulative_sigma, max_error_sigma
    ):
                
        # Define the grid
        domain_size = 1.0
        delta_x = domain_size / grid_size
        # Create a 1D axis from the center of the first cell to the last cell
        x_axis = jnp.linspace(delta_x/2, domain_size - delta_x/2, grid_size)
        
        # Compute exact solutions using vectorized mapping
        u_ex_x = jax.vmap(lambda y: jax.vmap(lambda x: u_num_exact_x(x=y, y=x, t=t))(x_axis))(x_axis)
        u_ex_y = jax.vmap(lambda y: jax.vmap(lambda x: u_num_exact_y(x=y, y=x, t=t))(x_axis))(x_axis)
        
        # Numerical displacements (assumed already jnp.arrays)
        u_num_x = u_num_displ[0, :, :]
        u_num_y = u_num_displ[1, :, :]
        
        # Calculate L2 error (squared sum over grid)
        norm_error_u = jnp.sum((u_ex_x - u_num_x) ** 2 + (u_ex_y - u_num_y) ** 2)
        norm_u = jnp.sum(u_ex_x ** 2 + u_ex_y ** 2)
        max_error_u_current = jnp.max(jnp.abs(jnp.stack([u_ex_x - u_num_x, u_ex_y - u_num_y], axis=0)))
        
        # Compute stress components from exact solutions via vectorized mapping
        _U_js_ex = jax.vmap(lambda y: jax.vmap(lambda x: U_js(x=y, y=x, t=t))(x_axis))(x_axis)
        _U_jd_ex = jax.vmap(lambda y: jax.vmap(lambda x: U_jd(x=y, y=x, t=t))(x_axis))(x_axis)
        _U_jxy_ex = jax.vmap(lambda y: jax.vmap(lambda x: U_jxy(x=y, y=x, t=t))(x_axis))(x_axis)
        
        sigma_xx_ex = -(c_k_led * _U_js_ex + c_mu_led * _U_jd_ex)
        sigma_yy_ex = -(c_k_led * _U_js_ex - c_mu_led * _U_jd_ex)
        sigma_xy_ex = -(c_mu_led * _U_jxy_ex)
        
        sigma_xx_num = -(c_k_led * U_num_tilde[2, :, :] + c_mu_led * U_num_tilde[3, :, :])
        sigma_yy_num = -(c_k_led * U_num_tilde[2, :, :] - c_mu_led * U_num_tilde[3, :, :])
        sigma_xy_num = -(c_mu_led * U_num_tilde[4, :, :])
        
        norm_error_sigma = jnp.sum((sigma_xx_ex - sigma_xx_num) ** 2 +
                                (sigma_yy_ex - sigma_yy_num) ** 2 +
                                (sigma_xy_ex - sigma_xy_num) ** 2)
        norm_sigma = jnp.sum(sigma_xx_ex ** 2 + sigma_yy_ex ** 2 + sigma_xy_ex ** 2)
        max_error_sigma_current = jnp.max(jnp.abs(jnp.stack([
            sigma_xx_ex - sigma_xx_num,
            sigma_yy_ex - sigma_yy_num,
            sigma_xy_ex - sigma_xy_num
        ], axis=0)))
        
        # Update cumulative values and maximum errors
        cumulative_error_u += norm_error_u
        cumulative_u += norm_u
        max_error_u = jnp.maximum(max_error_u, max_error_u_current)
        
        cumulative_error_sigma += norm_error_sigma
        cumulative_sigma += norm_sigma
        max_error_sigma = jnp.maximum(max_error_sigma, max_error_sigma_current)
        
        # Return updated error metrics (and possibly other data for plotting)
        return (cumulative_error_u, cumulative_u, max_error_u,
                cumulative_error_sigma, cumulative_sigma, max_error_sigma)

    def post_process(self, i, show_plot=False):
        # Write the results. We'll use JAX compute_backend for the post-processing
        if not isinstance(self.f_0, jnp.ndarray):
            # If the compute_backend is warp, we need to drop the last dimension added by warp for 2D simulations
            wp.synchronize_device()
            wp.synchronize()
            # U_num_tilde = wp.to_jax(self.stepper_collide.macroscopic_LED(self.f_0, self.U_num_tilde, wp.float32(0)))[..., 0]
            U_num_tilde = wp.to_jax(self.U_num_tilde)[..., 0]
            u_num_displ = wp.to_jax(self.u_num_displ_1)[..., 0]

        # fields = {"u_x": u_num_displ[0],
        #           "u_y": u_num_displ[1],
        #           "abs_u": np.sqrt(np.square(u_num_displ[0]) + np.square(u_num_displ[1])),
        #           "sigma_xx": -(wp.c_k_led * U_num_tilde[2] + wp.c_mu_led * U_num_tilde[3]),
        #           "sigma_yy": -(wp.c_k_led * U_num_tilde[2] - wp.c_mu_led * U_num_tilde[3]),
        #           "sigma_xy": -(wp.c_mu_led * U_num_tilde[4])}
        
        # save_fields_vtk(fields, timestep=i, prefix="2d_sine_wave")
        # save_image(fields["sigma_xx"], timestep=i, prefix="2d_sine_wave")

        # Compare solutions on cuts through 2d plane
        t = np.float32(i * wp.delta_t_led)  #
        grid_size = self.grid_shape[0]
        plot_index = int(0.57 * (grid_size-1))
        assert (plot_index >=0) and (plot_index <= grid_size-1), "Plotting index invalid"
        plot_index_num = plot_index
        domain_size = 1
        delta_x = domain_size/grid_size
        x_cut = delta_x * (plot_index + 0.5)
        y_cut = delta_x * (plot_index + 0.5)
        
        x_axis_num = np.linspace(delta_x/2, domain_size-delta_x/2, num=grid_size)

        cumulative_error_u = jnp.array(self.cumulative_error_u, dtype=jnp.float32)
        cumulative_u         = jnp.array(self.cumulative_u, dtype=jnp.float32)
        cumulative_error_sigma = jnp.array(self.cumulative_error_sigma, dtype=jnp.float32)
        cumulative_sigma     = jnp.array(self.cumulative_sigma, dtype=jnp.float32)
        max_error_u          = jnp.array(self.max_error_u, dtype=jnp.float32)
        max_error_sigma      = jnp.array(self.max_error_sigma, dtype=jnp.float32)

        (self.cumulative_error_u,
         self.cumulative_u,
         self.max_error_u,
         self.cumulative_error_sigma,
         self.cumulative_sigma,
         self.max_error_sigma) = self.post_process_jax(U_num_tilde,
                                                       u_num_displ,
                                                       grid_size,
                                                       t,
                                                       self.u_num_exact_x_jax,
                                                       self.u_num_exact_y_jax,
                                                       self.U_js_jax,
                                                       self.U_jd_jax,
                                                       self.U_jxy_jax,
                                                       wp.c_k_led,
                                                       wp.c_mu_led,
                                                       cumulative_error_u,
                                                       cumulative_u,
                                                       max_error_u,
                                                       cumulative_error_sigma,
                                                       cumulative_sigma,
                                                       max_error_sigma)
        """
        Approximate error calculation
        """
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
        # # norm_error_sigma = np.linalg.norm(np.stack([(sigma_xx_ex - sigma_xx_num), (sigma_yy_ex - sigma_yy_num), (sigma_xy_ex - sigma_xy_num)], axis=0))
        # norm_error_sigma = np.sum((sigma_xx_ex - sigma_xx_num) ** 2 + (sigma_yy_ex - sigma_yy_num) ** 2 + (sigma_xy_ex - sigma_xy_num) ** 2)
        # norm_sigma = np.sum((sigma_xx_ex) ** 2 + (sigma_yy_ex) ** 2 + (sigma_xy_ex) ** 2)
        # # norm_sigma = np.linalg.norm(np.stack([sigma_xx_ex, sigma_yy_ex, sigma_xy_ex], axis=0))
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
    grid_size = 64  # Number of grid cells along one dimension
    grid_shape = (grid_size, grid_size)
    num_steps = 200  # Number of collision/streaming steps
    pp_interval = 1  # Post process interval
    domain_size = 1  # Size of domain in meters
    delta_x_led = domain_size/grid_size
    total_time = 1  # Total real world time
    delta_t_led = total_time/num_steps
    c_led = delta_x_led/delta_t_led
    c_k_led = 1.1
    c_mu_led = 0.4
    stability_factor = 2.0*np.sqrt(c_k_led**2+c_mu_led**2.0)/c_led

    wp.grid_size = wp.constant(grid_size)
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

    stime = time.time()
    simulation = SineWave2D_LED_jaxpp(grid_shape, velocity_set, compute_backend, precision_policy)
    simulation.run(num_steps=num_steps, post_process_interval=pp_interval, show_plot=True)
    print(f"took {time.time() - stime:.2} seconds")

