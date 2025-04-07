import pytest
import numpy as np
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium import QuadraticEquilibrium, Equilibrium_LED
from xlb.operator.collision import BGK_LED
from xlb.grid import grid_factory
from xlb import DefaultConfig
import xlb.velocity_set
import warp as wp

# # Running the simulation
grid_size = 400  # Number of grid cells along one dimension
grid_shape = (grid_size, grid_size)
num_steps = int(2.5 * grid_size)  # Number of collision/streaming steps
pp_interval = int(num_steps+1)  # Post process interval
domain_size = 1  # Size of domain in meters
delta_x_led = domain_size/grid_size
total_time = 1  # Total real world time
delta_t_led = total_time/num_steps
c_led = delta_x_led/delta_t_led
c_k_led = 0.8  # Like in Fig.4 on page 17 from the paper.
c_mu_led = 0.7
stability_factor = 2.0*np.sqrt(c_k_led**2+c_mu_led**2.0)/c_led


# Setting problem parameters as constants to warp for ease of use and performance
wp.grid_size = wp.constant(grid_size)
wp.c_mu_led = wp.constant(c_mu_led)
wp.c_k_led = wp.constant(c_k_led)
wp.delta_t_led = wp.constant(delta_t_led)
wp.delta_x_led = wp.constant(delta_x_led)
wp.c_led = wp.constant(c_led)


def init_xlb_env(velocity_set):
    vel_set = velocity_set(precision_policy=xlb.PrecisionPolicy.FP32FP32, compute_backend=ComputeBackend.WARP)
    xlb.init(
        default_precision_policy=xlb.PrecisionPolicy.FP32FP32,
        default_backend=ComputeBackend.WARP,
        velocity_set=vel_set,
    )


@pytest.mark.parametrize(
    "dim,velocity_set,grid_shape,omega",
    [   
        (2, xlb.velocity_set.D2Q4, (100, 100), 2),
        # (2, xlb.velocity_set.D2Q9, (100, 100), 0.6),
        # (2, xlb.velocity_set.D2Q9, (100, 100), 1.0),
        # (3, xlb.velocity_set.D3Q19, (50, 50, 50), 0.6),
        # (3, xlb.velocity_set.D3Q19, (50, 50, 50), 1.0),
        # (3, xlb.velocity_set.D3Q27, (50, 50, 50), 0.6),
        # (3, xlb.velocity_set.D3Q27, (50, 50, 50), 1.0),
    ],
)
def test_bgk_collision_warp(dim, velocity_set, grid_shape, omega):
    init_xlb_env(velocity_set)  # Done
    my_grid = grid_factory(grid_shape)  # Done

    # rho = my_grid.create_field(cardinality=1, fill_value=1.0)
    U_num_tilde = my_grid.create_field(cardinality=5, fill_value=0)  # Done

    # compute_macro = QuadraticEquilibrium()
    compute_macro = Equilibrium_LED()  # Done

    f_eq = my_grid.create_field(cardinality=20)  # Done
    # f_eq = my_grid.create_field(cardinality=DefaultConfig.velocity_set.q)
    f_eq = compute_macro(U_num_tilde, f_eq)

    compute_collision = BGK_LED()
    f_orig = my_grid.create_field(cardinality=20)
    f_orig = compute_macro(U_num_tilde, f_orig)

    f_out = my_grid.create_field(cardinality=20)
    f_out = compute_collision(f_orig, f_eq, f_out, omega)

    f_eq = f_eq.numpy()
    f_out = f_out.numpy()
    f_orig = f_orig.numpy()
    print(f_eq)
    print(f_out)
    print(f_orig)

    # f - self.compute_dtype(omega) * (f - feq)
    # assert np.allclose(f_out, f_orig - omega * (f_orig - f_eq), atol=1e-5)

    manual_fout = omega * f_eq + (1 - omega) * f_orig
    assert np.allclose(f_out, manual_fout, atol=1e-5), f"{f_out}/n{manual_fout}"

if __name__ == "__main__":
    pytest.main()
