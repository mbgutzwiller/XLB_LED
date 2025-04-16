import pytest
import numpy as np
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.grid import grid_factory
from xlb import DefaultConfig
import xlb.operator
import xlb.operator.boundary_condition
from xlb.operator.boundary_masker import IndicesBoundaryMasker_LED
import xlb.operator.equilibrium
import warp as wp
import os


"""
Test for square domain.
"""


def init_xlb_env(velocity_set):
    vel_set = velocity_set(precision_policy=xlb.PrecisionPolicy.FP32FP32, compute_backend=ComputeBackend.WARP)
    xlb.init(
        default_precision_policy=xlb.PrecisionPolicy.FP32FP32,
        default_backend=ComputeBackend.WARP,
        velocity_set=vel_set,
    )


@pytest.mark.parametrize(
    "dim,velocity_set,grid_shape",
    [
        (2, xlb.velocity_set.D2Q4, (4, 4)),
        # (2, xlb.velocity_set.D2Q9, (100, 100)),
        # (2, xlb.velocity_set.D2Q9, (100, 100)),
        # (3, xlb.velocity_set.D3Q19, (50, 50, 50)),
        # (3, xlb.velocity_set.D3Q19, (50, 50, 50)),
    ],
)
def test_bc_dirichlet_warp(dim, velocity_set, grid_shape):
    init_xlb_env(velocity_set)
    my_grid = grid_factory(grid_shape)
    velocity_set = DefaultConfig.velocity_set
    print(velocity_set.opp_indices)

    missing_mask = my_grid.create_field(cardinality=velocity_set.q, dtype=xlb.Precision.BOOL)

    bc_mask = my_grid.create_field(cardinality=1, dtype=xlb.Precision.UINT8)

    indices_boundary_masker = IndicesBoundaryMasker_LED(velocity_set=velocity_set)

    # Make indices for boundary conditions (sphere)
    nr = grid_shape[0]
    x = np.arange(nr)
    y = np.arange(nr)
    # z = np.arange(nr)
    if dim == 2:
        X, Y = np.meshgrid(x, y)
        indices = np.where((X==0) | (X==nr-1) | (Y==0) | (Y==nr-1))
    else:
        raise NotImplementedError("Only 2D available for now")

    indices = [tuple(indices[i]) for i in range(velocity_set.d)]

    dirichlet_bc_led = xlb.operator.boundary_condition.DirichletBC_LED(
        velocity_set=velocity_set,
        indices=indices,
    )

    bc_mask, missing_mask = indices_boundary_masker([dirichlet_bc_led], bc_mask, missing_mask, start_index=None)
    print(bc_mask)
    print(missing_mask)
    # Generate a random field with the same shape
    if dim == 2:
        random_field = np.random.rand(velocity_set.q * 5, grid_shape[0], grid_shape[1], 1).astype(np.float32)
    else:
        raise NotImplementedError

    # Add the random field to f_pre
    f_pre = wp.array(random_field)
    f_post = wp.array(random_field)
    f = wp.array(random_field)

    # f = my_grid.create_field(cardinality=velocity_set.q * 5, dtype=xlb.Precision.FP32)
    # f_pre = my_grid.create_field(cardinality=velocity_set.q * 5, dtype=xlb.Precision.FP32)
    # f_post = my_grid.create_field(
        # cardinality=velocity_set.q * 5, dtype=xlb.Precision.FP32, fill_value=1.0
    # )  # Arbitrary value so that we can check if the values are changed outside the boundary

    # f = equilibrium_bc(f_pre, f_post, bc_mask, missing_mask)
    wp.c_k_led = wp.constant(0.)  # Set to zero so we can check at least the stress components of the f since the function on the boundary can vary.
    wp.c_mu_led = wp.constant(0.)  # The post stress components are the negative of the opposite direction of the pre streaming components.
    f = dirichlet_bc_led(f_pre, f_post, bc_mask, missing_mask)

    f = f.numpy()
    f_post = f_post.numpy()
    # diff_f = (f-f_post)


    for i in range(4):
        if dim == 2:
            for j in range(2):
                if missing_mask[i] == wp.uint8(1):
                    assert np.allclose(-f[velocity_set.opp_indices[i] * 5 + j, indices[0], indices[1]], f_post[i * 5 + j, indices[0], indices[1]]), f"what flows in in post stream should what flows ot in pre stream."
            for j in range(2, 5):
                if missing_mask[i] == wp.uint8(1):
                    print(f_post[i * 5 + j, indices[0], indices[1]])
                    print(f[velocity_set.opp_indices[i] * 5 + j, indices[0], indices[1]])
                    assert np.allclose(f[velocity_set.opp_indices[i] * 5 + j, indices[0], indices[1]], f_post[i * 5 + j, indices[0], indices[1]]), f"what flows in in post stream should what flows ot in pre stream."
        else:
            raise NotImplementedError


if __name__ == "__main__":
    os.environ["JAX_PLATFORMS"] = "cpu"  # Sometimes 
    wp.c_k_led = wp.constant(1.1)
    wp.c_mu_led = wp.constant(0.4)
    num_steps = 200
    grid_size = 128
    physical_time = 1
    domain_size = 1
    wp.delta_t_led = wp.constant(physical_time/num_steps)
    wp.delta_x_led = wp.constant(domain_size/grid_size)
    wp.c_led = wp.constant(wp.delta_x_led/wp.delta_t_led)
    pytest.main()
