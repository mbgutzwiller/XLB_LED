import pytest
import numpy as np
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium import Equilibrium_LED
from xlb.operator.macroscopic import Macroscopic_LED
from xlb.grid import grid_factory
from xlb import DefaultConfig


def init_xlb_env(velocity_set):
    vel_set = velocity_set(precision_policy=xlb.PrecisionPolicy.FP32FP32, compute_backend=ComputeBackend.WARP)
    xlb.init(
        default_precision_policy=xlb.PrecisionPolicy.FP32FP32,
        default_backend=ComputeBackend.WARP,
        velocity_set=vel_set,
    )


@pytest.mark.parametrize(
    "dim,velocity_set,grid_shape,U_num_tilde",
    [
        (2, xlb.velocity_set.D2Q4, (100, 100), 0),
        # (2, xlb.velocity_set.D2Q9, (100, 100), 1.0, 0.0),
        # (2, xlb.velocity_set.D2Q9, (100, 100), 1.1, 1.0),
        # (2, xlb.velocity_set.D2Q9, (100, 100), 1.1, 2.0),
        # (2, xlb.velocity_set.D2Q9, (50, 50), 1.1, 2.0),
        # # (3, xlb.velocity_set.D3Q19, (50, 50, 50), 1.0, 0.0),
        # (3, xlb.velocity_set.D3Q19, (50, 50, 50), 1.1, 1.0),  # TODO: Uncommenting will cause a Warp error. Needs investigation.
        # (3, xlb.velocity_set.D3Q19, (50, 50, 50), 1.1, 2.0),  # TODO: Uncommenting will cause a Warp error. Needs investigation.
    ],
)
def test_macroscopic_warp(dim, velocity_set, grid_shape, U_num_tilde):
    init_xlb_env(velocity_set)  # Done
    my_grid = grid_factory(grid_shape)  # Done
    dim_q = DefaultConfig.velocity_set.q

    U_num_tilde_field = my_grid.create_field(cardinality=5, fill_value=U_num_tilde)  # Done
    # velocity_field = my_grid.create_field(cardinality=dim, fill_value=velocity)

    f_eq = my_grid.create_field(cardinality=dim_q * 5)  # Done
    f_eq = Equilibrium_LED()(U_num_tilde_field, f_eq)

    compute_macro = Macroscopic_LED()
    U_num_tilde_calc = my_grid.create_field(cardinality=5)
    # u_calc = my_grid.create_field(cardinality=dim)

    U_num_tilde_calc = compute_macro(f_eq, U_num_tilde_calc)
    # U_num_tilde_calc = compute_macro(f_eq, U_num_tilde_calc, 0)    
    
    assert np.allclose(U_num_tilde_calc.numpy(), np.array([0, 0, 0, 0, 0])), f"Computed U_num_tilde should be close to initialized U_num_tilde {np.array([0, 0, 0, 0, 0])} but is {U_num_tilde_calc.numpy()}"
    # assert np.allclose(u_calc.numpy(), velocity), f"Computed velocity should be close to initialized velocity {velocity}"


if __name__ == "__main__":
    pytest.main()
