import pytest
import numpy as np
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium import Equilibrium_LED
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
    "dim,velocity_set,grid_shape",
    [
        (2, xlb.velocity_set.D2Q4, (50, 50)),
        # (2, xlb.velocity_set.D2Q9, (50, 50)),
        # (2, xlb.velocity_set.D2Q9, (100, 100)),
        # (3, xlb.velocity_set.D3Q19, (50, 50, 50)),
        # (3, xlb.velocity_set.D3Q19, (100, 100, 100)),
        # (3, xlb.velocity_set.D3Q27, (50, 50, 50)),
        # (3, xlb.velocity_set.D3Q27, (100, 100, 100)),
    ],
)
def test_quadratic_equilibrium_warp(dim, velocity_set, grid_shape):
    init_xlb_env(velocity_set)
    my_grid = grid_factory(grid_shape)

    U_num_tilde = my_grid.create_field(cardinality=5, fill_value=0)

    f_eq = my_grid.create_field(cardinality=20)

    compute_macro = Equilibrium_LED()
    U_num_tilde = compute_macro(U_num_tilde, f_eq)

    U_num_tilde_np = U_num_tilde.numpy()

    assert np.allclose(U_num_tilde_np, np.array([0, 0, 0, 0, 0])), "U_num_tilde should be 0.0 in every entry."
    # TODO: check other statistics


# @pytest.fixture(autouse=True)
# def setup_xlb_env(request):
#     dim, velocity_set, grid_shape = request.param
#     init_xlb_env(velocity_set)

if __name__ == "__main__":
    pytest.main()
