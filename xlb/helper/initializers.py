from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.equilibrium import Equilibrium_LED


def initialize_eq(f, grid, velocity_set, precision_policy, compute_backend, rho=None, u=None):
    if rho is None:
        rho = grid.create_field(cardinality=1, fill_value=1.0, dtype=precision_policy.compute_precision)
    if u is None:
        u = grid.create_field(cardinality=velocity_set.d, fill_value=0.0, dtype=precision_policy.compute_precision)
    equilibrium = QuadraticEquilibrium()

    if compute_backend == ComputeBackend.JAX:
        f = equilibrium(rho, u)

    elif compute_backend == ComputeBackend.WARP:
        f = equilibrium(rho, u, f)

    del rho, u

    return f

def initialize_eq_LED(f, grid, precision_policy, compute_backend):
    if f is None:
        U_0 = grid.create_field(cardinality=20, fill_value=0.0, dtype=precision_policy.compute_precision)
    equilibrium = Equilibrium_LED()

    if compute_backend == ComputeBackend.JAX:
        f = equilibrium(U_0)

    elif compute_backend == ComputeBackend.WARP:
        U_0 = grid.create_field(cardinality=20, fill_value=0.0, dtype=precision_policy.compute_precision)
        f = equilibrium(U_0, f)
    del U_0
    return f
