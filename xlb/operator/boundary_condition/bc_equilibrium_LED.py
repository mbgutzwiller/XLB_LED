"""
Base class for boundary conditions in a LBM simulation.
"""

import jax.numpy as jnp
from jax import jit
import jax.lax as lax
from functools import partial
import warp as wp
from typing import Tuple, Any

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium.equilibrium import Equilibrium
from xlb.operator.equilibrium import Equilibrium_LED
from xlb.operator.operator import Operator
from xlb.operator.boundary_condition.boundary_condition_LED import (
    ImplementationStep_LED,
    BoundaryCondition_LED,
)


class EquilibriumBC_LED(BoundaryCondition_LED):
    """
    Full Bounce-back boundary condition for a lattice Boltzmann method simulation.
    """

    def __init__(
        self,
        U_num_tilde = Tuple[float, float, float, float, float],
        equilibrium_operator: Operator = None,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
        indices=None,
        mesh_vertices=None,
    ):
        # Store the equilibrium information
        self.U_num_tilde = U_num_tilde
        self.equilibrium_operator = Equilibrium_LED() if equilibrium_operator is None else equilibrium_operator
        # Raise error if equilibrium operator is not a subclass of Equilibrium
        if not issubclass(type(self.equilibrium_operator), Equilibrium):
            raise ValueError("Equilibrium operator must be a subclass of Equilibrium")

        # Call the parent constructor
        super().__init__(
            ImplementationStep_LED.STREAMING,
            velocity_set,
            precision_policy,
            compute_backend,
            indices,
            mesh_vertices,
        )

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, f_pre, f_post, bc_mask, missing_mask):
        feq = self.equilibrium_operator(jnp.array(self.U_num_tilde))
        new_shape = feq.shape + (1,) * self.velocity_set.d
        feq = lax.broadcast_in_dim(feq, new_shape, [0])
        boundary = bc_mask == self.id

        return jnp.where(boundary, feq, f_post)

    def _construct_warp(self):
        # Set local constants TODO: This is a hack and should be fixed with warp update
        _u_vec = wp.vec(5, dtype=self.compute_dtype)
        _u = _u_vec(self.U_num_tilde[0], self.U_num_tilde[1], self.U_num_tilde[2], self.U_num_tilde[3], self.U_num_tilde[4])

        # Construct the functional for this BC
        @wp.func
        def functional(
            index: Any,
            timestep: Any,
            missing_mask: Any,
            f_0: Any,
            f_1: Any,
            f_pre: Any,
            f_post: Any,
        ):
            _f = self.equilibrium_operator.warp_functional(_u)
            return _f

        # Use the parent class's kernel and pass the functional
        kernel = self._construct_kernel(functional)

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_pre, f_post, bc_mask, missing_mask):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[f_pre, f_post, bc_mask, missing_mask],
            dim=f_pre.shape[1:],
        )
        return f_post
