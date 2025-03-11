"""
Base class for boundary conditions in a LBM simulation.
"""

import jax.numpy as jnp
from jax import jit
import jax.lax as lax
from functools import partial
import warp as wp
from typing import Any

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.boundary_condition.boundary_condition_LED import (
    ImplementationStep_LED,
    BoundaryCondition_LED,
)


class DirichletBC_LED(BoundaryCondition_LED):
    """
    Halfway Bounce-back boundary condition for a lattice Boltzmann method simulation.

    TODO: Implement moving boundary conditions for this
    """

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
        indices=None,
        mesh_vertices=None,
    ):
        # Call the parent constructor
        super().__init__(
            ImplementationStep_LED.STREAMING,
            velocity_set,
            precision_policy,
            compute_backend,
            indices,
            mesh_vertices,
        )

        # This BC needs padding for finding missing directions when imposed on a geometry that is in the domain interior
        self.needs_padding = True

    # @Operator.register_backend(ComputeBackend.JAX)  # TODO: jax implementation of dirichlet BC
    # @partial(jit, static_argnums=(0))
    # def jax_implementation(self, f_pre, f_post, bc_mask, missing_mask):
    #     boundary = bc_mask == self.id
    #     new_shape = (self.velocity_set.q,) + boundary.shape[1:]
    #     boundary = lax.broadcast_in_dim(boundary, new_shape, tuple(range(self.velocity_set.d + 1)))
    #     return jnp.where(
    #         jnp.logical_and(missing_mask, boundary),
    #         f_pre[self.velocity_set.opp_indices],
    #         f_post,
    #     )

    def _construct_warp(self):
        # Set local constants
        _opp_indices = self.velocity_set.opp_indices
        _vel_c = wp.vec(2, dtype=self.compute_dtype)


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
            dudt_D_tilde: Any,
        ):
            # Post-streaming values are only modified at missing direction
            _f = f_post
            # this is [[1, 0, -1, 0],
            #          [0, 1, 0, -1]]
            _c = self.velocity_set.c
            # TODO: add S_ij
            for l in range(self.velocity_set.q):
                # If the mask is missing then take the opposite index
                for m in range(2):
                    if missing_mask[l * 5 + m] == wp.uint8(1):
                        # Get the pre-streaming distribution function in oppisite direction
                        # return negative of velocities
                        _f[l * 5 + m] = -f_pre[_opp_indices[l] * 5 + m]
                        # Add contribution of S_ij*u_D_tilde
                        _f[l * 5 + m] += self.compute_dtype(0.5) * dudt_D_tilde[m]
                for m in range(2, 5):
                    if missing_mask[l * 5 + m] == wp.uint8(1):
                        # Get the pre-streaming distribution function in oppisite direction
                        _f[l * 5 + m] = f_pre[_opp_indices[l] * 5 + m]
                
                for i, j, l in zip(_c[0], _c[1], range(4)):
                    _f[l * 5 + 2] += i * wp.c_k_led * dudt_D_tilde[0]   + j * wp.c_k_led * dudt_D_tilde[1]
                    _f[l * 5 + 3] += i * wp.c_mu_led * dudt_D_tilde[0]  - j * wp.c_mu_led * dudt_D_tilde[1]
                    _f[l * 5 + 4] += j * wp.c_mu_led * dudt_D_tilde[0]  + i * wp.c_mu_led * dudt_D_tilde[1]
                    raise UserWarning("Use t+0.5dt for dudt_D_tilde and boundary of domain for x, y, not node positions")




            # # TODO: add S_ij
            # for l in range(self.velocity_set.q):
            #     # If the mask is missing then take the opposite index
            #     for m in range(2):
            #         if missing_mask[l * 5 + m] == wp.uint8(1):
            #             # Get the pre-streaming distribution function in oppisite direction
            #             # return negative of velocities
            #             _f[l * 5 + m] = -f_pre[_opp_indices[l] * 5 + m]
            #     for m in range(2, 5):
            #         if missing_mask[l * 5 + m] == wp.uint8(1):
            #             # Get the pre-streaming distribution function in oppisite direction
            #             # return the same value for stresses
            #             _f[l * 5 + m] = f_pre[_opp_indices[l] * 5 + m]
            return _f

        kernel = self._construct_kernel(functional)

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_pre, f_post, bc_mask, missing_mask, u_D_tilde):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[f_pre, f_post, bc_mask, missing_mask, u_D_tilde],
            dim=f_pre.shape[1:],
        )
        return f_post
