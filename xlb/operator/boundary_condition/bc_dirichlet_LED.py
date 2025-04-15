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
        # TODO: maybe..?
        self.needs_padding = True


    def _construct_warp(self):
        # Set local constants
        _opp_indices = self.velocity_set.opp_indices
        _dudt_D_tilde_vector_vec = wp.vec(2, dtype=self.compute_dtype)
        
        @wp.func
        def dudt_tilde_func(x: wp.float32, y: wp.float32, t: wp.float32):
            _dudt_D_tilde = _dudt_D_tilde_vector_vec(0.)
            # 0 BC for hard reflection.
            _dudt_D_tilde[0] = self.compute_dtype(0.) #4.*wp.pi*wp.sin(4.*wp.pi*x)*wp.sin(2.*wp.pi*y)*wp.cos(4.*wp.pi*(t - 1./10.))
            _dudt_D_tilde[1] = self.compute_dtype(0.) #4.*wp.pi*wp.sin(4.*wp.pi*x)*wp.sin(2.*wp.pi*y)*wp.cos(4.*wp.pi*(t + 3./10.))
            return _dudt_D_tilde
        
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
            # Post-streaming values are only modified at missing direction
            _f = f_post
            for l in range(self.velocity_set.q):
                # If the mask is missing (true, 1) then take the opposite index
                for m in range(2):
                    if missing_mask[l] == wp.uint8(1):
                        # Get the pre-streaming distribution function in oppisite direction
                        # return negative of velocities
                        _f[l * 5 + m] = -f_pre[_opp_indices[l] * 5 + m]
                for m in range(2, 5):
                    if missing_mask[l] == wp.uint8(1):
                        # Get the pre-streaming distribution function in oppisite direction
                        _f[l * 5 + m] = f_pre[_opp_indices[l] * 5 + m]
            
            # for i, j in zip(_vel_c[0], _vel_c[1]):
            # TODO: make this more efficient/cleaner, maybe use c from velocity set and access directions with index, define boundary prior.
            for l in range(4):
                if missing_mask[l] == wp.uint8(1):
                    if l == 0:
                        i = 1
                        j = 0
                        _x = 0.
                        _y = (self.compute_dtype(index[1]) + self.compute_dtype(0.5)) * wp.delta_x_led
                    elif l == 1:
                        i = 0
                        j = 1
                        _x = (self.compute_dtype(index[0]) + self.compute_dtype(0.5)) * wp.delta_x_led
                        _y = 0. 
                    elif l == 2:
                        i = -1
                        j = 0
                        _x = 1.
                        _y = (self.compute_dtype(index[1]) + self.compute_dtype(0.5)) * wp.delta_x_led
                    elif l == 3:
                        i = 0
                        j = -1
                        _x = (self.compute_dtype(index[0]) + self.compute_dtype(0.5)) * wp.delta_x_led
                        _y = 1.
                    _dudt_D_tilde = dudt_tilde_func(_x, _y, (self.compute_dtype(timestep)+self.compute_dtype(0.5))*wp.delta_t_led)
                    # Add contribution of S_ij*u_D_tilde
                    _f[l * 5 + 0] += self.compute_dtype(0.5) * _dudt_D_tilde[0]
                    _f[l * 5 + 1] += self.compute_dtype(0.5) * _dudt_D_tilde[1]
                    _f[l * 5 + 2] += (self.compute_dtype(i) * wp.c_k_led ** 0.5 * _dudt_D_tilde[0]  + self.compute_dtype(j) * wp.c_k_led ** 0.5 * _dudt_D_tilde[1]) / wp.c_led
                    _f[l * 5 + 3] += (self.compute_dtype(i) * wp.c_mu_led ** 0.5 * _dudt_D_tilde[0]  - self.compute_dtype(j) * wp.c_mu_led ** 0.5 * _dudt_D_tilde[1]) / wp.c_led
                    _f[l * 5 + 4] += (self.compute_dtype(j) * wp.c_mu_led ** 0.5 * _dudt_D_tilde[0]  + self.compute_dtype(i) * wp.c_mu_led ** 0.5 * _dudt_D_tilde[1]) / wp.c_led

            return _f

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
