import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator import Operator
from functools import partial


class Displacement_LED(Operator):
    """
    Half time integration step for displacement solution.
    """

    @Operator.register_backend(ComputeBackend.JAX)  #  TODO: jax implementation of time integration of displacement solution
    @partial(jit, static_argnums=(0,))
    def jax_implementation(self, U_num_tilde: jnp.ndarray, u_num_displ: jnp.ndarray):
        raise NotImplementedError("is hardcoded right now")
        # return u_num_displ + U_num_tilde[:2] / 400  #hardcoded for numsteps=200, but works like this

    def _construct_warp(self):
        # Set local constants TODO: This is a hack and should be fixed with warp update
        _u_num_vec = wp.vec(2, dtype=self.compute_dtype)
        
        # Construct the functional
        @wp.func
        def functional(U_num_tilde: Any, uxy_num: Any):
            u_num_vec = _u_num_vec()
            vx = U_num_tilde[0]
            vy = U_num_tilde[1]
            u_num_vec[0] = uxy_num[0] + self.compute_dtype(0.5) * wp.delta_t_led * vx
            u_num_vec[1] = uxy_num[1] + self.compute_dtype(0.5) * wp.delta_t_led * vy
            return u_num_vec
        
        
        # Construct the warp kernel
        @wp.kernel
        def kernel(
            U_num_tilde: wp.array4d(dtype=Any),
            u_num_displ: wp.array4d(dtype=Any),
            u_num_out: wp.array4d(dtype=Any),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)  # TODO: Warp needs to fix this

            # Load needed values
            _uxy_num = _u_num_vec()
            _U_num_tilde = _u_num_vec()
            for l in range(2):
                _uxy_num[l] = u_num_displ[l, index[0], index[1], index[2]]
            for l in range(5):
                _U_num_tilde[l] = U_num_tilde[l, index[0], index[1], index[2]]

            # Compute the collision
            _u_num_out = functional(_U_num_tilde, _uxy_num)

            # Write the result
            for l in range(2):
                u_num_out[l, index[0], index[1], index[2]] = self.store_dtype(_u_num_out[l])

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, U_num_tilde, u_num_displ, u_num_out):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                U_num_tilde,
                u_num_displ,
                u_num_out
            ],
            dim=U_num_tilde.shape[1:],
        )
        return u_num_out
