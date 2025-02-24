from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator


class ZeroMoment_LED(Operator):
    """A class to compute the zeroth moment (density) of distribution functions."""

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0), inline=True)
    def jax_implementation(self, f):
        # return jnp.sum(f, axis=0, keepdims=True)
        Nx, Ny = f.shape[1], f.shape[2]
        return f.reshape(4, 5, Nx, Ny).sum(axis=0)

    def _construct_warp(self):
        _f_vec = wp.vec(20, dtype=self.compute_dtype)
        _u_num_vec = wp.vec(5, dtype=self.compute_dtype)

        @wp.func
        def functional(f: _f_vec):
            # This initializes the rho with value 0 and correct precision
            # rho = self.compute_dtype(0.0)
            U_num_tilde = _u_num_vec()
            for l in range(self.velocity_set.q):
                for m in range(5):
                    U_num_tilde[m] += f[l * 5 + m]  # TODO: add body load B_hat
            return U_num_tilde

        @wp.kernel
        def kernel(
            f: wp.array4d(dtype=Any),
            U_num_tilde: wp.array4d(dtype=Any),
        ):
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            _f = _f_vec()
            for l in range(20):
                _f[l] = f[l, index[0], index[1], index[2]]
            # _rho = functional(_f)
            _U_num_tilde = functional(_f)

            # rho[0, index[0], index[1], index[2]] = _rho
            for d in range(5):
                U_num_tilde[d, index[0], index[1], index[2]] = self.store_dtype(_U_num_tilde[d])

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f, U_num_tilde):
        wp.launch(self.warp_kernel, inputs=[f, U_num_tilde], dim=U_num_tilde.shape[1:])
        return U_num_tilde
