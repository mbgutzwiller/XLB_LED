from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.macroscopic.zero_moment_LED import ZeroMoment_LED
from xlb.operator.macroscopic.first_moment import FirstMoment


class Macroscopic_LED(Operator):
    """A class to compute both zero and first moments of distribution functions U_num_tilde."""

    def __init__(self, *args, **kwargs):
        self.zero_moment_LED = ZeroMoment_LED(*args, **kwargs)
        super().__init__(*args, **kwargs)

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0), inline=True)
    def jax_implementation(self, f):
        return self.zero_moment_LED(f)

    def _construct_warp(self):
        zero_moment_func = self.zero_moment_LED.warp_functional
        _f_vec = wp.vec(20, dtype=self.compute_dtype)

        @wp.func
        def functional(f: _f_vec, index: wp.vec3i, t: wp.float32):
            # TODO: seems unnecessarily nested
            return zero_moment_func(f, index, t)

        @wp.kernel
        def kernel(
            f: wp.array4d(dtype=Any),
            U_num_tilde: wp.array4d(dtype=Any),
            t: wp.float32
        ):
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            _f = _f_vec()
            _t = self.compute_dtype(t)
            for l in range(20):
                _f[l] = f[l, index[0], index[1], index[2]]
            _U_num_tilde = functional(_f, index, _t)

            for d in range(5):
                U_num_tilde[d, index[0], index[1], index[2]] = self.store_dtype(_U_num_tilde[d])

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f, U_num_tilde, t):
        wp.launch(
            self.warp_kernel,
            inputs=[f, U_num_tilde, t],
            dim=U_num_tilde.shape[1:],
        )
        return U_num_tilde
