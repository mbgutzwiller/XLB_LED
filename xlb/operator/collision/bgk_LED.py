import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb.operator.collision.collision import Collision
from xlb.operator import Operator
from functools import partial


class BGK_LED(Collision):
    """
    BGK collision operator for linear elastodynamics LBM.
    """

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0,))
    def jax_implementation(self, f: jnp.ndarray, feq: jnp.ndarray, omega):
        _omega = self.compute_dtype(omega)
        return _omega * feq + (1 - _omega) * f

    def _construct_warp(self):
        # Set local constants TODO: This is a hack and should be fixed with warp update
        _f_vec = wp.vec(20, dtype=self.compute_dtype)

        # Construct the functional
        @wp.func
        def functional(f: Any, feq: Any, omega: Any):
            _omega = self.compute_dtype(omega)
            return _omega * feq + (self.compute_dtype(1) - _omega) * f

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            f: wp.array4d(dtype=Any),
            feq: wp.array4d(dtype=Any),
            fout: wp.array4d(dtype=Any),
            omega: Any,
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)  # TODO: Warp needs to fix this

            # Load needed values
            _f = _f_vec()
            _feq = _f_vec()
            for l in range(20):
                _f[l] = f[l, index[0], index[1], index[2]]
                _feq[l] = feq[l, index[0], index[1], index[2]]

            # Compute the collision
            _fout = functional(_f, _feq, omega)

            # Write the result
            for l in range(20):
                fout[l, index[0], index[1], index[2]] = self.store_dtype(_fout[l])

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f, feq, fout, omega):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                f,
                feq,
                fout,
                omega,
            ],
            dim=f.shape[1:],
        )
        return fout
