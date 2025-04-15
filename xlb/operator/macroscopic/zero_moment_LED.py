from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any


from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator


class ZeroMoment_LED(Operator):
    """A class to compute the zeroth moment (density) of distribution functions."""

    # @Operator.register_backend(ComputeBackend.JAX)
    # @partial(jit, static_argnums=(0), inline=True)
    # def jax_implementation(self, f):
    #     # return jnp.sum(f, axis=0, keepdims=True)
    #     Nx, Ny = f.shape[1], f.shape[2]
    #     return f.reshape(4, 5, Nx, Ny)sum(axis=0)

    def _construct_warp(self):
        _f_vec = wp.vec(20, dtype=self.compute_dtype)
        _u_num_vec = wp.vec(5, dtype=self.compute_dtype)
        _x_y_vec = wp.vec(2, dtype=self.compute_dtype)
        _B_vec = wp.vec(2, dtype=self.compute_dtype)

        @wp.func
        def functional(f: _f_vec, index: wp.vec3i, t: wp.float32):
            # Target manufactured solution
            loc = _x_y_vec()
            # loc[0] = (self.compute_dtype(index[0]) + self.compute_dtype(0.5)) * wp.delta_x_led
            # loc[1] = (self.compute_dtype(index[1]) + self.compute_dtype(0.5)) * wp.delta_x_led
            _B = _B_vec()
            _B[0] = self.compute_dtype(0.) #wp.c_mu_led*(8.*wp.pi**2.*wp.cos(4.*wp.pi*loc[0])*wp.cos(2.*wp.pi*loc[1])*wp.sin(4.*wp.pi*(t + 3./10.)) + 16.*wp.pi**2.*wp.sin(4.*wp.pi*loc[0])*wp.sin(2.*wp.pi*loc[1])*wp.sin(4.*wp.pi*(t - 1./10.))) - wp.c_mu_led*(8.*wp.pi**2.*wp.cos(4.*wp.pi*loc[0])*wp.cos(2.*wp.pi*loc[1])*wp.sin(4.*wp.pi*(t + 3./10.)) - 4.*wp.pi**2.*wp.sin(4.*wp.pi*loc[0])*wp.sin(2.*wp.pi*loc[1])*wp.sin(4.*wp.pi*(t - 1./10.))) - wp.c_k_led*(8.*wp.pi**2.*wp.cos(4.*wp.pi*loc[0])*wp.cos(2.*wp.pi*loc[1])*wp.sin(4.*wp.pi*(t + 3./10.)) - 16.*wp.pi**2.*wp.sin(4.*wp.pi*loc[0])*wp.sin(2.*wp.pi*loc[1])*wp.sin(4.*wp.pi*(t - 1./10.))) - 16.*wp.pi**2.*wp.sin(4.*wp.pi*loc[0])*wp.sin(2.*wp.pi*loc[1])*wp.sin(4.*wp.pi*(t - 1./10.))
            _B[1] = self.compute_dtype(0.) #wp.c_mu_led*(8.*wp.pi**2.*wp.cos(4.*wp.pi*loc[0])*wp.cos(2.*wp.pi*loc[1])*wp.sin(4.*wp.pi*(t - 1./10.)) + 4.*wp.pi**2.*wp.sin(4.*wp.pi*loc[0])*wp.sin(2.*wp.pi*loc[1])*wp.sin(4.*wp.pi*(t + 3./10.))) - wp.c_k_led*(8.*wp.pi**2.*wp.cos(4.*wp.pi*loc[0])*wp.cos(2.*wp.pi*loc[1])*wp.sin(4.*wp.pi*(t - 1./10.)) - 4.*wp.pi**2.*wp.sin(4.*wp.pi*loc[0])*wp.sin(2.*wp.pi*loc[1])*wp.sin(4.*wp.pi*(t + 3./10.))) - wp.c_mu_led*(8.*wp.pi**2.*wp.cos(4.*wp.pi*loc[0])*wp.cos(2.*wp.pi*loc[1])*wp.sin(4.*wp.pi*(t - 1./10.)) - 16.*wp.pi**2.*wp.sin(4.*wp.pi*loc[0])*wp.sin(2.*wp.pi*loc[1])*wp.sin(4.*wp.pi*(t + 3./10.))) - 16.*wp.pi**2.*wp.sin(4.*wp.pi*loc[0])*wp.sin(2.*wp.pi*loc[1])*wp.sin(4.*wp.pi*(t + 3./10.))

            _U_num_tilde = _u_num_vec(self.compute_dtype(0.))

            for l in range(4):
                for m in range(5):                        
                    _U_num_tilde[m] += f[l * 5 + m]
            for m in range(2):
                _U_num_tilde[m] += _B[m] * wp.delta_t_led * self.compute_dtype(0.5)
            return _U_num_tilde

        @wp.kernel
        def kernel(
            f: wp.array4d(dtype=Any),
            U_num_tilde: wp.array4d(dtype=Any),
            t: wp.float32,
        ):
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            _f = _f_vec()
            for l in range(20):
                _f[l] = f[l, index[0], index[1], index[2]]
            # _rho = functional(_f)
            _t = self.compute_dtype(t)
            _U_num_tilde = functional(_f, index, _t)

            # rho[0, index[0], index[1], index[2]] = _rho
            for d in range(5):
                U_num_tilde[d, index[0], index[1], index[2]] = self.store_dtype(_U_num_tilde[d])

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f, U_num_tilde, t):
        wp.launch(self.warp_kernel, inputs=[f, U_num_tilde, t], dim=U_num_tilde.shape[1:])
        return U_num_tilde
