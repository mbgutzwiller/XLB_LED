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
        def functional(f: _f_vec, index: wp.vec3i, t: Any):
            # Target manufactured solution
            loc = _x_y_vec()

            loc[0] = (self.compute_dtype(index[0]) + self.compute_dtype(0.5)) * wp.float64(wp.delta_x_led)
            loc[1] = (self.compute_dtype(index[1]) + self.compute_dtype(0.5)) * wp.float64(wp.delta_x_led)
            B = _B_vec()
            _c_mu = wp.float64(wp.c_mu_led)
            _c_k = wp.float64(wp.c_k_led)
            _8p0 = wp.float64(8.0)
            _2p0 = wp.float64(2.0)
            _10p0 = wp.float64(10.0)
            _7p0 = wp.float64(7.0)
            _5p0 = wp.float64(5.)
            _16p0 = wp.float64(16.0)
            _3p0 = wp.float64(3.0)
            _25p0 = wp.float64(25.0)
            _597p0 = wp.float64(597.0)
            _28p0 = wp.float64(28.0)
            _112p0 = wp.float64(112.)
            _48p0 = wp.float64(48.)
            _64p0 = wp.float64(64.)
            _4p0 = wp.float64(4.0)
            _1p0 = wp.float64(1.0)
            _20p0 = wp.float64(20.0)
            _96p0 = wp.float64(96.)
            pi = wp.float64(wp.pi)

            B[0] = _c_mu*_c_mu*(_8p0*pi**_2p0*wp.cos(_2p0*pi*(t/_10p0 - loc[1]))*wp.sin(_4p0*pi*((_7p0*t)/_10p0 - loc[0]))*wp.cos(_4p0*pi*(t + _2p0/_5p0)) - _16p0*pi**_2p0*wp.cos(_2p0*pi*((_4p0*t)/_5p0 - loc[1]))*wp.sin(_4p0*pi*((_3p0*t)/_10p0 - loc[0]))*wp.sin(_4p0*pi*(t - _1p0/_10p0))) - _c_mu*_c_mu*(_8p0*pi**_2p0*wp.cos(_2p0*pi*(t/_10p0 - loc[1]))*wp.sin(_4p0*pi*((_7p0*t)/_10p0 - loc[0]))*wp.cos(_4p0*pi*(t + _2p0/_5p0)) + _4p0*pi**_2p0*wp.cos(_2p0*pi*((_4p0*t)/_5p0 - loc[1]))*wp.sin(_4p0*pi*((_3p0*t)/_10p0 - loc[0]))*wp.sin(_4p0*pi*(t - _1p0/_10p0))) - _c_k*_c_k*(_8p0*pi**_2p0*wp.cos(_2p0*pi*(t/_10p0 - loc[1]))*wp.sin(_4p0*pi*((_7p0*t)/_10p0 - loc[0]))*wp.cos(_4p0*pi*(t + _2p0/_5p0)) + _16p0*pi**_2p0*wp.cos(_2p0*pi*((_4p0*t)/_5p0 - loc[1]))*wp.sin(_4p0*pi*((_3p0*t)/_10p0 - loc[0]))*wp.sin(_4p0*pi*(t - _1p0/_10p0))) - (_48p0*pi**_2p0*wp.cos(_4p0*pi*((_3p0*t)/_10p0 - loc[0]))*wp.cos(_2p0*pi*((_4p0*t)/_5p0 - loc[1]))*wp.cos(_4p0*pi*(t - _1p0/_10p0)))/_5p0 + (_96p0*pi**_2p0*wp.cos(_4p0*pi*((_3p0*t)/_10p0 - loc[0]))*wp.sin(_2p0*pi*((_4p0*t)/_5p0 - loc[1]))*wp.sin(_4p0*pi*(t - _1p0/_10p0)))/_25p0 + _20p0*pi**_2p0*wp.cos(_2p0*pi*((_4p0*t)/_5p0 - loc[1]))*wp.sin(_4p0*pi*((_3p0*t)/_10p0 - loc[0]))*wp.sin(_4p0*pi*(t - _1p0/_10p0)) + (_64p0*pi**_2p0*wp.sin(_4p0*pi*((_3p0*t)/_10p0 - loc[0]))*wp.sin(_2p0*pi*((_4p0*t)/_5p0 - loc[1]))*wp.cos(_4p0*pi*(t - _1p0/_10p0)))/_5p0
            B[1] = (_8p0*pi**_2p0*wp.cos(_4p0*pi*((_7p0*t)/_10p0 - loc[0]))*wp.cos(_2p0*pi*(t/_10p0 - loc[1]))*wp.sin(_4p0*pi*(t + _2p0/_5p0)))/_5p0 - _c_mu*_c_mu*(_4p0*pi**_2p0*wp.cos(_4p0*pi*((_7p0*t)/_10p0 - loc[0]))*wp.sin(_2p0*pi*(t/_10p0 - loc[1]))*wp.cos(_4p0*pi*(t + _2p0/_5p0)) - _8p0*pi**_2p0*wp.cos(_4p0*pi*((_3p0*t)/_10p0 - loc[0]))*wp.sin(_2p0*pi*((_4p0*t)/_5p0 - loc[1]))*wp.sin(_4p0*pi*(t - _1p0/_10p0))) - _c_mu*_c_mu*(_16p0*pi**_2p0*wp.cos(_4p0*pi*((_7p0*t)/_10p0 - loc[0]))*wp.sin(_2p0*pi*(t/_10p0 - loc[1]))*wp.cos(_4p0*pi*(t + _2p0/_5p0)) + _8p0*pi**_2p0*wp.cos(_4p0*pi*((_3p0*t)/_10p0 - loc[0]))*wp.sin(_2p0*pi*((_4p0*t)/_5p0 - loc[1]))*wp.sin(_4p0*pi*(t - _1p0/_10p0))) - _c_k*_c_k*(_4p0*pi**_2p0*wp.cos(_4p0*pi*((_7p0*t)/_10p0 - loc[0]))*wp.sin(_2p0*pi*(t/_10p0 - loc[1]))*wp.cos(_4p0*pi*(t + _2p0/_5p0)) + _8p0*pi**_2p0*wp.cos(_4p0*pi*((_3p0*t)/_10p0 - loc[0]))*wp.sin(_2p0*pi*((_4p0*t)/_5p0 - loc[1]))*wp.sin(_4p0*pi*(t - _1p0/_10p0))) + (_597p0*pi**_2p0*wp.cos(_4p0*pi*((_7p0*t)/_10p0 - loc[0]))*wp.sin(_2p0*pi*(t/_10p0 - loc[1]))*wp.cos(_4p0*pi*(t + _2p0/_5p0)))/_25p0 + (_28p0*pi**_2p0*wp.cos(_2p0*pi*(t/_10p0 - loc[1]))*wp.sin(_4p0*pi*((_7p0*t)/_10p0 - loc[0]))*wp.cos(_4p0*pi*(t + _2p0/_5p0)))/_25p0 - (_112p0*pi**_2p0*wp.sin(_4p0*pi*((_7p0*t)/_10p0 - loc[0]))*wp.sin(_2p0*pi*(t/_10p0 - loc[1]))*wp.sin(_4p0*pi*(t + _2p0/_5p0)))/_5p0
            # B[0] = wp.c_mu_led*wp.c_mu_led*(8.*wp.pi**2.*wp.cos(2.*wp.pi*(t/10. - loc[1]))*wp.sin(4.*wp.pi*((7.*t)/10. - loc[0]))*wp.cos(4.*wp.pi*(t + 2./5.)) - 16.*wp.pi**2.*wp.cos(2.*wp.pi*((4.*t)/5. - loc[1]))*wp.sin(4.*wp.pi*((3.*t)/10. - loc[0]))*wp.sin(4.*wp.pi*(t - 1./10.))) - wp.c_mu_led*wp.c_mu_led*(8.*wp.pi**2.*wp.cos(2.*wp.pi*(t/10. - loc[1]))*wp.sin(4.*wp.pi*((7.*t)/10. - loc[0]))*wp.cos(4.*wp.pi*(t + 2./5.)) + 4.*wp.pi**2.*wp.cos(2.*wp.pi*((4.*t)/5. - loc[1]))*wp.sin(4.*wp.pi*((3.*t)/10. - loc[0]))*wp.sin(4.*wp.pi*(t - 1./10.))) - wp.c_k_led*wp.c_k_led*(8.*wp.pi**2.*wp.cos(2.*wp.pi*(t/10. - loc[1]))*wp.sin(4.*wp.pi*((7.*t)/10. - loc[0]))*wp.cos(4.*wp.pi*(t + 2./5.)) + 16.*wp.pi**2.*wp.cos(2.*wp.pi*((4.*t)/5. - loc[1]))*wp.sin(4.*wp.pi*((3.*t)/10. - loc[0]))*wp.sin(4.*wp.pi*(t - 1./10.))) - (48.*wp.pi**2.*wp.cos(4.*wp.pi*((3.*t)/10. - loc[0]))*wp.cos(2.*wp.pi*((4.*t)/5. - loc[1]))*wp.cos(4.*wp.pi*(t - 1./10.)))/5. + (96.*wp.pi**2.*wp.cos(4.*wp.pi*((3.*t)/10. - loc[0]))*wp.sin(2.*wp.pi*((4.*t)/5. - loc[1]))*wp.sin(4.*wp.pi*(t - 1./10.)))/25. + 20.*wp.pi**2.*wp.cos(2.*wp.pi*((4.*t)/5. - loc[1]))*wp.sin(4.*wp.pi*((3.*t)/10. - loc[0]))*wp.sin(4.*wp.pi*(t - 1./10.)) + (64.*wp.pi**2.*wp.sin(4.*wp.pi*((3.*t)/10. - loc[0]))*wp.sin(2.*wp.pi*((4.*t)/5. - loc[1]))*wp.cos(4.*wp.pi*(t - 1./10.)))/5.
            # B[1] = (8.*wp.pi**2.*wp.cos(4.*wp.pi*((7.*t)/10. - loc[0]))*wp.cos(2.*wp.pi*(t/10. - loc[1]))*wp.sin(4.*wp.pi*(t + 2./5.)))/5. - wp.c_mu_led*wp.c_mu_led*(4.*wp.pi**2.*wp.cos(4.*wp.pi*((7.*t)/10. - loc[0]))*wp.sin(2.*wp.pi*(t/10. - loc[1]))*wp.cos(4.*wp.pi*(t + 2./5.)) - 8.*wp.pi**2.*wp.cos(4.*wp.pi*((3.*t)/10. - loc[0]))*wp.sin(2.*wp.pi*((4.*t)/5. - loc[1]))*wp.sin(4.*wp.pi*(t - 1./10.))) - wp.c_mu_led*wp.c_mu_led*(16.*wp.pi**2.*wp.cos(4.*wp.pi*((7.*t)/10. - loc[0]))*wp.sin(2.*wp.pi*(t/10. - loc[1]))*wp.cos(4.*wp.pi*(t + 2./5.)) + 8.*wp.pi**2.*wp.cos(4.*wp.pi*((3.*t)/10. - loc[0]))*wp.sin(2.*wp.pi*((4.*t)/5. - loc[1]))*wp.sin(4.*wp.pi*(t - 1./10.))) - wp.c_k_led*wp.c_k_led*(4.*wp.pi**2.*wp.cos(4.*wp.pi*((7.*t)/10. - loc[0]))*wp.sin(2.*wp.pi*(t/10. - loc[1]))*wp.cos(4.*wp.pi*(t + 2./5.)) + 8.*wp.pi**2.*wp.cos(4.*wp.pi*((3.*t)/10. - loc[0]))*wp.sin(2.*wp.pi*((4.*t)/5. - loc[1]))*wp.sin(4.*wp.pi*(t - 1./10.))) + (597.*wp.pi**2.*wp.cos(4.*wp.pi*((7.*t)/10. - loc[0]))*wp.sin(2.*wp.pi*(t/10. - loc[1]))*wp.cos(4.*wp.pi*(t + 2./5.)))/25. + (28.*wp.pi**2.*wp.cos(2.*wp.pi*(t/10. - loc[1]))*wp.sin(4.*wp.pi*((7.*t)/10. - loc[0]))*wp.cos(4.*wp.pi*(t + 2./5.)))/25. - (112.*wp.pi**2.*wp.sin(4.*wp.pi*((7.*t)/10. - loc[0]))*wp.sin(2.*wp.pi*(t/10. - loc[1]))*wp.sin(4.*wp.pi*(t + 2./5.)))/5.

            _U_num_tilde = _u_num_vec()

            for l in range(4):
                for m in range(5):                        
                    _U_num_tilde[m] += f[l * 5 + m]
            for m in range(2):
                _U_num_tilde[m] += B[m] * wp.delta_t_led * self.compute_dtype(0.5)
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
