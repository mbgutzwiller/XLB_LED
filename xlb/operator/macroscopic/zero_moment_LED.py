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
        raise NotImplementedError("Maybe works, not tested yet")
        # Nx, Ny = f.shape[1], f.shape[2]
        # return f.reshape(4, 5, Nx, Ny).sum(axis=0)

    def _construct_warp(self):
        _f_vec = wp.vec(20, dtype=self.compute_dtype)
        _u_num_vec = wp.vec(5, dtype=self.compute_dtype)
        _x_y_vec = wp.vec(2, dtype=self.compute_dtype)
        _B_vec = wp.vec(2, dtype=self.compute_dtype)

        @wp.func
        def functional(f: _f_vec, index: wp.vec3i, t: wp.float32):
            # Target manufactured solution
            loc = _x_y_vec()

            loc[0] = (self.compute_dtype(index[0]) + self.compute_dtype(0.5)) * wp.delta_x_led
            loc[1] = (self.compute_dtype(index[1]) + self.compute_dtype(0.5)) * wp.delta_x_led
            # @(t,x,y)[wp.pi**2.*wp.cos(wp.pi*(t*(3.0./1.0e+1)-x)*4.0)*wp.cos(wp.pi*(t*(4.0./5..0)-y)*2.0)*wp.cos(wp.pi*(t-1.0./1.0e+1)*4.0)*(-4.8e+1./5..0)-wp.pi**2.*wp.cos(wp.pi*(t/1.0e+1-y)*2.0)*wp.sin(wp.pi*(t*(7.0./1.0e+1)-x)*4.0)*wp.cos(wp.pi*(t+2.0./5..0)*4.0)*(4.4e+1./5..0)+wp.pi**2.*wp.cos(wp.pi*(t*(3.0./1.0e+1)-x)*4.0)*wp.sin(wp.pi*(t*(4.0./5..0)-y)*2.0)*wp.sin(wp.pi*(t-1.0./1.0e+1)*4.0)*(9.6e+1./2.5e+1)-wp.pi**2.*wp.cos(wp.pi*(t*(4.0./5..0)-y)*2.0)*wp.sin(wp.pi*(t*(3.0./1.0e+1)-x)*4.0)*wp.sin(wp.pi*(t-1.0./1.0e+1)*4.0)*(2.8e+1./5..0)+wp.pi**2.*wp.sin(wp.pi*(t*(3.0./1.0e+1)-x)*4.0)*wp.sin(wp.pi*(t*(4.0./5..0)-y)*2.0)*wp.cos(wp.pi*(t-1.0./1.0e+1)*4.0)*(6.4e+1./5..0);wp.pi**2.*wp.cos(wp.pi*(t*(7.0./1.0e+1)-x)*4.0)*wp.cos(wp.pi*(t/1.0e+1-y)*2.0)*wp.sin(wp.pi*(t+2.0./5..0)*4.0)*(8.0./5..0)+wp.pi**2.*wp.cos(wp.pi*(t*(7.0./1.0e+1)-x)*4.0)*wp.sin(wp.pi*(t/1.0e+1-y)*2.0)*wp.cos(wp.pi*(t+2.0./5..0)*4.0)*(2.87e+2./2.5e+1)+wp.pi**2.*wp.cos(wp.pi*(t/1.0e+1-y)*2.0)*wp.sin(wp.pi*(t*(7.0./1.0e+1)-x)*4.0)*wp.cos(wp.pi*(t+2.0./5..0)*4.0)*(2.8e+1./2.5e+1)-wp.pi**2.*wp.cos(wp.pi*(t*(3.0./1.0e+1)-x)*4.0)*wp.sin(wp.pi*(t*(4.0./5..0)-y)*2.0)*wp.sin(wp.pi*(t-1.0./1.0e+1)*4.0)*(4.4e+1./5..0)-wp.pi**2.*wp.sin(wp.pi*(t*(7.0./1.0e+1)-x)*4.0)*wp.sin(wp.pi*(t/1.0e+1-y)*2.0)*wp.sin(wp.pi*(t+2.0./5..0)*4.0)*(1.12e+2./5..0)]
            B = _B_vec()
            B[0] = wp.c_mu_led*wp.c_mu_led*(8.*wp.pi**2.*wp.cos(2.*wp.pi*(t/10. - loc[1]))*wp.sin(4.*wp.pi*((7.*t)/10. - loc[0]))*wp.cos(4.*wp.pi*(t + 2./5.)) - 16.*wp.pi**2.*wp.cos(2.*wp.pi*((4.*t)/5. - loc[1]))*wp.sin(4.*wp.pi*((3.*t)/10. - loc[0]))*wp.sin(4.*wp.pi*(t - 1./10.))) - wp.c_mu_led*wp.c_mu_led*(8.*wp.pi**2.*wp.cos(2.*wp.pi*(t/10. - loc[1]))*wp.sin(4.*wp.pi*((7.*t)/10. - loc[0]))*wp.cos(4.*wp.pi*(t + 2./5.)) + 4.*wp.pi**2.*wp.cos(2.*wp.pi*((4.*t)/5. - loc[1]))*wp.sin(4.*wp.pi*((3.*t)/10. - loc[0]))*wp.sin(4.*wp.pi*(t - 1./10.))) - wp.c_k_led*wp.c_k_led*(8.*wp.pi**2.*wp.cos(2.*wp.pi*(t/10. - loc[1]))*wp.sin(4.*wp.pi*((7.*t)/10. - loc[0]))*wp.cos(4.*wp.pi*(t + 2./5.)) + 16.*wp.pi**2.*wp.cos(2.*wp.pi*((4.*t)/5. - loc[1]))*wp.sin(4.*wp.pi*((3.*t)/10. - loc[0]))*wp.sin(4.*wp.pi*(t - 1./10.))) - (48.*wp.pi**2.*wp.cos(4.*wp.pi*((3.*t)/10. - loc[0]))*wp.cos(2.*wp.pi*((4.*t)/5. - loc[1]))*wp.cos(4.*wp.pi*(t - 1./10.)))/5. + (96.*wp.pi**2.*wp.cos(4.*wp.pi*((3.*t)/10. - loc[0]))*wp.sin(2.*wp.pi*((4.*t)/5. - loc[1]))*wp.sin(4.*wp.pi*(t - 1./10.)))/25. + 20.*wp.pi**2.*wp.cos(2.*wp.pi*((4.*t)/5. - loc[1]))*wp.sin(4.*wp.pi*((3.*t)/10. - loc[0]))*wp.sin(4.*wp.pi*(t - 1./10.)) + (64.*wp.pi**2.*wp.sin(4.*wp.pi*((3.*t)/10. - loc[0]))*wp.sin(2.*wp.pi*((4.*t)/5. - loc[1]))*wp.cos(4.*wp.pi*(t - 1./10.)))/5.
            B[1] = (8.*wp.pi**2.*wp.cos(4.*wp.pi*((7.*t)/10. - loc[0]))*wp.cos(2.*wp.pi*(t/10. - loc[1]))*wp.sin(4.*wp.pi*(t + 2./5.)))/5. - wp.c_mu_led*wp.c_mu_led*(4.*wp.pi**2.*wp.cos(4.*wp.pi*((7.*t)/10. - loc[0]))*wp.sin(2.*wp.pi*(t/10. - loc[1]))*wp.cos(4.*wp.pi*(t + 2./5.)) - 8.*wp.pi**2.*wp.cos(4.*wp.pi*((3.*t)/10. - loc[0]))*wp.sin(2.*wp.pi*((4.*t)/5. - loc[1]))*wp.sin(4.*wp.pi*(t - 1./10.))) - wp.c_mu_led*wp.c_mu_led*(16.*wp.pi**2.*wp.cos(4.*wp.pi*((7.*t)/10. - loc[0]))*wp.sin(2.*wp.pi*(t/10. - loc[1]))*wp.cos(4.*wp.pi*(t + 2./5.)) + 8.*wp.pi**2.*wp.cos(4.*wp.pi*((3.*t)/10. - loc[0]))*wp.sin(2.*wp.pi*((4.*t)/5. - loc[1]))*wp.sin(4.*wp.pi*(t - 1./10.))) - wp.c_k_led*wp.c_k_led*(4.*wp.pi**2.*wp.cos(4.*wp.pi*((7.*t)/10. - loc[0]))*wp.sin(2.*wp.pi*(t/10. - loc[1]))*wp.cos(4.*wp.pi*(t + 2./5.)) + 8.*wp.pi**2.*wp.cos(4.*wp.pi*((3.*t)/10. - loc[0]))*wp.sin(2.*wp.pi*((4.*t)/5. - loc[1]))*wp.sin(4.*wp.pi*(t - 1./10.))) + (597.*wp.pi**2.*wp.cos(4.*wp.pi*((7.*t)/10. - loc[0]))*wp.sin(2.*wp.pi*(t/10. - loc[1]))*wp.cos(4.*wp.pi*(t + 2./5.)))/25. + (28.*wp.pi**2.*wp.cos(2.*wp.pi*(t/10. - loc[1]))*wp.sin(4.*wp.pi*((7.*t)/10. - loc[0]))*wp.cos(4.*wp.pi*(t + 2./5.)))/25. - (112.*wp.pi**2.*wp.sin(4.*wp.pi*((7.*t)/10. - loc[0]))*wp.sin(2.*wp.pi*(t/10. - loc[1]))*wp.sin(4.*wp.pi*(t + 2./5.)))/5.
            # B[0] = -wp.float32(wp.float32(wp.c_k_led*wp.c_k_led))**wp.float32(wp.float32(2.0))*(-wp.float32(8.0)*wp.float32(wp.pi)**wp.float32(wp.float32(2.0))*wp.sin(wp.float32(wp.pi)*(wp.float32(-2.8)*t + wp.float32(4.0)*loc[0]))*wp.cos(wp.float32(wp.pi)*(wp.float32(-0.2)*t + wp.float32(wp.float32(2.0))*loc[1]))*wp.cos(wp.float32(wp.pi)*(wp.float32(4.0)*t + wp.float32(1.6))) - wp.float32(16.0)*wp.float32(wp.pi)**wp.float32(wp.float32(2.0))*wp.sin(wp.float32(wp.pi)*(-wp.float32(1.2)*t + wp.float32(4.0)*loc[0]))*wp.sin(wp.float32(wp.pi)*(wp.float32(4.0)*t - wp.float32(0.4)))*wp.cos(wp.float32(wp.pi)*(-wp.float32(1.6)*t + wp.float32(wp.float32(2.0))*loc[1]))) - wp.float32(wp.c_mu_led*wp.c_mu_led)**wp.float32(wp.float32(2.0))*(-wp.float32(8.0)*wp.float32(wp.pi)**wp.float32(wp.float32(2.0))*wp.sin(wp.float32(wp.pi)*(wp.float32(-2.8)*t + wp.float32(4.0)*loc[0]))*wp.cos(wp.float32(wp.pi)*(wp.float32(-0.2)*t + wp.float32(wp.float32(2.0))*loc[1]))*wp.cos(wp.float32(wp.pi)*(wp.float32(4.0)*t + wp.float32(1.6))) - wp.float32(4.0)*wp.float32(wp.pi)**wp.float32(wp.float32(2.0))*wp.sin(wp.float32(wp.pi)*(-wp.float32(1.2)*t + wp.float32(4.0)*loc[0]))*wp.sin(wp.float32(wp.pi)*(wp.float32(4.0)*t - wp.float32(0.4)))*wp.cos(wp.float32(wp.pi)*(-wp.float32(1.6)*t + wp.float32(wp.float32(2.0))*loc[1]))) - wp.float32(wp.c_mu_led*wp.c_mu_led)**wp.float32(wp.float32(2.0))*(wp.float32(8.0)*wp.float32(wp.pi)**wp.float32(wp.float32(2.0))*wp.sin(wp.float32(wp.pi)*(wp.float32(-2.8)*t + wp.float32(4.0)*loc[0]))*wp.cos(wp.float32(wp.pi)*(wp.float32(-0.2)*t + wp.float32(wp.float32(2.0))*loc[1]))*wp.cos(wp.float32(wp.pi)*(wp.float32(4.0)*t + wp.float32(1.6))) - wp.float32(16.0)*wp.float32(wp.pi)**wp.float32(wp.float32(2.0))*wp.sin(wp.float32(wp.pi)*(-wp.float32(1.2)*t + wp.float32(4.0)*loc[0]))*wp.sin(wp.float32(wp.pi)*(wp.float32(4.0)*t - wp.float32(0.4)))*wp.cos(wp.float32(wp.pi)*(-wp.float32(1.6)*t + wp.float32(wp.float32(2.0))*loc[1]))) + wp.float32(12.8)*wp.float32(wp.pi)**wp.float32(wp.float32(2.0))*wp.sin(wp.float32(wp.pi)*(-wp.float32(1.6)*t + wp.float32(wp.float32(2.0))*loc[1]))*wp.sin(wp.float32(wp.pi)*(-wp.float32(1.2)*t + wp.float32(4.0)*loc[0]))*wp.cos(wp.float32(wp.pi)*(wp.float32(4.0)*t - wp.float32(0.4))) - wp.float32(3.84)*wp.float32(wp.pi)**wp.float32(wp.float32(2.0))*wp.sin(wp.float32(wp.pi)*(-wp.float32(1.6)*t + wp.float32(wp.float32(2.0))*loc[1]))*wp.sin(wp.float32(wp.pi)*(wp.float32(4.0)*t - wp.float32(0.4)))*wp.cos(wp.float32(wp.pi)*(-wp.float32(1.2)*t + wp.float32(4.0)*loc[0])) - wp.float32(20.0)*wp.float32(wp.pi)**wp.float32(wp.float32(2.0))*wp.sin(wp.float32(wp.pi)*(-wp.float32(1.2)*t + wp.float32(4.0)*loc[0]))*wp.sin(wp.float32(wp.pi)*(wp.float32(4.0)*t - wp.float32(0.4)))*wp.cos(wp.float32(wp.pi)*(-wp.float32(1.6)*t + wp.float32(wp.float32(2.0))*loc[1])) - wp.float32(9.6)*wp.float32(wp.pi)**wp.float32(wp.float32(2.0))*wp.cos(wp.float32(wp.pi)*(-wp.float32(1.6)*t + wp.float32(wp.float32(2.0))*loc[1]))*wp.cos(wp.float32(wp.pi)*(-wp.float32(1.2)*t + wp.float32(4.0)*loc[0]))*wp.cos(wp.float32(wp.pi)*(wp.float32(4.0)*t - wp.float32(0.4)))
            # B[1] = -wp.float32(wp.float32(wp.c_k_led*wp.c_k_led))**wp.float32(wp.float32(2.0))*(-wp.float32(8.0)*wp.float32(wp.pi)**wp.float32(wp.float32(2.0))*wp.sin(wp.float32(wp.pi)*(-wp.float32(1.6)*t + wp.float32(wp.float32(2.0))*loc[1]))*wp.sin(wp.float32(wp.pi)*(wp.float32(4.0)*t - wp.float32(0.4)))*wp.cos(wp.float32(wp.pi)*(-wp.float32(1.2)*t + wp.float32(4.0)*loc[0])) - wp.float32(4.0)*wp.float32(wp.pi)**wp.float32(wp.float32(2.0))*wp.sin(wp.float32(wp.pi)*(wp.float32(-0.2)*t + wp.float32(wp.float32(2.0))*loc[1]))*wp.cos(wp.float32(wp.pi)*(wp.float32(-2.8)*t + wp.float32(4.0)*loc[0]))*wp.cos(wp.float32(wp.pi)*(wp.float32(4.0)*t + wp.float32(1.6)))) - wp.float32(wp.c_mu_led*wp.c_mu_led)**wp.float32(wp.float32(2.0))*(-wp.float32(8.0)*wp.float32(wp.pi)**wp.float32(wp.float32(2.0))*wp.sin(wp.float32(wp.pi)*(-wp.float32(1.6)*t + wp.float32(wp.float32(2.0))*loc[1]))*wp.sin(wp.float32(wp.pi)*(wp.float32(4.0)*t - wp.float32(0.4)))*wp.cos(wp.float32(wp.pi)*(-wp.float32(1.2)*t + wp.float32(4.0)*loc[0])) - wp.float32(16.0)*wp.float32(wp.pi)**wp.float32(wp.float32(2.0))*wp.sin(wp.float32(wp.pi)*(wp.float32(-0.2)*t + wp.float32(wp.float32(2.0))*loc[1]))*wp.cos(wp.float32(wp.pi)*(wp.float32(-2.8)*t + wp.float32(4.0)*loc[0]))*wp.cos(wp.float32(wp.pi)*(wp.float32(4.0)*t + wp.float32(1.6)))) + wp.float32(wp.c_mu_led*wp.c_mu_led)**wp.float32(wp.float32(2.0))*(-wp.float32(8.0)*wp.float32(wp.pi)**wp.float32(wp.float32(2.0))*wp.sin(wp.float32(wp.pi)*(-wp.float32(1.6)*t + wp.float32(wp.float32(2.0))*loc[1]))*wp.sin(wp.float32(wp.pi)*(wp.float32(4.0)*t - wp.float32(0.4)))*wp.cos(wp.float32(wp.pi)*(-wp.float32(1.2)*t + wp.float32(4.0)*loc[0])) + wp.float32(4.0)*wp.float32(wp.pi)**wp.float32(wp.float32(2.0))*wp.sin(wp.float32(wp.pi)*(wp.float32(-0.2)*t + wp.float32(wp.float32(2.0))*loc[1]))*wp.cos(wp.float32(wp.pi)*(wp.float32(-2.8)*t + wp.float32(4.0)*loc[0]))*wp.cos(wp.float32(wp.pi)*(wp.float32(4.0)*t + wp.float32(1.6)))) - wp.float32(22.4)*wp.float32(wp.pi)**wp.float32(wp.float32(2.0))*wp.sin(wp.float32(wp.pi)*(wp.float32(-2.8)*t + wp.float32(4.0)*loc[0]))*wp.sin(wp.float32(wp.pi)*(wp.float32(-0.2)*t + wp.float32(wp.float32(2.0))*loc[1]))*wp.sin(wp.float32(wp.pi)*(wp.float32(4.0)*t + wp.float32(1.6))) - wp.float32(1.12)*wp.float32(wp.pi)**wp.float32(wp.float32(2.0))*wp.sin(wp.float32(wp.pi)*(wp.float32(-2.8)*t + wp.float32(4.0)*loc[0]))*wp.cos(wp.float32(wp.pi)*(wp.float32(-0.2)*t + wp.float32(wp.float32(2.0))*loc[1]))*wp.cos(wp.float32(wp.pi)*(wp.float32(4.0)*t + wp.float32(1.6))) - wp.float32(23.88)*wp.float32(wp.pi)**wp.float32(wp.float32(2.0))*wp.sin(wp.float32(wp.pi)*(wp.float32(-0.2)*t + wp.float32(wp.float32(2.0))*loc[1]))*wp.cos(wp.float32(wp.pi)*(wp.float32(-2.8)*t + wp.float32(4.0)*loc[0]))*wp.cos(wp.float32(wp.pi)*(wp.float32(4.0)*t + wp.float32(1.6))) + wp.float32(1.6)*wp.float32(wp.pi)**wp.float32(wp.float32(2.0))*wp.sin(wp.float32(wp.pi)*(wp.float32(4.0)*t + wp.float32(1.6)))*wp.cos(wp.float32(wp.pi)*(wp.float32(-2.8)*t + wp.float32(4.0)*loc[0]))*wp.cos(wp.float32(wp.pi)*(wp.float32(-0.2)*t + wp.float32(wp.float32(2.0))*loc[1]))
            
            # B[0] = -wp.c_k_led*wp.c_k_led**2.0*(-8.0*wp.pi**2.0*wp.sin(wp.pi*(-2.8.*t + 4.0*loc[0]))*wp.cos(wp.pi*(-0.2.*t + 2.0*loc[1]))*wp.cos(wp.pi*(4.0*t + 1.6)) - 16.0*wp.pi**2.0*wp.sin(wp.pi*(-1.2.*t + 4.0*loc[0]))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.6*t + 2.0*loc[1]))) - wp.c_mu_led*wp.c_mu_led**2.0*(-8.0*wp.pi**2.0*wp.sin(wp.pi*(-2.8.*t + 4.0*loc[0]))*wp.cos(wp.pi*(-0.2.*t + 2.0*loc[1]))*wp.cos(wp.pi*(4.0*t + 1.6)) - 4.0*wp.pi**2.0*wp.sin(wp.pi*(-1.2.*t + 4.0*loc[0]))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.6*t + 2.0*loc[1]))) - wp.c_mu_led*wp.c_mu_led**2.0*(8.0*wp.pi**2.0*wp.sin(wp.pi*(-2.8.*t + 4.0*loc[0]))*wp.cos(wp.pi*(-0.2.*t + 2.0*loc[1]))*wp.cos(wp.pi*(4.0*t + 1.6)) - 16.0*wp.pi**2.0*wp.sin(wp.pi*(-1.2.*t + 4.0*loc[0]))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.6*t + 2.0*loc[1]))) + 12.8.*wp.pi**2.0*wp.sin(wp.pi*(-1.6*t + 2.0*loc[1]))*wp.sin(wp.pi*(-1.2.*t + 4.0*loc[0]))*wp.cos(wp.pi*(4.0*t - 0.4)) - 3.84*wp.pi**2.0*wp.sin(wp.pi*(-1.6*t + 2.0*loc[1]))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.2.*t + 4.0*loc[0])) - 20.0*wp.pi**2.0*wp.sin(wp.pi*(-1.2.*t + 4.0*loc[0]))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.6*t + 2.0*loc[1])) - 9.6*wp.pi**2.0*wp.cos(wp.pi*(-1.6*t + 2.0*loc[1]))*wp.cos(wp.pi*(-1.2.*t + 4.0*loc[0]))*wp.cos(wp.pi*(4.0*t - 0.4))
            # B[1] = -wp.c_k_led*wp.c_k_led**2.0*(-8.0*wp.pi**2.0*wp.sin(wp.pi*(-1.6*t + 2.0*loc[1]))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.2.*t + 4.0*loc[0])) - 4.0*wp.pi**2.0*wp.sin(wp.pi*(-0.2.*t + 2.0*loc[1]))*wp.cos(wp.pi*(-2.8.*t + 4.0*loc[0]))*wp.cos(wp.pi*(4.0*t + 1.6))) - wp.c_mu_led*wp.c_mu_led**2.0*(-8.0*wp.pi**2.0*wp.sin(wp.pi*(-1.6*t + 2.0*loc[1]))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.2.*t + 4.0*loc[0])) - 16.0*wp.pi**2.0*wp.sin(wp.pi*(-0.2.*t + 2.0*loc[1]))*wp.cos(wp.pi*(-2.8.*t + 4.0*loc[0]))*wp.cos(wp.pi*(4.0*t + 1.6))) + wp.c_mu_led*wp.c_mu_led**2.0*(-8.0*wp.pi**2.0*wp.sin(wp.pi*(-1.6*t + 2.0*loc[1]))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.2.*t + 4.0*loc[0])) + 4.0*wp.pi**2.0*wp.sin(wp.pi*(-0.2.*t + 2.0*loc[1]))*wp.cos(wp.pi*(-2.8.*t + 4.0*loc[0]))*wp.cos(wp.pi*(4.0*t + 1.6))) - 22.4.*wp.pi**2.0*wp.sin(wp.pi*(-2.8.*t + 4.0*loc[0]))*wp.sin(wp.pi*(-0.2.*t + 2.0*loc[1]))*wp.sin(wp.pi*(4.0*t + 1.6)) - 1.12*wp.pi**2.0*wp.sin(wp.pi*(-2.8.*t + 4.0*loc[0]))*wp.cos(wp.pi*(-0.2.*t + 2.0*loc[1]))*wp.cos(wp.pi*(4.0*t + 1.6)) - 23.88*wp.pi**2.0*wp.sin(wp.pi*(-0.2.*t + 2.0*loc[1]))*wp.cos(wp.pi*(-2.8.*t + 4.0*loc[0]))*wp.cos(wp.pi*(4.0*t + 1.6)) + 1.6*wp.pi**2.0*wp.sin(wp.pi*(4.0*t + 1.6))*wp.cos(wp.pi*(-2.8.*t + 4.0*loc[0]))*wp.cos(wp.pi*(-0.2.*t + 2.0*loc[1]))

            _U_num_tilde = _u_num_vec(0.)

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
