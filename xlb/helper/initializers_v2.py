from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.equilibrium import Equilibrium_LED
from xlb.operator.operator import Operator
import warp as wp

from typing import Any


def initialize_f_U_num_LED(f, grid, precision_policy, compute_backend):
    if f is None:
        U_0 = grid.create_field(cardinality=20, fill_value=0.0, dtype=precision_policy.compute_precision)
    
    equilibrium = Equilibrium_LED()

    if compute_backend == ComputeBackend.JAX:
        f = equilibrium(U_0)  # TODO

    elif compute_backend == ComputeBackend.WARP:
        U_0 = grid.create_field(cardinality=5, fill_value=0.0, dtype=precision_policy.compute_precision)
        # U_1 = grid.create_field(cardinality=5, fill_value=0.0, dtype=precision_policy.compute_precision)
        u_num_displ_0 = grid.create_field(cardinality=5, fill_value=0.0, dtype=precision_policy.compute_precision)
        u_num_displ_1 = grid.create_field(cardinality=5, fill_value=0.0, dtype=precision_policy.compute_precision)
        f = equilibrium(U_0, f)
    return f, U_0, u_num_displ_0, u_num_displ_1

class Initializer_LED(Operator):
    def __init__(self, velocity_set=None, precision_policy=None, compute_backend=None):
        super().__init__(velocity_set, precision_policy, compute_backend)
    
    def _construct_warp(self):
        _u_num_displ_vector_vec = wp.vec(2, dtype=self.compute_dtype)
        _U_vector_vec = wp.vec(5, dtype=self.compute_dtype)
        _f_vector_vec = wp.vec(20, dtype=self.compute_dtype)

        # Analytical functions to set initial f correctly (non trivial in contrast to fluid LBM)
        @wp.func
        def u_num_displ_func(x: Any, y: Any, t: Any):
            _u_num_displ = _u_num_displ_vector_vec()
            _4 = wp.float64(4.)
            _03 = wp.float64(0.3)
            _2 = wp.float64(2.)
            _08 = wp.float64(0.8)
            _01 = wp.float64(0.1)
            _04 = wp.float64(0.4)
            _07 = wp.float64(0.7)
            pi = wp.float64(wp.pi)
            _u_num_displ[0] = wp.sin(_4*pi*(x-_03*t)) * wp.cos(_2*pi*(y-_08*t)) * wp.sin(_4*pi*(t-_01))
            _u_num_displ[1] = wp.cos(_4*pi*(x-_07*t)) * wp.sin(_2*pi*(y-_01*t)) * wp.cos(_4*pi*(t+_04))
            return _u_num_displ

        # This is the U_num_tilde, not the displacement.
        @wp.func
        def U_func(x: Any, y: Any, t: Any):
            _U = _U_vector_vec()
            _1p6 = wp.float64(1.6)
            _2p0 = wp.float64(2.0)
            _1p2 = wp.float64(1.2)
            _4p0 = wp.float64(4.0)
            _0p4 = wp.float64(0.4)
            _0p2 = wp.float64(0.2)
            _2p8 = wp.float64(2.8)
            pi = wp.float64(wp.pi)
            _U[0] = _1p6*pi*wp.sin(pi*(-_1p6*t + _2p0*y))*wp.sin(pi*(-_1p2*t + _4p0*x))*wp.sin(pi*(_4p0*t - _0p4)) + _4p0*pi*wp.sin(pi*(-_1p2*t + _4p0*x))*wp.cos(pi*(-_1p6*t + _2p0*y))*wp.cos(pi*(_4p0*t - _0p4)) - _1p2*pi*wp.sin(pi*(_4p0*t - _0p4))*wp.cos(pi*(-_1p6*t + _2p0*y))*wp.cos(pi*(-_1p2*t + _4p0*x))
            _U[1] = _2p8*pi*wp.sin(pi*(-_2p8*t + _4p0*x))*wp.sin(pi*(-_0p2*t + _2p0*y))*wp.cos(pi*(_4p0*t + _1p6)) - _4p0*pi*wp.sin(pi*(-_0p2*t + _2p0*y))*wp.sin(pi*(_4p0*t + _1p6))*wp.cos(pi*(-_2p8*t + _4p0*x)) - _0p2*pi*wp.cos(pi*(-_2p8*t + _4p0*x))*wp.cos(pi*(-_0p2*t + _2p0*y))*wp.cos(pi*(_4p0*t + _1p6))
            _U[2] = -wp.float64(wp.c_k_led)*(_4p0*pi*wp.sin(pi*(_4p0*t - _0p4))*wp.cos(pi*(-_1p6*t + _2p0*y))*wp.cos(pi*(-_1p2*t + _4p0*x))+_2p0*pi*wp.cos(pi*(-_2p8*t + _4p0*x))*wp.cos(pi*(-_0p2*t + _2p0*y))*wp.cos(pi*(_4p0*t + _1p6)))
            _U[3] = -wp.float64(wp.c_mu_led)*(_4p0*pi*wp.sin(pi*(_4p0*t - _0p4))*wp.cos(pi*(-_1p6*t + _2p0*y))*wp.cos(pi*(-_1p2*t + _4p0*x))-_2p0*pi*wp.cos(pi*(-_2p8*t + _4p0*x))*wp.cos(pi*(-_0p2*t + _2p0*y))*wp.cos(pi*(_4p0*t + _1p6)))
            _U[4] = -wp.float64(wp.c_mu_led)*(-_2p0*pi*wp.sin(pi*(-_1p6*t + _2p0*y))*wp.sin(pi*(-_1p2*t + _4p0*x))*wp.sin(pi*(_4p0*t - _0p4))-_4p0*pi*wp.sin(pi*(-_2p8*t + _4p0*x))*wp.sin(pi*(-_0p2*t + _2p0*y))*wp.cos(pi*(_4p0*t + _1p6)))
            return _U
        
        @wp.func
        def dUdx_func(x: Any, y: Any, t: Any):
            _dUdx = _U_vector_vec()
            _8p0 = wp.float64(8.0)
            _2p0 = wp.float64(2.0)
            _6p4 = wp.float64(6.4)
            _16p0 = wp.float64(16.0)
            _4p0 = wp.float64(4.0)
            _4p8 = wp.float64(4.8)
            _2p8 = wp.float64(2.8)
            _1p6 = wp.float64(1.6)
            _0p4 = wp.float64(0.4)
            _0p2 = wp.float64(0.2)
            _0p8 = wp.float64(0.8)
            _1p2 = wp.float64(1.2)
            _11p2 = wp.float64(11.2)
            pi = wp.float64(wp.pi)
            _c_k = wp.float64(wp.c_k_led)
            _c_mu = wp.float64(wp.c_mu_led)
            _dUdx[0] = _6p4*pi**_2p0*wp.sin(pi*(-_1p6*t + _2p0*y))*wp.sin(pi*(_4p0*t - _0p4))*wp.cos(pi*(-_1p2*t + _4p0*x)) + _4p8*pi**_2p0*wp.sin(pi*(-_1p2*t + _4p0*x))*wp.sin(pi*(_4p0*t - _0p4))*wp.cos(pi*(-_1p6*t + _2p0*y)) + _16p0*pi**_2p0*wp.cos(pi*(-_1p6*t + _2p0*y))*wp.cos(pi*(-_1p2*t + _4p0*x))*wp.cos(pi*(_4p0*t - _0p4))
            _dUdx[1] = _16p0*pi**_2p0*wp.sin(pi*(-_2p8*t + _4p0*x))*wp.sin(pi*(-_0p2*t + _2p0*y))*wp.sin(pi*(_4p0*t + _1p6)) + _0p8*pi**_2p0*wp.sin(pi*(-_2p8*t + _4p0*x))*wp.cos(pi*(-_0p2*t + _2p0*y))*wp.cos(pi*(_4p0*t + _1p6)) + _11p2*pi**_2p0*wp.sin(pi*(-_0p2*t + _2p0*y))*wp.cos(pi*(-_2p8*t + _4p0*x))*wp.cos(pi*(_4p0*t + _1p6))
            _dUdx[2] = -_c_k*(-_8p0*pi**_2p0*wp.sin(pi*(-_2p8*t + _4p0*x))*wp.cos(pi*(-_0p2*t + _2p0*y))*wp.cos(pi*(_4p0*t + _1p6)) - _16p0*pi**_2p0*wp.sin(pi*(-_1p2*t + _4p0*x))*wp.sin(pi*(_4p0*t - _0p4))*wp.cos(pi*(-_1p6*t + _2p0*y)))
            _dUdx[3] = -_c_mu*(_8p0*pi**_2p0*wp.sin(pi*(-_2p8*t + _4p0*x))*wp.cos(pi*(-_0p2*t + _2p0*y))*wp.cos(pi*(_4p0*t + _1p6)) - _16p0*pi**_2p0*wp.sin(pi*(-_1p2*t + _4p0*x))*wp.sin(pi*(_4p0*t - _0p4))*wp.cos(pi*(-_1p6*t + _2p0*y))) 
            _dUdx[4] = -_c_mu*(-_8p0*pi**_2p0*wp.sin(pi*(-_1p6*t + _2p0*y))*wp.sin(pi*(_4p0*t - _0p4))*wp.cos(pi*(-_1p2*t + _4p0*x)) - _16p0*pi**_2p0*wp.sin(pi*(-_0p2*t + _2p0*y))*wp.cos(pi*(-_2p8*t + _4p0*x))*wp.cos(pi*(_4p0*t + _1p6)))
            return _dUdx
        
        @wp.func
        def dUdy_func(x: Any, y: Any, t: Any):
            _dUdy = _U_vector_vec()
            _8p0 = wp.float64(8.0)
            _2p0 = wp.float64(2.0)
            _5p6 = wp.float64(5.6)
            _2p4 = wp.float64(2.4)
            _3p2 = wp.float64(3.2)
            _4p0 = wp.float64(4.0)
            _2p8 = wp.float64(2.8)
            _1p6 = wp.float64(1.6)
            _0p4 = wp.float64(0.4)
            _0p2 = wp.float64(0.2)
            _1p2 = wp.float64(1.2)
            pi = wp.float64(wp.pi)
            _c_k = wp.float64(wp.c_k_led)
            _c_mu = wp.float64(wp.c_mu_led)
            _dUdy[0] = -_8p0*pi**_2p0*wp.sin(pi*(-_1p6*t + _2p0*y))*wp.sin(pi*(-_1p2*t + _4p0*x))*wp.cos(pi*(_4p0*t - _0p4)) + _2p4*pi**_2p0*wp.sin(pi*(-_1p6*t + _2p0*y))*wp.sin(pi*(_4p0*t - _0p4))*wp.cos(pi*(-_1p2*t + _4p0*x)) + _3p2*pi**_2p0*wp.sin(pi*(-_1p2*t + _4p0*x))*wp.sin(pi*(_4p0*t - _0p4))*wp.cos(pi*(-_1p6*t + _2p0*y))
            _dUdy[1] = _5p6*pi**_2p0*wp.sin(pi*(-_2p8*t + _4p0*x))*wp.cos(pi*(-_0p2*t + _2p0*y))*wp.cos(pi*(_4p0*t + _1p6)) + _0p4*pi**_2p0*wp.sin(pi*(-_0p2*t + _2p0*y))*wp.cos(pi*(-_2p8*t + _4p0*x))*wp.cos(pi*(_4p0*t + _1p6)) - _8p0*pi**_2p0*wp.sin(pi*(_4p0*t + _1p6))*wp.cos(pi*(-_2p8*t + _4p0*x))*wp.cos(pi*(-_0p2*t + _2p0*y))
            _dUdy[2] = -_c_k*(-_8p0*pi**_2p0*wp.sin(pi*(-_1p6*t + _2p0*y))*wp.sin(pi*(_4p0*t - _0p4))*wp.cos(pi*(-_1p2*t + _4p0*x)) - _4p0*pi**_2p0*wp.sin(pi*(-_0p2*t + _2p0*y))*wp.cos(pi*(-_2p8*t + _4p0*x))*wp.cos(pi*(_4p0*t + _1p6)))
            _dUdy[3] = -_c_mu*(-_8p0*pi**_2p0*wp.sin(pi*(-_1p6*t + _2p0*y))*wp.sin(pi*(_4p0*t - _0p4))*wp.cos(pi*(-_1p2*t + _4p0*x)) + _4p0*pi**_2p0*wp.sin(pi*(-_0p2*t + _2p0*y))*wp.cos(pi*(-_2p8*t + _4p0*x))*wp.cos(pi*(_4p0*t + _1p6)))
            _dUdy[4] = -_c_mu*(-_8p0*pi**_2p0*wp.sin(pi*(-_2p8*t + _4p0*x))*wp.cos(pi*(-_0p2*t + _2p0*y))*wp.cos(pi*(_4p0*t + _1p6)) - _4p0*pi**_2p0*wp.sin(pi*(-_1p2*t + _4p0*x))*wp.sin(pi*(_4p0*t - _0p4))*wp.cos(pi*(-_1p6*t + _2p0*y)))
            return _dUdy

        @wp.func
        def B_func(x: Any, y: Any, t: Any):
            _B = _U_vector_vec()
            # # _B[0] = wp.pi**2.*wp.cos(wp.pi*(t*(3.0/1.0e+1)-x)*4.0)*wp.cos(wp.pi*(t*(4.0/5.0)-y)*2.0)*wp.cos(wp.pi*(t-1.0/1.0e+1)*4.0)*(-4.8e+1/5.0)-wp.pi**2.*wp.cos(wp.pi*(t/1.0e+1-y)*2.0)*wp.sin(wp.pi*(t*(7.0/1.0e+1)-x)*4.0)*wp.cos(wp.pi*(t+2.0/5.0)*4.0)*(4.4e+1/5.0)+wp.pi**2.*wp.cos(wp.pi*(t*(3.0/1.0e+1)-x)*4.0)*wp.sin(wp.pi*(t*(4.0/5.0)-y)*2.0)*wp.sin(wp.pi*(t-1.0/1.0e+1)*4.0)*(9.6e+1/2.5e+1)-wp.pi**2.*wp.cos(wp.pi*(t*(4.0/5.0)-y)*2.0)*wp.sin(wp.pi*(t*(3.0/1.0e+1)-x)*4.0)*wp.sin(wp.pi*(t-1.0/1.0e+1)*4.0)*(2.8e+1/5.0)+wp.pi**2.*wp.sin(wp.pi*(t*(3.0/1.0e+1)-x)*4.0)*wp.sin(wp.pi*(t*(4.0/5.0)-y)*2.0)*wp.cos(wp.pi*(t-1.0/1.0e+1)*4.0)*(6.4e+1/5.0)
            # # _B[1] = wp.pi**2.*wp.cos(wp.pi*(t*(7.0/1.0e+1)-x)*4.0)*wp.cos(wp.pi*(t/1.0e+1-y)*2.0)*wp.sin(wp.pi*(t+2.0/5.0)*4.0)*(8.0/5.0)+wp.pi**2.*wp.cos(wp.pi*(t*(7.0/1.0e+1)-x)*4.0)*wp.sin(wp.pi*(t/1.0e+1-y)*2.0)*wp.cos(wp.pi*(t+2.0/5.0)*4.0)*(2.87e+2/2.5e+1)+wp.pi**2.*wp.cos(wp.pi*(t/1.0e+1-y)*2.0)*wp.sin(wp.pi*(t*(7.0/1.0e+1)-x)*4.0)*wp.cos(wp.pi*(t+2.0/5.0)*4.0)*(2.8e+1/2.5e+1)-wp.pi**2.*wp.cos(wp.pi*(t*(3.0/1.0e+1)-x)*4.0)*wp.sin(wp.pi*(t*(4.0/5.0)-y)*2.0)*wp.sin(wp.pi*(t-1.0/1.0e+1)*4.0)*(4.4e+1/5.0)-wp.pi**2.*wp.sin(wp.pi*(t*(7.0/1.0e+1)-x)*4.0)*wp.sin(wp.pi*(t/1.0e+1-y)*2.0)*wp.sin(wp.pi*(t+2.0/5.0)*4.0)*(1.12e+2/5.0)
            
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
            _B[0] = _c_mu*_c_mu*(_8p0*pi**_2p0*wp.cos(_2p0*pi*(t/_10p0 - y))*wp.sin(_4p0*pi*((_7p0*t)/_10p0 - x))*wp.cos(_4p0*pi*(t + _2p0/_5p0)) - _16p0*pi**_2p0*wp.cos(_2p0*pi*((_4p0*t)/_5p0 - y))*wp.sin(_4p0*pi*((_3p0*t)/_10p0 - x))*wp.sin(_4p0*pi*(t - _1p0/_10p0))) - _c_mu*_c_mu*(_8p0*pi**_2p0*wp.cos(_2p0*pi*(t/_10p0 - y))*wp.sin(_4p0*pi*((_7p0*t)/_10p0 - x))*wp.cos(_4p0*pi*(t + _2p0/_5p0)) + _4p0*pi**_2p0*wp.cos(_2p0*pi*((_4p0*t)/_5p0 - y))*wp.sin(_4p0*pi*((_3p0*t)/_10p0 - x))*wp.sin(_4p0*pi*(t - _1p0/_10p0))) - _c_k*_c_k*(_8p0*pi**_2p0*wp.cos(_2p0*pi*(t/_10p0 - y))*wp.sin(_4p0*pi*((_7p0*t)/_10p0 - x))*wp.cos(_4p0*pi*(t + _2p0/_5p0)) + _16p0*pi**_2p0*wp.cos(_2p0*pi*((_4p0*t)/_5p0 - y))*wp.sin(_4p0*pi*((_3p0*t)/_10p0 - x))*wp.sin(_4p0*pi*(t - _1p0/_10p0))) - (_48p0*pi**_2p0*wp.cos(_4p0*pi*((_3p0*t)/_10p0 - x))*wp.cos(_2p0*pi*((_4p0*t)/_5p0 - y))*wp.cos(_4p0*pi*(t - _1p0/_10p0)))/_5p0 + (_96p0*pi**_2p0*wp.cos(_4p0*pi*((_3p0*t)/_10p0 - x))*wp.sin(_2p0*pi*((_4p0*t)/_5p0 - y))*wp.sin(_4p0*pi*(t - _1p0/_10p0)))/_25p0 + _20p0*pi**_2p0*wp.cos(_2p0*pi*((_4p0*t)/_5p0 - y))*wp.sin(_4p0*pi*((_3p0*t)/_10p0 - x))*wp.sin(_4p0*pi*(t - _1p0/_10p0)) + (_64p0*pi**_2p0*wp.sin(_4p0*pi*((_3p0*t)/_10p0 - x))*wp.sin(_2p0*pi*((_4p0*t)/_5p0 - y))*wp.cos(_4p0*pi*(t - _1p0/_10p0)))/_5p0
            _B[1] = (_8p0*pi**_2p0*wp.cos(_4p0*pi*((_7p0*t)/_10p0 - x))*wp.cos(_2p0*pi*(t/_10p0 - y))*wp.sin(_4p0*pi*(t + _2p0/_5p0)))/_5p0 - _c_mu*_c_mu*(_4p0*pi**_2p0*wp.cos(_4p0*pi*((_7p0*t)/_10p0 - x))*wp.sin(_2p0*pi*(t/_10p0 - y))*wp.cos(_4p0*pi*(t + _2p0/_5p0)) - _8p0*pi**_2p0*wp.cos(_4p0*pi*((_3p0*t)/_10p0 - x))*wp.sin(_2p0*pi*((_4p0*t)/_5p0 - y))*wp.sin(_4p0*pi*(t - _1p0/_10p0))) - _c_mu*_c_mu*(_16p0*pi**_2p0*wp.cos(_4p0*pi*((_7p0*t)/_10p0 - x))*wp.sin(_2p0*pi*(t/_10p0 - y))*wp.cos(_4p0*pi*(t + _2p0/_5p0)) + _8p0*pi**_2p0*wp.cos(_4p0*pi*((_3p0*t)/_10p0 - x))*wp.sin(_2p0*pi*((_4p0*t)/_5p0 - y))*wp.sin(_4p0*pi*(t - _1p0/_10p0))) - _c_k*_c_k*(_4p0*pi**_2p0*wp.cos(_4p0*pi*((_7p0*t)/_10p0 - x))*wp.sin(_2p0*pi*(t/_10p0 - y))*wp.cos(_4p0*pi*(t + _2p0/_5p0)) + _8p0*pi**_2p0*wp.cos(_4p0*pi*((_3p0*t)/_10p0 - x))*wp.sin(_2p0*pi*((_4p0*t)/_5p0 - y))*wp.sin(_4p0*pi*(t - _1p0/_10p0))) + (_597p0*pi**_2p0*wp.cos(_4p0*pi*((_7p0*t)/_10p0 - x))*wp.sin(_2p0*pi*(t/_10p0 - y))*wp.cos(_4p0*pi*(t + _2p0/_5p0)))/_25p0 + (_28p0*pi**_2p0*wp.cos(_2p0*pi*(t/_10p0 - y))*wp.sin(_4p0*pi*((_7p0*t)/_10p0 - x))*wp.cos(_4p0*pi*(t + _2p0/_5p0)))/_25p0 - (_112p0*pi**_2p0*wp.sin(_4p0*pi*((_7p0*t)/_10p0 - x))*wp.sin(_2p0*pi*(t/_10p0 - y))*wp.sin(_4p0*pi*(t + _2p0/_5p0)))/_5p0
            # _B[0] = -wp.c_k_led**2.0*(-8.0*wp.pi**2.0*wp.sin(wp.pi*(-2.8*t + 4.0*x))*wp.cos(wp.pi*(-0.2*t + 2.0*y))*wp.cos(wp.pi*(4.0*t + 1.6)) - 16.0*wp.pi**2.0*wp.sin(wp.pi*(-1.2*t + 4.0*x))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.6*t + 2.0*y))) - wp.c_mu_led**2.0*(-8.0*wp.pi**2.0*wp.sin(wp.pi*(-2.8*t + 4.0*x))*wp.cos(wp.pi*(-0.2*t + 2.0*y))*wp.cos(wp.pi*(4.0*t + 1.6)) - 4.0*wp.pi**2.0*wp.sin(wp.pi*(-1.2*t + 4.0*x))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.6*t + 2.0*y))) - wp.c_mu_led**2.0*(8.0*wp.pi**2.0*wp.sin(wp.pi*(-2.8*t + 4.0*x))*wp.cos(wp.pi*(-0.2*t + 2.0*y))*wp.cos(wp.pi*(4.0*t + 1.6)) - 16.0*wp.pi**2.0*wp.sin(wp.pi*(-1.2*t + 4.0*x))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.6*t + 2.0*y))) + 12.8*wp.pi**2.0*wp.sin(wp.pi*(-1.6*t + 2.0*y))*wp.sin(wp.pi*(-1.2*t + 4.0*x))*wp.cos(wp.pi*(4.0*t - 0.4)) - 3.84*wp.pi**2.0*wp.sin(wp.pi*(-1.6*t + 2.0*y))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.2*t + 4.0*x)) - 20.0*wp.pi**2.0*wp.sin(wp.pi*(-1.2*t + 4.0*x))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.6*t + 2.0*y)) - 9.6*wp.pi**2.0*wp.cos(wp.pi*(-1.6*t + 2.0*y))*wp.cos(wp.pi*(-1.2*t + 4.0*x))*wp.cos(wp.pi*(4.0*t - 0.4))
            # _B[1] = -wp.c_k_led**2.0*(-8.0*wp.pi**2.0*wp.sin(wp.pi*(-1.6*t + 2.0*y))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.2*t + 4.0*x)) - 4.0*wp.pi**2.0*wp.sin(wp.pi*(-0.2*t + 2.0*y))*wp.cos(wp.pi*(-2.8*t + 4.0*x))*wp.cos(wp.pi*(4.0*t + 1.6))) - wp.c_mu_led**2.0*(-8.0*wp.pi**2.0*wp.sin(wp.pi*(-1.6*t + 2.0*y))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.2*t + 4.0*x)) - 16.0*wp.pi**2.0*wp.sin(wp.pi*(-0.2*t + 2.0*y))*wp.cos(wp.pi*(-2.8*t + 4.0*x))*wp.cos(wp.pi*(4.0*t + 1.6))) + wp.c_mu_led**2.0*(-8.0*wp.pi**2.0*wp.sin(wp.pi*(-1.6*t + 2.0*y))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.2*t + 4.0*x)) + 4.0*wp.pi**2.0*wp.sin(wp.pi*(-0.2*t + 2.0*y))*wp.cos(wp.pi*(-2.8*t + 4.0*x))*wp.cos(wp.pi*(4.0*t + 1.6))) - 22.4*wp.pi**2.0*wp.sin(wp.pi*(-2.8*t + 4.0*x))*wp.sin(wp.pi*(-0.2*t + 2.0*y))*wp.sin(wp.pi*(4.0*t + 1.6)) - 1.12*wp.pi**2.0*wp.sin(wp.pi*(-2.8*t + 4.0*x))*wp.cos(wp.pi*(-0.2*t + 2.0*y))*wp.cos(wp.pi*(4.0*t + 1.6)) - 23.88*wp.pi**2.0*wp.sin(wp.pi*(-0.2*t + 2.0*y))*wp.cos(wp.pi*(-2.8*t + 4.0*x))*wp.cos(wp.pi*(4.0*t + 1.6)) + 1.6*wp.pi**2.0*wp.sin(wp.pi*(4.0*t + 1.6))*wp.cos(wp.pi*(-2.8*t + 4.0*x))*wp.cos(wp.pi*(-0.2*t + 2.0*y))
            _B[2] = wp.float64(0.0)
            _B[3] = wp.float64(0.0)
            _B[4] = wp.float64(0.0)
            return _B
        
        @wp.func
        def phi_x_lb_func(_U: Any):
            _phi_x = _U_vector_vec()
            _phi_x[0] = (wp.c_k_led*_U[2]+wp.c_mu_led*_U[3])/wp.c_led
            _phi_x[1] = wp.c_mu_led*_U[4]/wp.c_led
            _phi_x[2] = wp.c_k_led*_U[0]/wp.c_led
            _phi_x[3] = wp.c_mu_led*_U[0]/wp.c_led
            _phi_x[4] = wp.c_mu_led*_U[1]/wp.c_led
            return _phi_x

        @wp.func
        def phi_y_lb_func(_U: Any):
            _phi_y = _U_vector_vec()
            _c = wp.float64(wp.c_led)
            _c_k = wp.float64(wp.c_k_led)
            _c_mu = wp.float64(wp.c_mu_led)
            _phi_y[0] = _c_mu*_U[4]/_c
            _phi_y[1] = (_c_k*_U[2]-_c_mu*_U[3])/_c
            _phi_y[2] = _c_k*_U[1]/_c
            _phi_y[3] = -_c_mu*_U[1]/_c
            _phi_y[4] = _c_mu*_U[0]/_c
            return _phi_y
        
            
        #2nd order IC with relatively high error due to precision issues
        @wp.kernel
        def initial_conditions_v2_kernel(f: wp.array4d(dtype=Any), U_num_tilde: wp.array4d(dtype=Any), u_num_displ_out: wp.array4d(dtype=Any)):
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)
            x = (self.compute_dtype(index[0]) + self.compute_dtype(0.5)) * wp.float64(wp.delta_x_led)
            y = (self.compute_dtype(index[1]) + self.compute_dtype(0.5)) * wp.float64(wp.delta_x_led)
            t = self.compute_dtype(0.0)
            #evaluate relevant properties from analytical solutions
            _u_num_displ = u_num_displ_func(x, y, t)
            _U = U_func(x, y, t)
            _B = B_func(x, y, t)*wp.delta_t_led

            _dUdx = dUdx_func(x, y, t)*wp.delta_x_led
            _dUdy = dUdy_func(x, y, t)*wp.delta_x_led
            _f = _f_vector_vec()
            _f0 = wp.float64(0.25)*((_U+wp.float64(wp.float64(2.0))*phi_x_lb_func(_U)) + wp.float64(0.5)*(-(_B+wp.float64(wp.float64(2.0))*phi_x_lb_func(_B)) - _dUdx - phi_x_lb_func(_dUdx) + phi_y_lb_func(_dUdy) + wp.float64(wp.float64(2.0))*(phi_x_lb_func(phi_x_lb_func(_dUdx)+phi_y_lb_func(_dUdy)))))
            _f1 = wp.float64(0.25)*((_U+wp.float64(wp.float64(2.0))*phi_y_lb_func(_U)) + wp.float64(0.5)*(-(_B+wp.float64(wp.float64(2.0))*phi_y_lb_func(_B)) - _dUdy + phi_x_lb_func(_dUdx) - phi_y_lb_func(_dUdy) + wp.float64(wp.float64(2.0))*(phi_y_lb_func(phi_x_lb_func(_dUdx)+phi_y_lb_func(_dUdy)))))
            _f2 = wp.float64(0.25)*((_U-wp.float64(wp.float64(2.0))*phi_x_lb_func(_U)) + wp.float64(0.5)*(-(_B-wp.float64(wp.float64(2.0))*phi_x_lb_func(_B)) + _dUdx - phi_x_lb_func(_dUdx) + phi_y_lb_func(_dUdy) - wp.float64(wp.float64(2.0))*(phi_x_lb_func(phi_x_lb_func(_dUdx)+phi_y_lb_func(_dUdy)))))
            _f3 = wp.float64(0.25)*((_U-wp.float64(wp.float64(2.0))*phi_y_lb_func(_U)) + wp.float64(0.5)*(-(_B-wp.float64(wp.float64(2.0))*phi_y_lb_func(_B)) + _dUdy + phi_x_lb_func(_dUdx) - phi_y_lb_func(_dUdy) - wp.float64(wp.float64(2.0))*(phi_y_lb_func(phi_x_lb_func(_dUdx)+phi_y_lb_func(_dUdy)))))
            
            # # dir x
            # _f0 = 0.25*((_U+2.0*phi_x_lb(_U)) + 0.5*(-(_B+2.0*phi_x_lb(_B)) - _dUdx - phi_x_lb(_dUdx) + phi_y_lb(_dUdy) + 2.0*(phi_x_lb(phi_x_lb(_dUdx)+phi_y_lb(_dUdy)))))
            # #  dir y
            # _f1 = 0.25*((_U+2.0*phi_y_lb(_U)) + 0.5*(-(_B+2.0*phi_y_lb(_B)) - _dUdy + phi_x_lb(_dUdx) - phi_y_lb(_dUdy) + 2.0*(phi_y_lb(phi_x_lb(_dUdx)+phi_y_lb(_dUdy)))))
            # # dir -x
            # _f2 = 0.25*((_U-2.0*phi_x_lb(_U)) + 0.5*(-(_B-2.0*phi_x_lb(_B)) + _dUdx - phi_x_lb(_dUdx) + phi_y_lb(_dUdy) - 2.0*(phi_x_lb(phi_x_lb(_dUdx)+phi_y_lb(_dUdy)))))
            # # dir -y
            # _f3 = 0.25*((_U-2.0*phi_y_lb(_U)) + 0.5*(-(_B-2.0*phi_y_lb(_B)) + _dUdy + phi_x_lb(_dUdx) - phi_y_lb(_dUdy) - 2.0*(phi_y_lb(phi_x_lb(_dUdx)+phi_y_lb(_dUdy)))))
            for s in range(5):
                _f[s] = _f0[s]
                _f[s + 5] = _f1[s]
                _f[s + 10] = _f2[s]
                _f[s + 15] = _f3[s]
            
            for l in range(20):
                f[l, index[0], index[1], index[2]] = self.store_dtype(_f[l])
            for l in range(5):
                U_num_tilde[l, index[0], index[1], index[2]] = self.store_dtype(_U[l])
            for l in range(2):
                u_num_displ_out[l, index[0], index[1], index[2]] = self.store_dtype(_u_num_displ[l])
        
        return None, initial_conditions_v2_kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f, U_num_tilde, u_num_displ_out):
        wp.launch(
            self.warp_kernel,
            inputs=[f, U_num_tilde, u_num_displ_out],
            dim=U_num_tilde.shape[1:],
        )
        return f, U_num_tilde, u_num_displ_out
   